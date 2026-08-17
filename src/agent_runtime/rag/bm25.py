"""Sparse BM25 (Okapi) retrieval over indexed chunks.

Hand-rolled instead of pulled from a dependency for two reasons: offline CI must
stay deterministic without network installs, and the index can be rebuilt
in-process from the chunk text already persisted in the vector index, so there is
no second artefact that can drift out of sync with it.

At corpus sizes beyond this MVP the inverted index belongs in OpenSearch or
Elasticsearch, which own incremental updates, sharding, and persistence.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional, Sequence


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._term_freq: dict[str, Counter] = {}
        self._doc_len: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._postings: dict[str, list[str]] = {}
        self._avg_doc_len = 0.0

    @property
    def size(self) -> int:
        return len(self._doc_len)

    def build(self, documents: Sequence[tuple[str, list[str]]]) -> None:
        """documents: list of (chunk_id, tokens)."""
        self._term_freq.clear()
        self._doc_len.clear()
        self._doc_freq.clear()
        self._postings.clear()
        for chunk_id, tokens in documents:
            freq = Counter(tokens)
            self._term_freq[chunk_id] = freq
            self._doc_len[chunk_id] = len(tokens)
            for term in freq:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
                self._postings.setdefault(term, []).append(chunk_id)
        total_len = sum(self._doc_len.values())
        self._avg_doc_len = total_len / len(self._doc_len) if self._doc_len else 0.0

    def _idf(self, term: str) -> float:
        doc_freq = self._doc_freq.get(term, 0)
        if doc_freq == 0:
            return 0.0
        corpus_size = len(self._doc_len)
        # The trailing +1 keeps IDF non-negative for terms present in most
        # documents, which otherwise drag relevant chunks below irrelevant ones.
        return math.log((corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(
        self,
        query_tokens: Sequence[str],
        top_n: int,
        allowed: Optional[Iterable[str]] = None,
    ) -> list[tuple[str, float]]:
        if not query_tokens or not self._doc_len:
            return []
        allowed_set = set(allowed) if allowed is not None else None
        scores: dict[str, float] = {}
        for term in set(query_tokens):
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for chunk_id in self._postings.get(term, ()):
                if allowed_set is not None and chunk_id not in allowed_set:
                    continue
                term_count = self._term_freq[chunk_id][term]
                length_norm = (
                    self._doc_len[chunk_id] / self._avg_doc_len if self._avg_doc_len else 1.0
                )
                denominator = term_count + self.k1 * (1 - self.b + self.b * length_norm)
                if denominator <= 0:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                    term_count * (self.k1 + 1)
                ) / denominator
        # Tie-break on chunk_id so identical scores rank identically across runs.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_n]
