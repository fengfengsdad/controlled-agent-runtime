"""Near-duplicate suppression and diversity-aware selection.

Overlapping chunk windows mean the same sentences legitimately appear in several
chunks, so relevance ranking alone tends to fill the context window with
restatements of one passage. Two controls counter that: shingle-based
near-duplicate removal before reranking, and Maximal Marginal Relevance with a
per-source cap during final selection.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from agent_runtime.models.schemas import Citation
from agent_runtime.rag.text import shingles


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def suppress_near_duplicates(
    citations: Sequence[Citation],
    threshold: float,
    shingle_size: int = 5,
) -> Tuple[List[Citation], int]:
    """Keep the highest-ranked member of each near-duplicate group.

    Input order is assumed to be best-first, so the survivor is the one the
    upstream ranking already preferred.
    """
    kept: List[Citation] = []
    kept_shingles: List[set] = []
    for citation in citations:
        fingerprint = shingles(citation.text, shingle_size)
        if any(jaccard(fingerprint, seen) >= threshold for seen in kept_shingles):
            continue
        kept.append(citation)
        kept_shingles.append(fingerprint)
    return kept, len(citations) - len(kept)


def _normalise(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lowest, highest = min(values), max(values)
    span = highest - lowest
    if span <= 0:
        return [1.0] * len(values)
    return [(value - lowest) / span for value in values]


def mmr_select(
    citations: Sequence[Citation],
    top_k: int,
    lambda_: float,
    similarity: Callable[[Citation, Citation], float],
    per_source_cap: Optional[int] = None,
) -> Tuple[List[Citation], int]:
    """Greedy MMR selection, returning the picks and how many the cap blocked.

    Relevance is min-max normalised across the candidate set so it is comparable
    with the similarity term regardless of which reranker produced the scores.
    """
    if not citations or top_k <= 0:
        return [], 0

    relevance = dict(
        zip(
            (c.chunk_id for c in citations),
            _normalise([c.score for c in citations]),
        )
    )
    remaining = list(citations)
    selected: List[Citation] = []
    per_source: Dict[str, int] = {}
    capped = 0

    while remaining and len(selected) < top_k:
        best: Optional[Citation] = None
        best_value = float("-inf")
        for candidate in remaining:
            if (
                per_source_cap is not None
                and per_source.get(candidate.source, 0) >= per_source_cap
            ):
                continue
            if selected:
                redundancy = max(similarity(candidate, chosen) for chosen in selected)
            else:
                redundancy = 0.0
            value = lambda_ * relevance[candidate.chunk_id] - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_value = value
                best = candidate
        if best is None:
            # Every remaining candidate is blocked by the per-source cap.
            capped += len(remaining)
            break
        remaining.remove(best)
        selected.append(best)
        per_source[best.source] = per_source.get(best.source, 0) + 1

    return selected, capped


def make_vector_similarity(
    vectors: Dict[str, Sequence[float]],
    shingle_size: int = 5,
) -> Callable[[Citation, Citation], float]:
    """Cosine similarity over chunk embeddings, falling back to text shingles.

    Embeddings are the better redundancy signal, but an index written before a
    schema change may not have them for every chunk, so the lexical fallback
    keeps selection working instead of silently treating everything as unique.
    """

    def similarity(left: Citation, right: Citation) -> float:
        left_vec = vectors.get(left.chunk_id)
        right_vec = vectors.get(right.chunk_id)
        if left_vec and right_vec and len(left_vec) == len(right_vec):
            return sum(a * b for a, b in zip(left_vec, right_vec))
        return jaccard(
            shingles(left.text, shingle_size), shingles(right.text, shingle_size)
        )

    return similarity
