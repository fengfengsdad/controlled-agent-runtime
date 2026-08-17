"""Multi-stage hybrid retrieval.

Pipeline: structured pre-filter -> parallel sparse/dense recall at candidate_k
-> reciprocal rank fusion -> near-duplicate suppression -> rerank -> MMR
selection with a per-source cap -> top_k.

Vector-only and BM25-only modes, and a rerank on/off switch, are kept so each
stage can be A/B compared against a baseline rather than assumed to help.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from agent_runtime.config import settings
from agent_runtime.models.schemas import Citation, RetrievalFilter, RetrievalTrace
from agent_runtime.rag.bm25 import Bm25Index
from agent_runtime.rag.diversity import (
    make_vector_similarity,
    mmr_select,
    suppress_near_duplicates,
)
from agent_runtime.rag.fusion import FusedHit, reciprocal_rank_fusion
from agent_runtime.rag.rerank import NONE, Reranker, RerankerUnavailable, build_reranker
from agent_runtime.rag.store import VectorStore
from agent_runtime.rag.text import tokenize

VECTOR = "vector"
BM25 = "bm25"
HYBRID = "hybrid"
MODES = (VECTOR, BM25, HYBRID)

# RRF scores are ~1/60 scale, so keep enough precision to order them.
SCORE_PRECISION = 6


@dataclass
class RetrievalResult:
    citations: list
    trace: RetrievalTrace


class HybridRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self._bm25: Optional[Bm25Index] = None
        self._bm25_signature: Optional[tuple] = None
        self._reranker: Optional[Reranker] = None
        self._reranker_provider: Optional[str] = None

    def _index_signature(self) -> tuple:
        records = self.vector_store.records
        # Cheap staleness check so a reindex rebuilds the sparse index without
        # the caller having to remember to invalidate it.
        return (
            len(records),
            records[0]["chunk_id"] if records else None,
            records[-1]["chunk_id"] if records else None,
        )

    def _ensure_bm25(self) -> Bm25Index:
        signature = self._index_signature()
        if self._bm25 is None or self._bm25_signature != signature:
            index = Bm25Index(k1=settings.bm25_k1, b=settings.bm25_b)
            index.build(
                [
                    (rec["chunk_id"], tokenize(rec["text"]))
                    for rec in self.vector_store.records
                ]
            )
            self._bm25 = index
            self._bm25_signature = signature
        return self._bm25

    def _reranker_for(self, enabled: bool) -> Optional[Reranker]:
        if not enabled:
            return None
        provider = (settings.reranker_provider or NONE).lower()
        if provider == NONE:
            return None
        # Cached by provider because a cross-encoder loads model weights and must
        # not be reconstructed per query.
        if self._reranker_provider != provider:
            self._reranker = build_reranker(provider)
            self._reranker_provider = provider
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        filters: Optional[RetrievalFilter] = None,
        rerank: Optional[bool] = None,
    ) -> RetrievalResult:
        mode = (mode or settings.retrieval_mode).lower()
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode '{mode}', expected one of {MODES}")
        top_k = top_k or settings.rag_top_k
        candidate_k = max(settings.candidate_k, top_k)

        records = self.vector_store.records
        if filters is not None and not filters.is_empty():
            allowed = [r for r in records if filters.matches(r.get("metadata") or {})]
        else:
            allowed = list(records)

        query_tokens = tokenize(query)
        trace = RetrievalTrace(
            mode=mode,
            corpus_size=len(records),
            filtered_out=len(records) - len(allowed),
            query_terms=len(query_tokens),
        )
        if not allowed:
            return RetrievalResult(citations=[], trace=trace)

        rankings: dict[str, list[str]] = {}
        raw_scores: dict[str, dict[str, float]] = {}

        if mode in (VECTOR, HYBRID):
            hits = self.vector_store.vector_search(query, allowed, candidate_k)
            rankings[VECTOR] = [chunk_id for chunk_id, _ in hits]
            raw_scores[VECTOR] = dict(hits)

        if mode in (BM25, HYBRID):
            allowed_ids = {r["chunk_id"] for r in allowed}
            hits = self._ensure_bm25().search(query_tokens, candidate_k, allowed=allowed_ids)
            rankings[BM25] = [chunk_id for chunk_id, _ in hits]
            raw_scores[BM25] = dict(hits)

        trace.candidate_counts = {name: len(ids) for name, ids in rankings.items()}

        if len(rankings) == 1:
            # Single retriever: keep its own score so the baseline stays
            # interpretable instead of reporting a fusion score of one input.
            name, ordered_ids = next(iter(rankings.items()))
            fused = [
                FusedHit(chunk_id=chunk_id, score=raw_scores[name][chunk_id], ranks={name: rank})
                for rank, chunk_id in enumerate(ordered_ids, start=1)
            ]
        else:
            fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k)
            trace.rrf_k = settings.rrf_k

        trace.fused_count = len(fused)
        record_map = self.vector_store.record_map()
        candidates = [
            self._to_citation(hit, record_map[hit.chunk_id], raw_scores, rank)
            for rank, hit in enumerate(fused[: settings.rerank_candidates], start=1)
            if hit.chunk_id in record_map
        ]

        # Suppress restatements before reranking: overlapping chunk windows
        # otherwise spend the rerank budget scoring the same passage twice.
        candidates, trace.deduplicated = suppress_near_duplicates(
            candidates, settings.dedup_threshold, settings.dedup_shingle_size
        )

        rerank_enabled = settings.rerank_enabled if rerank is None else rerank
        candidates = self._apply_rerank(query, candidates, rerank_enabled, trace)

        similarity = make_vector_similarity(
            {rec["chunk_id"]: rec["vector"] for rec in allowed},
            settings.dedup_shingle_size,
        )
        selected, trace.source_capped = mmr_select(
            candidates,
            top_k=top_k,
            lambda_=settings.mmr_lambda,
            similarity=similarity,
            per_source_cap=settings.per_source_cap,
        )
        trace.mmr_lambda = settings.mmr_lambda
        trace.returned = len(selected)
        return RetrievalResult(citations=selected, trace=trace)

    def _apply_rerank(
        self,
        query: str,
        candidates: List[Citation],
        enabled: bool,
        trace: RetrievalTrace,
    ) -> List[Citation]:
        reranker = self._reranker_for(enabled)
        if reranker is None or not candidates:
            trace.reranker = None if reranker is None else reranker.name
            return candidates
        try:
            scores = reranker.score(query, candidates)
        except RerankerUnavailable as exc:
            # Relevance degrades to the fusion order; availability is preserved.
            trace.reranker = f"{reranker.name}:unavailable"
            trace.rerank_error = str(exc)
            return candidates
        trace.reranker = reranker.name
        trace.reranked = len(candidates)
        for citation, score in zip(candidates, scores):
            citation.rerank_score = round(float(score), SCORE_PRECISION)
            citation.score = citation.rerank_score
        ordered = sorted(
            candidates, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id)
        )
        for rank, citation in enumerate(ordered, start=1):
            citation.rerank_rank = rank
        return ordered

    @staticmethod
    def _to_citation(
        hit: FusedHit, record: dict, raw_scores: dict, fusion_rank: int
    ) -> Citation:
        vector_score = raw_scores.get(VECTOR, {}).get(hit.chunk_id)
        bm25_score = raw_scores.get(BM25, {}).get(hit.chunk_id)
        base_score = round(float(hit.score), SCORE_PRECISION)
        return Citation(
            chunk_id=hit.chunk_id,
            source=record["source"],
            score=base_score,
            fusion_score=base_score,
            fusion_rank=fusion_rank,
            text=record["text"],
            metadata=record.get("metadata") or {},
            retrievers=sorted(hit.ranks),
            vector_score=(
                round(float(vector_score), SCORE_PRECISION) if vector_score is not None else None
            ),
            vector_rank=hit.ranks.get(VECTOR),
            bm25_score=(
                round(float(bm25_score), SCORE_PRECISION) if bm25_score is not None else None
            ),
            bm25_rank=hit.ranks.get(BM25),
        )
