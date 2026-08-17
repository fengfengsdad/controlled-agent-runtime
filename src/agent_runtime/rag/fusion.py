"""Reciprocal Rank Fusion for merging multiple retriever rankings.

Rank-based rather than score-weighted: BM25 scores and cosine similarities are
not on a comparable scale, so a weighted blend needs a per-corpus alpha that
drifts as the corpus changes. RRF needs no tuning and is unaffected by outlier
scores. The cost is losing score magnitude — it cannot distinguish "first place
far ahead of second" from "top two nearly tied".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass
class FusedHit:
    chunk_id: str
    score: float
    ranks: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> list[FusedHit]:
    """rankings: retriever name -> ordered chunk_ids, best first."""
    hits: dict[str, FusedHit] = {}
    for retriever, chunk_ids in rankings.items():
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            hit = hits.get(chunk_id)
            if hit is None:
                hit = FusedHit(chunk_id=chunk_id, score=0.0)
                hits[chunk_id] = hit
            hit.score += 1.0 / (k + rank)
            hit.ranks[retriever] = rank
    ordered = sorted(hits.values(), key=lambda hit: (-hit.score, hit.chunk_id))
    return ordered[:top_n] if top_n is not None else ordered
