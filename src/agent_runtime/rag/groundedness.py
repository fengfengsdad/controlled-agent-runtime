from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

from agent_runtime.models.schemas import (
    Citation,
    ClaimSupport,
    DeliveryPlan,
    GroundednessResult,
)
from agent_runtime.rag.text import tokenize

LABEL_RE = re.compile(r"\[S(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

REFUSAL_NO_EVIDENCE = "NO_RETRIEVED_EVIDENCE"
REFUSAL_RELEVANCE = "RELEVANCE_BELOW_FLOOR"
REFUSAL_COVERAGE = "CITATION_COVERAGE_BELOW_FLOOR"


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [p.strip() for p in _SENTENCE_RE.split(stripped) if p.strip()]


def extract_claims(plan: DeliveryPlan) -> list[str]:
    """Groundable assertions: the summary and named risks.

    Task titles are proposed work, not evidence claims, so they are not
    included in citation coverage.
    """
    claims: list[str] = []
    for part in [plan.summary, *plan.risks]:
        claims.extend(split_sentences(part))
    return claims


def strip_labels(text: str) -> str:
    return LABEL_RE.sub("", text).strip()


def lexical_support(claim: str, chunk_text: str) -> float:
    claim_tokens = set(tokenize(strip_labels(claim)))
    chunk_tokens = set(tokenize(chunk_text))
    if not claim_tokens or not chunk_tokens:
        return 0.0
    overlap = claim_tokens & chunk_tokens
    return round(len(overlap) / len(claim_tokens), 4)


def check_groundedness(
    plan: DeliveryPlan,
    citations: Sequence[Citation],
    label_to_chunk_id: Mapping[str, str],
    coverage_floor: float,
) -> GroundednessResult:
    by_id = {c.chunk_id: c for c in citations}
    label_lookup = {raw.strip("[]"): chunk_id for raw, chunk_id in label_to_chunk_id.items()}

    claims = extract_claims(plan)
    labeled = 0
    hallucinated: list[str] = []
    support: list[ClaimSupport] = []
    seen_hallucinations: set[str] = set()

    for claim in claims:
        found = LABEL_RE.findall(claim)
        if found:
            labeled += 1
        for number in found:
            key = f"S{number}"
            marker = f"[{key}]"
            chunk_id = label_lookup.get(key)
            if chunk_id is None:
                if marker not in seen_hallucinations:
                    hallucinated.append(marker)
                    seen_hallucinations.add(marker)
                continue
            citation = by_id.get(chunk_id)
            snippet = (citation.text[:240] if citation else "")
            support.append(
                ClaimSupport(
                    label=key,
                    chunk_id=chunk_id,
                    source=citation.source if citation else "",
                    quoted_snippet=snippet,
                    support_score=lexical_support(claim, citation.text if citation else ""),
                    claim=strip_labels(claim),
                )
            )

    total = len(claims)
    coverage = round(labeled / total, 4) if total else 0.0
    mean_support = (
        round(sum(item.support_score for item in support) / len(support), 4) if support else 0.0
    )
    confidence = round(coverage * mean_support, 4) if support else 0.0
    refused = coverage < coverage_floor
    reason = REFUSAL_COVERAGE if refused else None
    return GroundednessResult(
        citation_coverage=coverage,
        confidence=confidence,
        labeled_claims=labeled,
        total_claims=total,
        hallucinated_labels=hallucinated,
        support=support,
        refused=refused,
        refusal_reason=reason,
    )


def drop_hallucinated_labels(plan: DeliveryPlan, valid_keys: Mapping[str, str]) -> DeliveryPlan:
    allowed = {key.strip("[]") for key in valid_keys}

    def clean(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            key = f"S{match.group(1)}"
            return match.group(0) if key in allowed else ""

        cleaned = LABEL_RE.sub(replace, text)
        return re.sub(r" +", " ", cleaned).strip()

    plan.summary = clean(plan.summary)
    plan.risks = [clean(risk) for risk in plan.risks]
    for task in plan.tasks:
        task.title = clean(task.title)
        task.acceptance_criteria = [clean(item) for item in task.acceptance_criteria]
    return plan


def citation_relevance(citation: Citation) -> float:
    """Best retrieval signal, not just the final ordering score.

    A lexical stub reranker can score 0.0 on a genuine vector hit when the
    query and chunk share no exact tokens (`idempotent` vs `idempotency`).
    The refuse-to-answer gate should look at whether any stage thought the
    hit was relevant.
    """
    scores = [citation.score]
    for value in (citation.fusion_score, citation.vector_score, citation.bm25_score):
        if value is not None:
            scores.append(value)
    return max(scores)


def retrieval_refusal_reason(
    citations: Sequence[Citation], relevance_floor: float
) -> Optional[str]:
    if not citations:
        return REFUSAL_NO_EVIDENCE
    max_score = max(citation_relevance(c) for c in citations)
    if max_score < relevance_floor:
        return REFUSAL_RELEVANCE
    return None
