from __future__ import annotations

from agent_runtime.models.schemas import Citation, DeliveryPlan
from agent_runtime.rag.groundedness import (
    REFUSAL_COVERAGE,
    REFUSAL_NO_EVIDENCE,
    REFUSAL_RELEVANCE,
    check_groundedness,
    drop_hallucinated_labels,
    lexical_support,
    retrieval_refusal_reason,
)


def test_coverage_counts_labelled_summary_and_risks():
    plan = DeliveryPlan(
        summary="Use idempotency keys on the write path [S1].",
        risks=["Approval gates must still run [S1]."],
    )
    citations = [
        Citation(
            chunk_id="c1",
            source="a.md",
            score=0.8,
            text="Idempotency keys prevent duplicate writes. Approval gates protect tools.",
        )
    ]
    result = check_groundedness(plan, citations, {"S1": "c1"}, coverage_floor=0.5)
    assert result.total_claims == 2
    assert result.labeled_claims == 2
    assert result.citation_coverage == 1.0
    assert result.refused is False
    assert result.support[0].chunk_id == "c1"
    assert result.support[0].support_score > 0


def test_hallucinated_labels_are_dropped_and_counted():
    plan = DeliveryPlan(
        summary="This cites a real chunk [S1] and a fake one [S9].",
        risks=[],
    )
    citations = [
        Citation(chunk_id="c1", source="a.md", score=0.8, text="real chunk about retries.")
    ]
    cleaned = drop_hallucinated_labels(plan, {"S1": "c1"})
    assert "[S1]" in cleaned.summary
    assert "[S9]" not in cleaned.summary
    result = check_groundedness(cleaned, citations, {"S1": "c1"}, coverage_floor=0.5)
    assert result.hallucinated_labels == []
    raw = DeliveryPlan(
        summary="This cites a real chunk [S1] and a fake one [S9].",
        risks=[],
    )
    raw_result = check_groundedness(raw, citations, {"S1": "c1"}, coverage_floor=0.5)
    assert "[S9]" in raw_result.hallucinated_labels


def test_low_coverage_refuses():
    plan = DeliveryPlan(
        summary="An unsupported assertion with no citation.",
        risks=["Another unsupported risk."],
    )
    result = check_groundedness(plan, [], {}, coverage_floor=0.5)
    assert result.citation_coverage == 0.0
    assert result.refused is True
    assert result.refusal_reason == REFUSAL_COVERAGE


def test_retrieval_refusal_reasons():
    assert retrieval_refusal_reason([], 0.1) == REFUSAL_NO_EVIDENCE
    weak = [Citation(chunk_id="c1", source="a.md", score=0.05, text="weak")]
    assert retrieval_refusal_reason(weak, 0.12) == REFUSAL_RELEVANCE
    reranked_zero = [
        Citation(
            chunk_id="c1",
            source="a.md",
            score=0.0,
            text="vector hit with no lexical overlap",
            fusion_score=0.4,
        )
    ]
    assert retrieval_refusal_reason(reranked_zero, 0.12) is None
    strong = [Citation(chunk_id="c1", source="a.md", score=0.4, text="strong")]
    assert retrieval_refusal_reason(strong, 0.12) is None


def test_lexical_support_is_token_overlap():
    score = lexical_support(
        "idempotency keys prevent duplicate writes",
        "Idempotency keys prevent duplicate writes on the webhook path.",
    )
    assert score > 0.8
    assert lexical_support("zzzz", "nothing overlapping here") == 0.0
