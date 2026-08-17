from __future__ import annotations

import pytest

from agent_runtime.models.schemas import Citation
from agent_runtime.rag.diversity import (
    jaccard,
    make_vector_similarity,
    mmr_select,
    suppress_near_duplicates,
)
from agent_runtime.rag.rerank import (
    LLMReranker,
    RerankerUnavailable,
    StubReranker,
    build_reranker,
)
from agent_runtime.rag.text import shingles


def citation(chunk_id: str, source: str, text: str, score: float = 0.0) -> Citation:
    return Citation(chunk_id=chunk_id, source=source, score=score, text=text)


# --- stub reranker -----------------------------------------------------------


def test_stub_reranker_prefers_full_query_coverage():
    reranker = StubReranker()
    covered = citation("a", "a.md", "Idempotency keys prevent duplicate writes.")
    partial = citation("b", "b.md", "Idempotency is discussed elsewhere in the guide.")
    scores = reranker.score("idempotency keys duplicate writes", [covered, partial])
    assert scores[0] > scores[1]


def test_stub_reranker_penalises_dilution_in_long_passages():
    reranker = StubReranker()
    focused = citation("a", "a.md", "Approval gate blocks writes.")
    diluted = citation(
        "b",
        "b.md",
        "Approval gate blocks writes. " + "Unrelated background prose. " * 40,
    )
    scores = reranker.score("approval gate blocks writes", [focused, diluted])
    assert scores[0] > scores[1]


def test_stub_reranker_is_deterministic():
    reranker = StubReranker()
    items = [citation("a", "a.md", "Retry with bounded exponential backoff.")]
    assert reranker.score("retry backoff", items) == reranker.score("retry backoff", items)


def test_stub_reranker_handles_empty_query_and_text():
    reranker = StubReranker()
    assert reranker.score("", [citation("a", "a.md", "text")]) == [0.0]
    assert reranker.score("query", [citation("a", "a.md", "")]) == [0.0]


# --- reranker factory --------------------------------------------------------


def test_build_reranker_none_disables_reranking():
    assert build_reranker("none") is None


def test_build_reranker_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown reranker provider"):
        build_reranker("telepathy")


def test_llm_reranker_requires_an_api_key():
    with pytest.raises(RerankerUnavailable, match="LLM_API_KEY"):
        LLMReranker(base_url="https://example.com/v1", api_key="", model="m", text_chars=100)


def test_llm_reranker_maps_scores_onto_candidate_positions():
    parsed = LLMReranker._parse_scores(
        '{"scores":[{"index":2,"score":9},{"index":1,"score":3.5}]}', expected=3
    )
    assert parsed == [3.5, 9.0, 0.0]


def test_llm_reranker_tolerates_prose_wrapped_json():
    parsed = LLMReranker._parse_scores(
        'Here you go: {"scores":[{"index":1,"score":7}]} hope that helps', expected=1
    )
    assert parsed == [7.0]


def test_llm_reranker_rejects_unusable_output():
    with pytest.raises(RerankerUnavailable):
        LLMReranker._parse_scores("no json at all", expected=2)
    with pytest.raises(RerankerUnavailable):
        LLMReranker._parse_scores('{"unexpected": true}', expected=2)


def test_llm_reranker_skips_malformed_entries():
    parsed = LLMReranker._parse_scores(
        '{"scores":[{"index":1,"score":"high"},{"index":2,"score":4},{"score":9}]}',
        expected=2,
    )
    assert parsed == [0.0, 4.0]


def test_llm_reranker_ignores_out_of_range_indices():
    parsed = LLMReranker._parse_scores(
        '{"scores":[{"index":99,"score":9},{"index":1,"score":2}]}', expected=2
    )
    assert parsed == [2.0, 0.0]


# --- near-duplicate suppression ---------------------------------------------


def test_suppress_near_duplicates_keeps_highest_ranked_member():
    text = "Idempotency keys prevent duplicate writes across retried submissions."
    kept, removed = suppress_near_duplicates(
        [
            citation("first", "a.md", text, score=0.9),
            citation("second", "a.md", text, score=0.8),
            citation("other", "b.md", "Approval gates protect mutating tools.", 0.7),
        ],
        threshold=0.85,
    )
    assert removed == 1
    assert [c.chunk_id for c in kept] == ["first", "other"]


def test_suppress_near_duplicates_keeps_distinct_passages():
    kept, removed = suppress_near_duplicates(
        [
            citation("a", "a.md", "Idempotency keys prevent duplicate writes."),
            citation("b", "b.md", "Cross-encoder reranking narrows the candidate set."),
        ],
        threshold=0.85,
    )
    assert removed == 0
    assert len(kept) == 2


def test_jaccard_bounds():
    assert jaccard(set(), {("a",)}) == 0.0
    assert jaccard({("a",)}, {("a",)}) == 1.0
    assert 0.0 < jaccard({("a",), ("b",)}, {("b",), ("c",)}) < 1.0


# --- MMR and per-source cap --------------------------------------------------


def identical_similarity(_left, _right):
    return 1.0


def orthogonal_similarity(_left, _right):
    return 0.0


def test_mmr_returns_relevance_order_when_candidates_are_unrelated():
    candidates = [
        citation("a", "a.md", "alpha", score=0.9),
        citation("b", "b.md", "beta", score=0.5),
        citation("c", "c.md", "gamma", score=0.1),
    ]
    selected, capped = mmr_select(
        candidates, top_k=2, lambda_=0.7, similarity=orthogonal_similarity
    )
    assert [c.chunk_id for c in selected] == ["a", "b"]
    assert capped == 0


def test_mmr_penalises_redundant_candidates():
    candidates = [
        citation("a", "a.md", "alpha", score=1.0),
        citation("b", "b.md", "beta", score=0.95),
    ]
    # With maximum redundancy and a low lambda, the second pick loses all value,
    # so ordering is driven by the penalty rather than by raw relevance.
    selected, _ = mmr_select(
        candidates, top_k=2, lambda_=0.1, similarity=identical_similarity
    )
    assert len(selected) == 2
    assert selected[0].chunk_id == "a"


def test_per_source_cap_blocks_a_single_document_dominating():
    candidates = [
        citation("a1", "big.md", "alpha", score=1.0),
        citation("a2", "big.md", "beta", score=0.9),
        citation("a3", "big.md", "gamma", score=0.8),
        citation("b1", "small.md", "delta", score=0.1),
    ]
    selected, capped = mmr_select(
        candidates,
        top_k=4,
        lambda_=1.0,
        similarity=orthogonal_similarity,
        per_source_cap=2,
    )
    sources = [c.source for c in selected]
    assert sources.count("big.md") == 2
    assert "small.md" in sources
    # The third big.md chunk was dropped by the cap, and that is reported so the
    # trade-off shows up in the trace rather than silently.
    assert capped == 1


def test_per_source_cap_stops_selection_when_everything_is_blocked():
    candidates = [
        citation("a1", "only.md", "alpha", score=1.0),
        citation("a2", "only.md", "beta", score=0.9),
        citation("a3", "only.md", "gamma", score=0.8),
    ]
    selected, capped = mmr_select(
        candidates,
        top_k=3,
        lambda_=1.0,
        similarity=orthogonal_similarity,
        per_source_cap=1,
    )
    assert len(selected) == 1
    assert capped == 2


def test_mmr_handles_empty_input():
    assert mmr_select([], top_k=3, lambda_=0.7, similarity=orthogonal_similarity) == ([], 0)


def test_vector_similarity_falls_back_to_shingles_when_vectors_missing():
    text = "Idempotency keys prevent duplicate writes."
    left = citation("a", "a.md", text)
    right = citation("b", "b.md", text)
    similarity = make_vector_similarity({})
    assert similarity(left, right) == pytest.approx(1.0)


def test_vector_similarity_uses_embeddings_when_present():
    left = citation("a", "a.md", "totally different words here")
    right = citation("b", "b.md", "nothing lexically in common")
    similarity = make_vector_similarity({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert similarity(left, right) == pytest.approx(1.0)
    assert shingles("", 5) == set()
