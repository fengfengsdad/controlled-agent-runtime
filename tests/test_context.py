from __future__ import annotations

from agent_runtime.models.schemas import Citation
from agent_runtime.rag.context import build_context, count_tokens, reset_tokenizer_cache


def _citation(chunk_id: str, source: str, score: float, text: str) -> Citation:
    return Citation(chunk_id=chunk_id, source=source, score=score, text=text)


def test_labels_are_assigned_in_score_order():
    citations = [
        _citation("c-low", "a.md", 0.2, "Low scoring passage about retries."),
        _citation("c-high", "b.md", 0.9, "High scoring passage about idempotency keys."),
    ]
    bundle = build_context(citations, budget_tokens=400)
    assert [item.label for item in bundle.selected] == ["S1", "S2"]
    assert bundle.label_to_chunk_id["S1"] == "c-high"
    assert "[S1]" in bundle.context_text
    assert bundle.dropped_count == 0
    assert bundle.tokens_used <= bundle.token_budget


def test_budget_drops_overflow_chunks_and_records_them():
    long_a = "Idempotency keys prevent duplicate writes. " * 40
    long_b = "Approval gates protect mutating tools. " * 40
    citations = [
        _citation("keep", "a.md", 0.9, long_a),
        _citation("drop", "b.md", 0.8, long_b),
    ]
    small_budget = count_tokens(f"[S1] source=a.md score=0.9\n{long_a}") + 5
    bundle = build_context(citations, budget_tokens=small_budget)
    assert bundle.selected[0].chunk_id == "keep"
    assert "drop" in bundle.dropped_chunk_ids
    assert bundle.dropped_count >= 1


def test_last_chunk_truncates_on_sentence_boundary():
    text = (
        "First sentence about idempotency keys stays intact. "
        "Second sentence should be dropped because the budget is tight."
    )
    citations = [_citation("c1", "a.md", 0.9, text)]
    header = "[S1] source=a.md score=0.9\n"
    first = "First sentence about idempotency keys stays intact."
    budget = count_tokens(header + first) + 2
    bundle = build_context(citations, budget_tokens=budget)
    assert bundle.selected
    kept = bundle.selected[0]
    assert kept.truncated is True
    assert "First sentence" in kept.text
    assert "Second sentence" not in kept.text
    assert not kept.text.endswith("intact. Second")


def test_per_source_cap_drops_extra_chunks_from_same_document():
    citations = [
        _citation("a1", "same.md", 0.9, "First chunk from the same source."),
        _citation("a2", "same.md", 0.8, "Second chunk from the same source."),
        _citation("b1", "other.md", 0.7, "Chunk from a different source."),
    ]
    bundle = build_context(citations, budget_tokens=400, per_source_cap=1)
    selected_ids = [item.chunk_id for item in bundle.selected]
    assert selected_ids == ["a1", "b1"]
    assert "a2" in bundle.dropped_chunk_ids


def test_empty_citations_produce_empty_context():
    bundle = build_context([], budget_tokens=100)
    assert bundle.context_text == ""
    assert bundle.dropped_count == 0
    assert bundle.tokens_used == 0


def test_heuristic_fallback_when_tiktoken_missing(monkeypatch):
    reset_tokenizer_cache()
    monkeypatch.setattr("agent_runtime.rag.context._get_encoding", lambda: None)
    text = "abcd" * 10
    assert count_tokens(text) == (len(text) + 3) // 4
    reset_tokenizer_cache()
