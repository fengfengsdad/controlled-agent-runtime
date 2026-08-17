from __future__ import annotations

import pytest

from agent_runtime.models.schemas import RetrievalFilter
from agent_runtime.rag.bm25 import Bm25Index
from agent_runtime.rag.fusion import reciprocal_rank_fusion
from agent_runtime.rag.retriever import HybridRetriever
from agent_runtime.rag.store import SourceDocument, StubEmbedder, VectorStore
from agent_runtime.rag.text import tokenize

CORPUS = [
    SourceDocument(
        source="idempotency.md",
        text=(
            "Every mutating request carries an idempotency key. Replaying a "
            "submission returns the original workflow instead of duplicating "
            "side effects."
        ),
        metadata={"doc_type": "markdown", "classification": "internal", "owner": "platform"},
    ),
    SourceDocument(
        source="approvals.md",
        text=(
            "Write operations pause at an approval gate until a human reviewer "
            "resolves them. Read-only tools bypass the gate."
        ),
        metadata={"doc_type": "markdown", "classification": "internal", "owner": "platform"},
    ),
    SourceDocument(
        source="ledger_spec.md",
        text=(
            "The webhook_idempotency_ledger table stores delivery attempts and "
            "their terminal outcome for reconciliation."
        ),
        metadata={"doc_type": "markdown", "classification": "internal", "owner": "payments"},
    ),
    SourceDocument(
        source="salary_bands.md",
        text=(
            "Compensation bands and reviewer allocations for the payments "
            "engineering group are recorded here."
        ),
        metadata={"doc_type": "markdown", "classification": "restricted", "owner": "hr"},
    ),
]


@pytest.fixture()
def retriever(tmp_path):
    store = VectorStore(tmp_path / "vector_store_test", StubEmbedder())
    store.clear()
    store.add_documents(CORPUS)
    return HybridRetriever(store)


def test_bm25_ranks_exact_term_match_first(retriever):
    result = retriever.retrieve("webhook_idempotency_ledger", mode="bm25")
    assert result.citations
    assert result.citations[0].source == "ledger_spec.md"
    assert result.citations[0].bm25_rank == 1
    assert result.citations[0].bm25_score > 0


def test_bm25_scoring_is_deterministic_across_rebuilds():
    documents = [(f"chunk-{i}", tokenize(doc.text)) for i, doc in enumerate(CORPUS)]
    first, second = Bm25Index(), Bm25Index()
    first.build(documents)
    second.build(documents)
    query = tokenize("idempotency key replay")
    assert first.search(query, top_n=4) == second.search(query, top_n=4)


def test_bm25_respects_allowed_id_restriction():
    documents = [(f"chunk-{i}", tokenize(doc.text)) for i, doc in enumerate(CORPUS)]
    index = Bm25Index()
    index.build(documents)
    query = tokenize("reviewer")
    unrestricted = index.search(query, top_n=10)
    restricted = index.search(query, top_n=10, allowed={"chunk-1"})
    assert len(unrestricted) > len(restricted)
    assert [chunk_id for chunk_id, _ in restricted] == ["chunk-1"]


def test_bm25_has_no_subword_matching():
    """Known tokenizer limitation, asserted so it stays a decision not a surprise.

    `webhook_idempotency_ledger` is one token, so a query for `idempotency`
    cannot match it lexically. Dense retrieval is what covers this gap, which is
    part of why the default mode is hybrid rather than BM25 alone.
    """
    documents = [(f"chunk-{i}", tokenize(doc.text)) for i, doc in enumerate(CORPUS)]
    index = Bm25Index()
    index.build(documents)
    matched = {chunk_id for chunk_id, _ in index.search(tokenize("idempotency"), top_n=10)}
    assert "chunk-2" not in matched


def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion(
        {"vector": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, k=60
    )
    by_id = {hit.chunk_id: hit for hit in fused}
    # "b" is 1st and 2nd; "c" appears in one ranking only, so agreement wins.
    assert by_id["b"].score > by_id["c"].score
    assert by_id["b"].ranks == {"vector": 2, "bm25": 1}
    assert by_id["d"].ranks == {"bm25": 3}


def test_hybrid_combines_both_retrievers_and_records_ranks(retriever):
    result = retriever.retrieve("idempotency key replay protection", mode="hybrid")
    assert result.trace.mode == "hybrid"
    assert set(result.trace.candidate_counts) == {"vector", "bm25"}
    assert result.trace.rrf_k == 60
    assert result.citations
    top = result.citations[0]
    assert top.source == "idempotency.md"
    # A hit found by both retrievers must carry both provenance ranks.
    agreed = [c for c in result.citations if set(c.retrievers) == {"bm25", "vector"}]
    assert agreed
    assert all(c.vector_rank and c.bm25_rank for c in agreed)


def test_hybrid_retains_rare_term_match_that_vector_alone_ranks_lower(retriever):
    query = "webhook_idempotency_ledger reconciliation"
    vector_only = retriever.retrieve(query, mode="vector")
    hybrid = retriever.retrieve(query, mode="hybrid")

    def rank_of(result, source):
        for index, citation in enumerate(result.citations, start=1):
            if citation.source == source:
                return index
        return None

    hybrid_rank = rank_of(hybrid, "ledger_spec.md")
    vector_rank = rank_of(vector_only, "ledger_spec.md")
    assert hybrid_rank is not None
    assert vector_rank is None or hybrid_rank <= vector_rank


def test_vector_mode_without_rerank_orders_by_cosine(retriever):
    result = retriever.retrieve(
        "approval gate human reviewer", mode="vector", rerank=False
    )
    assert result.trace.candidate_counts.keys() == {"vector"}
    assert result.trace.rrf_k is None
    assert result.trace.reranker is None
    top = result.citations[0]
    assert top.retrievers == ["vector"]
    # Without a reranker the single-retriever score is the ordering score.
    assert top.score == top.vector_score == top.fusion_score
    assert -1.0 <= top.score <= 1.0


def test_rerank_replaces_ordering_score_but_preserves_fusion_score(retriever):
    result = retriever.retrieve("approval gate human reviewer", mode="vector")
    top = result.citations[0]
    assert result.trace.reranker == "stub"
    assert result.trace.reranked >= 1
    assert top.rerank_score is not None
    assert top.rerank_rank == 1
    assert top.score == top.rerank_score
    # The pre-rerank signal survives so the reranker's effect stays measurable.
    assert top.fusion_score == top.vector_score


def test_classification_filter_excludes_documents_before_scoring(retriever):
    filters = RetrievalFilter(classifications=["internal"])
    result = retriever.retrieve("reviewer allocations for payments", filters=filters)
    assert result.trace.filtered_out == 1
    assert "salary_bands.md" not in {c.source for c in result.citations}


def test_owner_filter_narrows_to_single_source(retriever):
    result = retriever.retrieve(
        "delivery attempts", filters=RetrievalFilter(owners=["payments"])
    )
    assert {c.source for c in result.citations} == {"ledger_spec.md"}
    assert result.trace.filtered_out == 3


def test_filter_matching_nothing_returns_no_citations(retriever):
    result = retriever.retrieve(
        "idempotency", filters=RetrievalFilter(owners=["nobody"])
    )
    assert result.citations == []
    assert result.trace.returned == 0


def test_top_k_is_respected(retriever):
    result = retriever.retrieve("approval idempotency ledger reviewer", top_k=2)
    assert len(result.citations) == 2


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retriever.retrieve("anything", mode="magic")


def test_sparse_index_rebuilds_after_documents_are_added(retriever):
    retriever.retrieve("idempotency", mode="bm25")
    retriever.vector_store.add_documents(
        [
            SourceDocument(
                source="rollback.md",
                text="Index rollback restores the previous manifest version.",
                metadata={"classification": "internal", "owner": "platform"},
            )
        ]
    )
    result = retriever.retrieve("rollback manifest", mode="bm25")
    assert result.citations[0].source == "rollback.md"


def test_citations_carry_governance_metadata(retriever):
    result = retriever.retrieve("idempotency key")
    metadata = result.citations[0].metadata
    assert metadata["classification"] == "internal"
    assert metadata["doc_type"] == "markdown"
    assert metadata["source"] == metadata["source"]
    assert "chunk_index" in metadata
