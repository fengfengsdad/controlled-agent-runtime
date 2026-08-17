from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from agent_runtime.api.main import create_app

    return TestClient(create_app())


def test_health_reports_retrieval_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["retrieval_mode"] == "hybrid"
    assert body["embedding_provider"] == "stub"
    assert body["indexed_chunks"] >= 1


def test_search_returns_citations_with_retrieval_trace(client):
    response = client.post(
        "/v1/retrieval/search",
        json={"query": "idempotency keys and approval gates"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    trace = body["trace"]
    assert trace["mode"] == "hybrid"
    assert set(trace["candidate_counts"]) == {"vector", "bm25"}
    assert trace["returned"] == len(body["citations"])
    top = body["citations"][0]
    assert top["chunk_id"] and top["source"]
    assert top["metadata"]["classification"] == "internal"


def test_search_honours_mode_override(client):
    body = client.post(
        "/v1/retrieval/search",
        json={"query": "approval gates", "mode": "bm25"},
    ).json()
    assert body["trace"]["mode"] == "bm25"
    assert set(body["trace"]["candidate_counts"]) == {"bm25"}
    assert body["trace"]["rrf_k"] is None


def test_search_reports_rerank_and_diversity_stages(client):
    body = client.post(
        "/v1/retrieval/search",
        json={"query": "idempotency keys and approval gates"},
    ).json()
    trace = body["trace"]
    assert trace["reranker"] == "stub"
    assert trace["reranked"] >= 1
    assert trace["mmr_lambda"] == 0.7
    top = body["citations"][0]
    assert top["rerank_rank"] == 1
    assert top["fusion_rank"] is not None
    assert top["score"] == top["rerank_score"]


def test_search_can_disable_rerank_for_baseline_comparison(client):
    payload = {"query": "approval gates protect mutating tools", "mode": "vector"}
    with_rerank = client.post("/v1/retrieval/search", json=payload).json()
    without = client.post(
        "/v1/retrieval/search", json={**payload, "rerank": False}
    ).json()
    assert with_rerank["trace"]["reranker"] == "stub"
    assert without["trace"]["reranker"] is None
    assert without["trace"]["reranked"] == 0
    baseline = without["citations"][0]
    assert baseline["rerank_score"] is None
    assert baseline["score"] == baseline["fusion_score"]


def test_search_rejects_unknown_mode(client):
    response = client.post(
        "/v1/retrieval/search",
        json={"query": "anything", "mode": "magic"},
    )
    assert response.status_code == 400
    assert "unknown retrieval mode" in response.json()["detail"]


def test_search_applies_classification_filter(client):
    body = client.post(
        "/v1/retrieval/search",
        json={
            "query": "idempotency",
            "filters": {"classifications": ["top-secret"]},
        },
    ).json()
    assert body["citations"] == []
    assert body["trace"]["filtered_out"] >= 1


def test_search_validates_empty_query(client):
    response = client.post("/v1/retrieval/search", json={"query": ""})
    assert response.status_code == 422


def test_workflow_audit_records_retrieval_stages(client):
    created = client.post(
        "/v1/workflows",
        json={
            "requirement": "Add idempotent retry handling for webhook failures.",
            "change_id": "CHG-1001",
            "auto_approve": True,
        },
    ).json()
    events = client.get(f"/v1/workflows/{created['workflow_id']}/audit").json()["events"]
    retrieval = next(e for e in events if e["event_type"] == "retrieval_completed")
    payload = retrieval["payload"]
    assert payload["mode"] == "hybrid"
    assert payload["corpus_size"] >= 1
    assert payload["reranker"] == "stub"
    assert payload["selected"]
    assert payload["selected"][0]["retrievers"]
    assert payload["selected"][0]["rerank_rank"] == 1
