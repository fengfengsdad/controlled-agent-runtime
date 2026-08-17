from __future__ import annotations

from pathlib import Path

import pytest

# Isolate test data from local developer state.
TEST_DATA = Path(__file__).resolve().parent / "_tmp_data"


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    data = tmp_path / "data"
    corpus = data / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "sample.md").write_text(
        "Idempotency keys prevent duplicate writes. Approval gates protect mutating tools.",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("CHECKPOINT_DB", str(data / "checkpoints.db"))
    monkeypatch.setenv("AUDIT_DB", str(data / "audit.db"))
    monkeypatch.setenv("VECTOR_DIR", str(data / "vector_store"))
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "stub")
    monkeypatch.setenv("TOOL_TRANSPORT", "mcp")

    from agent_runtime import config as config_module
    from agent_runtime.graph import runtime as runtime_module

    config_module.reload_settings()
    runtime_module._runtime = None
    yield
    runtime_module._runtime = None
    config_module.reload_settings()
