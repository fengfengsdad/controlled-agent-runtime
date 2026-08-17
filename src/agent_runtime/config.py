from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    data_dir: Path = Path("./data")
    corpus_dir: Path = Path("./data/corpus")
    checkpoint_db: Path = Path("./data/checkpoints.db")
    audit_db: Path = Path("./data/audit.db")
    vector_dir: Path = Path("./data/vector_store")

    llm_provider: str = "stub"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    prompt_version: str = "v1"

    embedding_provider: str = "stub"
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 80

    # Retrieval: vector | bm25 | hybrid. vector/bm25 are kept as A/B baselines.
    retrieval_mode: str = "hybrid"
    candidate_k: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Reranking: stub (deterministic lexical, offline) | llm | cross_encoder | none
    reranker_provider: str = "stub"
    rerank_enabled: bool = True
    rerank_candidates: int = 20
    rerank_text_chars: int = 900
    cross_encoder_model: str = "BAAI/bge-reranker-base"

    # Near-duplicate suppression and diversity selection
    dedup_threshold: float = 0.85
    dedup_shingle_size: int = 5
    mmr_lambda: float = 0.7
    per_source_cap: int = 2

    # Applied to every ingested document until per-source governance metadata
    # exists; retrieval filters read these labels.
    default_classification: str = "internal"
    default_owner: str = "platform-engineering"

    max_payload_chars: int = 12000
    require_approval_for_writes: bool = True

    # Tool transport: mcp (real stdio client/server session) | local
    tool_transport: str = "mcp"
    mcp_server_module: str = "agent_runtime.tools.mcp_requirement_server"

    host: str = "0.0.0.0"
    port: int = 8080

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.audit_db.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()


def reload_settings() -> Settings:
    """Re-read configuration into the existing singleton.

    Modules bind `settings` at import time, so rebinding this module attribute
    leaves them pointing at the stale instance — which silently let a developer
    `.env` reach the test suite. Mutating in place is what actually propagates.
    """
    fresh = Settings()
    for name in type(fresh).model_fields:
        setattr(settings, name, getattr(fresh, name))
    return settings
