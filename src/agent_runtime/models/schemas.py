from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    value = uuid4().hex
    return f"{prefix}{value}" if prefix else value


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Citation(BaseModel):
    chunk_id: str
    source: str
    # Final ordering score: the rerank score when reranking ran, else the
    # fusion/base retrieval score. `fusion_score` always keeps the pre-rerank
    # value so a reranker's effect stays measurable.
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrievers: List[str] = Field(default_factory=list)
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    fusion_score: Optional[float] = None
    fusion_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None


class RetrievalFilter(BaseModel):
    """Structured pre-filter applied before scoring.

    Filtering before retrieval rather than after is deliberate: once an
    out-of-scope chunk reaches the context window the model can leak its content
    even if the citation is stripped from the response.
    """

    sources: List[str] = Field(default_factory=list)
    doc_types: List[str] = Field(default_factory=list)
    classifications: List[str] = Field(default_factory=list)
    owners: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.sources or self.doc_types or self.classifications or self.owners)

    def matches(self, metadata: Dict[str, Any]) -> bool:
        checks = (
            (self.sources, metadata.get("source")),
            (self.doc_types, metadata.get("doc_type")),
            (self.classifications, metadata.get("classification")),
            (self.owners, metadata.get("owner")),
        )
        for allowed, value in checks:
            if allowed and value not in allowed:
                return False
        return True


class RetrievalTrace(BaseModel):
    """Per-query record of what each retrieval stage did, for auditability."""

    mode: str
    corpus_size: int = 0
    filtered_out: int = 0
    query_terms: int = 0
    candidate_counts: Dict[str, int] = Field(default_factory=dict)
    fused_count: int = 0
    deduplicated: int = 0
    reranker: Optional[str] = None
    reranked: int = 0
    rerank_error: Optional[str] = None
    source_capped: int = 0
    mmr_lambda: Optional[float] = None
    returned: int = 0
    rrf_k: Optional[int] = None


class DeliveryTask(BaseModel):
    title: str
    owner_role: str = "engineer"
    estimate_days: float = 1.0
    dependencies: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)


class LabeledChunk(BaseModel):
    label: str
    chunk_id: str
    source: str
    score: float
    text: str
    truncated: bool = False


class ContextBundle(BaseModel):
    """Token-budgeted, labelled context assembled for generation."""

    context_text: str = ""
    selected: List[LabeledChunk] = Field(default_factory=list)
    label_to_chunk_id: Dict[str, str] = Field(default_factory=dict)
    tokens_used: int = 0
    token_budget: int = 0
    tokenizer: str = "heuristic"
    dropped_count: int = 0
    dropped_chunk_ids: List[str] = Field(default_factory=list)
    truncated_count: int = 0


class ClaimSupport(BaseModel):
    label: str
    chunk_id: str
    source: str = ""
    quoted_snippet: str = ""
    support_score: float = 0.0
    claim: str = ""


class GroundednessResult(BaseModel):
    citation_coverage: float = 0.0
    confidence: float = 0.0
    labeled_claims: int = 0
    total_claims: int = 0
    hallucinated_labels: List[str] = Field(default_factory=list)
    support: List[ClaimSupport] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None


class DeliveryPlan(BaseModel):
    summary: str
    risks: List[str] = Field(default_factory=list)
    tasks: List[DeliveryTask] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    prompt_version: str = "v1"
    model: str = "stub"
    support: List[ClaimSupport] = Field(default_factory=list)
    confidence: float = 0.0
    citation_coverage: float = 0.0
    refusal_reason: Optional[str] = None
    context_labels: Dict[str, str] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    requirement: str = Field(..., min_length=10, max_length=12000)
    change_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    auto_approve: bool = False


class ApprovalRequest(BaseModel):
    approved: bool
    reviewer: str = "human"
    note: str = ""


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    change_id: str
    idempotency_key: str
    plan: Optional[DeliveryPlan] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AuditEventType(str, Enum):
    WORKFLOW_STARTED = "workflow_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    TOOL_INVOKED = "tool_invoked"
    CONTEXT_ASSEMBLED = "context_assembled"
    PLAN_GENERATED = "plan_generated"
    GROUNDEDNESS_CHECKED = "groundedness_checked"
    APPROVAL_RESOLVED = "approval_resolved"
    WORKFLOW_COMPLETED = "workflow_completed"


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt_"))
    workflow_id: str
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=utc_now)
    payload: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    prompt_version: str
    retrieval_mode: str = "hybrid"
    rag_top_k: int = 4
    candidate_k: int = 20
    indexed_chunks: int = 0
    context_token_budget: int = 800
    relevance_floor: float = 0.12
    coverage_floor: float = 0.5


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    mode: Optional[str] = Field(default=None, description="vector | bm25 | hybrid")
    filters: Optional[RetrievalFilter] = None
    rerank: Optional[bool] = Field(
        default=None, description="override the configured rerank stage for this query"
    )


class RetrievalResponse(BaseModel):
    query: str
    citations: List[Citation] = Field(default_factory=list)
    trace: RetrievalTrace


class EvidenceResponse(BaseModel):
    """Reconstructs why a plan was produced or refused."""

    workflow_id: str
    status: WorkflowStatus
    retrieval_trace: Optional[RetrievalTrace] = None
    citations: List[Citation] = Field(default_factory=list)
    context: Optional[ContextBundle] = None
    groundedness: Optional[GroundednessResult] = None
    support: List[ClaimSupport] = Field(default_factory=list)
    plan_summary: Optional[str] = None
    refusal_reason: Optional[str] = None
