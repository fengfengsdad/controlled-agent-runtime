from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent_runtime.audit.store import AuditStore
from agent_runtime.checkpoint.store import CheckpointStore
from agent_runtime.config import settings
from agent_runtime.llm.client import LLMClient, get_llm_client
from agent_runtime.models.schemas import (
    AuditEventType,
    Citation,
    ContextBundle,
    DeliveryPlan,
    EvidenceResponse,
    GroundednessResult,
    RetrievalFilter,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowStatus,
    new_id,
    utc_now,
)
from agent_runtime.rag.context import build_context
from agent_runtime.rag.groundedness import (
    check_groundedness,
    drop_hallucinated_labels,
    retrieval_refusal_reason,
)
from agent_runtime.rag.retriever import HybridRetriever
from agent_runtime.rag.store import VectorStore, get_embedder, ingest_corpus
from agent_runtime.security.controls import SecurityError, redact_secrets, validate_payload
from agent_runtime.tools.mcp_client import RequirementLookupMcpClient
from agent_runtime.tools.requirement_lookup import RequirementLookupTool
from agent_runtime.tools.retry import with_backoff


class AgentState(TypedDict, total=False):
    workflow_id: str
    change_id: str
    idempotency_key: str
    requirement: str
    auto_approve: bool
    citations: List[Dict[str, Any]]
    retrieval_trace: Dict[str, Any]
    context_bundle: Dict[str, Any]
    groundedness: Dict[str, Any]
    tool_context: str
    plan: Dict[str, Any]
    status: str
    error: Optional[str]
    created_at: str
    updated_at: str


class RuntimeService:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self.audit = AuditStore(settings.audit_db)
        self.checkpoints = CheckpointStore(settings.checkpoint_db)
        self.embedder = get_embedder()
        self.vector_store = VectorStore(settings.vector_dir, self.embedder)
        self.retriever = HybridRetriever(self.vector_store)
        self.llm: LLMClient = get_llm_client()
        if settings.tool_transport == "mcp":
            self.requirement_tool = RequirementLookupMcpClient(
                server_module=settings.mcp_server_module
            )
        else:
            self.requirement_tool = RequirementLookupTool()
        self._ensure_corpus_indexed()
        self.graph = self._build_graph()

    def _ensure_corpus_indexed(self) -> None:
        if not (settings.vector_dir / "index.json").exists():
            ingest_corpus(settings.corpus_dir, self.vector_store)

    def reindex_corpus(self) -> int:
        return ingest_corpus(settings.corpus_dir, self.vector_store)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        filters: Optional[RetrievalFilter] = None,
        rerank: Optional[bool] = None,
    ):
        """Retrieval without the surrounding workflow, for evaluation and A/B runs."""
        return self.retriever.retrieve(
            query, top_k=top_k, mode=mode, filters=filters, rerank=rerank
        )

    def _persist(self, state: AgentState) -> None:
        state["updated_at"] = utc_now().isoformat()
        self.checkpoints.save(
            workflow_id=state["workflow_id"],
            idempotency_key=state["idempotency_key"],
            status=state["status"],
            state=dict(state),
            updated_at=state["updated_at"],
        )

    def _node_start(self, state: AgentState) -> AgentState:
        self.audit.append(
            state["workflow_id"],
            AuditEventType.WORKFLOW_STARTED,
            {
                "change_id": state["change_id"],
                "requirement_preview": redact_secrets(state["requirement"])[:200],
            },
        )
        state["status"] = WorkflowStatus.RUNNING.value
        self._persist(state)
        return state

    def _node_retrieve(self, state: AgentState) -> AgentState:
        result = with_backoff(lambda: self.retriever.retrieve(state["requirement"]))
        citations = result.citations
        state["citations"] = [c.model_dump() for c in citations]
        state["retrieval_trace"] = result.trace.model_dump()
        self.audit.append(
            state["workflow_id"],
            AuditEventType.RETRIEVAL_COMPLETED,
            {
                "mode": result.trace.mode,
                "corpus_size": result.trace.corpus_size,
                "filtered_out": result.trace.filtered_out,
                "candidate_counts": result.trace.candidate_counts,
                "fused_count": result.trace.fused_count,
                "deduplicated": result.trace.deduplicated,
                "reranker": result.trace.reranker,
                "reranked": result.trace.reranked,
                "source_capped": result.trace.source_capped,
                "mmr_lambda": result.trace.mmr_lambda,
                "hit_count": len(citations),
                "sources": [c.source for c in citations],
                "selected": [
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "score": c.score,
                        "retrievers": c.retrievers,
                        "vector_rank": c.vector_rank,
                        "bm25_rank": c.bm25_rank,
                        "fusion_rank": c.fusion_rank,
                        "rerank_rank": c.rerank_rank,
                    }
                    for c in citations
                ],
            },
        )
        self._persist(state)
        return state

    def _node_context(self, state: AgentState) -> AgentState:
        citations = [Citation.model_validate(c) for c in state.get("citations", [])]
        bundle = build_context(
            citations,
            budget_tokens=settings.context_token_budget,
            per_source_cap=settings.per_source_cap,
        )
        state["context_bundle"] = bundle.model_dump()
        self.audit.append(
            state["workflow_id"],
            AuditEventType.CONTEXT_ASSEMBLED,
            {
                "token_budget": bundle.token_budget,
                "tokens_used": bundle.tokens_used,
                "tokenizer": bundle.tokenizer,
                "selected": len(bundle.selected),
                "dropped_count": bundle.dropped_count,
                "dropped_chunk_ids": bundle.dropped_chunk_ids,
                "truncated_count": bundle.truncated_count,
                "labels": list(bundle.label_to_chunk_id),
            },
        )
        reason = retrieval_refusal_reason(citations, settings.relevance_floor)
        if reason:
            return self._complete_refusal(
                state,
                reason,
                GroundednessResult(refused=True, refusal_reason=reason),
            )
        self._persist(state)
        return state

    def _node_tool(self, state: AgentState) -> AgentState:
        result = self.requirement_tool.lookup(state["change_id"])
        self.audit.append(
            state["workflow_id"],
            AuditEventType.TOOL_INVOKED,
            {
                "tool": result.tool_name,
                "ok": result.ok,
                "error": result.error,
                "transport": getattr(result, "transport", settings.tool_transport),
            },
        )
        state["tool_context"] = (
            str(result.data) if result.ok else f"tool_error={result.error}"
        )
        self._persist(state)
        return state

    def _node_plan(self, state: AgentState) -> AgentState:
        citations = [Citation.model_validate(c) for c in state.get("citations", [])]
        bundle = ContextBundle.model_validate(state.get("context_bundle") or {})
        plan = with_backoff(
            lambda: self.llm.generate_plan(
                requirement=state["requirement"],
                citations=citations,
                tool_context=state.get("tool_context", ""),
                context_text=bundle.context_text,
            )
        )
        plan.context_labels = bundle.label_to_chunk_id
        state["plan"] = plan.model_dump()
        self.audit.append(
            state["workflow_id"],
            AuditEventType.PLAN_GENERATED,
            {
                "task_count": len(plan.tasks),
                "model": plan.model,
                "prompt_version": plan.prompt_version,
            },
        )
        self._persist(state)
        return state

    def _node_groundedness(self, state: AgentState) -> AgentState:
        citations = [Citation.model_validate(c) for c in state.get("citations", [])]
        bundle = ContextBundle.model_validate(state.get("context_bundle") or {})
        plan = DeliveryPlan.model_validate(state.get("plan") or {"summary": ""})
        plan = drop_hallucinated_labels(plan, bundle.label_to_chunk_id)
        result = check_groundedness(
            plan, citations, bundle.label_to_chunk_id, settings.coverage_floor
        )
        plan.support = result.support
        plan.citation_coverage = result.citation_coverage
        plan.confidence = result.confidence
        plan.refusal_reason = result.refusal_reason
        plan.context_labels = bundle.label_to_chunk_id
        state["plan"] = plan.model_dump()
        if result.refused:
            return self._complete_refusal(state, result.refusal_reason or "", result)
        state["groundedness"] = result.model_dump()
        self.audit.append(
            state["workflow_id"],
            AuditEventType.GROUNDEDNESS_CHECKED,
            {
                "citation_coverage": result.citation_coverage,
                "confidence": result.confidence,
                "labeled_claims": result.labeled_claims,
                "total_claims": result.total_claims,
                "hallucinated_labels": result.hallucinated_labels,
                "refused": False,
            },
        )
        self._persist(state)
        return state

    def _complete_refusal(
        self,
        state: AgentState,
        reason: str,
        groundedness: GroundednessResult,
    ) -> AgentState:
        citations = [Citation.model_validate(c) for c in state.get("citations", [])]
        bundle = ContextBundle.model_validate(state.get("context_bundle") or {})
        plan = DeliveryPlan(
            summary="Insufficient retrieved evidence to produce a grounded plan.",
            risks=[],
            tasks=[],
            citations=citations,
            prompt_version=settings.prompt_version,
            model="none",
            support=groundedness.support,
            confidence=0.0,
            citation_coverage=groundedness.citation_coverage,
            refusal_reason=reason,
            context_labels=bundle.label_to_chunk_id,
        )
        state["plan"] = plan.model_dump()
        state["groundedness"] = groundedness.model_dump()
        state["status"] = WorkflowStatus.INSUFFICIENT_EVIDENCE.value
        self.audit.append(
            state["workflow_id"],
            AuditEventType.GROUNDEDNESS_CHECKED,
            {
                "citation_coverage": groundedness.citation_coverage,
                "confidence": groundedness.confidence,
                "hallucinated_labels": groundedness.hallucinated_labels,
                "refused": True,
                "refusal_reason": reason,
            },
        )
        self.audit.append(
            state["workflow_id"],
            AuditEventType.WORKFLOW_COMPLETED,
            {"status": state["status"], "refusal_reason": reason},
        )
        self._persist(state)
        return state

    def _node_approval_gate(self, state: AgentState) -> AgentState:
        if settings.require_approval_for_writes and not state.get("auto_approve"):
            state["status"] = WorkflowStatus.AWAITING_APPROVAL.value
            self._persist(state)
            return state
        state["status"] = WorkflowStatus.COMPLETED.value
        self.audit.append(
            state["workflow_id"],
            AuditEventType.APPROVAL_RESOLVED,
            {"approved": True, "mode": "auto"},
        )
        self.audit.append(
            state["workflow_id"],
            AuditEventType.WORKFLOW_COMPLETED,
            {"status": state["status"]},
        )
        self._persist(state)
        return state

    def _route_after_context(self, state: AgentState) -> str:
        if state.get("status") == WorkflowStatus.INSUFFICIENT_EVIDENCE.value:
            return "refuse"
        return "plan"

    def _route_after_groundedness(self, state: AgentState) -> str:
        if state.get("status") == WorkflowStatus.INSUFFICIENT_EVIDENCE.value:
            return "done"
        return "approval"

    def _route_after_approval(self, state: AgentState) -> str:
        if state.get("status") == WorkflowStatus.AWAITING_APPROVAL.value:
            return "wait"
        return "done"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("start", self._node_start)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("tool", self._node_tool)
        graph.add_node("context", self._node_context)
        graph.add_node("plan", self._node_plan)
        graph.add_node("groundedness", self._node_groundedness)
        graph.add_node("approval_gate", self._node_approval_gate)
        graph.set_entry_point("start")
        graph.add_edge("start", "retrieve")
        graph.add_edge("retrieve", "tool")
        graph.add_edge("tool", "context")
        graph.add_conditional_edges(
            "context",
            self._route_after_context,
            {"refuse": END, "plan": "plan"},
        )
        graph.add_edge("plan", "groundedness")
        graph.add_conditional_edges(
            "groundedness",
            self._route_after_groundedness,
            {"done": END, "approval": "approval_gate"},
        )
        graph.add_conditional_edges(
            "approval_gate",
            self._route_after_approval,
            {"wait": END, "done": END},
        )
        return graph.compile()

    def start_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        requirement = validate_payload(request.requirement)
        idempotency_key = request.idempotency_key or new_id("idem_")
        existing = self.checkpoints.get_by_idempotency_key(idempotency_key)
        if existing:
            return self._to_response(existing["state"])

        created = utc_now().isoformat()
        state: AgentState = {
            "workflow_id": new_id("wf_"),
            "change_id": request.change_id or "CHG-1002",
            "idempotency_key": idempotency_key,
            "requirement": requirement,
            "auto_approve": request.auto_approve,
            "citations": [],
            "retrieval_trace": {},
            "context_bundle": {},
            "groundedness": {},
            "tool_context": "",
            "plan": {},
            "status": WorkflowStatus.PENDING.value,
            "error": None,
            "created_at": created,
            "updated_at": created,
        }
        try:
            final_state = self.graph.invoke(state)
            return self._to_response(final_state)
        except SecurityError as exc:
            state["status"] = WorkflowStatus.FAILED.value
            state["error"] = str(exc)
            self._persist(state)
            return self._to_response(state)
        except Exception as exc:  # noqa: BLE001
            state["status"] = WorkflowStatus.FAILED.value
            state["error"] = str(exc)
            self._persist(state)
            return self._to_response(state)

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowResponse]:
        row = self.checkpoints.get(workflow_id)
        if row is None:
            return None
        return self._to_response(row["state"])

    def resolve_approval(
        self, workflow_id: str, approved: bool, reviewer: str, note: str = ""
    ) -> Optional[WorkflowResponse]:
        row = self.checkpoints.get(workflow_id)
        if row is None:
            return None
        state: AgentState = dict(row["state"])
        if state.get("status") != WorkflowStatus.AWAITING_APPROVAL.value:
            return self._to_response(state)

        self.audit.append(
            workflow_id,
            AuditEventType.APPROVAL_RESOLVED,
            {"approved": approved, "reviewer": reviewer, "note": note},
        )
        if approved:
            state["status"] = WorkflowStatus.COMPLETED.value
        else:
            state["status"] = WorkflowStatus.REJECTED.value
        self.audit.append(
            workflow_id,
            AuditEventType.WORKFLOW_COMPLETED,
            {"status": state["status"]},
        )
        self._persist(state)
        return self._to_response(state)

    def list_audit(self, workflow_id: str) -> List[Dict[str, Any]]:
        return [e.model_dump(mode="json") for e in self.audit.list_for_workflow(workflow_id)]

    def get_evidence(self, workflow_id: str) -> Optional[EvidenceResponse]:
        row = self.checkpoints.get(workflow_id)
        if row is None:
            return None
        state = row["state"]
        plan = None
        if state.get("plan"):
            plan = DeliveryPlan.model_validate(state["plan"])
        trace = None
        if state.get("retrieval_trace"):
            from agent_runtime.models.schemas import RetrievalTrace

            trace = RetrievalTrace.model_validate(state["retrieval_trace"])
        context = None
        if state.get("context_bundle"):
            context = ContextBundle.model_validate(state["context_bundle"])
        groundedness = None
        if state.get("groundedness"):
            groundedness = GroundednessResult.model_validate(state["groundedness"])
        citations = [Citation.model_validate(c) for c in state.get("citations", [])]
        return EvidenceResponse(
            workflow_id=state["workflow_id"],
            status=WorkflowStatus(state["status"]),
            retrieval_trace=trace,
            citations=citations,
            context=context,
            groundedness=groundedness,
            support=plan.support if plan else [],
            plan_summary=plan.summary if plan else None,
            refusal_reason=plan.refusal_reason if plan else None,
        )

    def _to_response(self, state: Dict[str, Any]) -> WorkflowResponse:
        plan = None
        if state.get("plan"):
            plan = DeliveryPlan.model_validate(state["plan"])
        return WorkflowResponse(
            workflow_id=state["workflow_id"],
            status=WorkflowStatus(state["status"]),
            change_id=state["change_id"],
            idempotency_key=state["idempotency_key"],
            plan=plan,
            error=state.get("error"),
            created_at=state["created_at"],
            updated_at=state.get("updated_at", state["created_at"]),
        )


_runtime: Optional[RuntimeService] = None


def get_runtime() -> RuntimeService:
    global _runtime
    if _runtime is None:
        _runtime = RuntimeService()
    return _runtime
