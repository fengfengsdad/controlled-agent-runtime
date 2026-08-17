from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from agent_runtime.config import settings
from agent_runtime.graph.runtime import get_runtime
from agent_runtime.models.schemas import (
    ApprovalRequest,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
    WorkflowRequest,
    WorkflowResponse,
)
from agent_runtime.observability.tracing import setup_tracing
from agent_runtime.security.controls import SecurityError


def create_app() -> FastAPI:
    settings.ensure_dirs()
    setup_tracing()
    app = FastAPI(
        title="Controlled Agent Runtime",
        description=(
            "Engineering-change delivery agent runtime with RAG, approval gates, "
            "audit trail, and restart-safe checkpoints."
        ),
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_provider=settings.llm_provider,
            embedding_provider=settings.embedding_provider,
            prompt_version=settings.prompt_version,
            retrieval_mode=settings.retrieval_mode,
            rag_top_k=settings.rag_top_k,
            candidate_k=settings.candidate_k,
            indexed_chunks=len(get_runtime().vector_store.records),
        )

    @app.post(
        "/v1/retrieval/search",
        response_model=RetrievalResponse,
        tags=["retrieval"],
        summary="Run retrieval on its own, with per-request mode and filters",
        description=(
            "Exposes the retrieval stage without generation so relevance can be "
            "inspected and A/B compared across vector, bm25, and hybrid modes, "
            "with the rerank stage switchable per query."
        ),
    )
    def search(request: RetrievalRequest) -> RetrievalResponse:
        try:
            result = get_runtime().retrieve(
                query=request.query,
                top_k=request.top_k,
                mode=request.mode,
                filters=request.filters,
                rerank=request.rerank,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RetrievalResponse(
            query=request.query, citations=result.citations, trace=result.trace
        )

    @app.post(
        "/v1/workflows",
        response_model=WorkflowResponse,
        tags=["workflows"],
        summary="Start an engineering-change planning workflow",
    )
    def start_workflow(request: WorkflowRequest) -> WorkflowResponse:
        runtime = get_runtime()
        try:
            return runtime.start_workflow(request)
        except SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/v1/workflows/{workflow_id}",
        response_model=WorkflowResponse,
        tags=["workflows"],
    )
    def get_workflow(workflow_id: str) -> WorkflowResponse:
        result = get_runtime().get_workflow(workflow_id)
        if result is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        return result

    @app.post(
        "/v1/workflows/{workflow_id}/approval",
        response_model=WorkflowResponse,
        tags=["workflows"],
    )
    def approve_workflow(workflow_id: str, body: ApprovalRequest) -> WorkflowResponse:
        result = get_runtime().resolve_approval(
            workflow_id=workflow_id,
            approved=body.approved,
            reviewer=body.reviewer,
            note=body.note,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        return result

    @app.get("/v1/workflows/{workflow_id}/audit", tags=["audit"])
    def get_audit(workflow_id: str):
        runtime = get_runtime()
        if runtime.get_workflow(workflow_id) is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        return {"workflow_id": workflow_id, "events": runtime.list_audit(workflow_id)}

    @app.post("/v1/admin/reindex", tags=["ops"])
    def reindex():
        count = get_runtime().reindex_corpus()
        return {"indexed_chunks": count}

    @app.exception_handler(SecurityError)
    async def security_handler(_, exc: SecurityError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "agent_runtime.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
