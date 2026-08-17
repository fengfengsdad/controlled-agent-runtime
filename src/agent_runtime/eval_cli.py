from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from agent_runtime.config import reload_settings, settings
from agent_runtime.graph.runtime import RuntimeService
from agent_runtime.models.schemas import WorkflowRequest, WorkflowStatus
from agent_runtime.security.controls import SecurityError, validate_payload


def load_cases(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_overrides(offline: bool, mode: Optional[str], rerank: Optional[bool]) -> None:
    """Pin providers, retrieval mode, and rerank before the runtime reads them.

    Offline is the default so a developer `.env` pointing at a hosted model
    cannot silently turn the regression suite into a billed network test.
    """
    if offline:
        os.environ["LLM_PROVIDER"] = "stub"
        os.environ["EMBEDDING_PROVIDER"] = "stub"
        os.environ["RERANKER_PROVIDER"] = "stub"
    if mode:
        os.environ["RETRIEVAL_MODE"] = mode
    if rerank is not None:
        os.environ["RERANK_ENABLED"] = "true" if rerank else "false"
    reload_settings()


def run_eval(
    cases_path: Optional[Path] = None,
    offline: bool = True,
    mode: Optional[str] = None,
    rerank: Optional[bool] = None,
) -> dict:
    root = Path(__file__).resolve().parents[2]
    cases_path = cases_path or (root / "eval" / "cases" / "smoke.json")
    cases = load_cases(cases_path)
    _apply_overrides(offline=offline, mode=mode, rerank=rerank)
    runtime = RuntimeService()

    passed = 0
    results = []
    for case in cases:
        name = case["name"]
        expect = case["expect"]
        ok = False
        detail = ""
        try:
            if expect == "reject_security":
                try:
                    validate_payload(case["requirement"])
                    detail = "expected security rejection"
                except SecurityError:
                    ok = True
                    detail = "rejected"
            else:
                response = runtime.start_workflow(
                    WorkflowRequest(
                        requirement=case["requirement"],
                        change_id=case.get("change_id", "CHG-1002"),
                        idempotency_key=case.get("idempotency_key"),
                        auto_approve=case.get("auto_approve", True),
                    )
                )
                if expect == "completed":
                    ok = response.status == WorkflowStatus.COMPLETED and response.plan is not None
                    detail = response.status.value
                elif expect == "awaiting_approval":
                    ok = response.status == WorkflowStatus.AWAITING_APPROVAL
                    detail = response.status.value
                else:
                    detail = f"unknown expect={expect}"
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
        if ok:
            passed += 1
        results.append({"name": name, "ok": ok, "detail": detail})

    return {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": round(passed / max(len(cases), 1), 4),
        "config": {
            "offline": offline,
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "retrieval_mode": settings.retrieval_mode,
            "rag_top_k": settings.rag_top_k,
            "candidate_k": settings.candidate_k,
            "reranker": settings.reranker_provider if settings.rerank_enabled else None,
            "mmr_lambda": settings.mmr_lambda,
            "per_source_cap": settings.per_source_cap,
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation harness")
    parser.add_argument("--cases", type=Path, default=None, help="path to a cases JSON file")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use configured providers instead of forcing offline stubs",
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid"],
        default=None,
        help="override retrieval mode for this run",
    )
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=None,
        help="force the rerank stage on",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        default=None,
        help="force the rerank stage off, for measuring what it contributes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_eval(
        cases_path=args.cases,
        offline=not args.live,
        mode=args.mode,
        rerank=args.rerank,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
