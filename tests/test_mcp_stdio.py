from __future__ import annotations

from agent_runtime.graph.runtime import RuntimeService
from agent_runtime.models.schemas import WorkflowRequest, WorkflowStatus
from agent_runtime.tools.mcp_client import RequirementLookupMcpClient


def test_mcp_stdio_lookup_direct():
    client = RequirementLookupMcpClient()
    result = client.lookup("CHG-1001")
    assert result.ok is True
    assert result.transport == "mcp-stdio"
    assert result.data["owner"] == "payments-platform"


def test_mcp_stdio_lookup_missing():
    client = RequirementLookupMcpClient()
    result = client.lookup("CHG-9999")
    assert result.ok is False
    assert "not found" in (result.error or "")


def test_workflow_uses_mcp_transport(monkeypatch):
    monkeypatch.setenv("TOOL_TRANSPORT", "mcp")
    from agent_runtime import config as config_module
    from agent_runtime.config import Settings
    from agent_runtime.graph import runtime as runtime_module

    config_module.settings = Settings()
    runtime_module._runtime = None

    runtime = RuntimeService()
    response = runtime.start_workflow(
        WorkflowRequest(
            requirement=(
                "Add idempotent retries for webhook delivery failures "
                "using idempotency keys."
            ),
            change_id="CHG-1001",
            auto_approve=True,
        )
    )
    assert response.status == WorkflowStatus.COMPLETED
    events = runtime.list_audit(response.workflow_id)
    tool_events = [e for e in events if e["event_type"] == "tool_invoked"]
    assert tool_events
    assert tool_events[0]["payload"]["transport"] == "mcp-stdio"
    assert tool_events[0]["payload"]["ok"] is True
