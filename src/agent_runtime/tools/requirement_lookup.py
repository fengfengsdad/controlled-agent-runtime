from __future__ import annotations

from agent_runtime.tools.catalog import lookup_requirement
from agent_runtime.tools.mcp_client import RequirementLookupMcpClient, ToolResult


class RequirementLookupTool:
    """Backward-compatible local lookup. Prefer MCP client in RuntimeService."""

    def lookup(self, change_id: str) -> ToolResult:
        return self.invoke(change_id)

    def invoke(self, change_id: str) -> ToolResult:
        result = lookup_requirement(change_id)
        return ToolResult(
            tool_name="requirement_lookup",
            ok=bool(result["ok"]),
            data=result["data"] or {},
            error=result["error"],
            transport="local-direct",
        )


__all__ = [
    "RequirementLookupMcpClient",
    "RequirementLookupTool",
    "ToolResult",
]
