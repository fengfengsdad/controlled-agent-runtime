"""MCP stdio server exposing synthetic read-only requirement_lookup tool.

Speaks a practical subset of the Model Context Protocol over newline-delimited
JSON-RPC on stdin/stdout (stdio transport). stderr is reserved for logs.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from agent_runtime.security.controls import ALLOWED_TOOLS, SecurityError, validate_payload
from agent_runtime.tools.catalog import lookup_requirement
from agent_runtime.tools.mcp_protocol import (
    PROTOCOL_VERSION,
    decode_message,
    encode_message,
    make_error,
    make_result,
)

TOOL_NAME = "requirement_lookup"


def _tool_spec() -> Dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Look up a synthetic engineering-change requirement by change_id. "
            "Read-only; returns owner, status, and constraints."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "change_id": {
                    "type": "string",
                    "description": "Engineering change identifier, e.g. CHG-1001",
                }
            },
            "required": ["change_id"],
        },
    }


def _handle_initialize(req_id: Any) -> Dict[str, Any]:
    return make_result(
        req_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "controlled-agent-requirement-server",
                "version": "0.1.0",
            },
        },
    )


def _handle_tools_list(req_id: Any) -> Dict[str, Any]:
    return make_result(req_id, {"tools": [_tool_spec()]})


def _handle_tools_call(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name != TOOL_NAME:
        return make_error(req_id, -32601, f"unknown tool: {name}")
    if TOOL_NAME not in ALLOWED_TOOLS:
        return make_error(req_id, -32000, "tool not allowlisted")

    change_id = arguments.get("change_id", "")
    try:
        change_id = validate_payload(str(change_id)) if change_id else ""
    except SecurityError as exc:
        return make_result(
            req_id,
            {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            },
        )

    result = lookup_requirement(change_id)
    payload = {
        "ok": result["ok"],
        "error": result["error"],
        "data": result["data"],
        "transport": "mcp-stdio",
    }
    return make_result(
        req_id,
        {
            "content": [
                {
                    "type": "text",
                    "text": __import__("json").dumps(payload, ensure_ascii=False),
                }
            ],
            "isError": not bool(result["ok"]),
            "structuredContent": payload,
        },
    )


def handle_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    req_id = message.get("id")

    # Notifications have no id and expect no response.
    if req_id is None and method and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return _handle_initialize(req_id)
    if method == "tools/list":
        return _handle_tools_list(req_id)
    if method == "tools/call":
        return _handle_tools_call(req_id, message.get("params") or {})
    if method == "ping":
        return make_result(req_id, {})

    if req_id is not None:
        return make_error(req_id, -32601, f"method not found: {method}")
    return None


def serve_stdio() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = decode_message(line)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"invalid mcp message: {exc}\n")
            sys.stderr.flush()
            continue
        response = handle_message(message)
        if response is not None:
            sys.stdout.write(encode_message(response))
            sys.stdout.flush()


def main() -> None:
    serve_stdio()


if __name__ == "__main__":
    main()
