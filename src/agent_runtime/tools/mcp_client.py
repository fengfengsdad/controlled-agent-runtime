"""MCP stdio client that spawns the requirement_lookup server subprocess."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Dict, Optional

from agent_runtime.tools.mcp_protocol import (
    PROTOCOL_VERSION,
    decode_message,
    encode_message,
    make_notification,
    make_request,
)


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    data: dict
    error: Optional[str] = None
    transport: str = "mcp-stdio"


class McpStdioError(RuntimeError):
    pass


class RequirementLookupMcpClient:
    """Starts a real MCP stdio server session per lookup (simple + auditable)."""

    def __init__(
        self,
        server_module: str = "agent_runtime.tools.mcp_requirement_server",
        python_executable: Optional[str] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.server_module = server_module
        self.python_executable = python_executable or sys.executable
        self.timeout_sec = timeout_sec

    def _spawn(self) -> Popen:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure src layout imports resolve when launched as module.
        src_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(
            [src_root, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [src_root]
        )
        return Popen(
            [self.python_executable, "-m", self.server_module],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            env=env,
            bufsize=1,
        )

    def _exchange(
        self,
        proc: Popen,
        message: Dict[str, Any],
        expect_response: bool = True,
    ) -> Optional[Dict[str, Any]]:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(encode_message(message))
        proc.stdin.flush()
        if not expect_response:
            return None
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise McpStdioError(f"MCP server closed stdout unexpectedly. stderr={stderr!r}")
        return decode_message(line)

    def lookup(self, change_id: str) -> ToolResult:
        proc = self._spawn()
        try:
            init_result = self._exchange(
                proc,
                make_request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {
                            "name": "controlled-agent-runtime",
                            "version": "0.1.0",
                        },
                    },
                    req_id=1,
                ),
            )
            if not init_result or "result" not in init_result:
                raise McpStdioError(f"initialize failed: {init_result}")

            self._exchange(
                proc,
                make_notification("notifications/initialized"),
                expect_response=False,
            )

            tools = self._exchange(proc, make_request("tools/list", req_id=2))
            tool_names = [t.get("name") for t in (tools or {}).get("result", {}).get("tools", [])]
            if "requirement_lookup" not in tool_names:
                raise McpStdioError(f"requirement_lookup not advertised: {tool_names}")

            call = self._exchange(
                proc,
                make_request(
                    "tools/call",
                    {
                        "name": "requirement_lookup",
                        "arguments": {"change_id": change_id},
                    },
                    req_id=3,
                ),
            )
            if not call or "result" not in call:
                err = (call or {}).get("error", {})
                raise McpStdioError(f"tools/call failed: {err or call}")

            result = call["result"]
            structured = result.get("structuredContent")
            if structured is None:
                # Fallback: parse first text content block.
                content = result.get("content") or []
                text = content[0]["text"] if content else "{}"
                structured = json.loads(text)

            return ToolResult(
                tool_name="requirement_lookup",
                ok=bool(structured.get("ok")),
                data=structured.get("data") or {},
                error=structured.get("error"),
                transport="mcp-stdio",
            )
        finally:
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.kill()
            proc.wait(timeout=2)
