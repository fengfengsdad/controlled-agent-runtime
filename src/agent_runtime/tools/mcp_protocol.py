"""Minimal MCP JSON-RPC helpers for newline-delimited stdio transport."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

PROTOCOL_VERSION = "2024-11-05"


def make_request(
    method: str, params: Optional[Dict[str, Any]] = None, req_id: int = 1
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def make_notification(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message


def make_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def encode_message(message: Dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False) + "\n"


def decode_message(line: str) -> Dict[str, Any]:
    return json.loads(line)
