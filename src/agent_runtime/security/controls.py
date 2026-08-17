from __future__ import annotations

import re

from agent_runtime.config import settings

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+safety", re.IGNORECASE),
]

_SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9]{10,}"), "sk-[REDACTED]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
]


class SecurityError(ValueError):
    pass


def validate_payload(text: str) -> str:
    if not text or not text.strip():
        raise SecurityError("empty payload")
    if len(text) > settings.max_payload_chars:
        raise SecurityError(
            f"payload exceeds max_payload_chars={settings.max_payload_chars}"
        )
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise SecurityError("prompt-injection pattern rejected")
    return text.strip()


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


ALLOWED_TOOLS = frozenset({"requirement_lookup", "rag_search"})
