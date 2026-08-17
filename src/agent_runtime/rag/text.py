from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) <= size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}
