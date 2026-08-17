from __future__ import annotations

import re
from typing import Optional, Sequence

from agent_runtime.models.schemas import Citation, ContextBundle, LabeledChunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_tiktoken_encoding = None
_tiktoken_tried = False


def reset_tokenizer_cache() -> None:
    """Test helper: drop the cached encoder so the fallback path can be forced."""
    global _tiktoken_encoding, _tiktoken_tried
    _tiktoken_encoding = None
    _tiktoken_tried = False


def tokenizer_name() -> str:
    return "tiktoken" if _get_encoding() is not None else "heuristic"


def _get_encoding():
    global _tiktoken_encoding, _tiktoken_tried
    if not _tiktoken_tried:
        _tiktoken_tried = True
        try:
            import tiktoken

            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _tiktoken_encoding = None
    return _tiktoken_encoding


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, (len(text) + 3) // 4)


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = _SENTENCE_RE.split(stripped)
    return [p.strip() for p in parts if p.strip()]


def truncate_to_budget(text: str, budget: int) -> tuple[str, bool]:
    """Keep whole sentences that fit; hard-cut only if the first sentence overflows."""
    if budget <= 0:
        return "", True
    if count_tokens(text) <= budget:
        return text, False

    kept: list[str] = []
    for sentence in _split_sentences(text):
        candidate = " ".join([*kept, sentence]) if kept else sentence
        if count_tokens(candidate) <= budget:
            kept.append(sentence)
        else:
            break
    if kept:
        return " ".join(kept), True

    if _get_encoding() is None:
        char_budget = max(0, budget * 4)
        return text[:char_budget].rstrip(), True

    lo, hi, best = 0, len(text), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        piece = text[:mid]
        if count_tokens(piece) <= budget:
            best = piece
            lo = mid + 1
        else:
            hi = mid - 1
    return best.rstrip(), True


def _format_block(label: str, citation: Citation, body: str) -> str:
    return f"[{label}] source={citation.source} score={citation.score}\n{body}"


def build_context(
    citations: Sequence[Citation],
    budget_tokens: int,
    per_source_cap: Optional[int] = None,
) -> ContextBundle:
    """Pack scored citations into a labelled context that fits a token budget.

    Selection is score order. A per-source cap is applied here as well as at
    retrieval so a later config change cannot let one document monopolise the
    window. The last included chunk is truncated on a sentence boundary rather
    than mid-token. Chunks that do not fit are counted, not silently dropped.
    """
    ordered = sorted(citations, key=lambda c: c.score, reverse=True)
    selected: list[LabeledChunk] = []
    dropped: list[str] = []
    per_source: dict[str, int] = {}
    used = 0
    truncated_count = 0
    index = 0

    for citation in ordered:
        if per_source_cap is not None and per_source.get(citation.source, 0) >= per_source_cap:
            dropped.append(citation.chunk_id)
            continue
        remaining = budget_tokens - used
        if remaining <= 0:
            dropped.append(citation.chunk_id)
            continue

        index += 1
        label = f"S{index}"
        full = _format_block(label, citation, citation.text)
        full_tokens = count_tokens(full)
        body = citation.text
        truncated = False
        if full_tokens > remaining:
            header = _format_block(label, citation, "")
            header_tokens = count_tokens(header)
            body_budget = remaining - header_tokens
            body, truncated = truncate_to_budget(citation.text, body_budget)
            if not body.strip():
                index -= 1
                dropped.append(citation.chunk_id)
                continue
            full = _format_block(label, citation, body)
            full_tokens = count_tokens(full)
            if full_tokens > remaining:
                index -= 1
                dropped.append(citation.chunk_id)
                continue

        selected.append(
            LabeledChunk(
                label=label,
                chunk_id=citation.chunk_id,
                source=citation.source,
                score=citation.score,
                text=body,
                truncated=truncated,
            )
        )
        used += full_tokens
        per_source[citation.source] = per_source.get(citation.source, 0) + 1
        if truncated:
            truncated_count += 1

    blocks = [
        f"[{item.label}] source={item.source} score={item.score}\n{item.text}"
        for item in selected
    ]
    context_text = "\n\n".join(blocks)
    return ContextBundle(
        context_text=context_text,
        selected=selected,
        label_to_chunk_id={item.label: item.chunk_id for item in selected},
        tokens_used=count_tokens(context_text) if context_text else 0,
        token_budget=budget_tokens,
        tokenizer=tokenizer_name(),
        dropped_count=len(dropped),
        dropped_chunk_ids=dropped,
        truncated_count=truncated_count,
    )
