from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import httpx

from agent_runtime.config import settings
from agent_runtime.models.schemas import Citation
from agent_runtime.rag.text import tokenize as _tokenize

DOC_TYPES = {".md": "markdown", ".txt": "text"}


@dataclass
class SourceDocument:
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class StubEmbedder(Embedder):
    """Deterministic bag-of-words embedding for local/offline runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _tokenize(text):
                digest = hashlib.sha256(token.encode()).digest()
                idx = int.from_bytes(digest[:2], "big") % self.dim
                sign = 1.0 if digest[2] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAICompatibleEmbedder(Embedder):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


def get_embedder() -> Embedder:
    if settings.embedding_provider == "openai_compatible":
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY required for openai_compatible embeddings")
        return OpenAICompatibleEmbedder(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.embedding_model,
        )
    return StubEmbedder()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        self.path = path
        self.embedder = embedder
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "index.json"
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.index_file.exists():
            self._records = json.loads(self.index_file.read_text(encoding="utf-8"))
        else:
            self._records = []

    def _save(self) -> None:
        self.index_file.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self._records = []
        self._save()

    @property
    def records(self) -> list[dict]:
        return self._records

    def record_map(self) -> dict:
        return {rec["chunk_id"]: rec for rec in self._records}

    def add_documents(
        self, documents: Sequence[Union[SourceDocument, tuple]]
    ) -> int:
        """documents: SourceDocument entries, or legacy (source, text) tuples."""
        normalised = [
            doc if isinstance(doc, SourceDocument) else SourceDocument(doc[0], doc[1])
            for doc in documents
        ]
        all_chunks: list[tuple[str, str, str, dict]] = []
        for doc in normalised:
            chunks = chunk_text(doc.text, settings.chunk_size, settings.chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha1(
                    f"{doc.source}:{i}:{chunk[:32]}".encode()
                ).hexdigest()[:12]
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "source": doc.source,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    }
                )
                metadata.setdefault("doc_type", "text")
                metadata.setdefault("classification", settings.default_classification)
                metadata.setdefault("owner", settings.default_owner)
                all_chunks.append((chunk_id, doc.source, chunk, metadata))
        if not all_chunks:
            return 0
        vectors = self.embedder.embed([c[2] for c in all_chunks])
        for (chunk_id, source, chunk, metadata), vector in zip(all_chunks, vectors):
            self._records.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": chunk,
                    "metadata": metadata,
                    "vector": vector,
                }
            )
        self._save()
        return len(all_chunks)

    def vector_search(
        self,
        query: str,
        records: Optional[Sequence[dict]] = None,
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        candidates = self._records if records is None else records
        if not candidates:
            return []
        query_vec = self.embedder.embed([query])[0]
        scored = [
            (rec["chunk_id"], _cosine(query_vec, rec["vector"])) for rec in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_n] if top_n is not None else scored

    def search(self, query: str, top_k: int | None = None) -> list[Citation]:
        """Vector-only retrieval, retained as the A/B baseline."""
        top_k = top_k or settings.rag_top_k
        record_map = self.record_map()
        results: list[Citation] = []
        for rank, (chunk_id, score) in enumerate(
            self.vector_search(query, top_n=top_k), start=1
        ):
            rec = record_map[chunk_id]
            results.append(
                Citation(
                    chunk_id=chunk_id,
                    source=rec["source"],
                    score=round(float(score), 4),
                    text=rec["text"],
                    metadata=rec.get("metadata") or {},
                    retrievers=["vector"],
                    vector_score=round(float(score), 4),
                    vector_rank=rank,
                )
            )
        return results


def document_metadata(path: Path) -> dict:
    updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "doc_type": DOC_TYPES.get(path.suffix.lower(), "text"),
        "classification": settings.default_classification,
        "owner": settings.default_owner,
        "updated_at": updated_at.isoformat(),
    }


def ingest_corpus(corpus_dir: Path, store: VectorStore) -> int:
    documents: list[SourceDocument] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if path.is_file() and path.suffix.lower() in DOC_TYPES:
            documents.append(
                SourceDocument(
                    source=path.name,
                    text=path.read_text(encoding="utf-8"),
                    metadata=document_metadata(path),
                )
            )
    if not documents:
        return 0
    store.clear()
    return store.add_documents(documents)
