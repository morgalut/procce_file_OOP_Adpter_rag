"""Embed prepared chunks; persistence is optional and adapter-injected.

No PDF extraction, LLM metadata generation, or database-specific code belongs here.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol, Sequence


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def save_vectors(self, document_id: str, vectors: tuple["VectorizedChunk", ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class VectorizedChunk:
    chunk_id: str
    document_id: str
    text: str
    embedding: tuple[float, ...]
    model: str


@dataclass(frozen=True, slots=True)
class VectorizeDocument:
    embedder: Embedder
    model: str
    store: VectorStore | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not self.model.strip() or self.batch_size < 1:
            raise ValueError("model must be nonempty and batch_size positive")

    async def execute(self, document_id: str, chunks: Sequence[Any]) -> tuple[VectorizedChunk, ...]:
        if not document_id.strip():
            raise ValueError("document_id must not be blank")
        if not chunks:
            return ()
        if any(getattr(c, "document_id", None) != document_id for c in chunks):
            raise ValueError("chunk document provenance mismatch")
        result: list[VectorizedChunk] = []
        dimension: int | None = None
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            texts = [getattr(c, "text") for c in batch]
            if any(not isinstance(t, str) or not t.strip() for t in texts):
                raise ValueError("cannot embed empty chunks")
            vectors = await self.embedder.embed(texts)
            if len(vectors) != len(batch):
                raise ValueError("embedding count mismatch")
            for idx, (chunk, vector) in enumerate(zip(batch, vectors, strict=True)):
                if not vector or any(not isinstance(v, (int, float)) or not isfinite(v) for v in vector):
                    raise ValueError("embedding contains invalid values")
                if dimension is None:
                    dimension = len(vector)
                if len(vector) != dimension:
                    raise ValueError("embedding dimension mismatch")
                chunk_id = getattr(chunk, "chunk_id", None)
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError("chunk_id is required")
                result.append(VectorizedChunk(chunk_id=chunk_id, document_id=document_id,
                                              text=texts[idx], embedding=tuple(float(v) for v in vector),
                                              model=self.model))
        output = tuple(result)
        if self.store is not None:
            await self.store.save_vectors(document_id, output)
        return output
