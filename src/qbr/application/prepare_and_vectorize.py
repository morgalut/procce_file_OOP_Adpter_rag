"""Explicit vector preparation stage, independent of PDF upload.

Uses the V18 ChunkSource/TextChunk and V19 VectorizeDocument contracts.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from qbr.application.prepare_chunks import PrepareChunks
from qbr.application.vectorize_document import VectorizeDocument, VectorizedChunk
from qbr.ports.chunker import ChunkSource


@dataclass(frozen=True)
class IndexedVector:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    embedding: tuple[float, ...]
    model: str
    section: str | None
    page_numbers: tuple[int, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class PrepareAndVectorize:
    prepare: PrepareChunks
    vectorize: VectorizeDocument
    store: object

    async def execute(self, document_id: str, sources: Sequence[ChunkSource]) -> tuple[IndexedVector, ...]:
        chunks = self.prepare.execute(sources, document_id=document_id)
        # V19 VectorizeDocument requires chunk_id; V18 TextChunk supplies chunk_index.
        indexed = []
        for chunk in chunks:
            stable_key = f'{document_id}:{chunk.chunk_index}:{chunk.text}'
            chunk_id = sha256(stable_key.encode('utf-8')).hexdigest()
            indexed.append(_EmbeddingInput(document_id, chunk_id, chunk.text))
        # Disable implicit writes from VectorizeDocument to ensure metadata is saved
        # together with vectors by the provenance-aware pgvector adapter.
        if self.vectorize.store is not None:
            raise ValueError('VectorizeDocument.store must be None for this orchestration')
        vectors = await self.vectorize.execute(document_id, indexed)
        output = tuple(IndexedVector(
            chunk_id=v.chunk_id, document_id=document_id, chunk_index=c.chunk_index,
            text=v.text, embedding=v.embedding, model=v.model,
            section=c.section, page_numbers=c.page_numbers, source_ids=c.source_ids,
        ) for c, v in zip(chunks, vectors, strict=True))
        await self.store.save_vectors(document_id, output)
        return output


@dataclass(frozen=True)
class _EmbeddingInput:
    document_id: str
    chunk_id: str
    text: str
