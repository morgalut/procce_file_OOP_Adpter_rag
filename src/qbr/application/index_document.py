"""Index existing document PDF into pgvector without re-running extraction."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexDocument:
    loader: object
    pipeline: object

    async def execute(self, document_id: str) -> dict:
        sources = await self.loader.load(document_id)
        vectors = await self.pipeline.execute(document_id, sources)
        return {
            'document_id': document_id,
            'status': 'completed',
            'chunks_created': len(vectors),
            'embedding_model': vectors[0].model if vectors else None,
            'dimensions': len(vectors[0].embedding) if vectors else None,
        }
