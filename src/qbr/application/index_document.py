"""Index existing document PDF into pgvector without re-running extraction."""
from __future__ import annotations
from dataclasses import dataclass
from qbr.infrastructure.event_tracking import stage


@dataclass(frozen=True)
class IndexDocument:
    loader: object
    pipeline: object

    async def execute(self, document_id: str) -> dict:
        async with stage("vector_source.load", input_summary={"document_id": document_id}) as details:
            sources = await self.loader.load(document_id)
            details.finish(source_count=len(sources))
        vectors = await self.pipeline.execute(document_id, sources)
        return {
            'document_id': document_id,
            'status': 'completed',
            'chunks_created': len(vectors),
            'embedding_model': vectors[0].model if vectors else None,
            'dimensions': len(vectors[0].embedding) if vectors else None,
        }
