"""Embed a search query and retrieve grounded pgvector matches."""
from __future__ import annotations
from dataclasses import dataclass
from uuid import UUID
from qbr.infrastructure.event_tracking import stage


@dataclass(frozen=True)
class VectorSearch:
    embedder: object
    store: object
    model: str

    async def execute(self, query: str, *, top_k: int = 5, document_id: str | None = None):
        if not query.strip() or len(query) > 4000:
            raise ValueError('query must be 1..4000 non-whitespace characters')
        if not 1 <= top_k <= 100:
            raise ValueError('top_k must be between 1 and 100')
        if document_id is not None:
            document_id = str(UUID(document_id))
        async with stage("embedding.query", input_summary={"query_length": len(query), "model": self.model}) as details:
            vectors = await self.embedder.embed([query])
            if len(vectors) != 1:
                raise ValueError('embedding provider returned incorrect count')
            details.finish(vector_count=1, dimensions=len(vectors[0]), model=self.model)
        async with stage("vector_database.search", input_summary={"top_k": top_k, "document_filter": document_id, "model": self.model}) as details:
            matches = await self.store.search(vectors[0], limit=top_k, document_id=document_id, model=self.model)
            details.finish(result_count=len(matches))
            return matches
