from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100)
    document_id: UUID | None = None

class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    model: str
    distance: float
    similarity: float
    section: str | None
    page_numbers: list[int]
    source_ids: list[str]

class SearchResponse(BaseModel):
    results: list[SearchHit]
