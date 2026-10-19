from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from qbr.api.contracts.vectors import SearchHit

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class AnswerRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=30)
    document_id: UUID | None = None

class Fact(StrictModel):
    statement: str
    chunk_ids: list[str]

class AnswerDraft(StrictModel):
    answer: str
    summary: str
    facts: list[Fact]
    insufficient_evidence: bool

class AnswerResponse(AnswerDraft):
    query: str
    sources: list[SearchHit]
