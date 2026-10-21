"""Auditable context supplied to every document-review request."""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field

class PageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_number: int = Field(ge=1)
    text: str
    tables: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class DocumentReviewContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    total_pages: int = Field(ge=1)
    included_pages: list[int]
    pages: list[PageContext]
    candidate_metadata: dict
    validation_findings: list[dict] = Field(default_factory=list)
    complete: bool
    context_type: str = "source_pages"

    def model_context(self) -> str:
        """Serialized source pages plus a manifest; the metadata is sent separately."""
        import json
        return json.dumps(self.model_dump(exclude={"candidate_metadata"}), ensure_ascii=False)
