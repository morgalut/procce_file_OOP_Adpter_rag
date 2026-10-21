"""HTTP response contracts only."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from qbr.ports.extraction_store import ExtractionRecord

class UploadResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: str
    status: str
    incomplete: bool
    page_count: int | None
    fragment_count: int | None

class DocumentResult(UploadResult):
    filename: str
    metadata: dict[str, Any] | None
    validation: dict[str, Any] | None
    extraction_diagnostics: list[dict[str, Any]] = Field(default_factory=list)

class ValidationDetails(BaseModel):
    document_id: str
    status: str
    incomplete: bool
    validation: dict[str, Any] | None
    fragments: list[dict[str, Any]]
    evidence: list[dict[str, Any]]


def as_document_result(record: ExtractionRecord) -> DocumentResult:
    return DocumentResult(document_id=record.document_id, status=record.status,
        incomplete=record.incomplete, page_count=record.page_count,
        fragment_count=record.fragment_count, filename=record.filename,
        metadata=record.metadata, validation=record.validation,
        extraction_diagnostics=[{
            "fragment_id": f.get("fragment_id"),
            "page_number": f.get("page_number"),
            "failure_code": f.get("failure_code"),
            "warnings": f.get("warnings", []),
            "fallback_used": f.get("fallback_used", False),
        } for f in record.fragments])
