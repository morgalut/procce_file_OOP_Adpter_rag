"""Application-facing persistence port; no SQLAlchemy imports."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True, slots=True)
class ExtractionRecord:
    document_id: str
    filename: str
    sha256: str
    status: str
    metadata: dict[str, Any] | None
    validation: dict[str, Any] | None
    page_count: int | None = None
    fragment_count: int | None = None
    incomplete: bool = False
    fragments: tuple[dict[str, Any], ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    storage_uri: str | None = None
    parser_version: str | None = None
    preprocessor_version: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None

class ExtractionStore(Protocol):
    async def save(self, record: ExtractionRecord) -> None: ...
    async def get(self, document_id: str) -> ExtractionRecord | None: ...
