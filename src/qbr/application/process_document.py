"""Explicit application orchestration; delegates extraction and persistence."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
from qbr.ports.pdf_extractor import PDFSource


class Extractor(Protocol):
    async def execute(self, source: PDFSource) -> Any: ...


class Persister(Protocol):
    async def execute(self, result: Any, *, filename: str, sha256: str,
                      storage_uri: str | None = None) -> Any: ...


class DocumentProvenanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessDocument:
    extractor: Extractor
    persister: Persister

    async def execute(self, source: PDFSource, *, filename: str,
                      sha256: str, storage_uri: str | None = None) -> Any:
        if not filename.strip() or not sha256.strip():
            raise ValueError("filename and sha256 are required")
        result = await self.extractor.execute(source)
        if result.document_id != source.document_id:
            raise DocumentProvenanceError("extraction document provenance mismatch")
        return await self.persister.execute(result, filename=filename,
                                            sha256=sha256, storage_uri=storage_uri)
