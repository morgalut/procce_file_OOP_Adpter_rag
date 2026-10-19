"""Use case: validate upload metadata, store, extract, persist, and read."""
from __future__ import annotations
from pathlib import Path
import re
from typing import Any, AsyncIterable, Protocol
from uuid import UUID, uuid4
from qbr.ports.document_storage import DocumentStorage, UploadRejected
from qbr.ports.pdf_extractor import PDFSource
from qbr.ports.extraction_store import ExtractionRecord, ExtractionStore
from qbr.application.process_document import ProcessDocument
from qbr.infrastructure.event_tracking import run_scope, stage

class ExtractionPipeline(Protocol):
    async def execute(self, source: PDFSource) -> Any: ...

class PersistenceService(Protocol):
    async def execute(self, result: Any, *, filename: str, sha256: str,
                      storage_uri: str | None = None) -> ExtractionRecord: ...

class DocumentNotFound(Exception):
    pass

class InvalidDocumentId(Exception):
    pass

class DocumentService:
    def __init__(self, *, pipeline: ExtractionPipeline, persistence: PersistenceService,
                 store: ExtractionStore, storage: DocumentStorage,
                 max_pdf_bytes: int = 20 * 1024 * 1024,
                 processor: ProcessDocument | None = None, tracker: Any = None):
        if max_pdf_bytes < 5:
            raise ValueError('max_pdf_bytes must be at least 5')
        self.tracker = tracker
        self.pipeline = pipeline
        self.persistence = persistence
        self.store = store
        self.storage = storage
        self.max_pdf_bytes = max_pdf_bytes
        # One orchestration path only: ProcessDocument owns extraction + persistence.
        # Legacy constructor arguments remain supported for existing API wiring.
        self.processor = processor or ProcessDocument(extractor=pipeline, persister=persistence)

    async def upload(self, *, filename: str | None, content_type: str | None,
                     content: AsyncIterable[bytes]) -> ExtractionRecord:
        if content_type not in {'application/pdf', 'application/octet-stream'}:
            raise UploadRejected('unsupported_media', 'Expected a PDF upload')
        name = (filename or 'document.pdf').replace('\\', '/').rsplit('/', 1)[-1]
        name = re.sub(r'[^A-Za-z0-9._-]', '_', name).strip('._')[:160] or 'document.pdf'
        if not name.lower().endswith('.pdf'):
            raise UploadRejected('unsupported_media', 'Expected .pdf filename')
        document_id = str(uuid4())
        if self.tracker is not None:
            async with run_scope(self.tracker, "document_upload", document_id):
                return await self._upload_tracked(document_id, name, content)
        return await self._upload_tracked(document_id, name, content)

    async def _upload_tracked(self, document_id: str, name: str, content: AsyncIterable[bytes]) -> ExtractionRecord:
        async with stage("storage.save_pdf"):
            stored = await self.storage.save_pdf(document_id, content, max_bytes=self.max_pdf_bytes)
        try:
            async with stage("document.process"):
                record = await self.processor.execute(
                PDFSource(document_id=document_id, path=Path(stored.uri)),
                filename=name, sha256=stored.sha256, storage_uri=stored.uri,
            )
            if record.document_id != document_id:
                raise RuntimeError("Persistence returned inconsistent document ID")
            return record
        except BaseException:
            async with stage("storage.cleanup"):
                await self.storage.delete(stored.uri)
            raise

    async def get(self, document_id: str) -> ExtractionRecord:
        try:
            normalized = str(UUID(document_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise InvalidDocumentId(document_id) from exc
        record = await self.store.get(normalized)
        if record is None:
            raise DocumentNotFound(normalized)
        return record
