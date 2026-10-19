"""Read already-stored PDF for vector indexing; no LLM or metadata re-extraction."""
from __future__ import annotations
from pathlib import Path
from uuid import UUID
from qbr.ports.chunker import ChunkSource


class StoredPDFSourceLoader:
    def __init__(self, document_service, *, max_pages: int = 200, max_file_bytes: int = 20 * 1024 * 1024):
        self.documents = document_service
        self.max_pages = max_pages
        self.max_file_bytes = max_file_bytes

    async def load(self, document_id: str) -> tuple[ChunkSource, ...]:
        document_id = str(UUID(document_id))
        record = await self.documents.get(document_id)
        if not record.storage_uri:
            raise ValueError('document has no stored PDF')
        # Never open arbitrary URI paths: only paths inside the configured storage root.
        storage = self.documents.storage
        root = getattr(storage, 'root', None) or getattr(storage, '_root', None) or getattr(storage, '_directory', None)
        if root is None:
            raise ValueError('PDF storage must expose its trusted root directory')
        root = Path(root).resolve()
        path = Path(record.storage_uri).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('PDF storage path is missing or outside configured root')
        if path.stat().st_size > self.max_file_bytes:
            raise ValueError('stored PDF exceeds configured size limit')
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError('PyMuPDF (pymupdf) is required') from exc
        sources = []
        with fitz.open(path) as pdf:
            if pdf.page_count > self.max_pages:
                raise ValueError('PDF exceeds maximum page count')
            for i, page in enumerate(pdf):
                text = page.get_text('text').strip()
                if text:
                    sources.append(ChunkSource(document_id=document_id, page_number=i + 1, text=text,
                                               source_ids=(f'pdf-page-{i + 1}',)))
        if not sources:
            raise ValueError('PDF has no extractable text; OCR required')
        return tuple(sources)
