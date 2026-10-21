"""Page-wise PyMuPDF extraction with stable table/cell provenance."""
from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import pymupdf
import importlib
import logging

logger = logging.getLogger(__name__)
from qbr.domain.exceptions import DocumentNotFoundError, ExtractionError, InvalidDocumentError
from qbr.ports.pdf_extractor import PDFCell, PDFExtractionOptions, PDFPage, PDFSource, PDFTable, PDFTextBlock

@dataclass(frozen=True, slots=True)
class PyMuPDFLimits:
    max_file_size_mb: int = 20
    max_pages: int = 100
    use_layout: bool = False
    layout_required: bool = False
    def __post_init__(self):
        if self.max_file_size_mb < 1 or self.max_pages < 1: raise ValueError('PDF limits must be positive')

class PyMuPDFAdapter:
    def __init__(self, limits: PyMuPDFLimits | None = None):
        self._limits = limits or PyMuPDFLimits()
        if self._limits.layout_required and not self._limits.use_layout:
            raise ValueError("layout_required requires use_layout=True")
        self._layout_active = False
        if self._limits.use_layout:
            try:
                # The package registers enhanced PyMuPDF layout functionality on import.
                # Do not assume that importing it alone proves a given page was improved.
                importlib.import_module("pymupdf_layout")
                self._layout_active = True
            except ImportError as exc:
                if self._limits.layout_required:
                    raise RuntimeError("pymupdf_layout is required but not installed") from exc
                logger.warning("pymupdf_layout unavailable; using standard PyMuPDF")

    @property
    def layout_active(self) -> bool:
        return self._layout_active

    def iter_pages(self, source: PDFSource, *, options: PDFExtractionOptions | None = None) -> AsyncIterator[PDFPage]:
        return self._iter_pages(source, options or PDFExtractionOptions())

    async def _iter_pages(self, source: PDFSource, options: PDFExtractionOptions) -> AsyncIterator[PDFPage]:
        loop = asyncio.get_running_loop()
        # A single worker owns the document; do not move PyMuPDF objects across workers.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='qbr-pdf') as pool:
            document = None
            try:
                document = await loop.run_in_executor(pool, self._open_document, source)
                count = document.page_count
                if count > self._limits.max_pages:
                    raise InvalidDocumentError('PDF exceeds maximum page count', context={'document_id': source.document_id, 'pages': count})
                if options.start_page > count:
                    raise InvalidDocumentError('Requested start page exceeds document length', context={'document_id': source.document_id, 'pages': count})
                last = min(options.end_page if options.end_page is not None else count, count)
                for number in range(options.start_page, last + 1):
                    try:
                        result = await loop.run_in_executor(pool, self._extract_page, document, number, options.extract_tables, self._layout_active)
                    except Exception as exc:
                        raise ExtractionError('Unable to extract PDF page', context={'document_id': source.document_id, 'page_number': number}) from exc
                    yield result
            finally:
                if document is not None:
                    await asyncio.shield(loop.run_in_executor(pool, document.close))

    def _open_document(self, source: PDFSource):
        path: Path = source.path
        if not path.is_file():
            raise DocumentNotFoundError('PDF source file does not exist', context={'document_id': source.document_id})
        if path.suffix.lower() != '.pdf': raise InvalidDocumentError('Source must have a .pdf extension')
        try: size = path.stat().st_size
        except OSError as exc: raise InvalidDocumentError('Unable to inspect PDF file') from exc
        if size == 0 or size > self._limits.max_file_size_mb * 1024 * 1024:
            raise InvalidDocumentError('PDF file size is empty or exceeds configured limit', context={'document_id': source.document_id, 'size_bytes': size})
        try: document = pymupdf.open(path)
        except (pymupdf.FileDataError, pymupdf.FileNotFoundError, OSError) as exc:
            raise InvalidDocumentError('PDF cannot be opened', context={'document_id': source.document_id}) from exc
        if document.needs_pass:
            document.close(); raise InvalidDocumentError('Password-protected PDFs are unsupported')
        if document.page_count == 0:
            document.close(); raise InvalidDocumentError('PDF contains no pages')
        return document

    @staticmethod
    def _extract_page(document, page_number: int, extract_tables: bool, layout_active: bool = False) -> PDFPage:
        page = document.load_page(page_number - 1)
        text = page.get_text('text', sort=True)
        blocks = []
        for i, block in enumerate(page.get_text('blocks', sort=True)):
            if len(block) > 6 and block[6] != 0: continue  # exclude image blocks
            if str(block[4]).strip():
                blocks.append(PDFTextBlock(f'p{page_number}-b{i}', str(block[4]), tuple(float(v) for v in block[:4])))
        tables = []
        warnings = []
        if layout_active:
            warnings.append("layout_plugin_loaded_not_page_verified")
        if extract_tables:
            try:
                finder = page.find_tables()
                for table_index, table in enumerate(finder.tables):
                    raw = table.extract()
                    if not raw: continue
                    rows = tuple(tuple(None if cell is None else str(cell) for cell in row) for row in raw)
                    table_id = f'p{page_number}-t{table_index}'
                    cells = []
                    # PyMuPDF table.cells is row-major, and includes None for missing geometry.
                    positions = getattr(table, 'cells', ())
                    ncols = max(len(row) for row in rows)
                    for ri, row in enumerate(rows):
                        for ci, value in enumerate(row):
                            idx = ri * ncols + ci
                            rect = positions[idx] if idx < len(positions) else None
                            bbox = tuple(float(v) for v in rect) if rect is not None else None
                            cells.append(PDFCell(f'{table_id}-r{ri}-c{ci}', ri, ci, value, bbox))
                    tables.append(PDFTable(rows, page_number, table_id, tuple(float(v) for v in table.bbox), tuple(cells)))
            except Exception:
                warnings.append('table_extraction_failed')
        if not text.strip() and not tables: warnings.append('no_extractable_content')
        return PDFPage(page_number, text, tuple(tables), tuple(warnings), tuple(blocks))
