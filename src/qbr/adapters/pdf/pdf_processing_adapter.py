"""Section-aware, bounded PDF preparation. Raw page data stays outside LLM payloads.

Keeps the existing fragment/table interfaces; adds section labels and source IDs.
No LLM, PDF parser, or filesystem dependency is required by this module.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from qbr.ports.pdf_extractor import PDFPage, PDFTable


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    max_fragment_chars: int = 6000
    max_table_rows_per_fragment: int = 40
    # The original page is available from PDFPage / the canonical store.
    # Repeating it in every fragment is unnecessary for LLM input.
    include_source_text: bool = False
    group_sections: bool = True

    def __post_init__(self) -> None:
        if self.max_fragment_chars < 100:
            raise ValueError("max_fragment_chars must be >= 100")
        if self.max_table_rows_per_fragment < 1:
            raise ValueError("max_table_rows_per_fragment must be positive")


@dataclass(frozen=True, slots=True)
class CleanTable:
    page_number: int
    table_index: int
    rows: tuple[tuple[str | None, ...], ...]
    original_rows: tuple[tuple[str | None, ...], ...]
    table_id: str = ""
    row_offset: int = 0
    cell_ids: tuple[tuple[str, ...], ...] = ()
    header_rows: tuple[tuple[str | None, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class CleanDocumentFragment:
    document_id: str
    page_number: int
    fragment_index: int
    text: str
    tables: tuple[CleanTable, ...] = ()
    source_text: str = ""
    warnings: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    section: str = ""
    content_type: str = "text"

    @property
    def has_content(self) -> bool:
        return bool(self.text or self.tables)


@runtime_checkable
class PDFProcessor(Protocol):
    def process_page(self, document_id: str, page: PDFPage) -> tuple[CleanDocumentFragment, ...]: ...


class SectionDetector:
    """Conservative, explicit QBR headings; no fuzzy classification of prose."""

    _HEADINGS = {
        "quarterly business review": "document_header",
        "executive overview": "executive_overview",
        "executive summary": "executive_overview",
        "goals for q2": "goals",
        "goals": "goals",
        "performance": "performance",
        "highlights": "highlights",
        "achievements": "highlights",
        "challenges": "challenges",
        "risks": "risks",
        "next steps": "next_steps",
        "action items": "next_steps",
    }

    def detect(self, text: str) -> str | None:
        candidate = text.strip().casefold().rstrip(":")
        if candidate in self._HEADINGS:
            return self._HEADINGS[candidate]
        if re.fullmatch(r"goals for q[1-4](?:\s+20\d{2})?", candidate):
            return "goals"
        return None


class PDFProcessingAdapter:
    _SPACES = re.compile(r"[ \t\f\v]+")
    _NEWLINES = re.compile(r"\n{3,}")
    _FOOTER = re.compile(r"^.{0,120}\bpage\s+\d+\s*$", re.IGNORECASE)

    def __init__(self, options: ProcessingOptions | None = None, *, detector: SectionDetector | None = None) -> None:
        self._options = options or ProcessingOptions()
        self._detector = detector or SectionDetector()

    @classmethod
    def normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("text must be str")
        value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
        value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch) != "Cc")
        return cls._NEWLINES.sub("\n\n", "\n".join(cls._SPACES.sub(" ", line).strip() for line in value.split("\n"))).strip()

    @staticmethod
    def _overlaps(block_bbox: tuple[float, ...], table_bbox: tuple[float, ...]) -> bool:
        x0, y0, x1, y1 = block_bbox
        a0, b0, a1, b1 = table_bbox
        intersection = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
        area = max(0.0, (x1 - x0) * (y1 - y0))
        return area > 0 and intersection / area >= 0.5

    def _text_blocks(self, page: PDFPage) -> list[tuple[str, str, float]]:
        if page.blocks:
            result = []
            for block in page.blocks:
                if any(table.bbox is not None and self._overlaps(block.bbox, table.bbox) for table in page.tables):
                    continue
                text = self.normalize_text(block.text)
                if text:
                    result.append((text, block.block_id, block.bbox[1]))
            return result
        # Geometry is unavailable: preserve original text to avoid losing facts.
        return [(self.normalize_text(page.text), f"p{page.page_number}-text", 0.0)] if page.text.strip() else []

    def _split(self, text: str) -> list[str]:
        """Bounded chunks, splitting by paragraph/line before hard character splits."""
        limit = self._options.max_fragment_chars
        result: list[str] = []
        current = ""
        for paragraph in re.split(r"\n\s*\n", text):
            for line in (paragraph.splitlines() if len(paragraph) > limit else [paragraph]):
                if not line:
                    continue
                if len(line) > limit:
                    if current:
                        result.append(current)
                        current = ""
                    result.extend(line[i:i + limit] for i in range(0, len(line), limit))
                    continue
                candidate = f"{current}\n{line}" if current else line
                if len(candidate) > limit:
                    result.append(current)
                    current = line
                else:
                    current = candidate
        if current:
            result.append(current)
        return result

    def _table(self, table: PDFTable, index: int) -> CleanTable:
        rows = tuple(tuple(self.normalize_text(c) if c is not None else None for c in row) for row in table.rows)
        table_id = table.table_id or f"p{table.page_number}-t{index}"
        cell_ids = tuple(tuple(f"{table_id}-r{ri}-c{ci}" for ci in range(len(row))) for ri, row in enumerate(rows))
        return CleanTable(table.page_number, index, rows, table.rows, table_id, 0, cell_ids)

    def process_page(self, document_id: str, page: PDFPage) -> tuple[CleanDocumentFragment, ...]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must not be blank")
        if not isinstance(page, PDFPage):
            raise TypeError("page must be PDFPage")
        source_text = page.text if self._options.include_source_text else ""
        warnings = tuple(page.warnings)
        # Geometric ordering interleaves narrative and tables; stable ties use index.
        events: list[tuple[float, int, str, object]] = []
        for idx, (text, source_id, top) in enumerate(self._text_blocks(page)):
            events.append((top, idx, "text", (text, source_id)))
        for idx, table in enumerate(page.tables):
            top = table.bbox[1] if table.bbox is not None else float("inf")
            events.append((top, idx, "table", (idx, table)))
        events.sort(key=lambda event: (event[0], event[1]))

        fragments: list[CleanDocumentFragment] = []
        current_section = "document_header"
        pending: list[str] = []
        pending_ids: list[str] = []
        pending_heading: tuple[str, str] | None = None

        def emit_text() -> None:
            if not pending:
                return
            content = "\n\n".join(pending)
            for part in self._split(content):
                fragments.append(CleanDocumentFragment(
                    document_id, page.page_number, len(fragments), part,
                    source_text=source_text, warnings=warnings,
                    source_ids=tuple(dict.fromkeys(pending_ids)),
                    section=current_section if self._options.group_sections else "",
                ))
            pending.clear()
            pending_ids.clear()

        for _, _, kind, payload in events:
            if kind == "text":
                text, source_id = payload
                # Blocks can include multiple headings and paragraphs.
                for line in re.split(r"\n\s*\n|\n(?=(?:Goals for Q[1-4]|Executive Overview|Performance|Highlights|Challenges|Risks|Next Steps)\s*$)", text):
                    line = line.strip()
                    if not line or self._FOOTER.fullmatch(line) or (" - QBR\nPage " in line and re.search(r"\bPage\s+\d+\s*$", line)):
                        continue
                    heading = self._detector.detect(line) if self._options.group_sections else None
                    if heading:
                        if pending_heading is not None and not pending:
                            pending.append(pending_heading[0])
                            pending_ids.append(pending_heading[1])
                        emit_text()
                        current_section = heading
                        pending_heading = (line, source_id)
                        continue
                    pending_heading = None
                    pending.append(line)
                    pending_ids.append(source_id)
            else:
                emit_text()
                pending_heading = None
                table_index, table = payload
                cleaned = self._table(table, table_index)
                size = self._options.max_table_rows_per_fragment
                headers = cleaned.rows[:1]
                for start in range(0, len(cleaned.rows), size):
                    part = CleanTable(
                        cleaned.page_number, cleaned.table_index,
                        cleaned.rows[start:start + size], cleaned.original_rows[start:start + size],
                        cleaned.table_id, start, cleaned.cell_ids[start:start + size],
                        headers if start else (),
                    )
                    fragments.append(CleanDocumentFragment(
                        document_id, page.page_number, len(fragments), "", tables=(part,),
                        source_text=source_text, warnings=warnings,
                        source_ids=(cleaned.table_id,), section=current_section,
                        content_type="table",
                    ))
        if pending_heading is not None and not pending:
            pending.append(pending_heading[0])
            pending_ids.append(pending_heading[1])
        emit_text()
        if not fragments:
            fragments.append(CleanDocumentFragment(
                document_id, page.page_number, 0, "", source_text=source_text,
                warnings=tuple(dict.fromkeys((*warnings, "empty_after_normalization"))),
            ))
        return tuple(fragments)

    async def iter_fragments(self, document_id: str, pages: AsyncIterator[PDFPage]) -> AsyncIterator[CleanDocumentFragment]:
        async for page in pages:
            for fragment in self.process_page(document_id, page):
                yield fragment
