"""Framework-independent, page-streaming PDF extraction contracts."""
from __future__ import annotations
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

BBox = tuple[float, float, float, float]

@dataclass(frozen=True, slots=True)
class PDFSource:
    document_id: str
    path: Path
    def __post_init__(self):
        if not self.document_id.strip(): raise ValueError('document_id must not be blank')
        if not isinstance(self.path, Path): raise TypeError('path must be a pathlib.Path')

@dataclass(frozen=True, slots=True)
class PDFCell:
    cell_id: str
    row_index: int
    column_index: int
    text: str | None
    bbox: BBox | None = None

@dataclass(frozen=True, slots=True)
class PDFTextBlock:
    block_id: str
    text: str
    bbox: BBox

@dataclass(frozen=True, slots=True)
class PDFTable:
    rows: tuple[tuple[str | None, ...], ...]
    page_number: int
    table_id: str = ''
    bbox: BBox | None = None
    cells: tuple[PDFCell, ...] = ()
    def __post_init__(self):
        if self.page_number < 1: raise ValueError('page_number must be >= 1')
        if not self.rows: raise ValueError('table must contain at least one row')

@dataclass(frozen=True, slots=True)
class PDFPage:
    page_number: int
    text: str
    tables: tuple[PDFTable, ...] = ()
    warnings: tuple[str, ...] = ()
    blocks: tuple[PDFTextBlock, ...] = ()
    def __post_init__(self):
        if self.page_number < 1: raise ValueError('page_number must be >= 1')
        if any(t.page_number != self.page_number for t in self.tables):
            raise ValueError('table page_number must match its containing page')
    @property
    def has_content(self) -> bool:
        return bool(self.text.strip() or self.tables)

@dataclass(frozen=True, slots=True)
class PDFExtractionOptions:
    extract_tables: bool = True
    start_page: int = 1
    end_page: int | None = None
    def __post_init__(self):
        if self.start_page < 1: raise ValueError('start_page must be >= 1')
        if self.end_page is not None and self.end_page < self.start_page:
            raise ValueError('end_page must be >= start_page')

@runtime_checkable
class PDFExtractor(Protocol):
    def iter_pages(self, source: PDFSource, *, options: PDFExtractionOptions | None = None) -> AsyncIterator[PDFPage]: ...
