"""Deterministic source-reference lookup for one cleaned PDF fragment.

The resolver does not use an LLM and does not change table rows.
It intentionally refuses ambiguous/unavailable source IDs.
"""
from __future__ import annotations

from dataclasses import dataclass

from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment


class SourceReferenceError(ValueError):
    """A source ID is unknown or its excerpt is not supported."""


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    source_id: str
    page_number: int
    text: str
    section: str
    kind: str


class EvidenceResolver:
    """Build a fragment-scoped registry; resolve exact source IDs and excerpts."""

    def __init__(self, fragment: CleanDocumentFragment) -> None:
        self._sources: dict[str, ResolvedSource] = {}
        for table in fragment.tables:
            table_id = table.table_id or f"p{table.page_number}-t{table.table_index}"
            for offset, row in enumerate(table.original_rows):
                row_number = table.row_offset + offset
                cell_ids = table.cell_ids[offset] if offset < len(table.cell_ids) else ()
                for col, value in enumerate(row):
                    source_id = cell_ids[col] if col < len(cell_ids) else f"{table_id}-r{row_number}-c{col}"
                    if value is not None:
                        self._add(ResolvedSource(source_id, fragment.page_number, value, fragment.section, "cell"))
        # Text block IDs refer to subregions, but the current fragment contract
        # does not carry per-block text. Avoid claiming block-level exactness.
        # A fragment-wide reference remains precise about its actual scope.
        if fragment.text:
            self._add(ResolvedSource(
                f"p{fragment.page_number}-f{fragment.fragment_index}",
                fragment.page_number, fragment.text, fragment.section, "fragment_text"
            ))

    def _add(self, source: ResolvedSource) -> None:
        if source.source_id in self._sources:
            raise SourceReferenceError(f"Duplicate source ID: {source.source_id}")
        self._sources[source.source_id] = source

    def resolve(self, source_id: str, *, quote: str | None = None) -> ResolvedSource:
        try:
            source = self._sources[source_id]
        except KeyError as exc:
            raise SourceReferenceError(f"Unknown source ID: {source_id}") from exc
        if quote is not None and (not quote.strip() or quote not in source.text):
            raise SourceReferenceError(f"Unsupported excerpt for source ID: {source_id}")
        return source

    def available_ids(self) -> tuple[str, ...]:
        return tuple(self._sources)
