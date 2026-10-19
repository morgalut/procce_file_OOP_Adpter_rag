
"""Deterministic source-reference lookup for one cleaned PDF fragment.

The resolver does not use an LLM and does not change table rows.
It refuses unknown, duplicate, and ambiguous source IDs.
Evidence quotations must occur in the exact referenced source;
whitespace-only PDF extraction differences may be normalized.
"""
from __future__ import annotations

import re
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


def normalize_whitespace(text: str) -> str:
    """Collapse Unicode whitespace without changing words or punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def is_supported_quote(quote: str, source: str) -> bool:
    """Allow exact or whitespace-equivalent contiguous quotations."""
    if not quote or not quote.strip():
        return False

    if quote in source:
        return True

    return normalize_whitespace(quote) in normalize_whitespace(source)


class EvidenceResolver:
    """Build a fragment-scoped registry; resolve source IDs and excerpts."""

    def __init__(self, fragment: CleanDocumentFragment) -> None:
        self._sources: dict[str, ResolvedSource] = {}

        for table in fragment.tables:
            table_id = (
                table.table_id
                or f"p{table.page_number}-t{table.table_index}"
            )

            for offset, row in enumerate(table.original_rows):
                row_number = table.row_offset + offset
                cell_ids = (
                    table.cell_ids[offset]
                    if offset < len(table.cell_ids)
                    else ()
                )

                for col, value in enumerate(row):
                    source_id = (
                        cell_ids[col]
                        if col < len(cell_ids)
                        else f"{table_id}-r{row_number}-c{col}"
                    )

                    if value is not None:
                        self._add(
                            ResolvedSource(
                                source_id=source_id,
                                page_number=fragment.page_number,
                                text=value,
                                section=fragment.section,
                                kind="cell",
                            )
                        )

        # The fragment contract does not contain per-block text.
        # Use only the supported fragment-wide text reference.
        if fragment.text:
            self._add(
                ResolvedSource(
                    source_id=(
                        f"p{fragment.page_number}-f"
                        f"{fragment.fragment_index}"
                    ),
                    page_number=fragment.page_number,
                    text=fragment.text,
                    section=fragment.section,
                    kind="fragment_text",
                )
            )

    def _add(self, source: ResolvedSource) -> None:
        if source.source_id in self._sources:
            raise SourceReferenceError(
                f"Duplicate source ID: {source.source_id}"
            )

        self._sources[source.source_id] = source

    def resolve(
        self,
        source_id: str,
        *,
        quote: str | None = None,
    ) -> ResolvedSource:
        try:
            source = self._sources[source_id]
        except KeyError as exc:
            raise SourceReferenceError(
                f"Unknown source ID: {source_id}"
            ) from exc

        if quote is not None:
            if not quote.strip():
                raise SourceReferenceError(
                    f"Empty evidence quotation for source ID: {source_id}"
                )

            if not is_supported_quote(quote, source.text):
                raise SourceReferenceError(
                    f"Unsupported excerpt for source ID: {source_id}; "
                    f"source_kind={source.kind}; "
                    f"quote_length={len(quote)}; "
                    f"source_length={len(source.text)}; "
                    "whitespace_match=False"
                )

        return source

    def available_ids(self) -> tuple[str, ...]:
        return tuple(self._sources)
