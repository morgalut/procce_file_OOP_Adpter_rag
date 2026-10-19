"""Deterministic checks that extracted table facts belong to the same source row.

This module does not invoke an LLM or change the existing 2-D table contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment
from qbr.domain.models import QBRMetadata


class GroundingError(ValueError):
    """A generated fact contradicts the structured fragment it came from."""


class MetadataGrounder(Protocol):
    def verify(self, metadata: QBRMetadata, fragment: CleanDocumentFragment) -> None: ...


@dataclass(frozen=True, slots=True)
class TableGrounder:
    """Check KPI values and action fields against one row, not arbitrary cells.

    Strict checks apply when a table fragment contains the corresponding records.
    Other fragment types remain subject to the adapter's existing evidence checks.
    """

    @staticmethod
    def _value(value: str | None) -> str:
        return (value or "").strip()

    def verify(self, metadata: QBRMetadata, fragment: CleanDocumentFragment) -> None:
        if not fragment.tables:
            return
        for kpi in metadata.kpis:
            matches = []
            for table in fragment.tables:
                for row in table.rows:
                    if row and self._value(row[0]).casefold() == kpi.name.strip().casefold():
                        matches.append((table, row))
            if not matches:
                raise GroundingError(f"KPI {kpi.name!r} not found in any table row")
            if not any(self._kpi_values_match(kpi, table, row) for table, row in matches):
                raise GroundingError(f"KPI {kpi.name!r} has values not supported by one source row")
        for action in metadata.next_steps:
            matches = [row for table in fragment.tables for row in table.rows
                       if row and self._value(row[0]).casefold() == action.description.strip().casefold()]
            if not matches:
                raise GroundingError(f"Action {action.description!r} not found in any table row")
            if not any(self._action_values_match(action, row) for row in matches):
                raise GroundingError(f"Action {action.description!r} has unsupported owner or target")

    def _kpi_values_match(self, kpi, table, row) -> bool:
        # When period headings are available, use them to preserve Q1/Q2 meaning.
        headers = table.header_rows or (table.rows[:1] if table.row_offset == 0 else ())
        header = [self._value(v).casefold() for v in headers[0]] if headers else []
        # Explicit Q1/Q2 mapping is only safe when both columns are present.
        if "q1" in header and "q2" in header:
            previous = self._value(row[header.index("q1")]) if header.index("q1") < len(row) else ""
            current = self._value(row[header.index("q2")]) if header.index("q2") < len(row) else ""
            return ((kpi.previous_value is None or self._value(kpi.previous_value) == previous)
                    and (kpi.current_value is None or self._value(kpi.current_value) == current))
        cells = [self._value(v) for v in row[1:]]
        return all(self._value(value) in cells for value in (kpi.previous_value, kpi.current_value) if value is not None)

    def _action_values_match(self, action, row) -> bool:
        cells = [self._value(v) for v in row[1:]]
        return all(self._value(value) in cells for value in (action.owner, action.target_date_text) if value is not None)
