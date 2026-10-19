"""Deterministic accumulation of independently extracted QBR fragments.

This module does not perform PDF extraction, source grounding, or business validation.
It preserves observations and provenance for those downstream components.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from qbr.domain.models import QBRMetadata


class Fragment(BaseModel):
    """One independently extracted portion of a source document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragment_id: str = Field(min_length=1)
    order: int = Field(ge=0, description="Stable source order, independent of arrival")
    metadata: QBRMetadata


class FieldConflict(BaseModel):
    """Conflicting top-level scalar observations; neither is discarded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str
    values: tuple[str, ...]
    fragment_ids: tuple[str, ...]


class AccumulationResult(BaseModel):
    """Snapshot of the accumulated metadata and its source lineage."""

    model_config = ConfigDict(extra="forbid")

    metadata: QBRMetadata
    fragment_ids: tuple[str, ...]
    conflicts: tuple[FieldConflict, ...]
    provenance: dict[str, tuple[str, ...]]


_SCALARS = ("customer_name", "reporting_period", "executive_summary")
_COLLECTIONS = ("goals", "kpis", "achievements", "challenges", "risks", "next_steps")
_EVIDENCE = ("customer_evidence", "period_evidence", "summary_evidence")


class MetadataAccumulator:
    """Accumulate QBR fragments without mutating input or dropping conflicts.

    Rebuilding from stored fragments makes the result independent of arrival order.
    A duplicate fragment ID with different content is rejected explicitly.
    """

    def __init__(self) -> None:
        self._fragments: dict[str, Fragment] = {}

    def add(self, fragment: Fragment) -> bool:
        """Add a fragment; return False for an identical retry.

        Raises ValueError for an ID reused with different content or source order.
        """
        existing = self._fragments.get(fragment.fragment_id)
        if existing is not None:
            if existing != fragment:
                raise ValueError(f"Fragment ID reused with different data: {fragment.fragment_id}")
            return False
        # Store a deep copy: frozen outer models do not freeze nested Pydantic models.
        self._fragments[fragment.fragment_id] = fragment.model_copy(deep=True)
        return True

    def snapshot(self) -> AccumulationResult:
        """Build a deterministic, detached snapshot of all accepted fragments."""
        ordered = sorted(self._fragments.values(), key=lambda f: (f.order, f.fragment_id))
        metadata = QBRMetadata()
        provenance: dict[str, list[str]] = {}
        scalar_values: dict[str, dict[str, list[str]]] = {name: {} for name in _SCALARS}
        collection_seen: dict[str, set[str]] = {name: set() for name in _COLLECTIONS}
        evidence_seen: dict[str, set[str]] = {name: set() for name in _EVIDENCE}

        for fragment in ordered:
            source = fragment.metadata
            for field in _SCALARS:
                value = getattr(source, field)
                if value is None or not value.strip():
                    continue
                observed = scalar_values[field]
                observed.setdefault(value, []).append(fragment.fragment_id)
                if getattr(metadata, field) is None:
                    setattr(metadata, field, value)
                provenance.setdefault(field, []).append(fragment.fragment_id)

            for field in _COLLECTIONS:
                destination = getattr(metadata, field)
                for item in getattr(source, field):
                    # Exact semantic equality only. Differing KPI values remain separate.
                    fingerprint = item.model_dump_json(exclude_none=False)
                    if fingerprint not in collection_seen[field]:
                        collection_seen[field].add(fingerprint)
                        destination.append(item.model_copy(deep=True))
                    provenance.setdefault(field, []).append(fragment.fragment_id)

            for field in _EVIDENCE:
                destination = getattr(metadata, field)
                for item in getattr(source, field):
                    fingerprint = item.model_dump_json(exclude_none=False)
                    if fingerprint not in evidence_seen[field]:
                        evidence_seen[field].add(fingerprint)
                        destination.append(item.model_copy(deep=True))
                    provenance.setdefault(field, []).append(fragment.fragment_id)

        conflicts = tuple(
            FieldConflict(
                field_name=field,
                values=tuple(observed),
                fragment_ids=tuple(dict.fromkeys(fid for ids in observed.values() for fid in ids)),
            )
            for field, observed in scalar_values.items()
            if len(observed) > 1
        )
        return AccumulationResult(
            metadata=metadata,
            fragment_ids=tuple(fragment.fragment_id for fragment in ordered),
            conflicts=conflicts,
            provenance={key: tuple(dict.fromkeys(ids)) for key, ids in provenance.items()},
        )

    def __len__(self) -> int:
        return len(self._fragments)
