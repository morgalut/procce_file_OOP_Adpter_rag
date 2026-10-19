"""Unit tests for deterministic QBR metadata accumulation.

Tests target the existing domain API; no database, PDF parser, or LLM needed.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from qbr.domain.enums import EvidenceStatus
from qbr.domain.metadata_accumulator import (
    AccumulationResult,
    FieldConflict,
    Fragment,
    MetadataAccumulator,
)
from qbr.domain.models import ActionItem, Evidence, Finding, Goal, KPIMetric, QBRMetadata


def fragment(fragment_id: str, order: int, **metadata: object) -> Fragment:
    return Fragment(fragment_id=fragment_id, order=order, metadata=QBRMetadata(**metadata))


def test_empty_accumulator() -> None:
    accumulator = MetadataAccumulator()
    result = accumulator.snapshot()
    assert len(accumulator) == 0
    assert result.metadata == QBRMetadata()
    assert result.fragment_ids == ()
    assert result.conflicts == ()
    assert result.provenance == {}
    assert isinstance(result, AccumulationResult)


def test_add_returns_true_and_increments_count() -> None:
    accumulator = MetadataAccumulator()
    assert accumulator.add(fragment("a", 0, customer_name="Acme")) is True
    assert len(accumulator) == 1


def test_identical_retry_is_idempotent() -> None:
    accumulator = MetadataAccumulator()
    item = fragment("a", 0, goals=[Goal(description="Grow")])
    assert accumulator.add(item) is True
    assert accumulator.add(item) is False
    assert len(accumulator) == 1
    assert len(accumulator.snapshot().metadata.goals) == 1


@pytest.mark.parametrize("changed", [
    fragment("same", 2, customer_name="Acme"),
    fragment("same", 1, customer_name="Other"),
])
def test_reused_id_with_different_content_is_rejected(changed: Fragment) -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("same", 1, customer_name="Acme"))
    with pytest.raises(ValueError, match="Fragment ID reused"):
        accumulator.add(changed)
    assert len(accumulator) == 1


@pytest.mark.parametrize("bad_id", ["", None])
def test_invalid_fragment_id_rejected(bad_id: str | None) -> None:
    with pytest.raises(ValidationError):
        Fragment(fragment_id=bad_id, order=0, metadata=QBRMetadata())


@pytest.mark.parametrize("bad_order", [-1, -10])
def test_negative_order_rejected(bad_order: int) -> None:
    with pytest.raises(ValidationError):
        fragment("bad", bad_order)


def test_unexpected_fragment_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        Fragment(fragment_id="x", order=1, metadata=QBRMetadata(), unknown=True)


def test_fragment_top_level_is_frozen() -> None:
    item = fragment("a", 0)
    with pytest.raises(ValidationError):
        item.order = 2


def test_out_of_order_arrivals_produce_same_snapshot() -> None:
    first = fragment("a", 1, customer_name="Acme", goals=[Goal(description="Grow")])
    second = fragment("b", 2, reporting_period="Q2 2026", goals=[Goal(description="Retain")])
    left, right = MetadataAccumulator(), MetadataAccumulator()
    for item in (second, first):
        left.add(item)
    for item in (first, second):
        right.add(item)
    assert left.snapshot() == right.snapshot()
    assert left.snapshot().fragment_ids == ("a", "b")


def test_equal_order_is_sorted_by_fragment_id() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("z", 1, customer_name="Second"))
    accumulator.add(fragment("a", 1, customer_name="First"))
    result = accumulator.snapshot()
    assert result.fragment_ids == ("a", "z")
    assert result.metadata.customer_name == "First"


@pytest.mark.parametrize("field", ["customer_name", "reporting_period", "executive_summary"])
def test_scalar_values_merge_with_provenance(field: str) -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, **{field: "Value"}))
    accumulator.add(fragment("b", 2, **{field: "Value"}))
    result = accumulator.snapshot()
    assert getattr(result.metadata, field) == "Value"
    assert result.conflicts == ()
    assert result.provenance[field] == ("a", "b")


@pytest.mark.parametrize("field", ["customer_name", "reporting_period", "executive_summary"])
def test_scalar_conflicts_preserve_all_observations(field: str) -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("later", 2, **{field: "Beta"}))
    accumulator.add(fragment("earlier", 1, **{field: "Alpha"}))
    result = accumulator.snapshot()
    assert getattr(result.metadata, field) == "Alpha"
    assert result.conflicts == (
        FieldConflict(field_name=field, values=("Alpha", "Beta"), fragment_ids=("earlier", "later")),
    )


def test_conflict_tracks_multiple_sources_for_same_value() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, customer_name="Acme"))
    accumulator.add(fragment("b", 2, customer_name="Acme"))
    accumulator.add(fragment("c", 3, customer_name="Globex"))
    result = accumulator.snapshot()
    assert result.conflicts[0].values == ("Acme", "Globex")
    assert result.conflicts[0].fragment_ids == ("a", "b", "c")
    assert result.provenance["customer_name"] == ("a", "b", "c")


def test_empty_scalar_does_not_override_existing_value() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, customer_name="Acme"))
    accumulator.add(fragment("b", 2, executive_summary="   "))
    result = accumulator.snapshot()
    assert result.metadata.customer_name == "Acme"
    assert result.metadata.executive_summary is None
    assert "executive_summary" not in result.provenance


@pytest.mark.parametrize("field,item", [
    ("goals", Goal(description="Increase adoption")),
    ("kpis", KPIMetric(name="MAU", current_value="500")),
    ("achievements", Finding(description="Launched")),
    ("challenges", Finding(description="Latency")),
    ("risks", Finding(description="Churn")),
    ("next_steps", ActionItem(description="Follow up", owner="Dana", target_date=date(2026, 11, 1))),
])
def test_collection_merges_and_deduplicates_exact_items(field: str, item: object) -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, **{field: [item]}))
    accumulator.add(fragment("b", 2, **{field: [item]}))
    result = accumulator.snapshot()
    assert getattr(result.metadata, field) == [item]
    assert result.provenance[field] == ("a", "b")


def test_different_kpi_values_are_preserved_for_conflict_validation() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, kpis=[KPIMetric(name="MAU", current_value="500")]))
    accumulator.add(fragment("b", 2, kpis=[KPIMetric(name="MAU", current_value="600")]))
    result = accumulator.snapshot()
    assert [k.current_value for k in result.metadata.kpis] == ["500", "600"]
    assert result.conflicts == ()  # KPI conflicts belong to ValidationEngine.


@pytest.mark.parametrize("field", ["customer_evidence", "period_evidence", "summary_evidence"])
def test_evidence_merges_deduplicates_and_tracks_sources(field: str) -> None:
    evidence = Evidence(page_number=2, source_text="Source excerpt", status=EvidenceStatus.SUPPORTED)
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, **{field: [evidence]}))
    accumulator.add(fragment("b", 2, **{field: [evidence]}))
    result = accumulator.snapshot()
    assert getattr(result.metadata, field) == [evidence]
    assert result.provenance[field] == ("a", "b")


def test_distinct_evidence_is_preserved() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, customer_evidence=[Evidence(page_number=1, source_text="Acme")]))
    accumulator.add(fragment("b", 2, customer_evidence=[Evidence(page_number=3, source_text="Acme")]))
    assert len(accumulator.snapshot().metadata.customer_evidence) == 2


def test_add_deep_copies_input_metadata() -> None:
    source = fragment("a", 1, goals=[Goal(description="Original")])
    accumulator = MetadataAccumulator()
    accumulator.add(source)
    source.metadata.goals[0].description = "Changed"
    source.metadata.goals.append(Goal(description="Injected"))
    result = accumulator.snapshot()
    assert [goal.description for goal in result.metadata.goals] == ["Original"]


def test_snapshot_is_detached_from_internal_state() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, goals=[Goal(description="Original")]))
    first = accumulator.snapshot()
    first.metadata.goals[0].description = "Changed"
    first.metadata.goals.append(Goal(description="Injected"))
    first.provenance["goals"] = ("fake",)
    second = accumulator.snapshot()
    assert [goal.description for goal in second.metadata.goals] == ["Original"]
    assert second.provenance["goals"] == ("a",)


def test_snapshots_are_stable_across_repeated_calls() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, customer_name="Acme"))
    assert accumulator.snapshot() == accumulator.snapshot()


def test_add_new_fragment_after_snapshot() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, customer_name="Acme"))
    before = accumulator.snapshot()
    accumulator.add(fragment("b", 2, reporting_period="Q3 2026"))
    after = accumulator.snapshot()
    assert before.metadata.reporting_period is None
    assert after.metadata.reporting_period == "Q3 2026"
    assert before.fragment_ids == ("a",)
    assert after.fragment_ids == ("a", "b")


def test_provenance_absent_for_missing_collections() -> None:
    accumulator = MetadataAccumulator()
    accumulator.add(fragment("a", 1, customer_name="Acme"))
    assert "kpis" not in accumulator.snapshot().provenance


def test_rejects_malformed_metadata() -> None:
    with pytest.raises(ValidationError):
        Fragment(fragment_id="a", order=0, metadata={"unknown_field": 1})
