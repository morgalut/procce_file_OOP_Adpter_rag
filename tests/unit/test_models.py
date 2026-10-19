"""Unit tests for the QBR domain contracts (Pydantic v2)."""

from datetime import date, datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from qbr.domain.enums import EvidenceStatus, IssueSeverity, ProcessingStatus, ValidationStatus
from qbr.domain.models import (
    ActionItem,
    Evidence,
    Finding,
    Goal,
    KPIMetric,
    QBRDocument,
    QBRMetadata,
    ValidationIssue,
    ValidationResult,
)


@pytest.fixture
def evidence() -> Evidence:
    return Evidence(page_number=1, source_text="  Active users: 875  ")


@pytest.fixture
def metadata(evidence: Evidence) -> QBRMetadata:
    return QBRMetadata(
        customer_name="Northwind Technologies",
        reporting_period="Q2 2026",
        executive_summary="  Platform adoption improved.  ",
        goals=[Goal(description="Increase adoption", evidence=[evidence])],
        kpis=[KPIMetric(name="Monthly Active Users", current_value="875", previous_value="640")],
        achievements=[Finding(description="SSO rollout completed")],
        next_steps=[ActionItem(description="Complete onboarding", owner="Sarah Levi")],
    )


def test_evidence_normalizes_text_and_defaults_to_unverified(evidence: Evidence) -> None:
    assert evidence.source_text == "Active users: 875"
    assert evidence.status is EvidenceStatus.UNVERIFIED


@pytest.mark.parametrize("page", [0, -1])
def test_evidence_rejects_invalid_pages(page: int) -> None:
    with pytest.raises(ValidationError):
        Evidence(page_number=page, source_text="Evidence")


@pytest.mark.parametrize("text", ["", "   "])
def test_required_text_rejects_blank_values(text: str) -> None:
    for model, field in (
        (Evidence, "source_text"),
        (Goal, "description"),
        (KPIMetric, "name"),
        (Finding, "description"),
        (ActionItem, "description"),
        (ValidationIssue, "code"),
    ):
        values = {field: text}
        if model is Evidence:
            values["page_number"] = 1
        if model is ValidationIssue:
            values.update(message="Message", severity=IssueSeverity.ERROR)
        with pytest.raises(ValidationError):
            model(**values)


def test_goal_finding_and_kpi_preserve_evidence(evidence: Evidence) -> None:
    goal = Goal(description="  Increase adoption ", evidence=[evidence])
    finding = Finding(description="  Completed migration  ", evidence=[evidence])
    kpi = KPIMetric(name=" Adoption ", current_value=" 68% ", evidence=[evidence])
    assert goal.description == "Increase adoption"
    assert finding.description == "Completed migration"
    assert kpi.name == "Adoption"
    assert kpi.current_value == "68%"
    assert all(item.evidence[0].page_number == 1 for item in (goal, finding, kpi))


def test_kpi_preserves_approximate_values_and_units() -> None:
    kpi = KPIMetric(name="Weekly active", current_value="~900", previous_value="590", unit="users")
    assert kpi.current_value == "~900"
    assert kpi.previous_value == "590"
    assert kpi.unit == "users"


def test_action_item_preserves_date_text_and_normalized_date() -> None:
    action = ActionItem(
        description="Complete onboarding",
        owner=" Sarah Levi ",
        target_date_text="15 July 2026",
        target_date=date(2026, 7, 15),
    )
    assert action.owner == "Sarah Levi"


