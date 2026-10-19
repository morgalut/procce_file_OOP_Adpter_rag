"""Typed domain models for QBR document processing.

Models represent extracted facts and provenance, not persistence entities.
"""

from datetime import date, datetime, timezone
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from qbr.domain.enums import (
    EvidenceStatus,
    IssueSeverity,
    ProcessingStatus,
    ValidationStatus,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DomainModel(BaseModel):
    """Strict base for all domain contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Evidence(DomainModel):
    """Trace a claim to a document page and exact supporting excerpt."""

    page_number: int = Field(ge=1)
    source_text: NonEmptyText
    source_id: NonEmptyText | None = None
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED


class Goal(DomainModel):
    description: NonEmptyText
    evidence: list[Evidence] = Field(default_factory=list)


class KPIMetric(DomainModel):
    name: NonEmptyText
    current_value: NonEmptyText | None = None
    previous_value: NonEmptyText | None = None
    unit: str | None = None
    period: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class Finding(DomainModel):
    description: NonEmptyText
    evidence: list[Evidence] = Field(default_factory=list)


class ActionItem(DomainModel):
    description: NonEmptyText
    owner: NonEmptyText | None = None
    target_date_text: NonEmptyText | None = None
    target_date: date | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class QBRMetadata(DomainModel):
    """Normalized extraction output; missing data is explicit."""

    customer_name: NonEmptyText | None = None
    reporting_period: NonEmptyText | None = None
    executive_summary: str | None = None
    goals: list[Goal] = Field(default_factory=list)
    kpis: list[KPIMetric] = Field(default_factory=list)
    achievements: list[Finding] = Field(default_factory=list)
    challenges: list[Finding] = Field(default_factory=list)
    risks: list[Finding] = Field(default_factory=list)
    next_steps: list[ActionItem] = Field(default_factory=list)
    customer_evidence: list[Evidence] = Field(default_factory=list)
    period_evidence: list[Evidence] = Field(default_factory=list)
    summary_evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("executive_summary")
    @classmethod
    def normalize_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class StrictSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
class ValidationIssue(DomainModel):
    code: NonEmptyText
    message: NonEmptyText
    severity: IssueSeverity
    field_path: str | None = None


class ValidationResult(DomainModel):
    status: ValidationStatus = ValidationStatus.PENDING
    issues: list[ValidationIssue] = Field(default_factory=list)


class QBRDocument(DomainModel):
    """Document aggregate, independent of SQLAlchemy and storage."""

    id: UUID = Field(default_factory=uuid4)
    filename: NonEmptyText
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: QBRMetadata | None = None
    validation: ValidationResult | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value
