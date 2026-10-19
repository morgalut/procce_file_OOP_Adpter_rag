"""Deterministic, composable rules for extracted QBR metadata.

Rules check structure and internal consistency; they do not establish that
quoted evidence is present in the original PDF. Source grounding is separate.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType
from dataclasses import dataclass
from typing import Protocol

from qbr.domain.enums import EvidenceStatus, IssueSeverity, ValidationStatus
from qbr.domain.models import Evidence, QBRMetadata, ValidationIssue, ValidationResult


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Business requirements for a validation run."""

    required_fields: tuple[str, ...] = (
        "customer_name",
        "reporting_period",
        "executive_summary",
    )
    require_evidence: bool = True
    max_page_number: int | None = None

    def __post_init__(self) -> None:
        supported = {"customer_name", "reporting_period", "executive_summary"}
        unknown = set(self.required_fields) - supported
        if unknown:
            raise ValueError(f"Unsupported required fields: {sorted(unknown)}")
        if self.max_page_number is not None and self.max_page_number < 1:
            raise ValueError("max_page_number must be positive")


class ValidationRule(Protocol):
    """Contract implemented by every independent validation rule."""

    def validate(
        self, metadata: QBRMetadata, policy: ValidationPolicy
    ) -> Iterable[ValidationIssue]: ...


def _issue(code: str, message: str, severity: IssueSeverity, path: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity=severity, field_path=path)


@dataclass(frozen=True, slots=True)
class RequiredFieldsRule:
    """Check mandatory top-level metadata fields."""

    def validate(self, metadata: QBRMetadata, policy: ValidationPolicy) -> Iterable[ValidationIssue]:
        for name in policy.required_fields:
            value = getattr(metadata, name)
            if value is None or (isinstance(value, str) and not value.strip()):
                yield _issue("required_field_missing", f"Required field '{name}' is missing", IssueSeverity.ERROR, name)


@dataclass(frozen=True, slots=True)
class KPICompletenessRule:
    """Flag KPIs that cannot report a current value."""

    def validate(self, metadata: QBRMetadata, policy: ValidationPolicy) -> Iterable[ValidationIssue]:
        for index, kpi in enumerate(metadata.kpis):
            if not kpi.current_value or not kpi.current_value.strip():
                yield _issue("kpi_current_value_missing", "KPI has no current value", IssueSeverity.WARNING, f"kpis[{index}].current_value")


@dataclass(frozen=True, slots=True)
class KPIConflictRule:
    """Detect different current values for the same named KPI and period."""

    def validate(self, metadata: QBRMetadata, policy: ValidationPolicy) -> Iterable[ValidationIssue]:
        seen: dict[tuple[str, str, str], tuple[str, int]] = {}
        for index, kpi in enumerate(metadata.kpis):
            if not kpi.current_value or not kpi.current_value.strip():
                continue
            key = (kpi.name.strip().casefold(), (kpi.period or metadata.reporting_period or "").strip().casefold(), (kpi.unit or "").strip().casefold())
            value = kpi.current_value.strip().casefold()
            previous = seen.get(key)
            if previous is None:
                seen[key] = (value, index)
            elif previous[0] != value:
                yield _issue("kpi_value_conflict", f"KPI conflicts with kpis[{previous[1]}]", IssueSeverity.ERROR, f"kpis[{index}].current_value")


@dataclass(frozen=True, slots=True)
class ActionItemRule:
    """Flag missing owners or deadlines without inventing their values."""

    def validate(self, metadata: QBRMetadata, policy: ValidationPolicy) -> Iterable[ValidationIssue]:
        for index, item in enumerate(metadata.next_steps):
            if not item.owner:
                yield _issue("action_owner_missing", "Action item has no owner", IssueSeverity.WARNING, f"next_steps[{index}].owner")
            if item.target_date is None and not item.target_date_text:
                yield _issue("action_deadline_missing", "Action item has no deadline", IssueSeverity.WARNING, f"next_steps[{index}].target_date")


def _evidence_groups(metadata: QBRMetadata) -> Iterable[tuple[str, list[Evidence]]]:
    """Yield all evidence locations with stable field paths."""
    yield "customer_evidence", metadata.customer_evidence
    yield "period_evidence", metadata.period_evidence
    yield "summary_evidence", metadata.summary_evidence
    for group_name in ("goals", "kpis", "achievements", "challenges", "risks", "next_steps"):
        for index, item in enumerate(getattr(metadata, group_name)):
            yield f"{group_name}[{index}].evidence", item.evidence


@dataclass(frozen=True, slots=True)
class EvidenceRule:
    """Check evidence presence and extraction-provided evidence status.

    UNVERIFIED means grounding has not yet been checked, not that it is valid.
    """

    def validate(self, metadata: QBRMetadata, policy: ValidationPolicy) -> Iterable[ValidationIssue]:
        for path, evidence_list in _evidence_groups(metadata):
            # Only require evidence for populated facts, not absent optional data.
            if policy.require_evidence and _has_claim(metadata, path) and not evidence_list:
                yield _issue("evidence_missing", "Extracted claim has no source evidence", IssueSeverity.WARNING, path)
            for index, evidence in enumerate(evidence_list):
                if evidence.status == EvidenceStatus.CONFLICTING:
                    yield _issue("evidence_conflicting", "Evidence is marked conflicting", IssueSeverity.ERROR, f"{path}[{index}].status")
                elif evidence.status in (EvidenceStatus.UNSUPPORTED, EvidenceStatus.MISSING):
                    yield _issue("evidence_unsupported", "Evidence is marked unsupported or missing", IssueSeverity.WARNING, f"{path}[{index}].status")


def _has_claim(metadata: QBRMetadata, path: str) -> bool:
    top_level = {
        "customer_evidence": metadata.customer_name,
        "period_evidence": metadata.reporting_period,
        "summary_evidence": metadata.executive_summary,
    }
    if path in top_level:
        return bool(top_level[path])
    return True  # Nested evidence groups always belong to populated items.


@dataclass(frozen=True, slots=True)
class EvidencePageRule:
    """Check page references against the PDF's known page count, if supplied."""

    def validate(self, metadata: QBRMetadata, policy: ValidationPolicy) -> Iterable[ValidationIssue]:
        if policy.max_page_number is None:
            return
        for path, evidence_list in _evidence_groups(metadata):
            for index, evidence in enumerate(evidence_list):
                if evidence.page_number > policy.max_page_number:
                    yield _issue("evidence_page_out_of_range", "Evidence references a page beyond the document", IssueSeverity.ERROR, f"{path}[{index}].page_number")


class ValidationPhase(StrEnum):
    INCREMENTAL = "incremental"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    name: str
    rule: ValidationRule
    phase: ValidationPhase


class RuleRegistry:
    """Immutable, uniquely named rule definitions in deterministic order."""

    def __init__(self, definitions: Iterable[RuleDefinition]) -> None:
        entries = tuple(definitions)
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate validation rule names")
        self._definitions = entries

    def for_phase(self, phase: ValidationPhase) -> tuple[ValidationRule, ...]:
        return tuple(item.rule for item in self._definitions if item.phase == phase)


DEFAULT_REGISTRY = RuleRegistry((
    RuleDefinition("kpi_completeness", KPICompletenessRule(), ValidationPhase.INCREMENTAL),
    RuleDefinition("action_items", ActionItemRule(), ValidationPhase.INCREMENTAL),
    RuleDefinition("evidence", EvidenceRule(), ValidationPhase.INCREMENTAL),
    RuleDefinition("evidence_pages", EvidencePageRule(), ValidationPhase.INCREMENTAL),
    RuleDefinition("required_fields", RequiredFieldsRule(), ValidationPhase.FINAL),
    RuleDefinition("kpi_conflicts", KPIConflictRule(), ValidationPhase.FINAL),
))

# Backwards compatibility for callers that import DEFAULT_RULES.
DEFAULT_RULES: tuple[ValidationRule, ...] = (
    *DEFAULT_REGISTRY.for_phase(ValidationPhase.INCREMENTAL),
    *DEFAULT_REGISTRY.for_phase(ValidationPhase.FINAL),
)


@dataclass(frozen=True, slots=True)
class SeverityPolicy:
    """Map the highest issue severity to the overall validation status."""

    outcomes: Mapping[IssueSeverity, ValidationStatus] = None  # type: ignore[assignment]
    empty_status: ValidationStatus = ValidationStatus.PASSED

    def __post_init__(self) -> None:
        outcomes = self.outcomes or {
            IssueSeverity.INFO: ValidationStatus.WARNING,
            IssueSeverity.WARNING: ValidationStatus.WARNING,
            IssueSeverity.ERROR: ValidationStatus.FAILED,
            IssueSeverity.CRITICAL: ValidationStatus.FAILED,
        }
        if set(outcomes) != set(IssueSeverity):
            raise ValueError("Severity policy must map every IssueSeverity")
        object.__setattr__(self, "outcomes", MappingProxyType(dict(outcomes)))

    def resolve(self, issues: Iterable[ValidationIssue]) -> ValidationStatus:
        severities = (issue.severity for issue in issues)
        priority = {
            IssueSeverity.INFO: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.ERROR: 2,
            IssueSeverity.CRITICAL: 3,
        }
        highest = max(severities, key=priority.__getitem__, default=None)
        return self.empty_status if highest is None else self.outcomes[highest]


class ValidationEngine:
    """Execute selected rules and calculate status using an injected policy.

    `validate` preserves the original public API and runs both phases.
    `validate_fragment` runs only rules that are meaningful before merging.
    """

    def __init__(
        self,
        rules: Iterable[ValidationRule] | None = None,
        policy: ValidationPolicy | None = None,
        *,
        registry: RuleRegistry = DEFAULT_REGISTRY,
        severity_policy: SeverityPolicy | None = None,
    ) -> None:
        if rules is not None and registry is not DEFAULT_REGISTRY:
            raise ValueError("Provide either rules or a custom registry, not both")
        self._custom_rules = tuple(rules) if rules is not None else None
        self._registry = registry
        self._policy = policy if policy is not None else ValidationPolicy()
        self._severity_policy = severity_policy or SeverityPolicy()

    def _run(self, metadata: QBRMetadata, rules: Iterable[ValidationRule]) -> ValidationResult:
        issues = [issue for rule in rules for issue in rule.validate(metadata, self._policy)]
        return ValidationResult(status=self._severity_policy.resolve(issues), issues=issues)

    def validate_fragment(self, fragment: QBRMetadata) -> ValidationResult:
        """Validate one extracted section; omit cross-section requirements."""
        if self._custom_rules is not None:
            raise ValueError("Fragment validation requires a phased rule registry")
        return self._run(fragment, self._registry.for_phase(ValidationPhase.INCREMENTAL))

    def validate(self, metadata: QBRMetadata) -> ValidationResult:
        """Validate complete metadata after all fragments have been merged."""
        if self._custom_rules is not None:
            return self._run(metadata, self._custom_rules)
        return self._run(metadata, (
            *self._registry.for_phase(ValidationPhase.INCREMENTAL),
            *self._registry.for_phase(ValidationPhase.FINAL),
        ))
