"""Behavioral tests for deterministic, phased QBR metadata validation."""

from dataclasses import FrozenInstanceError

import pytest

from qbr.domain.enums import EvidenceStatus, IssueSeverity, ValidationStatus
from qbr.domain.models import ActionItem, Evidence, Goal, KPIMetric, QBRMetadata, ValidationIssue
from qbr.domain.validation_rules import (
    DEFAULT_REGISTRY,
    DEFAULT_RULES,
    ActionItemRule,
    EvidencePageRule,
    EvidenceRule,
    KPICompletenessRule,
    KPIConflictRule,
    RequiredFieldsRule,
    RuleDefinition,
    RuleRegistry,
    SeverityPolicy,
    ValidationEngine,
    ValidationPhase,
    ValidationPolicy,
)


def complete_metadata(**overrides):
    values = {
        "customer_name": "Acme",
        "reporting_period": "Q2 2026",
        "executive_summary": "Adoption increased",
    }
    values.update(overrides)
    return QBRMetadata(**values)


def codes(result):
    return [issue.code for issue in result.issues]


def paths(result):
    return [issue.field_path for issue in result.issues]


def evidence(status=EvidenceStatus.SUPPORTED, page=1):
    return Evidence(page_number=page, source_text="Document excerpt", status=status)


@pytest.mark.parametrize("field", ["customer_name", "reporting_period", "executive_summary"])
def test_required_field_is_reported(field):
    metadata = complete_metadata(**{field: None})
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate(metadata)
    assert result.status == ValidationStatus.FAILED
    assert "required_field_missing" in codes(result)
    assert field in paths(result)


def test_required_fields_can_be_configured():
    policy = ValidationPolicy(required_fields=("customer_name",), require_evidence=False)
    result = ValidationEngine(policy=policy).validate(QBRMetadata(customer_name="Acme"))
    assert result.status == ValidationStatus.PASSED


@pytest.mark.parametrize("fields", [("unknown",), ("customer_name", "unknown")])
def test_invalid_required_fields_are_rejected(fields):
    with pytest.raises(ValueError, match="Unsupported required fields"):
        ValidationPolicy(required_fields=fields)


def test_policy_rejects_invalid_page_limit():
    with pytest.raises(ValueError, match="must be positive"):
        ValidationPolicy(max_page_number=0)


def test_policy_is_immutable():
    policy = ValidationPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.require_evidence = False


def test_missing_kpi_value_is_warning():
    metadata = complete_metadata(kpis=[KPIMetric(name="Active users")])
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate(metadata)
    assert result.status == ValidationStatus.WARNING
    assert codes(result) == ["kpi_current_value_missing"]
    assert paths(result) == ["kpis[0].current_value"]


def test_present_kpi_value_is_not_flagged():
    result = list(KPICompletenessRule().validate(
        complete_metadata(kpis=[KPIMetric(name="MAU", current_value="0")]),
        ValidationPolicy(),
    ))
    assert result == []


def test_conflicting_kpis_fail_final_validation():
    metadata = complete_metadata(kpis=[
        KPIMetric(name="Active Users", current_value="500"),
        KPIMetric(name="active users", current_value="600"),
    ])
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate(metadata)
    assert result.status == ValidationStatus.FAILED
    assert "kpi_value_conflict" in codes(result)
    assert "kpis[1].current_value" in paths(result)


def test_equal_kpis_with_case_and_whitespace_changes_do_not_conflict():
    metadata = complete_metadata(kpis=[
        KPIMetric(name="Active Users", current_value="500"),
        KPIMetric(name=" active users ", current_value=" 500 "),
    ])
    assert list(KPIConflictRule().validate(metadata, ValidationPolicy())) == []


@pytest.mark.parametrize("difference", ["unit", "period", "name"])
def test_distinct_kpi_keys_do_not_conflict(difference):
    first = {"name": "Usage", "current_value": "500", "unit": "users", "period": "Q2"}
    second = dict(first, current_value="600")
    second[difference] = {"unit": "sessions", "period": "Q3", "name": "Revenue"}[difference]
    metadata = complete_metadata(kpis=[KPIMetric(**first), KPIMetric(**second)])
    assert list(KPIConflictRule().validate(metadata, ValidationPolicy())) == []


def test_kpis_without_current_value_do_not_generate_conflicts():
    metadata = complete_metadata(kpis=[KPIMetric(name="Usage"), KPIMetric(name="Usage", current_value="10")])
    assert list(KPIConflictRule().validate(metadata, ValidationPolicy())) == []


def test_action_item_missing_owner_and_deadline():
    metadata = complete_metadata(next_steps=[ActionItem(description="Onboard team")])
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate(metadata)
    assert codes(result) == ["action_owner_missing", "action_deadline_missing"]
    assert result.status == ValidationStatus.WARNING


def test_action_item_with_owner_and_text_date_is_complete():
    metadata = complete_metadata(next_steps=[ActionItem(
        description="Onboard team", owner="Dana", target_date_text="July 15"
    )])
    assert list(ActionItemRule().validate(metadata, ValidationPolicy())) == []


def test_evidence_required_for_populated_claims():
    metadata = complete_metadata(goals=[Goal(description="Grow usage")])
    result = ValidationEngine().validate_fragment(metadata)
    assert "evidence_missing" in codes(result)
    assert "goals[0].evidence" in paths(result)
    assert "customer_evidence" in paths(result)


def test_evidence_not_required_for_missing_optional_claims():
    metadata = QBRMetadata()
    assert list(EvidenceRule().validate(metadata, ValidationPolicy())) == []


def test_evidence_requirement_can_be_disabled():
    metadata = complete_metadata(goals=[Goal(description="Grow usage")])
    assert list(EvidenceRule().validate(metadata, ValidationPolicy(require_evidence=False))) == []


@pytest.mark.parametrize("status,expected_code,expected_severity", [
    (EvidenceStatus.CONFLICTING, "evidence_conflicting", IssueSeverity.ERROR),
    (EvidenceStatus.UNSUPPORTED, "evidence_unsupported", IssueSeverity.WARNING),
    (EvidenceStatus.MISSING, "evidence_unsupported", IssueSeverity.WARNING),
])
def test_invalid_evidence_statuses(status, expected_code, expected_severity):
    metadata = QBRMetadata(customer_evidence=[evidence(status)])
    issues = list(EvidenceRule().validate(metadata, ValidationPolicy(require_evidence=False)))
    assert len(issues) == 1
    assert issues[0].code == expected_code
    assert issues[0].severity == expected_severity
    assert issues[0].field_path == "customer_evidence[0].status"


@pytest.mark.parametrize("status", [EvidenceStatus.SUPPORTED, EvidenceStatus.UNVERIFIED])
def test_supported_or_unverified_evidence_has_no_status_issue(status):
    metadata = QBRMetadata(customer_evidence=[evidence(status)])
    assert list(EvidenceRule().validate(metadata, ValidationPolicy(require_evidence=False))) == []


def test_page_number_above_document_limit_fails():
    metadata = complete_metadata(customer_evidence=[evidence(page=4)])
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False, max_page_number=3)).validate(metadata)
    assert codes(result) == ["evidence_page_out_of_range"]
    assert paths(result) == ["customer_evidence[0].page_number"]
    assert result.status == ValidationStatus.FAILED


def test_page_at_limit_is_valid():
    metadata = QBRMetadata(customer_evidence=[evidence(page=3)])
    assert list(EvidencePageRule().validate(metadata, ValidationPolicy(max_page_number=3))) == []


def test_page_limit_is_optional():
    metadata = QBRMetadata(customer_evidence=[evidence(page=999)])
    assert list(EvidencePageRule().validate(metadata, ValidationPolicy())) == []


def test_fragment_validation_does_not_require_top_level_fields():
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate_fragment(QBRMetadata())
    assert result.status == ValidationStatus.PASSED
    assert result.issues == []


def test_fragment_validation_does_not_check_kpi_conflicts():
    metadata = complete_metadata(kpis=[
        KPIMetric(name="MAU", current_value="5"),
        KPIMetric(name="MAU", current_value="6"),
    ])
    engine = ValidationEngine(policy=ValidationPolicy(require_evidence=False))
    assert "kpi_value_conflict" not in codes(engine.validate_fragment(metadata))
    assert "kpi_value_conflict" in codes(engine.validate(metadata))


def test_fragment_validation_runs_incremental_rules():
    metadata = QBRMetadata(next_steps=[ActionItem(description="Follow up")])
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate_fragment(metadata)
    assert codes(result) == ["action_owner_missing", "action_deadline_missing"]


def test_registry_preserves_order_and_filters_phases():
    registry = RuleRegistry((
        RuleDefinition("first", KPICompletenessRule(), ValidationPhase.INCREMENTAL),
        RuleDefinition("second", RequiredFieldsRule(), ValidationPhase.FINAL),
        RuleDefinition("third", ActionItemRule(), ValidationPhase.INCREMENTAL),
    ))
    assert [type(rule) for rule in registry.for_phase(ValidationPhase.INCREMENTAL)] == [
        KPICompletenessRule, ActionItemRule
    ]
    assert [type(rule) for rule in registry.for_phase(ValidationPhase.FINAL)] == [RequiredFieldsRule]


def test_registry_rejects_duplicate_rule_names():
    with pytest.raises(ValueError, match="Duplicate validation rule names"):
        RuleRegistry((
            RuleDefinition("same", KPICompletenessRule(), ValidationPhase.INCREMENTAL),
            RuleDefinition("same", ActionItemRule(), ValidationPhase.FINAL),
        ))


def test_default_registry_has_six_rules():
    assert len(DEFAULT_RULES) == 6
    assert len(DEFAULT_REGISTRY.for_phase(ValidationPhase.INCREMENTAL)) == 4
    assert len(DEFAULT_REGISTRY.for_phase(ValidationPhase.FINAL)) == 2


@pytest.mark.parametrize("severity,expected", [
    (IssueSeverity.INFO, ValidationStatus.WARNING),
    (IssueSeverity.WARNING, ValidationStatus.WARNING),
    (IssueSeverity.ERROR, ValidationStatus.FAILED),
    (IssueSeverity.CRITICAL, ValidationStatus.FAILED),
])
def test_severity_policy_resolves_each_severity(severity, expected):
    issue = ValidationIssue(code="test", message="Test issue", severity=severity)
    assert SeverityPolicy().resolve([issue]) == expected


def test_severity_policy_empty_issues_pass():
    assert SeverityPolicy().resolve([]) == ValidationStatus.PASSED


def test_severity_policy_uses_highest_severity():
    issues = [
        ValidationIssue(code="warn", message="Warning", severity=IssueSeverity.WARNING),
        ValidationIssue(code="error", message="Error", severity=IssueSeverity.ERROR),
    ]
    assert SeverityPolicy().resolve(iter(issues)) == ValidationStatus.FAILED


def test_severity_policy_supports_custom_outcomes():
    outcomes = {severity: ValidationStatus.WARNING for severity in IssueSeverity}
    policy = SeverityPolicy(outcomes=outcomes, empty_status=ValidationStatus.PENDING)
    assert policy.resolve([]) == ValidationStatus.PENDING
    issue = ValidationIssue(code="x", message="x", severity=IssueSeverity.CRITICAL)
    assert policy.resolve([issue]) == ValidationStatus.WARNING
    outcomes[IssueSeverity.CRITICAL] = ValidationStatus.FAILED
    assert policy.resolve([issue]) == ValidationStatus.WARNING


def test_severity_policy_rejects_incomplete_mapping():
    with pytest.raises(ValueError, match="must map every IssueSeverity"):
        SeverityPolicy(outcomes={IssueSeverity.ERROR: ValidationStatus.FAILED})


def test_custom_rule_is_injected_and_called():
    class TestRule:
        def validate(self, metadata, policy):
            return [ValidationIssue(code="custom", message="Custom issue", severity=IssueSeverity.WARNING)]

    result = ValidationEngine(rules=[TestRule()]).validate(QBRMetadata())
    assert codes(result) == ["custom"]
    assert result.status == ValidationStatus.WARNING


def test_custom_rules_cannot_run_fragment_validation():
    with pytest.raises(ValueError, match="requires a phased rule registry"):
        ValidationEngine(rules=[]).validate_fragment(QBRMetadata())


def test_custom_rules_and_registry_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either rules or a custom registry"):
        ValidationEngine(rules=[], registry=RuleRegistry(()))


def test_custom_registry_is_used_by_engine():
    registry = RuleRegistry((RuleDefinition("required", RequiredFieldsRule(), ValidationPhase.FINAL),))
    engine = ValidationEngine(registry=registry)
    assert engine.validate_fragment(QBRMetadata()).status == ValidationStatus.PASSED
    assert codes(engine.validate(QBRMetadata())) == ["required_field_missing"] * 3


def test_engine_collects_multiple_issues_in_rule_order():
    metadata = QBRMetadata(
        kpis=[KPIMetric(name="MAU"), KPIMetric(name="Revenue", current_value="10")],
        next_steps=[ActionItem(description="Follow up")],
    )
    result = ValidationEngine(policy=ValidationPolicy(require_evidence=False)).validate(metadata)
    assert codes(result) == [
        "kpi_current_value_missing",
        "action_owner_missing",
        "action_deadline_missing",
        "required_field_missing",
        "required_field_missing",
        "required_field_missing",
    ]
    assert result.status == ValidationStatus.FAILED


def test_engine_can_use_custom_severity_policy():
    mapping = {severity: ValidationStatus.WARNING for severity in IssueSeverity}
    metadata = QBRMetadata()
    result = ValidationEngine(
        policy=ValidationPolicy(require_evidence=False),
        severity_policy=SeverityPolicy(outcomes=mapping),
    ).validate(metadata)
    assert result.status == ValidationStatus.WARNING
