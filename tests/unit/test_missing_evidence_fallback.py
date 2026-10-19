"""Application-level evidence policy; no network or LLM API required."""
import pytest
from qbr.application.extract_metadata import MissingEvidenceError, require_claim_evidence
from qbr.domain.models import Evidence, Finding, KPIMetric, QBRMetadata


def test_rejects_scalar_without_evidence():
    with pytest.raises(MissingEvidenceError, match="customer_name"):
        require_claim_evidence(QBRMetadata(customer_name="Northwind"))


def test_rejects_kpi_without_evidence():
    with pytest.raises(MissingEvidenceError, match=r"kpis\[0\]"):
        require_claim_evidence(QBRMetadata(kpis=[KPIMetric(name="Users", current_value="875")]))


def test_rejects_one_unsupported_claim_among_supported_claims():
    supported = Evidence(page_number=1, source_text="Northwind")
    with pytest.raises(MissingEvidenceError, match=r"risks\[0\]"):
        require_claim_evidence(QBRMetadata(customer_name="Northwind", customer_evidence=[supported], risks=[Finding(description="Invented risk")]))


def test_empty_partial_result_is_allowed():
    require_claim_evidence(QBRMetadata())


def test_evidenced_claims_are_allowed():
    evidence = Evidence(page_number=1, source_text="Users")
    require_claim_evidence(QBRMetadata(kpis=[KPIMetric(name="Users", current_value="875", evidence=[evidence])]))
