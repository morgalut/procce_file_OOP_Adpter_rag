"""Unit tests for InstructorMetadataAdapter; no network or API key required."""
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from qbr.adapters.llm.instructor_adapter import (
    EvidenceVerificationError,
    InstructorMetadataAdapter,
    LLMExtractionSettings,
    MetadataExtractionError,
)
from qbr.adapters.pdf.pdf_processing_adapter import (
    CleanDocumentFragment,
    PDFProcessingAdapter,
    ProcessingOptions,
)
from qbr.domain.models import Evidence, Finding, KPIMetric, QBRMetadata
from qbr.ports.metadata_generator import MetadataExtractionRequest
from qbr.ports.pdf_extractor import PDFPage, PDFTable


def fragment(*, text="Customer: Northwind Technologies\nReporting Period: Q2 2026\nActive Users: 640 to 875", warnings=(), tables=()):
    return CleanDocumentFragment(
        document_id="northwind-q2", page_number=1, fragment_index=0,
        text=text, source_text=text, warnings=warnings, tables=tables,
    )


def adapter(response=None, *, side_effect=None, settings=None, processor=None):
    completions = type("FakeCompletions", (), {})()
    completions.create = AsyncMock(return_value=response if response is not None else QBRMetadata(), side_effect=side_effect)
    return InstructorMetadataAdapter(completions, settings or LLMExtractionSettings(model="test-model"), processor), completions.create


def supported_metadata():
    return QBRMetadata(
        customer_name="Northwind Technologies",
        customer_evidence=[Evidence(page_number=1, source_text="Customer: Northwind Technologies")],
        reporting_period="Q2 2026",
        period_evidence=[Evidence(page_number=1, source_text="Reporting Period: Q2 2026")],
        kpis=[KPIMetric(name="Active Users", current_value="875", previous_value="640", evidence=[Evidence(page_number=1, source_text="Active Users: 640 to 875")])],
    )


@pytest.mark.asyncio
async def test_supported_metadata_and_request_payload():
    service, create = adapter(supported_metadata())
    result = await service.generate_clean_fragment(fragment())
    assert result.document_id == "northwind-q2"
    assert result.page_number == 1
    assert result.metadata.kpis[0].current_value == "875"
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["response_model"] is QBRMetadata
    assert kwargs["max_retries"] == 2
    assert "untrusted data" in kwargs["messages"][0]["content"]
    payload = json.loads(kwargs["messages"][1]["content"])
    assert payload["page_number"] == 1
    assert payload["text"].startswith("Customer: Northwind")


@pytest.mark.asyncio
async def test_empty_fragment_skips_llm_and_preserves_warnings():
    service, create = adapter()
    result = await service.generate_clean_fragment(fragment(text="", warnings=("empty_page",)))
    create.assert_not_awaited()
    assert result.metadata == QBRMetadata()
    assert result.warnings == ("empty_page", "empty_fragment_skipped")


@pytest.mark.asyncio
async def test_warnings_propagated():
    service, _ = adapter()
    result = await service.generate_clean_fragment(fragment(warnings=("table_detection_failed",)))
    assert result.warnings == ("table_detection_failed",)


@pytest.mark.asyncio
async def test_wrong_page_evidence_rejected():
    value = QBRMetadata(risks=[Finding(description="Risk", evidence=[Evidence(page_number=2, source_text="Active Users")])])
    service, _ = adapter(value)
    with pytest.raises(EvidenceVerificationError, match="differs"):
        await service.generate_clean_fragment(fragment())


@pytest.mark.asyncio
async def test_fabricated_evidence_rejected():
    value = QBRMetadata(risks=[Finding(description="Risk", evidence=[Evidence(page_number=1, source_text="fabricated source")])])
    service, _ = adapter(value)
    with pytest.raises(EvidenceVerificationError, match="not present"):
        await service.generate_clean_fragment(fragment())


@pytest.mark.asyncio
async def test_evidence_from_original_text_accepted():
    item = fragment(text="Customer: Northwind Technologies")
    item = CleanDocumentFragment(document_id=item.document_id, page_number=1, fragment_index=0, text="Customer: Northwind Technologies", source_text="  Customer: Northwind Technologies\n")
    service, _ = adapter(QBRMetadata(customer_evidence=[Evidence(page_number=1, source_text="  Customer: Northwind Technologies")]))
    result = await service.generate_clean_fragment(item)
    assert len(result.metadata.customer_evidence) == 1


@pytest.mark.asyncio
async def test_invalid_structured_response_wrapped():
    service, _ = adapter({"kpis": [{"name": ""}]})
    with pytest.raises(MetadataExtractionError) as err:
        await service.generate_clean_fragment(fragment())
    assert isinstance(err.value.__cause__, ValidationError)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [TimeoutError("timeout"), ConnectionError("offline"), RuntimeError("rate limited")])
async def test_provider_errors_wrapped_with_cause(exc):
    service, _ = adapter(side_effect=exc)
    with pytest.raises(MetadataExtractionError) as err:
        await service.generate_clean_fragment(fragment())
    assert err.value.__cause__ is exc


@pytest.mark.asyncio
async def test_input_size_limit_prevents_llm_call():
    service, create = adapter(settings=LLMExtractionSettings(model="test", max_input_chars=500))
    with pytest.raises(ValueError, match="exceeds max_input_chars"):
        await service.generate_clean_fragment(fragment(text="X" * 800))
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("doc_id,page", [("", 1), ("northwind", 0)])
async def test_invalid_fragment_rejected(doc_id, page):
    service, create = adapter()
    invalid = CleanDocumentFragment(document_id=doc_id, page_number=page, fragment_index=0, text="hello")
    with pytest.raises(ValueError, match="fragment must have"):
        await service.generate_clean_fragment(invalid)
    create.assert_not_awaited()


@pytest.mark.parametrize("kwargs", [{"model": " "}, {"model": "test", "max_retries": -1}, {"model": "test", "max_input_chars": 499}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        LLMExtractionSettings(**kwargs)


@pytest.mark.asyncio
async def test_page_level_port_single_fragment():
    service, create = adapter(supported_metadata())
    request = MetadataExtractionRequest(document_id="northwind-q2", page=PDFPage(page_number=1, text=fragment().text))
    result = await service.generate_fragment(request)
    assert result.metadata.customer_name == "Northwind Technologies"
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_page_level_port_refuses_multiple_fragments():
    service, create = adapter(processor=PDFProcessingAdapter(ProcessingOptions(max_fragment_chars=100)))
    page = PDFPage(page_number=1, text="A" * 110)
    request = MetadataExtractionRequest(document_id="northwind-q2", page=page)
    with pytest.raises(ValueError, match="multiple clean fragments"):
        await service.generate_fragment(request)
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_page_level_port_refuses_text_and_table():
    service, create = adapter()
    page = PDFPage(page_number=1, text="Customer: Northwind", tables=(PDFTable(page_number=1, rows=(("KPI", "Value"), ("Users", "875"))),))
    with pytest.raises(ValueError, match="multiple clean fragments"):
        await service.generate_fragment(MetadataExtractionRequest(document_id="northwind-q2", page=page))
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_table_cell_evidence():
    page = PDFPage(page_number=1, text="", tables=(PDFTable(page_number=1, rows=(("KPI", "Value"), ("Users", "875"))),))
    table_fragment = PDFProcessingAdapter().process_page("northwind-q2", page)[0]
    metadata = QBRMetadata(kpis=[KPIMetric(name="Users", current_value="875", evidence=[Evidence(page_number=1, source_text="875")])])
    service, create = adapter(metadata)
    result = await service.generate_clean_fragment(table_fragment)
    assert result.metadata.kpis[0].current_value == "875"
    payload = json.loads(create.await_args.kwargs["messages"][1]["content"])
    assert payload["tables"][0]["rows"][1] == ["Users", "875"]


@pytest.mark.asyncio
async def test_rejects_table_evidence_not_present_in_cells():
    page = PDFPage(page_number=1, text="", tables=(PDFTable(page_number=1, rows=(("KPI", "Value"), ("Users", "875"))),))
    table_fragment = PDFProcessingAdapter().process_page("northwind-q2", page)[0]
    metadata = QBRMetadata(kpis=[KPIMetric(name="Users", evidence=[Evidence(page_number=1, source_text="900")])])
    service, _ = adapter(metadata)
    with pytest.raises(EvidenceVerificationError):
        await service.generate_clean_fragment(table_fragment)


@pytest.mark.asyncio
async def test_evidence_check_does_not_prove_claim_truth():
    """Existing guard only checks excerpt presence, not semantic correctness."""
    metadata = QBRMetadata(kpis=[KPIMetric(name="Active Users", current_value="999", evidence=[Evidence(page_number=1, source_text="Active Users: 640 to 875")])])
    service, _ = adapter(metadata)
    result = await service.generate_clean_fragment(fragment())
    assert result.metadata.kpis[0].current_value == "999"
    # Downstream ValidationEngine / evaluations must reject unsupported claims.
