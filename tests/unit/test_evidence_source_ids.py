"""V7 regression: source IDs are additive and must resolve to exact excerpts."""
import json
from unittest.mock import AsyncMock
import pytest

from qbr.domain.models import Evidence, QBRMetadata, KPIMetric
from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment, CleanTable
from qbr.adapters.llm.instructor_adapter import (
    InstructorMetadataAdapter, LLMExtractionSettings, EvidenceVerificationError,
)


def fragment():
    table = CleanTable(page_number=1, table_index=0,
        rows=(("Users", "640", "875"),), original_rows=(("Users", "640", "875"),),
        table_id="p1-t0", header_rows=(("Metric", "Q1", "Q2"),))
    return CleanDocumentFragment(document_id="doc", page_number=1,
        fragment_index=2, text="Customer adoption increased.",
        tables=(table,), section="performance")


def metadata(source_id, quote="Users"):
    return QBRMetadata(kpis=[KPIMetric(name="Users", previous_value="640",
        current_value="875", evidence=[Evidence(page_number=1,
            source_text=quote, source_id=source_id)])])


def adapter(result):
    client = type("Client", (), {})()
    client.create = AsyncMock(return_value=result)
    return InstructorMetadataAdapter(client, LLMExtractionSettings(model="test")), client


def test_evidence_source_id_optional_for_existing_records():
    evidence = Evidence(page_number=1, source_text="Users")
    assert evidence.source_id is None
    assert Evidence.model_validate(evidence.model_dump()).source_id is None


@pytest.mark.asyncio
async def test_valid_cell_source_id_preserves_original_rows():
    instance, client = adapter(metadata("p1-t0-r0-c0"))
    result = await instance.generate_clean_fragment(fragment())
    assert result.metadata.kpis[0].evidence[0].source_id == "p1-t0-r0-c0"
    payload = json.loads(client.create.await_args.kwargs["messages"][1]["content"])
    assert payload["tables"][0]["rows"] == [["Users", "640", "875"]]


@pytest.mark.asyncio
async def test_rejects_quote_from_different_cell_even_if_present_in_table():
    instance, _ = adapter(metadata("p1-t0-r0-c2"))
    with pytest.raises(EvidenceVerificationError, match="source_id"):
        await instance.generate_clean_fragment(fragment())


@pytest.mark.asyncio
async def test_rejects_unknown_source_id():
    instance, _ = adapter(metadata("p1-t0-r999-c0"))
    with pytest.raises(EvidenceVerificationError, match="source_id"):
        await instance.generate_clean_fragment(fragment())


@pytest.mark.asyncio
async def test_fragment_text_source_id_is_valid():
    instance, _ = adapter(QBRMetadata(executive_summary="Customer adoption increased.",
        summary_evidence=[Evidence(page_number=1, source_text="Customer adoption",
            source_id="p1-f2")]))
    result = await instance.generate_clean_fragment(fragment())
    assert result.metadata.summary_evidence[0].source_id == "p1-f2"
