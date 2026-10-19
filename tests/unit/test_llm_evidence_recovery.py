"""V6: source resolution and bounded deterministic-grounding recovery."""
import json
from unittest.mock import AsyncMock
import pytest

from qbr.adapters.llm.evidence_resolver import EvidenceResolver, SourceReferenceError
from qbr.adapters.llm.instructor_adapter import InstructorMetadataAdapter, LLMExtractionSettings, EvidenceVerificationError
from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment, CleanTable
from qbr.domain.models import QBRMetadata, KPIMetric, Evidence


def sample():
    table = CleanTable(page_number=1, table_index=0,
        rows=(("Users", "640", "875"),), original_rows=(("Users", "640", "875"),),
        table_id="p1-t0", header_rows=(("Metric", "Q1", "Q2"),))
    return CleanDocumentFragment(document_id="doc", page_number=1, fragment_index=2,
        text="Customer adoption increased.", tables=(table,), section="performance")


def extracted(current):
    return QBRMetadata(kpis=[KPIMetric(name="Users", previous_value="640", current_value=current,
        evidence=[Evidence(page_number=1, source_text="Users")])])


def test_resolve_exact_cell_and_reject_unknown():
    resolver = EvidenceResolver(sample())
    assert resolver.resolve("p1-t0-r0-c2", quote="875").text == "875"
    assert resolver.resolve("p1-f2", quote="adoption").kind == "fragment_text"
    with pytest.raises(SourceReferenceError):
        resolver.resolve("p1-t0-r0-c2", quote="640")
    with pytest.raises(SourceReferenceError):
        resolver.resolve("p1-t0-r999-c0")


@pytest.mark.asyncio
async def test_retry_recovers_from_wrong_kpi_and_preserves_table_rows():
    client = type("Client", (), {})()
    client.create = AsyncMock(side_effect=[extracted("74%"), extracted("875")])
    adapter = InstructorMetadataAdapter(client, LLMExtractionSettings(model="test", evidence_retries=1))
    result = await adapter.generate_clean_fragment(sample())
    assert result.metadata.kpis[0].current_value == "875"
    assert client.create.await_count == 2
    payload = json.loads(client.create.await_args_list[0].kwargs["messages"][1]["content"])
    assert payload["tables"][0]["rows"] == [["Users", "640", "875"]]
    assert "p1-t0-r0-c2" in payload["available_source_ids"]


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_without_accepting_bad_data():
    client = type("Client", (), {})()
    client.create = AsyncMock(return_value=extracted("74%"))
    adapter = InstructorMetadataAdapter(client, LLMExtractionSettings(model="test", evidence_retries=1))
    with pytest.raises(EvidenceVerificationError, match="after 2 attempt"):
        await adapter.generate_clean_fragment(sample())
    assert client.create.await_count == 2


@pytest.mark.asyncio
async def test_default_has_no_extra_calls():
    client = type("Client", (), {})()
    client.create = AsyncMock(return_value=extracted("74%"))
    adapter = InstructorMetadataAdapter(client, LLMExtractionSettings(model="test"))
    with pytest.raises(EvidenceVerificationError):
        await adapter.generate_clean_fragment(sample())
    assert client.create.await_count == 1
