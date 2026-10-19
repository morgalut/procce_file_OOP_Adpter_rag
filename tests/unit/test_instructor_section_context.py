"""Regression tests for V3 fragment -> LLM payload and strict evidence scope."""
import json
from unittest.mock import AsyncMock

import pytest

from qbr.adapters.llm.instructor_adapter import (
    EvidenceVerificationError, InstructorMetadataAdapter, LLMExtractionSettings,
)
from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment, CleanTable
from qbr.domain.models import Evidence, Finding, QBRMetadata


def make_adapter(metadata):
    client = type("Completions", (), {})()
    client.create = AsyncMock(return_value=metadata)
    return InstructorMetadataAdapter(client, LLMExtractionSettings(model="test-model")), client


def make_fragment(**changes):
    values = dict(document_id="northwind", page_number=1, fragment_index=0,
                  text="Operations adoption remains below target.",
                  source_text="FULL RAW PAGE - DO NOT SEND",
                  source_ids=("p1-b12",), section="challenges", content_type="text")
    values.update(changes)
    return CleanDocumentFragment(**values)


@pytest.mark.asyncio
async def test_payload_contains_section_and_source_ids_without_raw_page():
    adapter, client = make_adapter(QBRMetadata())
    await adapter.generate_clean_fragment(make_fragment())
    payload = json.loads(client.create.await_args.kwargs["messages"][1]["content"])
    assert payload["section"] == "challenges"
    assert payload["source_ids"] == ["p1-b12"]
    assert "FULL RAW PAGE" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_evidence_cannot_be_validated_from_unrelated_raw_page():
    metadata = QBRMetadata(challenges=[Finding(
        description="Unrelated", evidence=[Evidence(page_number=1, source_text="FULL RAW PAGE")])])
    adapter, _ = make_adapter(metadata)
    with pytest.raises(EvidenceVerificationError, match="not present"):
        await adapter.generate_clean_fragment(make_fragment())


@pytest.mark.asyncio
async def test_exact_fragment_evidence_is_accepted():
    metadata = QBRMetadata(challenges=[Finding(
        description="Low adoption", evidence=[Evidence(page_number=1, source_text="Operations adoption remains below target.")])])
    adapter, _ = make_adapter(metadata)
    result = await adapter.generate_clean_fragment(make_fragment())
    assert len(result.metadata.challenges) == 1


@pytest.mark.asyncio
async def test_table_payload_preserves_cell_identity_and_evidence():
    table = CleanTable(page_number=1, table_index=0,
        rows=(("Monthly Active Users", "640", "875"),),
        original_rows=(("Monthly Active Users", "640", "875"),),
        table_id="p1-t0", row_offset=1,
        cell_ids=(("p1-t0-r1-c0", "p1-t0-r1-c1", "p1-t0-r1-c2"),),
        header_rows=(("Metric", "Q1", "Q2"),))
    adapter, client = make_adapter(QBRMetadata())
    await adapter.generate_clean_fragment(make_fragment(text="", tables=(table,), section="performance", content_type="table"))
    payload = json.loads(client.create.await_args.kwargs["messages"][1]["content"])
    assert payload["tables"][0]["header"] == [["Metric", "Q1", "Q2"]]
    assert payload["tables"][0]["rows"][0] == ["Monthly Active Users", "640", "875"]
    assert payload["tables"][0]["row_refs"][0]["row_id"] == "p1-t0-r1"
    assert payload["tables"][0]["row_refs"][0]["cell_ids"][2] == "p1-t0-r1-c2"
