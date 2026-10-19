"""Real-document regression tests for PDF extraction -> deterministic cleaning.

Run: python -m pytest tests/integration/test_pdf_processing_real_qbr.py -v -s
The test reads the existing PDF fixture; it never fabricates its contents.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from qbr.adapters.pdf.pdf_processing_adapter import (
    CleanDocumentFragment,
    PDFProcessingAdapter,
    ProcessingOptions,
)
from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter
from qbr.ports.pdf_extractor import PDFSource

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "tests/fixtures/01_Northwind_Technologies_Q2_2026.pdf"
OUTPUT_PATH = ROOT / "tests/output/northwind_pdf_processing.json"
DOCUMENT_ID = "northwind-q2-2026"


def _table_as_dict(table):
    return {
        "page_number": table.page_number,
        "table_index": table.table_index,
        "rows": [list(row) for row in table.rows],
        "original_rows": [list(row) for row in table.original_rows],
        "table_id": table.table_id,
        "row_offset": table.row_offset,
        "cell_ids": [list(row) for row in table.cell_ids],
        "header_rows": [list(row) for row in table.header_rows],
    }


def _fragment_as_dict(fragment: CleanDocumentFragment):
    return {
        "document_id": fragment.document_id,
        "page_number": fragment.page_number,
        "fragment_index": fragment.fragment_index,
        "text": fragment.text,
        "tables": [_table_as_dict(table) for table in fragment.tables],
        "warnings": list(fragment.warnings),
        "has_content": fragment.has_content,
        "source_text": fragment.source_text,
        "source_ids": list(fragment.source_ids),
        "section": fragment.section,
        "content_type": fragment.content_type,
    }


@pytest.fixture(scope="module")
def real_pdf_path():
    assert PDF_PATH.is_file(), (
        f"Real QBR fixture missing: {PDF_PATH}\n"
        "Copy your Northwind PDF into tests/fixtures/ before running this test."
    )
    return PDF_PATH


@pytest.fixture(scope="module")
def extracted_data(real_pdf_path):
    return asyncio.run(_extract_data(real_pdf_path))


async def _extract_data(real_pdf_path):
    parser = PyMuPDFAdapter()
    processor = PDFProcessingAdapter()
    source = PDFSource(document_id=DOCUMENT_ID, path=real_pdf_path)
    pages = []
    fragments = []
    async for page in parser.iter_pages(source):
        pages.append(page)
        fragments.extend(processor.process_page(DOCUMENT_ID, page))
    return pages, fragments


@pytest.mark.asyncio
async def test_real_pdf_has_text_and_two_tables(extracted_data):
    pages, _ = extracted_data
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Northwind Technologies" in pages[0].text
    assert "Q2 2026" in pages[0].text
    assert len(pages[0].tables) == 2


@pytest.mark.asyncio
async def test_real_kpi_values_preserved(extracted_data):
    _, fragments = extracted_data
    table_fragments = [f for f in fragments if f.tables]
    all_rows = [row for f in table_fragments for table in f.tables for row in table.rows]
    assert ("Monthly Active Users", "640", "875") in all_rows
    assert ("Platform Adoption", "61%", "74%") in all_rows
    assert any(row[0] == "Support Tickets" and row[1:] == ("42", "29") for row in all_rows)


@pytest.mark.asyncio
async def test_real_actions_preserved(extracted_data):
    _, fragments = extracted_data
    rows = [row for f in fragments for t in f.tables for row in t.rows]
    assert ("Action", "Owner", "Target") in rows
    assert ("Complete Operations onboarding", "Sarah Levi", "15 July 2026") in rows


@pytest.mark.asyncio
async def test_fragments_keep_page_provenance(extracted_data):
    pages, fragments = extracted_data
    assert fragments
    assert [f.fragment_index for f in fragments] == list(range(len(fragments)))
    assert all(f.document_id == DOCUMENT_ID for f in fragments)
    assert all(f.page_number == 1 for f in fragments)
    # V3 stores canonical raw text on PDFPage, not on every LLM fragment.
    assert pages[0].text
    assert all(not f.source_text for f in fragments)
    assert all(f.source_ids for f in fragments)
    assert all(f.section for f in fragments)
    assert all(f.content_type in {"text", "table"} for f in fragments)
    assert all(f.has_content for f in fragments)


@pytest.mark.asyncio
async def test_cleaning_preserves_important_prose(extracted_data):
    _, fragments = extracted_data
    text = "\n".join(f.text for f in fragments)
    assert "Operations adoption remains below target" in text
    sections = {f.section for f in fragments}
    assert {"executive_overview", "goals", "performance", "highlights", "challenges", "risks", "next_steps"} <= sections
    assert any(f.section == "executive_overview" and "Q2 showed continued improvement" in f.text for f in fragments)
    assert any(f.section == "risks" and "Low adoption within Operations" in f.text for f in fragments)
    assert any(f.section == "next_steps" and f.tables for f in fragments)


@pytest.mark.asyncio
async def test_async_stream_matches_direct_processing(real_pdf_path, extracted_data):
    _, expected = extracted_data
    parser = PyMuPDFAdapter()
    processor = PDFProcessingAdapter()
    source = PDFSource(document_id=DOCUMENT_ID, path=real_pdf_path)
    actual = [f async for f in processor.iter_fragments(DOCUMENT_ID, parser.iter_pages(source))]
    assert actual == expected


@pytest.mark.asyncio
async def test_small_chunks_preserve_text_and_table_rows(extracted_data):
    pages, _ = extracted_data
    processor = PDFProcessingAdapter(ProcessingOptions(max_fragment_chars=200, max_table_rows_per_fragment=2))
    fragments = processor.process_page(DOCUMENT_ID, pages[0])
    text = "\n".join(f.text for f in fragments if f.text)
    # Reconstructing across line-based chunks must not discard semantic values.
    assert "Northwind Technologies" in text
    assert "Q2 2026" in text
    actual_rows = [row for f in fragments for table in f.tables for row in table.rows]
    expected_rows = [row for table in pages[0].tables for row in table.rows]
    assert actual_rows == expected_rows
    assert all(len(f.text) <= 200 for f in fragments)
    assert all(len(t.rows) <= 2 for f in fragments for t in f.tables)


@pytest.fixture(scope="module", autouse=True)
def export_actual_clean_data_for_llm(extracted_data):
    """Always create tests/output JSON when this test module is executed."""
    pages, fragments = extracted_data
    payload = {
        "document_id": DOCUMENT_ID,
        "source_file": str(PDF_PATH.relative_to(ROOT)),
        "stage": "pdf_preprocessing_only_no_llm_called",
        "page_count": len(pages),
        "fragment_count": len(fragments),
        "pages": [
            {
                "page_number": p.page_number,
                "raw_text": p.text,
                "raw_tables": [
                    {"page_number": t.page_number, "rows": [list(r) for r in t.rows]}
                    for t in p.tables
                ],
                "warnings": list(p.warnings),
            }
            for p in pages
        ],
        "clean_fragments_for_llm": [_fragment_as_dict(f) for f in fragments],
        "metadata_extraction": {
            "status": "not_executed",
            "note": "LLM metadata extraction is a later stage; this JSON contains cleaned PDF data only.",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved actual PDF data to: {OUTPUT_PATH}")
    assert OUTPUT_PATH.is_file()
    return OUTPUT_PATH


def test_output_file_contains_real_pdf_data(export_actual_clean_data_for_llm):
    saved = json.loads(export_actual_clean_data_for_llm.read_text(encoding="utf-8"))
    assert saved["document_id"] == DOCUMENT_ID
    assert saved["page_count"] >= 1
    assert saved["fragment_count"] >= 1
    assert saved["clean_fragments_for_llm"]
    assert "Northwind Technologies" in json.dumps(saved, ensure_ascii=False)
