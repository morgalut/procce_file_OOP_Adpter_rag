"""Real-PDF tests for PyMuPDFAdapter; generates PDFs in pytest tmp_path."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pymupdf
import pytest

from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter, PyMuPDFLimits
from qbr.domain.exceptions import DocumentNotFoundError, ExtractionError, InvalidDocumentError
from qbr.ports.pdf_extractor import PDFExtractionOptions, PDFPage, PDFSource


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A realistic two-page text-and-table QBR, generated during the test."""
    path = tmp_path / "sample_qbr.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 55), "QBR: Acme Corporation | Q2 2026", fontsize=15)
    page.insert_text((50, 85), "Executive summary: Adoption increased this quarter.")
    page.insert_text((50, 110), "Goal: Improve product adoption.")
    # Draw ruled table; find_tables() detects grid geometry.
    x_edges, y_edges = [50, 240, 350], [150, 178, 206, 234]
    for x in x_edges:
        page.draw_line((x, y_edges[0]), (x, y_edges[-1]), color=(0, 0, 0))
    for y in y_edges:
        page.draw_line((x_edges[0], y), (x_edges[-1], y), color=(0, 0, 0))
    for row, cells in enumerate([("KPI", "Current"), ("Active Users", "650"), ("Adoption", "78%")]):
        for col, cell in enumerate(cells):
            page.insert_text((x_edges[col] + 6, y_edges[row] + 19), cell, fontsize=10)
    page2 = doc.new_page()
    page2.insert_text((50, 60), "Risks: Integration delay.")
    page2.insert_text((50, 90), "Next step: Complete onboarding. Owner: Dana. Due: 2026-07-15.")
    doc.save(path)
    doc.close()
    return path


async def collect(path: Path, **kwargs) -> list[PDFPage]:
    source = PDFSource(document_id="qbr-001", path=path)
    return [page async for page in PyMuPDFAdapter().iter_pages(source, **kwargs)]


def test_extracts_text_and_page_numbers(sample_pdf: Path) -> None:
    pages = asyncio.run(collect(sample_pdf))
    assert [p.page_number for p in pages] == [1, 2]
    assert "Acme Corporation" in pages[0].text
    assert "Q2 2026" in pages[0].text
    assert "Integration delay" in pages[1].text
    assert all(p.has_content for p in pages)


def test_extracts_structured_kpi_table(sample_pdf: Path) -> None:
    pages = asyncio.run(collect(sample_pdf))
    assert pages[0].tables, "Expected the ruled KPI table to be detected"
    rows = pages[0].tables[0].rows
    assert rows[0] == ("KPI", "Current")
    assert ("Active Users", "650") in rows
    assert ("Adoption", "78%") in rows
    assert pages[0].tables[0].page_number == 1


def test_disable_table_extraction(sample_pdf: Path) -> None:
    pages = asyncio.run(collect(sample_pdf, options=PDFExtractionOptions(extract_tables=False)))
    assert all(not p.tables for p in pages)
    assert "Active Users" in pages[0].text


@pytest.mark.parametrize("start,end,expected", [(1, 1, [1]), (2, 2, [2]), (2, None, [2]), (1, 99, [1, 2])])
def test_page_ranges(sample_pdf: Path, start: int, end: int | None, expected: list[int]) -> None:
    pages = asyncio.run(collect(sample_pdf, options=PDFExtractionOptions(start_page=start, end_page=end)))
    assert [p.page_number for p in pages] == expected


def test_start_page_beyond_document(sample_pdf: Path) -> None:
    with pytest.raises(InvalidDocumentError, match="start page"):
        asyncio.run(collect(sample_pdf, options=PDFExtractionOptions(start_page=3)))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        asyncio.run(collect(tmp_path / "missing.pdf"))


def test_invalid_extension(tmp_path: Path) -> None:
    path = tmp_path / "fake.txt"
    path.write_bytes(b"not a pdf")
    with pytest.raises(InvalidDocumentError, match="extension"):
        asyncio.run(collect(path))


@pytest.mark.parametrize("payload", [b"", b"not actually a PDF"])
def test_empty_or_corrupt_pdf(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(payload)
    with pytest.raises(InvalidDocumentError):
        asyncio.run(collect(path))


def test_rejects_encrypted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "protected.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "Secret QBR")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader")
    doc.close()
    with pytest.raises(InvalidDocumentError, match="Password-protected"):
        asyncio.run(collect(path))


def test_rejects_excess_pages(sample_pdf: Path) -> None:
    adapter = PyMuPDFAdapter(PyMuPDFLimits(max_pages=1))
    async def run():
        return [p async for p in adapter.iter_pages(PDFSource("qbr-001", sample_pdf))]
    with pytest.raises(InvalidDocumentError, match="page count"):
        asyncio.run(run())


def test_rejects_oversized_file(sample_pdf: Path, tmp_path: Path) -> None:
    path = tmp_path / "oversized.pdf"
    path.write_bytes(sample_pdf.read_bytes() + b" " * (1024 * 1024 + 1))
    adapter = PyMuPDFAdapter(PyMuPDFLimits(max_file_size_mb=1))
    async def run():
        return [p async for p in adapter.iter_pages(PDFSource("qbr-001", path))]
    with pytest.raises(InvalidDocumentError, match="size"):
        asyncio.run(run())


def test_blank_page_warning(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    pages = asyncio.run(collect(path))
    assert pages[0].warnings == ("no_extractable_content",)
    assert not pages[0].has_content


def test_table_failure_retains_text_and_warning(sample_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_find_tables(self, *args, **kwargs):
        raise RuntimeError("table parser unavailable")
    monkeypatch.setattr(pymupdf.Page, "find_tables", broken_find_tables)
    pages = asyncio.run(collect(sample_pdf))
    assert "Acme Corporation" in pages[0].text
    assert "table_extraction_failed" in pages[0].warnings


def test_page_extraction_failure_has_page_context(sample_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_extract(*args, **kwargs):
        raise RuntimeError("simulated parser error")
    monkeypatch.setattr(PyMuPDFAdapter, "_extract_page", staticmethod(broken_extract))
    with pytest.raises(ExtractionError) as caught:
        asyncio.run(collect(sample_pdf))
    assert caught.value.context["page_number"] == 1


def test_limits_reject_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        PyMuPDFLimits(max_pages=0)
    with pytest.raises(ValueError):
        PyMuPDFLimits(max_file_size_mb=0)


def test_real_parser_output_is_serializable(sample_pdf: Path) -> None:
    pages = asyncio.run(collect(sample_pdf))
    payload = {
        "document_id": "qbr-001",
        "pages": [
            {"page_number": p.page_number, "text": p.text,
             "tables": [{"page_number": t.page_number, "rows": t.rows} for t in p.tables],
             "warnings": p.warnings}
            for p in pages
        ],
    }
    encoded = json.dumps(payload)
    assert '"page_number": 1' in encoded
    assert '"Acme Corporation' in encoded or 'Acme Corporation' in encoded
