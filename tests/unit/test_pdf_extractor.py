"""Contract tests for the framework-independent PDF extraction port.

No PDF parsing library or actual PDF file is required for these tests.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qbr.ports.pdf_extractor import (
    PDFExtractionOptions,
    PDFExtractor,
    PDFPage,
    PDFSource,
    PDFTable,
)


@pytest.mark.parametrize("document_id", ["", " ", "\n\t"])
def test_source_rejects_blank_document_id(document_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="document_id must not be blank"):
        PDFSource(document_id=document_id, path=tmp_path / "report.pdf")


def test_source_preserves_identity_and_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.pdf"
    source = PDFSource(document_id=" qbr-001 ", path=path)
    assert source.document_id == " qbr-001 "
    assert source.path == path
    # Port validates shape, not existence; adapter owns I/O checks.
    assert not path.exists()


@pytest.mark.parametrize("path", ["report.pdf", None, 42])
def test_source_rejects_non_path(path: object) -> None:
    with pytest.raises(TypeError, match="path must be a pathlib.Path"):
        PDFSource(document_id="doc-1", path=path)  # type: ignore[arg-type]


@pytest.mark.parametrize("page_number", [0, -1, -99])
def test_table_rejects_invalid_page_number(page_number: int) -> None:
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        PDFTable(rows=(("KPI", "Value"),), page_number=page_number)


def test_table_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="table must contain at least one row"):
        PDFTable(rows=(), page_number=1)


def test_table_preserves_rows_and_missing_cells() -> None:
    rows = (("KPI", "Q2"), ("Adoption", None), ("Users", "650"))
    table = PDFTable(rows=rows, page_number=3)
    assert table.rows == rows
    assert table.rows[1][1] is None
    assert table.page_number == 3


@pytest.mark.parametrize("page_number", [0, -1, -7])
def test_page_rejects_invalid_page_number(page_number: int) -> None:
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        PDFPage(page_number=page_number, text="content")


def test_page_rejects_table_from_another_page() -> None:
    table = PDFTable(rows=(("KPI",),), page_number=2)
    with pytest.raises(ValueError, match="table page_number must match"):
        PDFPage(page_number=1, text="", tables=(table,))


@pytest.mark.parametrize(
    ("text", "tables", "expected"),
    [
        ("content", (), True),
        ("  \n\t", (), False),
        ("", (), False),
        ("", (PDFTable(rows=(("KPI", "100"),), page_number=1),), True),
        ("   ", (PDFTable(rows=((None,),), page_number=1),), True),
    ],
)
def test_page_has_content(text: str, tables: tuple[PDFTable, ...], expected: bool) -> None:
    page = PDFPage(page_number=1, text=text, tables=tables)
    assert page.has_content is expected


def test_page_preserves_warnings_without_treating_them_as_content() -> None:
    page = PDFPage(page_number=2, text="", warnings=("Image-only page",))
    assert page.warnings == ("Image-only page",)
    assert page.has_content is False


def test_page_defaults_to_empty_immutable_collections() -> None:
    page = PDFPage(page_number=1, text="text")
    assert page.tables == ()
    assert page.warnings == ()


def test_extraction_options_defaults() -> None:
    options = PDFExtractionOptions()
    assert options.extract_tables is True
    assert options.start_page == 1
    assert options.end_page is None


@pytest.mark.parametrize("start_page", [0, -1])
def test_options_reject_invalid_start_page(start_page: int) -> None:
    with pytest.raises(ValueError, match="start_page must be >= 1"):
        PDFExtractionOptions(start_page=start_page)


@pytest.mark.parametrize("start_page,end_page", [(2, 1), (5, 4), (1, 0)])
def test_options_reject_end_before_start(start_page: int, end_page: int) -> None:
    with pytest.raises(ValueError, match="end_page must be >= start_page"):
        PDFExtractionOptions(start_page=start_page, end_page=end_page)


def test_options_accept_single_page_range() -> None:
    options = PDFExtractionOptions(start_page=5, end_page=5, extract_tables=False)
    assert (options.start_page, options.end_page, options.extract_tables) == (5, 5, False)


@pytest.mark.parametrize(
    "obj,attribute,replacement",
    [
        (PDFSource("id", Path("file.pdf")), "document_id", "other"),
        (PDFTable((("x",),), 1), "page_number", 2),
        (PDFPage(1, "x"), "text", "y"),
        (PDFExtractionOptions(), "start_page", 2),
    ],
)
def test_contracts_are_frozen(obj: object, attribute: str, replacement: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attribute, replacement)


class FakePDFExtractor:
    """Minimal adapter implementing the port without external dependencies."""

    def __init__(self, pages: tuple[PDFPage, ...]) -> None:
        self.pages = pages
        self.received_source: PDFSource | None = None
        self.received_options: PDFExtractionOptions | None = None

    async def iter_pages(
        self,
        source: PDFSource,
        *,
        options: PDFExtractionOptions | None = None,
    ):
        self.received_source = source
        self.received_options = options
        selected = options or PDFExtractionOptions()
        for page in self.pages:
            if page.page_number < selected.start_page:
                continue
            if selected.end_page is not None and page.page_number > selected.end_page:
                continue
            await asyncio.sleep(0)
            yield page


def test_fake_adapter_satisfies_runtime_protocol() -> None:
    adapter = FakePDFExtractor(())
    assert isinstance(adapter, PDFExtractor)


@pytest.mark.asyncio
async def test_async_iteration_yields_pages_in_order(tmp_path: Path) -> None:
    pages = (PDFPage(1, "first"), PDFPage(2, "second"), PDFPage(3, "third"))
    adapter = FakePDFExtractor(pages)
    source = PDFSource("doc-1", tmp_path / "report.pdf")
    options = PDFExtractionOptions(start_page=2, end_page=3)

    received = [page async for page in adapter.iter_pages(source, options=options)]

    assert [page.page_number for page in received] == [2, 3]
    assert [page.text for page in received] == ["second", "third"]
    assert adapter.received_source is source
    assert adapter.received_options is options


@pytest.mark.asyncio
async def test_async_iterator_supports_empty_result(tmp_path: Path) -> None:
    adapter = FakePDFExtractor(())
    source = PDFSource("doc-1", tmp_path / "report.pdf")
    assert [page async for page in adapter.iter_pages(source)] == []


def test_non_adapter_does_not_satisfy_protocol() -> None:
    class NoExtractor:
        pass

    assert not isinstance(NoExtractor(), PDFExtractor)
