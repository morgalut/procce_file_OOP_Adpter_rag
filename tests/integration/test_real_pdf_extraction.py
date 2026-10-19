"""Integration test: extract and inspect a real QBR PDF."""

import json
from pathlib import Path

import pytest

from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter
from qbr.ports.pdf_extractor import PDFSource


@pytest.mark.asyncio
async def test_real_pdf_extraction():
    pdf_path = Path("tests/fixtures/01_Northwind_Technologies_Q2_2026.pdf")

    assert pdf_path.is_file(), f"PDF not found: {pdf_path}"

    parser = PyMuPDFAdapter()

    source = PDFSource(
        document_id="real-qbr-test",
        path=pdf_path,
    )

    extracted_pages = []

    async for page in parser.iter_pages(source):
        page_data = {
            "page_number": page.page_number,
            "text": page.text,
            "tables": [
                {
                    "page_number": table.page_number,
                    "rows": [list(row) for row in table.rows],
                }
                for table in page.tables
            ],
            "warnings": list(page.warnings),
        }

        extracted_pages.append(page_data)

        print(f"\n{'=' * 70}")
        print(f"PAGE {page.page_number}")
        print(f"{'=' * 70}")

        print("\nEXTRACTED TEXT:")
        print(page.text or "[No text extracted]")

        print(f"\nTABLES FOUND: {len(page.tables)}")

        for index, table in enumerate(page.tables, start=1):
            print(f"\nTABLE {index}")
            for row in table.rows:
                print(" | ".join(str(cell or "") for cell in row))

        print("\nWARNINGS:", page.warnings)

    assert extracted_pages, "Parser returned no pages"

    output_path = Path("tests/output/real_pdf_extraction.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            {
                "document": str(pdf_path),
                "page_count": len(extracted_pages),
                "pages": extracted_pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nJSON OUTPUT: {output_path}")
    print(f"TOTAL PAGES: {len(extracted_pages)}")
