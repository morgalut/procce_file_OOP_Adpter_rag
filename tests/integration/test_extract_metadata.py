"""Integration tests: real PDF/parser/preprocessor, deterministic fake LLM."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from qbr.adapters.pdf.pdf_processing_adapter import PDFProcessingAdapter, ProcessingOptions
from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter
from qbr.application.extract_metadata import ExtractMetadata, ExtractionLimits, ExtractionPipelineError
from qbr.domain.models import Evidence, KPIMetric, QBRMetadata
from qbr.domain.validation_rules import ValidationEngine
from qbr.ports.metadata_generator import MetadataExtractionResult
from qbr.ports.pdf_extractor import PDFPage, PDFSource

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "tests" / "fixtures" / "01_Northwind_Technologies_Q2_2026.pdf"
OUTPUT = ROOT / "tests" / "output" / "northwind_metadata_pipeline.json"


class FakeMetadataGenerator:
    """Deterministic stub, never calls an LLM."""

    def __init__(self, *, fail: bool = False, wrong_provenance: bool = False):
        self.calls = []
        self.fail = fail
        self.wrong_provenance = wrong_provenance

    async def generate_clean_fragment(self, fragment):
        self.calls.append(fragment)
        if self.fail:
            raise TimeoutError("simulated LLM timeout")
        content = fragment.text + "\n" + "\n".join(
            " | ".join(cell or "" for cell in row)
            for table in fragment.tables for row in table.rows
        )
        metadata = QBRMetadata()
        if "Northwind Technologies" in content:
            metadata.customer_name = "Northwind Technologies"
            metadata.customer_evidence = [Evidence(
                page_number=fragment.page_number, source_text="Northwind Technologies"
            )]
        if "Q2 2026" in content:
            metadata.reporting_period = "Q2 2026"
            metadata.period_evidence = [Evidence(
                page_number=fragment.page_number, source_text="Q2 2026"
            )]
        if all(token in content for token in ("Monthly Active Users", "640", "875")):
            metadata.kpis.append(KPIMetric(
                name="Monthly Active Users", previous_value="640", current_value="875",
                evidence=[Evidence(page_number=fragment.page_number, source_text="Monthly Active Users")],
            ))
        return MetadataExtractionResult(
            document_id="incorrect-id" if self.wrong_provenance else fragment.document_id,
            page_number=fragment.page_number, metadata=metadata,
            warnings=("fake_llm_test_only",),
        )


def pipeline(generator=None, *, limits=None, processor=None, extractor=None):
    return ExtractMetadata(
        extractor=extractor or PyMuPDFAdapter(),
        processor=processor or PDFProcessingAdapter(),
        generator=generator or FakeMetadataGenerator(),
        validator=ValidationEngine(), limits=limits,
    )


def source(path=PDF):
    return PDFSource(document_id="northwind-q2-2026", path=path)


@pytest.fixture(scope="module")
def real_pdf():
    assert PDF.is_file(), f"Missing real fixture: {PDF}"
    return PDF


@pytest.mark.asyncio
async def test_real_pdf_end_to_end_with_fake_llm_and_export(real_pdf):
    fake = FakeMetadataGenerator()
    result = await pipeline(fake).execute(source(real_pdf))
    assert result.page_count >= 1
    assert result.fragment_count >= 1
    assert result.fragment_count == len(result.fragments) == len(fake.calls)
    assert result.metadata.customer_name == "Northwind Technologies"
    assert result.metadata.reporting_period == "Q2 2026"
    assert all(f.page_number >= 1 for f in result.fragments)
    assert all(f.fragment_id.startswith("page-") for f in result.fragments)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_pdf": str(real_pdf.relative_to(ROOT)),
        "document_id": result.document_id,
        "metadata_source": "deterministic_fake_generator_NOT_real_LLM",
        "page_count": result.page_count,
        "fragment_count": result.fragment_count,
        "clean_fragments_for_llm": [
            {
                "page_number": f.page_number,
                "fragment_index": f.fragment_index,
                "text": f.text,
                "tables": [
                    {"page_number": t.page_number, "table_index": t.table_index,
                     "rows": [list(row) for row in t.rows]}
                    for t in f.tables
                ],
                "warnings": list(f.warnings),
            }
            for f in fake.calls
        ],
        "metadata": result.metadata.model_dump(mode="json"),
        "validation": result.validation.model_dump(mode="json"),
        "conflicts": [c.model_dump(mode="json") for c in result.accumulation.conflicts],
        "fragment_outcomes": [
            {"fragment_id": f.fragment_id, "page_number": f.page_number,
             "fragment_index": f.fragment_index, "warnings": list(f.warnings),
             "validation": f.validation.model_dump(mode="json")}
            for f in result.fragments
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    assert json.loads(OUTPUT.read_text(encoding="utf-8"))["metadata"] == payload["metadata"]
    print(f"\nSaved real PDF processing + fake metadata output: {OUTPUT}")
    print(f"Pages={result.page_count}, fragments={result.fragment_count}, validation={result.validation.status}")


@pytest.mark.asyncio
async def test_kpi_is_grounded_in_real_pdf(real_pdf):
    result = await pipeline().execute(source(real_pdf))
    assert any(k.name == "Monthly Active Users" and k.previous_value == "640"
               and k.current_value == "875" for k in result.metadata.kpis)


@pytest.mark.asyncio
async def test_each_fragment_is_processed_once(real_pdf):
    fake = FakeMetadataGenerator()
    result = await pipeline(fake).execute(source(real_pdf))
    assert len(fake.calls) == result.fragment_count
    assert [(f.page_number, f.fragment_index) for f in fake.calls] == [
        (f.page_number, f.fragment_index) for f in result.fragments
    ]


@pytest.mark.asyncio
async def test_source_file_missing_is_pdf_stage_error(tmp_path):
    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline().execute(source(tmp_path / "missing.pdf"))
    assert err.value.stage == "pdf_or_limits"


@pytest.mark.asyncio
async def test_llm_failure_is_contextualized(real_pdf):
    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline(FakeMetadataGenerator(fail=True)).execute(source(real_pdf))
    assert err.value.stage == "llm"
    assert "simulated LLM timeout" in str(err.value)


@pytest.mark.asyncio
async def test_llm_provenance_mismatch_is_rejected(real_pdf):
    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline(FakeMetadataGenerator(wrong_provenance=True)).execute(source(real_pdf))
    assert err.value.stage == "pdf_or_limits"
    assert "provenance" in str(err.value).lower()


@pytest.mark.asyncio
async def test_max_fragments_is_enforced(real_pdf):
    processor = PDFProcessingAdapter(ProcessingOptions(max_fragment_chars=100))
    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline(limits=ExtractionLimits(max_fragments=1), processor=processor).execute(source(real_pdf))
    assert err.value.stage == "pdf_or_limits"
    assert "max_fragments" in str(err.value)


@pytest.mark.asyncio
async def test_empty_extraction_fails_closed():
    class EmptyExtractor:
        async def iter_pages(self, source, *, options=None):
            if False:
                yield PDFPage(page_number=1, text="")

    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline(extractor=EmptyExtractor()).execute(source())
    assert err.value.stage == "pdf"


@pytest.mark.asyncio
async def test_preprocessor_failure_has_stage_context():
    class OnePageExtractor:
        async def iter_pages(self, source, *, options=None):
            yield PDFPage(page_number=1, text="Hello")

    class BrokenProcessor:
        def process_page(self, document_id, page):
            raise RuntimeError("preprocessing broken")

    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline(extractor=OnePageExtractor(), processor=BrokenProcessor()).execute(source())
    assert err.value.stage == "preprocess"


@pytest.mark.asyncio
async def test_page_order_is_enforced():
    class UnorderedExtractor:
        async def iter_pages(self, source, *, options=None):
            yield PDFPage(page_number=2, text="Hello")
            yield PDFPage(page_number=1, text="World")

    with pytest.raises(ExtractionPipelineError) as err:
        await pipeline(extractor=UnorderedExtractor()).execute(source())
    assert err.value.stage == "pdf_or_limits"
    assert "unordered" in str(err.value)


def test_invalid_limits():
    with pytest.raises(ValueError):
        ExtractionLimits(max_pages=0)
    with pytest.raises(ValueError):
        ExtractionLimits(max_fragments=0)
