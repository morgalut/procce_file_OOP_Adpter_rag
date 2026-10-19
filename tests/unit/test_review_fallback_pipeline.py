"""Application-level fallback regression: no external LLM calls."""
from pathlib import Path
import pytest
from qbr.application.extract_metadata import ExtractMetadata
from qbr.adapters.llm.instructor_adapter import EvidenceVerificationError
from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment
from qbr.domain.enums import ProcessingStatus, ValidationStatus
from qbr.domain.models import Evidence, QBRMetadata, ValidationResult
from qbr.ports.metadata_generator import MetadataExtractionResult
from qbr.ports.pdf_extractor import PDFPage, PDFSource


class Extractor:
    async def iter_pages(self, source, *, options=None):
        yield PDFPage(page_number=1, text="Northwind")


class Processor:
    def process_page(self, document_id, page):
        return (CleanDocumentFragment(document_id, 1, 0, "Northwind"),)


class Generator:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    async def generate_clean_fragment(self, fragment):
        self.calls += 1
        if self.error:
            raise self.error
        return MetadataExtractionResult(document_id="doc", page_number=1, metadata=self.result)


class Validator:
    def __init__(self, status=ValidationStatus.PASSED):
        self.status = status

    def validate_fragment(self, metadata):
        return ValidationResult(status=self.status)

    def validate(self, metadata):
        return ValidationResult(status=self.status)


async def run(primary, fallback=None, status=ValidationStatus.PASSED):
    service = ExtractMetadata(extractor=Extractor(), processor=Processor(), generator=primary,
                              fallback_generator=fallback, validator=Validator(status))
    return await service.execute(PDFSource(document_id="doc", path=Path("unused.pdf")))


@pytest.mark.asyncio
async def test_fallback_recovers_missing_evidence():
    primary = Generator(QBRMetadata(customer_name="Northwind"))
    fallback = Generator(QBRMetadata(customer_name="Northwind", customer_evidence=[Evidence(page_number=1, source_text="Northwind")]))
    result = await run(primary, fallback)
    assert result.status == ProcessingStatus.COMPLETED
    assert not result.incomplete
    assert result.fragments[0].fallback_used
    assert result.metadata.customer_name == "Northwind"
    assert primary.calls == fallback.calls == 1


@pytest.mark.asyncio
async def test_unverified_quote_triggers_fallback():
    primary = Generator(error=EvidenceVerificationError("fabricated quote"))
    fallback = Generator(QBRMetadata())
    result = await run(primary, fallback)
    assert result.status == ProcessingStatus.COMPLETED
    assert result.fragments[0].fallback_used


@pytest.mark.asyncio
async def test_both_fail_review_required_and_no_unsupported_claims():
    primary = Generator(QBRMetadata(customer_name="invented"))
    fallback = Generator(QBRMetadata(customer_name="invented"))
    result = await run(primary, fallback)
    assert result.status == ProcessingStatus.REVIEW_REQUIRED
    assert result.incomplete
    assert result.metadata.customer_name is None
    assert result.fragments[0].failure_code
    assert result.fragments[0].error


@pytest.mark.asyncio
async def test_final_validation_failure_requires_review():
    result = await run(Generator(QBRMetadata()), status=ValidationStatus.FAILED)
    assert result.status == ProcessingStatus.REVIEW_REQUIRED
    assert result.review_required


@pytest.mark.asyncio
async def test_mismatched_fallback_provenance_requires_review():
    class WrongGenerator(Generator):
        async def generate_clean_fragment(self, fragment):
            return MetadataExtractionResult(document_id="other", page_number=1, metadata=QBRMetadata())
    result = await run(Generator(QBRMetadata(customer_name="Northwind")), WrongGenerator())
    assert result.status == ProcessingStatus.REVIEW_REQUIRED
    assert result.metadata.customer_name is None
