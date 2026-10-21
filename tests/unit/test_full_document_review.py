from pathlib import Path
import pytest
from qbr.application.reconcile_metadata import ReconcileMetadata
from qbr.domain.models import QBRMetadata
from qbr.domain.reconciliation_models import CorrectionProposal, DocumentReview, ReviewAction
from qbr.ports.pdf_extractor import PDFPage, PDFSource

class FakeExtractor:
    def __init__(self, pages):
        self.pages = pages
    async def iter_pages(self, source, *, options=None):
        for page in self.pages:
            yield page

class FakeReviewer:
    def __init__(self, review):
        self.review_result = review
        self.received = None
    async def review(self, *, document_text, metadata):
        self.received = document_text
        return self.review_result

def proposal(page, excerpt):
    return CorrectionProposal(
        action=ReviewAction.CORRECT_VALUE, field_path="kpis.0.current_value",
        original_value="49%", proposed_value="94%", reason="Incorrect KPI",
        page_number=page, source_excerpt=excerpt,
    )

@pytest.mark.asyncio
async def test_full_document_is_passed_and_correction_is_not_auto_applied():
    reviewer = FakeReviewer(DocumentReview(proposals=[proposal(2, "Retention: 94%")]))
    service = ReconcileMetadata(FakeExtractor([
        PDFPage(page_number=1, text="Overview"),
        PDFPage(page_number=2, text="Retention: 94%"),
    ]), reviewer)
    original = QBRMetadata(customer_name="Acme")
    result = await service.execute(PDFSource("doc1", Path("/unused.pdf")), original)
    assert '"page_number": 1' in reviewer.received
    assert '"page_number": 2' in reviewer.received
    assert result.proposals[0].status == "PENDING_APPROVAL"
    assert result.candidate_metadata == result.validated_metadata

@pytest.mark.asyncio
async def test_fabricated_evidence_is_rejected():
    reviewer = FakeReviewer(DocumentReview(proposals=[proposal(1, "Retention: 94%")]))
    service = ReconcileMetadata(FakeExtractor([PDFPage(page_number=1, text="Retention: 49%")]), reviewer)
    result = await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert result.proposals[0].status == "REJECTED"
    assert result.proposals[0].rejection_reason == "excerpt_not_found_on_page"

@pytest.mark.asyncio
async def test_context_limit_fails_without_truncation():
    reviewer = FakeReviewer(DocumentReview())
    service = ReconcileMetadata(FakeExtractor([PDFPage(page_number=1, text="A"*200)]), reviewer, max_document_chars=100)
    with pytest.raises(ValueError, match="context budget"):
        await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert reviewer.received is None

@pytest.mark.asyncio
async def test_multi_batch_covers_all_pages_and_runs_global_reconciliation():
    class CollectingReviewer:
        def __init__(self): self.calls = []
        async def review(self, *, document_text, metadata):
            self.calls.append(document_text)
            return DocumentReview()
    reviewer = CollectingReviewer()
    pages = [PDFPage(page_number=i, text="x"*100) for i in range(1, 5)]
    service = ReconcileMetadata(FakeExtractor(pages), reviewer, batch_chars=650)
    result = await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert result.review_batches == 4
    assert len(reviewer.calls) == 4
    assert all(any(f'"page_number": {i}' in call for call in reviewer.calls) for i in range(1, 5))

@pytest.mark.asyncio
async def test_numeric_claim_not_supported_by_evidence_rejected():
    reviewer = FakeReviewer(DocumentReview(proposals=[proposal(1, "Retention: 49%")]))
    service = ReconcileMetadata(FakeExtractor([PDFPage(page_number=1, text="Retention: 49%")]), reviewer)
    result = await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert result.proposals[0].status == "REJECTED"
    assert result.proposals[0].rejection_reason == "proposed_numbers_not_in_evidence"

@pytest.mark.asyncio
async def test_oversized_single_page_rejected_not_truncated():
    reviewer = FakeReviewer(DocumentReview())
    service = ReconcileMetadata(FakeExtractor([PDFPage(page_number=1, text="a"*1500)]), reviewer, batch_chars=650)
    with pytest.raises(ValueError, match="per-request context budget"):
        await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert reviewer.received is None

@pytest.mark.asyncio
async def test_noncontiguous_page_coverage_fails_closed():
    reviewer = FakeReviewer(DocumentReview())
    service = ReconcileMetadata(FakeExtractor([
        PDFPage(page_number=1, text="one"),
        PDFPage(page_number=3, text="three")]), reviewer)
    with pytest.raises(ValueError, match="Incomplete PDF page coverage"):
        await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert reviewer.received is None

@pytest.mark.asyncio
async def test_blank_page_fails_without_native_pdf_review():
    reviewer = FakeReviewer(DocumentReview())
    service = ReconcileMetadata(FakeExtractor([
        PDFPage(page_number=1, text="one"),
        PDFPage(page_number=2, text="")]), reviewer)
    with pytest.raises(ValueError, match="no extractable content"):
        await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())

@pytest.mark.asyncio
async def test_context_manifest_includes_metadata_and_table_provenance():
    import json
    from qbr.ports.pdf_extractor import PDFTable
    reviewer = FakeReviewer(DocumentReview())
    service = ReconcileMetadata(FakeExtractor([PDFPage(
        page_number=1, text="Summary",
        tables=(PDFTable(rows=(("Metric", "94%"),), page_number=1, table_id="p1-t0"),))]), reviewer)
    result = await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata(customer_name="Acme"))
    manifest = json.loads(reviewer.received)
    assert manifest["complete"] is True
    assert manifest["included_pages"] == [1]
    assert manifest["pages"][0]["tables"][0]["table_id"] == "p1-t0"
    assert result.candidate_metadata["customer_name"] == "Acme"

@pytest.mark.asyncio
async def test_metadata_included_in_batch_budget():
    reviewer = FakeReviewer(DocumentReview())
    service = ReconcileMetadata(FakeExtractor([PDFPage(page_number=1, text="Hello")]),
                                reviewer, batch_chars=100)
    with pytest.raises(ValueError, match="per-request context budget"):
        await service.execute(PDFSource("doc1", Path("/unused.pdf")), QBRMetadata())
    assert reviewer.received is None
