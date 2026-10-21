"""Complete source-page review with explicit coverage, bounded requests and safe proposals."""
from __future__ import annotations
from dataclasses import dataclass
import json
import re
from qbr.domain.models import QBRMetadata
from qbr.domain.review_context import DocumentReviewContext, PageContext
from qbr.domain.reconciliation_models import ReviewResult, VerifiedProposal, ReviewAction
from qbr.ports.pdf_extractor import PDFExtractor, PDFSource, PDFExtractionOptions
from qbr.ports.document_reviewer import DocumentReviewer
from qbr.infrastructure.event_tracking import stage, emit


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?%?", value))


def _review_size(context: str, metadata: QBRMetadata) -> int:
    """Conservative character budget including metadata (not a token guarantee)."""
    return len(context) + len(json.dumps(metadata.model_dump(mode="json"), ensure_ascii=False))


@dataclass(frozen=True)
class ReconcileMetadata:
    pdf_extractor: PDFExtractor
    reviewer: DocumentReviewer
    max_document_chars: int = 250_000
    max_pages: int = 150
    batch_chars: int = 30_000
    max_batches: int = 60
    native_reviewer: object | None = None

    async def execute(self, source: PDFSource, metadata: QBRMetadata) -> ReviewResult:
        if min(self.max_document_chars, self.batch_chars, self.max_pages, self.max_batches) < 1:
            raise ValueError("Review limits must be positive")
        pages: dict[int, PageContext] = {}
        async with stage("document_review.load_full_document"):
            async for page in self.pdf_extractor.iter_pages(source, options=PDFExtractionOptions()):
                if page.page_number in pages:
                    raise ValueError("Duplicate page number in PDF parser output")
                if len(pages) >= self.max_pages:
                    raise ValueError("Document exceeds review page limit")
                tables = [{"table_id": table.table_id or f"p{page.page_number}-t{i}",
                           "rows": [list(row) for row in table.rows],
                           "cells": [{"cell_id": c.cell_id, "row": c.row_index,
                                      "column": c.column_index, "text": c.text}
                                     for c in table.cells]}
                          for i, table in enumerate(page.tables)]
                pages[page.page_number] = PageContext(
                    page_number=page.page_number, text=page.text,
                    tables=tables, warnings=list(page.warnings))
        if not pages:
            raise ValueError("No pages returned from PDF parser")
        ordered = sorted(pages)
        if ordered != list(range(1, len(ordered) + 1)):
            raise ValueError("Incomplete PDF page coverage: missing or non-contiguous pages")
        if not any(p.text.strip() or p.tables for p in pages.values()):
            raise ValueError("No extractable content: OCR or native PDF review required")
        if any(not p.text.strip() and not p.tables for p in pages.values()) and self.native_reviewer is None:
            raise ValueError("Page with no extractable content: enable native PDF review or OCR")

        all_pages = [pages[n] for n in ordered]
        all_context = DocumentReviewContext(
            document_id=source.document_id, total_pages=len(all_pages),
            included_pages=ordered, pages=all_pages,
            candidate_metadata=metadata.model_dump(mode="json"), complete=True,
        )
        total_payload = all_context.model_context()
        if _review_size(total_payload, metadata) > self.max_document_chars:
            raise ValueError("Document exceeds overall review context budget")
        # One request for small documents. For large documents, each batch carries
        # an explicit manifest, metadata, and full source data for its own pages.
        batches: list[list[PageContext]] = []
        if _review_size(total_payload, metadata) <= self.batch_chars:
            batches = [all_pages]
        else:
            current: list[PageContext] = []
            for page in all_pages:
                candidate = current + [page]
                context = DocumentReviewContext(
                    document_id=source.document_id, total_pages=len(all_pages),
                    included_pages=[p.page_number for p in candidate], pages=candidate,
                    candidate_metadata=metadata.model_dump(mode="json"),
                    complete=len(candidate) == len(all_pages))
                if _review_size(context.model_context(), metadata) > self.batch_chars:
                    if not current:
                        raise ValueError(f"Page {page.page_number} exceeds per-request context budget")
                    batches.append(current)
                    current = [page]
                    single = DocumentReviewContext(
                        document_id=source.document_id, total_pages=len(all_pages),
                        included_pages=[page.page_number], pages=current,
                        candidate_metadata=metadata.model_dump(mode="json"), complete=False)
                    if _review_size(single.model_context(), metadata) > self.batch_chars:
                        raise ValueError(f"Page {page.page_number} exceeds per-request context budget")
                else:
                    current = candidate
            if current:
                batches.append(current)
        if len(batches) > self.max_batches:
            raise ValueError("Too many review batches")
        proposals = []
        questions = []
        covered: set[int] = set()
        for index, batch in enumerate(batches, 1):
            included = [p.page_number for p in batch]
            ctx = DocumentReviewContext(
                document_id=source.document_id, total_pages=len(all_pages),
                included_pages=included, pages=batch,
                candidate_metadata=metadata.model_dump(mode="json"),
                complete=len(batch) == len(all_pages))
            context = ctx.model_context()
            if _review_size(context, metadata) > self.batch_chars:
                raise ValueError("Review payload exceeds batch budget")
            async with stage("document_review.llm_compare", input_summary={
                "batch": index, "batch_count": len(batches),
                "pages": len(batch), "characters": len(context)}):
                review = await self.reviewer.review(document_text=context, metadata=metadata)
            covered.update(included)
            proposals.extend(review.proposals)
            questions.extend(review.unresolved_questions)
        if covered != set(ordered):
            raise ValueError("Not all document pages were submitted for review")

        # Cross-page review: source excerpts are rehydrated from actual pages, not
        # merely recycled LLM assertions. Never misrepresent this as full coverage.
        if len(batches) > 1:
            referenced = sorted({p.page_number for p in proposals if p.page_number in pages})
            if referenced:
                selected = [pages[n] for n in referenced]
                cross_ctx = DocumentReviewContext(
                    document_id=source.document_id, total_pages=len(all_pages),
                    included_pages=referenced, pages=selected,
                    candidate_metadata=metadata.model_dump(mode="json"), complete=False,
                    context_type="targeted_cross_page_reconciliation")
                payload = cross_ctx.model_context() + "\nBATCH FINDINGS:\n" + json.dumps(
                    [p.model_dump(mode="json") for p in proposals], ensure_ascii=False)
                if _review_size(payload, metadata) <= self.batch_chars:
                    async with stage("document_review.global_reconciliation",
                                     input_summary={"referenced_pages": len(referenced)}):
                        global_review = await self.reviewer.review(document_text=payload, metadata=metadata)
                    proposals.extend(global_review.proposals)
                    questions.extend(global_review.unresolved_questions)
                else:
                    questions.append("Cross-page reconciliation skipped: evidence exceeds context budget")
            else:
                questions.append("Cross-page reconciliation not performed: no source-backed proposals to correlate")
            questions.append("Cross-page relationships outside retrieved source pages were not globally verified")
        if self.native_reviewer is not None:
            async with stage("document_review.native_pdf_second_opinion",
                             input_summary={"pages": len(all_pages)}):
                native_review = await self.native_reviewer.review_pdf(pdf_path=source.path, metadata=metadata)
            proposals.extend(native_review.proposals)
            questions.extend(native_review.unresolved_questions)
        verified = []
        seen = set()
        for proposal in proposals:
            key = (proposal.action, proposal.field_path, proposal.page_number,
                   proposal.source_excerpt, proposal.proposed_value)
            if key in seen:
                continue
            seen.add(key)
            page = pages.get(proposal.page_number)
            sources = [] if page is None else [page.text] + [
                str(cell or "") for t in page.tables for row in t["rows"] for cell in row]
            valid = page is not None and any(
                _normalize(proposal.source_excerpt) in _normalize(s)
                for s in sources if s.strip())
            reason = None if valid else (
                "page_not_found" if page is None else "excerpt_not_found_on_page")
            if valid and proposal.action in {ReviewAction.CORRECT_VALUE, ReviewAction.ADD_MISSING}:
                proposed_numbers = _numbers(proposal.proposed_value or "")
                evidence_numbers = _numbers(proposal.source_excerpt)
                if proposed_numbers and not proposed_numbers.issubset(evidence_numbers):
                    valid, reason = False, "proposed_numbers_not_in_evidence"
            verified.append(VerifiedProposal(
                proposal=proposal, evidence_verified=valid,
                status="PENDING_APPROVAL" if valid else "REJECTED",
                rejection_reason=reason,
                semantic_status="NEEDS_SEMANTIC_OR_HUMAN_REVIEW" if valid else "NOT_EVALUATED"))
        await emit("document_review.completed", {
            "pages": len(all_pages), "pages_sent": len(covered), "batches": len(batches),
            "proposals": len(verified),
            "pending": sum(p.status == "PENDING_APPROVAL" for p in verified),
            "rejected": sum(p.status == "REJECTED" for p in verified)})
        original = metadata.model_dump(mode="json")
        return ReviewResult(
            document_id=source.document_id, page_count=len(all_pages),
            candidate_metadata=original, proposals=verified,
            unresolved_questions=list(dict.fromkeys(questions)),
            validated_metadata=original, review_batches=len(batches))
