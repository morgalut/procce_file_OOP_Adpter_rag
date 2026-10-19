"""Application orchestration: PDF -> clean fragments -> LLM -> validated QBR.

This use case is intentionally provider-neutral and performs no persistence.
The injected generator must support generate_clean_fragment().
Extraction is sequential and bounded by configurable limits. Evidence-quote
failures are recorded and skipped; other failures propagate with stage context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment
from qbr.adapters.llm.instructor_adapter import EvidenceVerificationError
from qbr.domain.metadata_accumulator import (
    AccumulationResult,
    Fragment,
    MetadataAccumulator,
)
from qbr.domain.models import QBRMetadata, ValidationResult
from qbr.domain.enums import ProcessingStatus, ValidationStatus
from qbr.ports.metadata_generator import MetadataExtractionResult
from qbr.ports.pdf_extractor import PDFExtractionOptions, PDFExtractor, PDFSource


@runtime_checkable
class FragmentProcessor(Protocol):
    def process_page(
        self, document_id: str, page: object
    ) -> tuple[CleanDocumentFragment, ...]: ...


@runtime_checkable
class CleanFragmentGenerator(Protocol):
    async def generate_clean_fragment(
        self, fragment: CleanDocumentFragment
    ) -> MetadataExtractionResult: ...


@runtime_checkable
class MetadataValidator(Protocol):
    def validate_fragment(self, metadata: QBRMetadata) -> ValidationResult: ...
    def validate(self, metadata: QBRMetadata) -> ValidationResult: ...


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    max_pages: int = 200
    max_fragments: int = 2_000

    def __post_init__(self) -> None:
        if self.max_pages < 1 or self.max_fragments < 1:
            raise ValueError("max_pages and max_fragments must be positive")


@dataclass(frozen=True, slots=True)
class FragmentOutcome:
    fragment_id: str
    page_number: int
    fragment_index: int
    warnings: tuple[str, ...]
    validation: ValidationResult | None
    error: str | None = None
    fallback_used: bool = False
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentExtractionResult:
    document_id: str
    page_count: int
    fragment_count: int
    accumulation: AccumulationResult
    validation: ValidationResult
    fragments: tuple[FragmentOutcome, ...]
    incomplete: bool = False

    @property
    def review_required(self) -> bool:
        """Evidence failures or unsuccessful final validation require review."""
        return self.status == ProcessingStatus.REVIEW_REQUIRED

    @property
    def status(self) -> ProcessingStatus:
        """Conservative business status; no unsupported facts are accepted."""
        if self.incomplete or self.validation.status != ValidationStatus.PASSED:
            return ProcessingStatus.REVIEW_REQUIRED
        return ProcessingStatus.COMPLETED

    @property
    def metadata(self) -> QBRMetadata:
        return self.accumulation.metadata


class ExtractionPipelineError(RuntimeError):
    """Raised with document/stage context; original cause is chained."""

    def __init__(self, document_id: str, stage: str, detail: str) -> None:
        self.document_id = document_id
        self.stage = stage
        super().__init__(f"document={document_id!r} stage={stage}: {detail}")


class ProvenanceMismatchError(ValueError):
    """Generator response refers to a different document or page."""


class MissingEvidenceError(ValueError):
    """The LLM produced business claims without supporting evidence."""


def require_claim_evidence(metadata: QBRMetadata) -> None:
    """Fail closed on unsupported facts, but allow truly empty partial results.

    This check is intentionally independent of the LLM provider and does not
    invent quotes. Exact excerpt/source-ID validation stays in the adapter.
    """
    scalar_claims = (
        ("customer_name", metadata.customer_name, metadata.customer_evidence),
        ("reporting_period", metadata.reporting_period, metadata.period_evidence),
        ("executive_summary", metadata.executive_summary, metadata.summary_evidence),
    )
    for field, value, evidence in scalar_claims:
        if value is not None and str(value).strip() and not evidence:
            raise MissingEvidenceError(f"{field} has a value but no evidence")
    for field in ("goals", "kpis", "achievements", "challenges", "risks", "next_steps"):
        for index, item in enumerate(getattr(metadata, field)):
            if not item.evidence:
                raise MissingEvidenceError(f"{field}[{index}] has no evidence")


class ExtractMetadata:
    """Process one authorized PDF source and return a reviewable result.

    - Only source-supplied pages are processed, in source order.
    - Each cleaned fragment is extracted once and independently validated.
    - The accumulator preserves conflicting observations and provenance.
    - Final validation occurs after accumulation.
    - No LLM tool calling, DB writes, model routing, or retry policy lives here.
    """

    def __init__(
        self,
        *,
        extractor: PDFExtractor,
        processor: FragmentProcessor,
        generator: CleanFragmentGenerator,
        validator: MetadataValidator,
        limits: ExtractionLimits | None = None,
        fallback_generator: CleanFragmentGenerator | None = None,
    ) -> None:
        self._extractor = extractor
        self._processor = processor
        self._generator = generator
        self._validator = validator
        self._fallback_generator = fallback_generator
        self._limits = limits or ExtractionLimits()

    @staticmethod
    def _check_response(response: MetadataExtractionResult, document_id: str, page_number: int) -> None:
        if response.document_id != document_id or response.page_number != page_number:
            raise ProvenanceMismatchError("LLM returned inconsistent document/page provenance")

    async def execute(
        self,
        source: PDFSource,
        *,
        options: PDFExtractionOptions | None = None,
    ) -> DocumentExtractionResult:
        """Run the pipeline; fail closed if any stage fails or limits are exceeded."""
        accumulator = MetadataAccumulator()
        outcomes: list[FragmentOutcome] = []
        page_count = 0
        fragment_count = 0
        last_page_number = 0
        incomplete = False

        try:
            async for page in self._extractor.iter_pages(source, options=options):
                if page.page_number <= last_page_number:
                    raise ValueError("PDF extractor returned duplicate or unordered pages")
                last_page_number = page.page_number
                page_count += 1
                if page_count > self._limits.max_pages:
                    raise ValueError("document exceeds max_pages")
                try:
                    fragments = self._processor.process_page(source.document_id, page)
                except Exception as exc:
                    raise ExtractionPipelineError(source.document_id, "preprocess", str(exc)) from exc
                for fragment in fragments:
                    fragment_count += 1
                    if fragment_count > self._limits.max_fragments:
                        raise ValueError("document exceeds max_fragments")
                    if fragment.document_id != source.document_id or fragment.page_number != page.page_number:
                        raise ValueError("preprocessor returned inconsistent document/page provenance")
                    fragment_id = f"page-{page.page_number:06d}-fragment-{fragment.fragment_index:06d}"
                    # Both missing evidence and failed quote/source verification are
                    # recoverable grounding failures. Never retry unrelated API errors.
                    response = None
                    grounding_error: MissingEvidenceError | EvidenceVerificationError | None = None
                    fallback_used = False
                    try:
                        candidate = await self._generator.generate_clean_fragment(fragment)
                        self._check_response(candidate, source.document_id, page.page_number)
                        require_claim_evidence(candidate.metadata)
                        response = candidate
                    except ProvenanceMismatchError as exc:
                        raise ExtractionPipelineError(source.document_id, "pdf_or_limits", f"{fragment_id}: {exc}") from exc
                    except (MissingEvidenceError, EvidenceVerificationError) as exc:
                        grounding_error = exc
                    except ExtractionPipelineError:
                        raise
                    except Exception as exc:
                        raise ExtractionPipelineError(source.document_id, "llm", f"{fragment_id}: {exc}") from exc

                    if grounding_error is not None and self._fallback_generator is not None:
                        try:
                            candidate = await self._fallback_generator.generate_clean_fragment(fragment)
                            self._check_response(candidate, source.document_id, page.page_number)
                            require_claim_evidence(candidate.metadata)
                            response = candidate
                            fallback_used = True
                        except ProvenanceMismatchError as exc:
                            raise ExtractionPipelineError(source.document_id, "pdf_or_limits", f"{fragment_id}: {exc}") from exc
                        except (MissingEvidenceError, EvidenceVerificationError) as exc:
                            grounding_error = MissingEvidenceError(
                                f"primary: {grounding_error}; fallback: {exc}"
                            )
                        except ExtractionPipelineError:
                            raise
                        except Exception as exc:
                            # A fallback outage does not destroy other extracted facts.
                            grounding_error = MissingEvidenceError(
                                f"primary: {grounding_error}; fallback unavailable: {type(exc).__name__}"
                            )

                    if response is None:
                        incomplete = True
                        code = ("evidence_verification_failed" if isinstance(grounding_error, EvidenceVerificationError)
                                else "missing_or_invalid_evidence")
                        outcomes.append(FragmentOutcome(
                            fragment_id=fragment_id,
                            page_number=page.page_number,
                            fragment_index=fragment.fragment_index,
                            warnings=(code, "fallback_exhausted" if self._fallback_generator else "fallback_not_configured"),
                            validation=None,
                            error=str(grounding_error),
                            failure_code=code,
                        ))
                        continue
                    try:
                        fragment_validation = self._validator.validate_fragment(response.metadata)
                        accumulator.add(Fragment(
                            fragment_id=fragment_id,
                            order=fragment_count - 1,
                            metadata=response.metadata,
                        ))
                    except Exception as exc:
                        raise ExtractionPipelineError(source.document_id, "accumulate", f"{fragment_id}: {exc}") from exc
                    outcomes.append(FragmentOutcome(
                        fragment_id=fragment_id,
                        page_number=page.page_number,
                        fragment_index=fragment.fragment_index,
                        warnings=response.warnings,
                        validation=fragment_validation,
                        fallback_used=fallback_used,
                    ))
        except ExtractionPipelineError:
            raise
        except Exception as exc:
            raise ExtractionPipelineError(source.document_id, "pdf_or_limits", str(exc)) from exc

        if page_count == 0:
            raise ExtractionPipelineError(source.document_id, "pdf", "extractor returned no pages")
        try:
            snapshot = accumulator.snapshot()
            validation = self._validator.validate(snapshot.metadata)
        except Exception as exc:
            raise ExtractionPipelineError(source.document_id, "final_validation", str(exc)) from exc
        return DocumentExtractionResult(
            document_id=source.document_id,
            page_count=page_count,
            fragment_count=fragment_count,
            accumulation=snapshot,
            validation=validation,
            fragments=tuple(outcomes),
            incomplete=incomplete,
        )
