"""Concrete composition root: PDF layout -> fragments -> Instructor agents -> validation.

The application ExtractMetadata use case calls each stage in this order; no
side channel sends raw PDF bytes to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter, PyMuPDFLimits
from qbr.adapters.pdf.pdf_processing_adapter import PDFProcessingAdapter, ProcessingOptions
from qbr.adapters.llm.instructor_adapter import InstructorMetadataAdapter, LLMExtractionSettings
from qbr.adapters.llm.category_agents import build_category_fallback
from qbr.application.extract_metadata import ExtractMetadata, ExtractionLimits
from qbr.domain.validation_rules import ValidationEngine


@dataclass(frozen=True, slots=True)
class ConnectedExtractionSettings:
    model: str
    use_layout: bool = True
    layout_required: bool = False
    max_pages: int = 100
    max_fragments: int = 2000
    max_fragment_chars: int = 4000
    max_input_chars: int = 18000
    schema_retries: int = 2
    enable_agent_recovery: bool = True
    enable_wake_up: bool = False


def build_connected_extractor(
    completions: Any,
    config: ConnectedExtractionSettings,
    *,
    readiness_probe: Callable[[], Awaitable[None]] | None = None,
) -> ExtractMetadata:
    """Return the actual document pipeline consumed by document processing.

    `completions` must be Instructor-patched. `readiness_probe` is provider-
    specific; never substitute a fabricated PDF request for a health probe.
    """
    if config.enable_wake_up and readiness_probe is None:
        raise ValueError('enable_wake_up requires a provider readiness_probe')
    pdf = PyMuPDFAdapter(PyMuPDFLimits(
        max_pages=config.max_pages,
        use_layout=config.use_layout,
        layout_required=config.layout_required,
    ))
    processor = PDFProcessingAdapter(ProcessingOptions(
        max_fragment_chars=config.max_fragment_chars,
    ))
    llm_settings = LLMExtractionSettings(
        model=config.model,
        max_retries=config.schema_retries,
        max_input_chars=config.max_input_chars,
        candidate_fallback_enabled=config.enable_agent_recovery,
        wake_up_enabled=config.enable_wake_up,
    )
    fallback = (
        build_category_fallback(completions, config.model)
        if config.enable_agent_recovery else None
    )
    generator = InstructorMetadataAdapter(
        completions=completions,
        settings=llm_settings,
        processor=processor,
        fallback_extractor=fallback,
        wake_up_probe=readiness_probe,
    )
    return ExtractMetadata(
        extractor=pdf,
        processor=processor,
        generator=generator,
        validator=ValidationEngine(),
        limits=ExtractionLimits(
            max_pages=config.max_pages,
            max_fragments=config.max_fragments,
        ),
    )
