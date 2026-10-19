
"""Dependency injection for the QBR metadata extraction pipeline."""

from __future__ import annotations

import logging
from typing import Any

import instructor
from openai import AsyncOpenAI

from qbr.application.extract_metadata import (
    ExtractMetadata,
    ExtractionLimits,
)
from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter
from qbr.adapters.pdf.pdf_processing_adapter import PDFProcessingAdapter
from qbr.adapters.llm.instructor_adapter import (
    InstructorMetadataAdapter,
    LLMExtractionSettings,
    StructuredChatCompletions,
)
from qbr.domain.validation_rules import ValidationEngine
from qbr.infrastructure.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


def build_completions_client(
    settings: AppSettings,
) -> StructuredChatCompletions:
    """Create an Instructor-enabled asynchronous LLM client."""

    provider = settings.llm_provider.lower()

    if provider != "openai":
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            "This factory currently supports 'openai'."
        )

    api_key = settings.effective_openai_api_key

    if not api_key:
        raise ValueError(
            "An OpenAI API key is required. "
            "Configure OPENAI_API_KEY or LLM_API_KEY in .env."
        )

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 60.0,
        "max_retries": 2,
    }

    if settings.llm_base_url:
        client_kwargs["base_url"] = settings.llm_base_url

    openai_client = AsyncOpenAI(**client_kwargs)

    instructor_client = instructor.from_openai(
        openai_client,
        mode=instructor.Mode.TOOLS,
    )

    return instructor_client.chat.completions


def build_llm_extraction_settings(
    settings: AppSettings,
) -> LLMExtractionSettings:
    """Map application settings to the extraction adapter settings."""

    return LLMExtractionSettings(
        model=settings.llm_model,
        max_retries=2,
        max_input_chars=18_000,
        evidence_retries=0,
    )


def create_pipeline() -> ExtractMetadata:
    """Build the complete PDF-to-QBR extraction pipeline."""

    settings = get_settings()

    logger.info(
        "Initializing QBR extraction pipeline: provider=%s model=%s",
        settings.llm_provider,
        settings.llm_model,
    )

    extractor = PyMuPDFAdapter()

    processor = PDFProcessingAdapter()

    validator = ValidationEngine()

    completions = build_completions_client(settings)

    generator = InstructorMetadataAdapter(
        completions=completions,
        settings=build_llm_extraction_settings(settings),
        processor=processor,
    )

    pipeline = ExtractMetadata(
        extractor=extractor,
        processor=processor,
        generator=generator,
        validator=validator,
        limits=ExtractionLimits(
            max_pages=settings.pdf_max_pages,
            max_fragments=2_000,
        ),
    )

    logger.info("QBR extraction pipeline initialized")

    return pipeline
