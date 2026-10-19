"""Provider-neutral contracts for evidence-grounded, page-level QBR metadata extraction.

The PDF adapter extracts content; a MetadataGenerator implementation interprets
that content into QBRMetadata. This port performs no LLM calls and no merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from qbr.domain.models import QBRMetadata
from qbr.ports.pdf_extractor import PDFPage


@dataclass(frozen=True, slots=True)
class MetadataExtractionRequest:
    """One document page plus stable provenance and extraction instructions.

    A caller invokes the generator separately for each page. Keeping page
    numbers in PDFPage makes evidence references unambiguous.
    """

    document_id: str
    page: PDFPage
    language_hint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-blank string")
        if not isinstance(self.page, PDFPage):
            raise TypeError("page must be a PDFPage")
        if self.language_hint is not None and not self.language_hint.strip():
            raise ValueError("language_hint must be non-blank when provided")


@dataclass(frozen=True, slots=True)
class MetadataExtractionResult:
    """Validated fragment with diagnostic information for downstream stages.

    `metadata` is a partial QBRMetadata for a single page, not the final
    document-level result. `warnings` describe uncertainty or omissions.
    """

    document_id: str
    page_number: int
    metadata: QBRMetadata
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-blank string")
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        if not isinstance(self.metadata, QBRMetadata):
            raise TypeError("metadata must be QBRMetadata")


@runtime_checkable
class MetadataGenerator(Protocol):
    """Asynchronous boundary implemented by an LLM-backed adapter.

    Implementations must:
    - Extract only claims supported by the supplied page's text/tables.
    - Return a QBRMetadata fragment, leaving absent fields empty or None.
    - Include Evidence(page_number, source_text) for extracted claims.
    - Preserve exact source excerpts rather than fabricate evidence.
    - Validate the model output against QBRMetadata before returning.
    - Raise adapter-specific errors for API/parse failures, not silently
      return a successful empty result when extraction failed.

    The application layer handles retries, fragment merging, validation,
    logging and persistence; they do not belong to this interface.
    """

    async def generate_fragment(
        self, request: MetadataExtractionRequest
    ) -> MetadataExtractionResult:
        """Extract metadata from exactly one page."""
        ...
