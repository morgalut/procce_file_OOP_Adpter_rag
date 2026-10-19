
"""Framework-independent domain exceptions for QBR processing.

Exception context must contain only safe, non-sensitive diagnostic values.
"""

from typing import Any, ClassVar


class QBRDomainError(Exception):
    """Base class for expected QBR domain failures."""

    code: ClassVar[str] = "qbr_domain_error"

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        if not message.strip():
            raise ValueError("Domain error message must not be empty")

        self.message = message
        self.context = dict(context) if context is not None else {}
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class DocumentError(QBRDomainError):
    """Base class for document-related failures."""

    code = "document_error"


class DocumentNotFoundError(DocumentError):
    """Requested document does not exist."""

    code = "document_not_found"


class DuplicateDocumentError(DocumentError):
    """Document already exists according to the deduplication policy."""

    code = "duplicate_document"


class InvalidDocumentError(DocumentError):
    """Document cannot be accepted for processing."""

    code = "invalid_document"


class ExtractionError(QBRDomainError):
    """Unable to extract required content from a document."""

    code = "extraction_failed"


class MetadataValidationError(QBRDomainError):
    """Extracted metadata violates an enforced business rule."""

    code = "metadata_validation_failed"


class EvidenceValidationError(MetadataValidationError):
    """A claimed source reference cannot be validated."""

    code = "evidence_validation_failed"


class InvalidStateTransitionError(QBRDomainError):
    """A requested document state transition is not allowed."""

    code = "invalid_state_transition"
