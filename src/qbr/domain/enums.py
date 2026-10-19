
"""Domain enumerations for the QBR Intelligence Pipeline.

This module contains only standard-library types.
No infrastructure or framework dependencies are allowed.
"""

from enum import StrEnum


class ProcessingStatus(StrEnum):
    """Overall document processing lifecycle."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class ValidationStatus(StrEnum):
    """Outcome of metadata validation."""

    PENDING = "pending"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class IssueSeverity(StrEnum):
    """Severity of a validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EvidenceStatus(StrEnum):
    """Source support for an extracted claim."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"
    MISSING = "missing"
    UNVERIFIED = "unverified"


class PipelineStage(StrEnum):
    """Identifies the stage currently being executed."""

    INGESTION = "ingestion"
    PDF_EXTRACTION = "pdf_extraction"
    NORMALIZATION = "normalization"
    METADATA_EXTRACTION = "metadata_extraction"
    VALIDATION = "validation"
    PERSISTENCE = "persistence"
    CHUNK_PREPARATION = "chunk_preparation"
    VECTORIZATION = "vectorization"


class EmbeddingStatus(StrEnum):
    """Lifecycle of a document chunk's embedding."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
