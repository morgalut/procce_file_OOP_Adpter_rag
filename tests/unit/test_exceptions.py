"""Unit tests for the QBR domain exception contracts."""

import pytest

from qbr.domain.exceptions import (
    DocumentError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EvidenceValidationError,
    ExtractionError,
    InvalidDocumentError,
    InvalidStateTransitionError,
    MetadataValidationError,
    QBRDomainError,
)


EXCEPTION_CASES = [
    (QBRDomainError, "qbr_domain_error", QBRDomainError),
    (DocumentError, "document_error", QBRDomainError),
    (DocumentNotFoundError, "document_not_found", DocumentError),
    (DuplicateDocumentError, "duplicate_document", DocumentError),
    (InvalidDocumentError, "invalid_document", DocumentError),
    (ExtractionError, "extraction_failed", QBRDomainError),
    (MetadataValidationError, "metadata_validation_failed", QBRDomainError),
    (EvidenceValidationError, "evidence_validation_failed", MetadataValidationError),
    (InvalidStateTransitionError, "invalid_state_transition", QBRDomainError),
]


@pytest.mark.parametrize("exception_type,expected_code,parent_type", EXCEPTION_CASES)
def test_exception_contract(exception_type, expected_code, parent_type):
    error = exception_type("Operation failed")

    assert isinstance(error, parent_type)
    assert isinstance(error, QBRDomainError)
    assert isinstance(error, Exception)
    assert error.code == expected_code
    assert error.message == "Operation failed"
    assert str(error) == "Operation failed"
    assert error.args == ("Operation failed",)
    assert error.context == {}


@pytest.mark.parametrize("exception_type,_,__", EXCEPTION_CASES)
def test_exception_supports_safe_context(exception_type, _, __):
    context = {"document_id": "doc-123", "retryable": False}
    error = exception_type("Failed", context=context)

    assert error.context == context
    assert error.context is not context


@pytest.mark.parametrize("exception_type,_,__", EXCEPTION_CASES)
def test_exception_rejects_blank_message(exception_type, _, __):
    with pytest.raises(ValueError, match="Domain error message must not be empty"):
        exception_type(" \t\n ")


def test_context_is_copied_on_construction():
    context = {"document_id": "original"}
    error = DocumentNotFoundError("Not found", context=context)

    context["document_id"] = "changed"
    context["new_key"] = "new value"

    assert error.context == {"document_id": "original"}


def test_context_is_only_shallow_copied():
    nested = {"pages": [1]}
    error = ExtractionError("Failed", context={"details": nested})

    nested["pages"].append(2)
    assert error.context["details"]["pages"] == [1, 2]


def test_catching_document_family_by_parent():
    with pytest.raises(DocumentError) as caught:
        raise DuplicateDocumentError("Duplicate", context={"document_id": "doc-123"})

    assert caught.value.code == "duplicate_document"


def test_catching_evidence_error_as_metadata_validation_error():
    with pytest.raises(MetadataValidationError) as caught:
        raise EvidenceValidationError("Missing supporting excerpt")

    assert caught.value.code == "evidence_validation_failed"


def test_catching_any_domain_error():
    with pytest.raises(QBRDomainError) as caught:
        raise InvalidStateTransitionError("Cannot transition from completed to pending")

    assert caught.value.code == "invalid_state_transition"