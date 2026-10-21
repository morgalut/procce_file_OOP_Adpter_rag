"""Explicit opt-in composition for full-document review."""
from __future__ import annotations
from importlib import import_module
from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter
from qbr.application.reconcile_metadata import ReconcileMetadata

def build_document_review_service(factory_path: str, *, max_document_chars: int = 250_000, batch_chars: int = 30_000, native_factory_path: str = ""):
    if ':' not in factory_path:
        raise ValueError('QBR_DOCUMENT_REVIEW_FACTORY must be module:function')
    module, function = factory_path.split(':', 1)
    reviewer = getattr(import_module(module), function)()
    if not callable(getattr(reviewer, 'review', None)):
        raise TypeError('Document review factory must return an object with async review()')
    native_reviewer = None
    if native_factory_path:
        if ':' not in native_factory_path:
            raise ValueError('QBR_NATIVE_PDF_REVIEW_FACTORY must be module:function')
        native_module, native_function = native_factory_path.split(':', 1)
        native_reviewer = getattr(import_module(native_module), native_function)()
        if not callable(getattr(native_reviewer, 'review_pdf', None)):
            raise TypeError('Native reviewer must implement review_pdf()')
    return ReconcileMetadata(PyMuPDFAdapter(), reviewer, max_document_chars=max_document_chars,
                             batch_chars=batch_chars, native_reviewer=native_reviewer)
