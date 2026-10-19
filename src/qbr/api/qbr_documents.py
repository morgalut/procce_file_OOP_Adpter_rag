"""Backward-compatible import path (V16–V22)."""
from qbr.api.endpoints.documents import (router, create_qbr_documents_router,
    upload_document, get_document, get_validation)
__all__ = ['router','create_qbr_documents_router','upload_document','get_document','get_validation']
