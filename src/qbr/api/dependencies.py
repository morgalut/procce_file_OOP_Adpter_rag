"""Resolve application services created during FastAPI lifespan."""
from __future__ import annotations
from typing import Any
from fastapi import HTTPException, Request
from qbr.application.document_service import DocumentService

def _service(request: Request, attribute: str, label: str) -> Any:
    service = getattr(request.app.state, attribute, None)
    if service is None:
        raise HTTPException(status_code=503, detail=f"{label} is not configured")
    return service

def get_document_service(request: Request) -> DocumentService:
    return _service(request, 'document_service', 'Document processing')

def get_index_document(request: Request) -> Any:
    return _service(request, 'index_document', 'Document indexing')

def get_vector_search(request: Request) -> Any:
    return _service(request, 'vector_search', 'Vector search')
