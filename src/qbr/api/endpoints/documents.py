"""Thin HTTP adapter: no persistence, extraction, or file system operations."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from qbr.api.dependencies import get_document_service
from qbr.api.contracts.documents import DocumentResult, UploadResult, ValidationDetails, as_document_result
from qbr.application.document_service import DocumentNotFound, DocumentService, InvalidDocumentId
from qbr.ports.document_storage import UploadRejected

router = APIRouter(prefix='/api/v1/qbr/documents', tags=['QBR documents'])
Service = Annotated[DocumentService, Depends(get_document_service)]


@router.post('', response_model=UploadResult, status_code=201)
async def upload_document(file: Annotated[UploadFile, File()], service: Service) -> UploadResult:
    async def chunks():
        while chunk := await file.read(1024 * 1024):
            yield chunk

    try:
        record = await service.upload(filename=file.filename, content_type=file.content_type, content=chunks())
    except UploadRejected as exc:
        code = {'too_large': 413, 'unsupported_media': 415, 'invalid_pdf': 422}.get(exc.code, 422)
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    finally:
        await file.close()
    return UploadResult.model_validate(
    as_document_result(record).model_dump(
        include={
            "document_id",
            "status",
            "incomplete",
            "page_count",
            "fragment_count",
        }
    )
)


async def _get_or_http_error(service: DocumentService, document_id: str):
    try:
        return await service.get(document_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail='Document not found') from exc
    except InvalidDocumentId as exc:
        raise HTTPException(status_code=422, detail='Invalid document UUID') from exc


@router.get('/{document_id}', response_model=DocumentResult)
async def get_document(document_id: str, service: Service) -> DocumentResult:
    return as_document_result(await _get_or_http_error(service, document_id))


@router.get('/{document_id}/validation', response_model=ValidationDetails)
async def get_validation(document_id: str, service: Service) -> ValidationDetails:
    record = await _get_or_http_error(service, document_id)
    return ValidationDetails(document_id=record.document_id, status=record.status,
                             incomplete=record.incomplete, validation=record.validation,
                             fragments=list(record.fragments), evidence=list(record.evidence))

def create_qbr_documents_router(*, service: DocumentService) -> APIRouter:
    """Explicit-service router for tests and embedding; same HTTP handlers."""
    isolated = APIRouter(prefix=router.prefix, tags=list(router.tags or []))

    @isolated.post('', response_model=UploadResult, status_code=201)
    async def upload(file: Annotated[UploadFile, File()]) -> UploadResult:
        return await upload_document(file=file, service=service)

    @isolated.get('/{document_id}', response_model=DocumentResult)
    async def read(document_id: str) -> DocumentResult:
        return await get_document(document_id=document_id, service=service)

    @isolated.get('/{document_id}/validation', response_model=ValidationDetails)
    async def validation(document_id: str) -> ValidationDetails:
        return await get_validation(document_id=document_id, service=service)

    return isolated
