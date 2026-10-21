"""Opt-in full-document review endpoint; no automatic metadata writes."""
from __future__ import annotations
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request
from qbr.api.dependencies import get_document_service
from qbr.application.document_service import DocumentService, DocumentNotFound, InvalidDocumentId
from qbr.domain.models import QBRMetadata
from qbr.domain.reconciliation_models import ReviewResult
from qbr.ports.pdf_extractor import PDFSource
from qbr.infrastructure.event_tracking import run_scope

def require_authenticated_reviewer(request: Request) -> str:
    """Fail closed until trusted auth middleware supplies a verified identity.

    Production deployments must ALSO enforce document-level authorization.
    """
    user_id = getattr(request.state, "authenticated_user_id", None)
    if not user_id:
        raise HTTPException(403, "Authenticated reviewer identity required")
    return str(user_id)

router = APIRouter(prefix="/api/v1/qbr/documents", tags=["QBR review"],
                   dependencies=[Depends(require_authenticated_reviewer)])

@router.post("/{document_id}/review", response_model=ReviewResult)
async def review_document(document_id: str, request: Request,
                          documents: Annotated[DocumentService, Depends(get_document_service)]) -> ReviewResult:
    service = getattr(request.app.state, "document_reviewer", None)
    if service is None:
        raise HTTPException(503, "Document review is not configured")
    try:
        record = await documents.get(document_id)
    except DocumentNotFound as exc:
        raise HTTPException(404, "Document not found") from exc
    except InvalidDocumentId as exc:
        raise HTTPException(422, "Invalid document ID") from exc
    if record.metadata is None or not record.storage_uri:
        raise HTTPException(409, "Document has no persisted metadata or PDF")
    storage_root = documents.storage._directory
    file_path = Path(record.storage_uri).resolve()
    if file_path.parent != storage_root or file_path.name != f"{record.document_id}.pdf":
        raise HTTPException(409, "Invalid document storage reference")
    if not file_path.is_file():
        raise HTTPException(409, "Original PDF is unavailable")
    metadata = QBRMetadata.model_validate(record.metadata)
    tracker = getattr(request.app.state, "pipeline_tracker", None)
    if tracker is None:
        result = await service.execute(PDFSource(record.document_id, file_path), metadata)
        repository = getattr(request.app.state, "review_repository", None)
        return await repository.save(result) if repository else result
    async with run_scope(tracker, "document_review", record.document_id):
        result = await service.execute(PDFSource(record.document_id, file_path), metadata)
        repository = getattr(request.app.state, "review_repository", None)
        return await repository.save(result) if repository else result

class ReviewDecisionRequest(BaseModel):
    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(min_length=3, max_length=2000)

@router.get("/reviews/{review_id}", response_model=ReviewResult)
async def get_review(review_id: UUID, request: Request):
    repo = getattr(request.app.state, "review_repository", None)
    if repo is None:
        raise HTTPException(503, "Review persistence unavailable")
    result = await repo.get(review_id)
    if result is None:
        raise HTTPException(404, "Review not found")
    return result

@router.post("/reviews/{review_id}/proposals/{proposal_index}/decision")
async def decide_review(
    review_id: UUID,
    proposal_index: int,
    body: ReviewDecisionRequest,
    request: Request,
):
    repo = getattr(request.app.state, "review_repository", None)

    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Review persistence unavailable",
        )

    # Identity must be verified by authentication middleware.
    reviewer_id = require_authenticated_reviewer(request)

    # Document-level authorization is required before making
    # a decision. The authenticated user must be authorized
    # to review this specific document.
    authorizer = getattr(request.app.state, "review_authorizer", None)

    if authorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Review authorization is not configured",
        )

    review = await repo.get(review_id)

    if review is None:
        raise HTTPException(
            status_code=404,
            detail="Review not found",
        )

    allowed = await authorizer.can_review(
        reviewer_id=reviewer_id,
        document_id=review.document_id,
    )

    if not allowed:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to review this document",
        )

    try:
        return await repo.decide(
            review_id=review_id,
            index=proposal_index,
            decision=body.decision,
            reviewer_id=reviewer_id,
            reason=body.reason,
        )

    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc

    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

    except Exception as exc:
        from sqlalchemy.exc import IntegrityError

        if isinstance(exc, IntegrityError):
            raise HTTPException(
                status_code=409,
                detail="Proposal has already been decided",
            ) from exc

        raise