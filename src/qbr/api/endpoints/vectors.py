
"""Vector API routes with runtime and explicit dependency injection."""

from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from qbr.api.contracts.vectors import (
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from qbr.api.dependencies import (
    get_index_document,
    get_vector_search,
)
from qbr.application.document_service import (
    DocumentNotFound,
    InvalidDocumentId,
)


def create_vector_router(
    *,
    index_dependency: Callable[..., Any] = get_index_document,
    search_dependency: Callable[..., Any] = get_vector_search,
) -> APIRouter:
    """Create vector routes using injectable service dependencies."""

    router = APIRouter(tags=["QBR vectors"])

    @router.post(
        "/api/v1/qbr/documents/{document_id}/prepare-vectors"
    )
    async def prepare_vectors(
        document_id: UUID,
        index: Any = Depends(index_dependency),
    ):
        """Prepare and index a document's vector embeddings."""

        try:
            return await index.execute(str(document_id))

        except (DocumentNotFound, LookupError) as exc:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            ) from exc

        except (ValueError, InvalidDocumentId) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

    @router.post(
        "/api/v1/qbr/search",
        response_model=SearchResponse,
    )
    async def search_vectors(
        payload: SearchRequest,
        search: Any = Depends(search_dependency),
    ) -> SearchResponse:
        """Search indexed document vectors."""

        try:
            hits = await search.execute(
                payload.query,
                top_k=payload.top_k,
                document_id=(
                    str(payload.document_id)
                    if payload.document_id
                    else None
                ),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        return SearchResponse(
            results=[
                SearchHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    text=hit.text,
                    model=hit.model,
                    distance=hit.distance,
                    similarity=max(
                        0.0,
                        min(1.0, 1.0 - hit.distance),
                    ),
                    section=hit.section,
                    page_numbers=list(hit.page_numbers),
                    source_ids=list(hit.source_ids),
                )
                for hit in hits
            ]
        )

    return router


def create_qbr_vectors_router(
    *,
    index_document: Any,
    vector_search: Any,
) -> APIRouter:
    """Create vector routes with explicitly supplied services."""

    async def injected_index() -> Any:
        return index_document

    async def injected_search() -> Any:
        return vector_search

    return create_vector_router(
        index_dependency=injected_index,
        search_dependency=injected_search,
    )


def create_qbr_vectors_runtime_router() -> APIRouter:
    """Create vector routes using application runtime dependencies."""

    return create_vector_router()
