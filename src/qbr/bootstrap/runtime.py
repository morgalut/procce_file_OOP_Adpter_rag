"""FastAPI composition root for document processing and optional V22 vector services."""
from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import import_module
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from qbr.adapters.persistence.postgres_repository import PostgresExtractionStore
from qbr.adapters.storage.local_pdf_storage import LocalPDFStorage
from qbr.application.document_service import DocumentService
from qbr.application.persist_extraction import PersistExtraction
from qbr.infrastructure.database import make_database
from qbr.infrastructure.event_tracking import PostgresPipelineTracker, TrackedService

logger = logging.getLogger(__name__)


def load_pipeline(factory_path: str):
    """Resolve a configured extraction factory; do not instantiate LLMs on import."""
    if ":" not in factory_path:
        raise ValueError("QBR_PIPELINE_FACTORY must be module:function")
    module_name, function_name = factory_path.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("QBR_PIPELINE_FACTORY must be module:function")
    factory = getattr(import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("QBR_PIPELINE_FACTORY target must be callable")
    pipeline = factory()
    if not callable(getattr(pipeline, "execute", None)):
        raise TypeError("Pipeline factory must return an object with execute(source)")
    return pipeline


def _enabled(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"true", "1", "yes"}


@asynccontextmanager
async def api_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create services once and release DB connections on shutdown.

    Vector services are enabled by default. Set QBR_VECTOR_ENABLED=false to
    run only the document API. With vectors disabled, vector endpoints return
    HTTP 503 through the existing API dependencies.
    """
    database_url = os.getenv("DATABASE_URL")
    factory_path = os.getenv("QBR_PIPELINE_FACTORY")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if not factory_path:
        raise RuntimeError("QBR_PIPELINE_FACTORY is required: module:function")
    vector_enabled = _enabled("QBR_VECTOR_ENABLED")

    # Fail before creating a connection pool if extraction is misconfigured.
    pipeline = load_pipeline(factory_path)
    engine, sessions = make_database(
        database_url,
        echo=_enabled("DATABASE_ECHO", "false"),
        pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
    )
    app.state.document_service = None
    app.state.index_document = None
    app.state.vector_search = None
    app.state.document_reviewer = None
    app.state.review_repository = None

    try:
        tracker = PostgresPipelineTracker(sessions) if _enabled("QBR_EVENT_TRACKING", "true") else None
        app.state.pipeline_tracker = tracker
        from qbr.adapters.persistence.review_repository import PostgresReviewRepository
        app.state.review_repository = PostgresReviewRepository(sessions)
        store = PostgresExtractionStore(sessions)
        documents = DocumentService(
            pipeline=pipeline,
            persistence=PersistExtraction(store),
            store=store,
            storage=LocalPDFStorage(Path(os.getenv("QBR_UPLOAD_DIR", "data/uploads"))),
            max_pdf_bytes=int(os.getenv("PDF_MAX_SIZE_MB", "20")) * 1024 * 1024,
            tracker=tracker,
        )
        app.state.document_service = documents
        review_factory = os.getenv("QBR_DOCUMENT_REVIEW_FACTORY", "").strip()
        if review_factory:
            from qbr.bootstrap.document_review_wiring import build_document_review_service
            app.state.document_reviewer = build_document_review_service(
                review_factory,
                max_document_chars=int(os.getenv("QBR_REVIEW_MAX_DOCUMENT_CHARS", "250000")),
                batch_chars=int(os.getenv("QBR_REVIEW_BATCH_CHARS", "30000")),
                native_factory_path=os.getenv("QBR_NATIVE_PDF_REVIEW_FACTORY", "").strip(),
            )

        if vector_enabled:
            # Deferred import: document-only deployments do not need vector SDKs.
            from qbr.bootstrap.vector_wiring import build_vector_services
            index, search = build_vector_services(
                document_service=documents,
                session_factory=sessions,
                embedding_model=os.getenv("QBR_EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_dimensions=int(os.getenv("QBR_EMBEDDING_DIMENSIONS", "1536")),
                chunk_max_tokens=int(os.getenv("QBR_CHUNK_MAX_TOKENS", "700")),
                chunk_overlap_tokens=int(os.getenv("QBR_CHUNK_OVERLAP_TOKENS", "70")),
            )
            app.state.index_document = TrackedService(index, "vector_index", tracker, document_arg=True) if tracker else index
            app.state.vector_search = TrackedService(search, "vector_search", tracker) if tracker else search
            logger.info("QBR vector services initialized")
        else:
            logger.info("QBR vector services disabled (QBR_VECTOR_ENABLED=false)")

        yield
    finally:
        app.state.vector_search = None
        app.state.document_reviewer = None
        app.state.review_repository = None
        app.state.index_document = None
        app.state.document_service = None
        app.state.pipeline_tracker = None
        await engine.dispose()
