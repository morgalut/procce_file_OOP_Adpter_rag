"""V22 vector composition. One embedder and one store shared by indexing/search."""
from __future__ import annotations

from typing import Any

from qbr.adapters.chunking.section_chunker import SectionChunker
from qbr.adapters.embedding.openai_embedder import OpenAIEmbedder
from qbr.adapters.persistence.pgvector_store import PgVectorStore
from qbr.application.index_document import IndexDocument
from qbr.application.prepare_and_vectorize import PrepareAndVectorize
from qbr.application.prepare_chunks import PrepareChunks
from qbr.application.vector_search import VectorSearch
from qbr.application.vector_source_loader import StoredPDFSourceLoader
from qbr.application.vectorize_document import VectorizeDocument


def build_vector_services(
    *,
    document_service: Any,
    session_factory: Any,
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: int = 1536,
    chunk_max_tokens: int = 700,
    chunk_overlap_tokens: int = 70,
    embedding_client: Any = None,
):
    """Return (IndexDocument, VectorSearch) using the existing DB session factory.

    Alembic 0002 uses vector(1536); other dimensions need a DB migration.
    """
    if embedding_dimensions != 1536:
        raise ValueError("Alembic migration 0002 requires 1536 embedding dimensions")
    if not embedding_model.strip():
        raise ValueError("embedding_model cannot be empty")
    if chunk_max_tokens < 1 or not 0 <= chunk_overlap_tokens < chunk_max_tokens:
        raise ValueError("chunk overlap must be nonnegative and less than chunk size")

    embedder = OpenAIEmbedder(
        client=embedding_client,
        model=embedding_model,
        dimensions=embedding_dimensions,
    )
    store = PgVectorStore(session_factory, dimensions=embedding_dimensions)
    chunker = SectionChunker(
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
    )
    pipeline = PrepareAndVectorize(
        prepare=PrepareChunks(chunker),
        vectorize=VectorizeDocument(embedder=embedder, model=embedding_model, store=None),
        store=store,
    )
    return (
        IndexDocument(loader=StoredPDFSourceLoader(document_service), pipeline=pipeline),
        VectorSearch(embedder=embedder, store=store, model=embedding_model),
    )


def build_vectors_router(*, document_service: Any, session_factory: Any, **kwargs: Any):
    """Backward-compatible helper for apps using explicit router injection.

    Do not call this in V27's main.py: V27 already registers runtime routes.
    """
    from qbr.api.qbr_vectors import create_qbr_vectors_router

    index, search = build_vector_services(
        document_service=document_service,
        session_factory=session_factory,
        **kwargs,
    )
    return create_qbr_vectors_router(index_document=index, vector_search=search)
