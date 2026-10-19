
"""Async PostgreSQL pgvector persistence adapter.

Responsibilities:
    - Validate embeddings before persistence.
    - Atomically replace vectors belonging to one document.
    - Serialize concurrent replacements using a document row lock.
    - Perform cosine similarity search with optional filters.
    - Return validated, typed vector search results.

Requirements:
    - PostgreSQL with pgvector extension.
    - Alembic migration 0002.
    - SQLAlchemy AsyncSession with asyncpg.
    - Vector dimensions fixed at 1536 by migration 0002.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


VECTOR_DIMENSIONS = 1536
MAX_SEARCH_LIMIT = 100


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One vector similarity search result."""

    chunk_id: str
    document_id: str
    text: str
    model: str
    distance: float
    section: str | None
    page_numbers: tuple[int, ...]
    source_ids: tuple[str, ...]


def _normalize_document_id(document_id: str) -> str:
    """Validate and normalize a document UUID."""

    try:
        return str(UUID(str(document_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"invalid document_id: {document_id!r}"
        ) from exc


def _vector_literal(
    values: Any,
    dimensions: int,
) -> str:
    """Convert a numeric embedding to a safe pgvector literal."""

    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "embedding must be an iterable of numeric values"
        ) from exc

    if len(values) != dimensions:
        raise ValueError(
            f"expected {dimensions} embedding dimensions, "
            f"got {len(values)}"
        )

    normalized: list[float] = []

    for index, value in enumerate(values):

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"embedding[{index}] must be a real number"
            )

        number = float(value)

        if not isfinite(number):
            raise ValueError(
                f"embedding[{index}] must be finite"
            )

        normalized.append(number)

    return "[" + ",".join(map(str, normalized)) + "]"


class PgVectorStore:
    """PostgreSQL adapter for document embeddings."""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        *,
        dimensions: int = VECTOR_DIMENSIONS,
    ) -> None:

        if dimensions != VECTOR_DIMENSIONS:
            raise ValueError(
                "migration 0002 fixes vector dimensions at 1536"
            )

        self._sessions = session_factory
        self.dimensions = dimensions

    async def save_vectors(
        self,
        document_id: str,
        vectors: tuple[Any, ...],
    ) -> None:
        """Atomically replace all vectors for one document."""

        document_id = _normalize_document_id(document_id)

        rows: list[dict[str, Any]] = []

        seen_chunk_ids: set[str] = set()
        seen_chunk_indexes: set[int] = set()

        # Validate all vectors before opening a write transaction.
        for index, item in enumerate(vectors):

            item_document_id = _normalize_document_id(
                item.document_id
            )

            if item_document_id != document_id:
                raise ValueError(
                    "cross-document vector rejected"
                )

            chunk_id = item.chunk_id

            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
            ):
                raise ValueError(
                    "chunk_id must be a nonempty string"
                )

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"duplicate chunk_id: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

            if (
                not isinstance(item.text, str)
                or not item.text.strip()
            ):
                raise ValueError(
                    "vector text is required"
                )

            if (
                not isinstance(item.model, str)
                or not item.model.strip()
            ):
                raise ValueError(
                    "embedding model is required"
                )

            chunk_index = getattr(
                item,
                "chunk_index",
                index,
            )

            if (
                isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
            ):
                raise ValueError(
                    "chunk_index must be a nonnegative integer"
                )

            if chunk_index in seen_chunk_indexes:
                raise ValueError(
                    f"duplicate chunk_index: {chunk_index}"
                )

            seen_chunk_indexes.add(chunk_index)

            embedding = _vector_literal(
                item.embedding,
                self.dimensions,
            )

            page_numbers = list(
                getattr(item, "page_numbers", ())
            )

            source_ids = list(
                getattr(item, "source_ids", ())
            )

            if any(
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
                for page in page_numbers
            ):
                raise ValueError(
                    "page_numbers must contain positive integers"
                )

            if any(
                not isinstance(source_id, str)
                or not source_id.strip()
                for source_id in source_ids
            ):
                raise ValueError(
                    "source_ids must contain nonempty strings"
                )

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": item.text,
                    "model": item.model,
                    "embedding": embedding,
                    "section": getattr(
                        item,
                        "section",
                        None,
                    ),
                    "page_numbers": json.dumps(
                        page_numbers
                    ),
                    "source_ids": json.dumps(
                        source_ids
                    ),
                }
            )

        # Lock the parent document so concurrent replacements
        # for the same document cannot interleave.
        lock_statement = text(
            """
            SELECT id
            FROM qbr_documents
            WHERE id = CAST(:document_id AS uuid)
            FOR UPDATE
            """
        )

        delete_statement = text(
            """
            DELETE FROM qbr_chunk_embeddings
            WHERE document_id = CAST(:document_id AS uuid)
            """
        )

        insert_statement = text(
            """
            INSERT INTO qbr_chunk_embeddings (
                chunk_id,
                document_id,
                chunk_index,
                text,
                embedding_model,
                embedding,
                section,
                page_numbers,
                source_ids
            )
            VALUES (
                :chunk_id,
                CAST(:document_id AS uuid),
                :chunk_index,
                :text,
                :model,
                CAST(:embedding AS vector(1536)),
                :section,
                CAST(:page_numbers AS jsonb),
                CAST(:source_ids AS jsonb)
            )
            """
        )

        async with self._sessions() as session:

            async with session.begin():

                locked = await session.execute(
                    lock_statement,
                    {"document_id": document_id},
                )

                if locked.first() is None:
                    raise LookupError(
                        f"document not found: {document_id}"
                    )

                await session.execute(
                    delete_statement,
                    {"document_id": document_id},
                )

                if rows:
                    await session.execute(
                        insert_statement,
                        rows,
                    )

        # session.begin() commits on success and rolls back
        # automatically if an exception occurs.

    async def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        document_id: str | None = None,
        model: str | None = None,
    ) -> tuple[VectorHit, ...]:
        """Search document chunks by cosine distance.

        Args:
            query_embedding: Query embedding with 1536 dimensions.
            limit: Maximum number of hits, between 1 and 100.
            document_id: Optional document UUID filter.
            model: Optional embedding model filter.

        Returns:
            Tuple of VectorHit results ordered by cosine distance.
            Empty tuple when no matching vectors exist.
        """

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_SEARCH_LIMIT
        ):
            raise ValueError(
                "limit must be between 1 and 100"
            )

        if document_id is not None:
            document_id = _normalize_document_id(
                document_id
            )

        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "model must be a nonempty string"
                )

        vector = _vector_literal(
            query_embedding,
            self.dimensions,
        )

        # Explicitly cast nullable bind parameters.
        #
        # Without the casts, asyncpg cannot infer the type
        # of parameters used in expressions such as:
        #
        #   :document_id IS NULL
        #
        # This previously raised AmbiguousParameterError.

        statement = text(
            """
            SELECT
                chunk_id,
                document_id::text AS document_id,
                text,
                embedding_model,
                section,
                page_numbers,
                source_ids,
                (
                    embedding <=>
                    CAST(:embedding AS vector(1536))
                ) AS distance

            FROM qbr_chunk_embeddings

            WHERE
                (
                    CAST(:document_id AS uuid) IS NULL
                    OR document_id = CAST(:document_id AS uuid)
                )
                AND
                (
                    CAST(:model AS text) IS NULL
                    OR embedding_model = CAST(:model AS text)
                )

            ORDER BY
                embedding <=>
                CAST(:embedding AS vector(1536))

            LIMIT :limit
            """
        )

        parameters = {
            "embedding": vector,
            "document_id": document_id,
            "model": model,
            "limit": limit,
        }

        async with self._sessions() as session:

            result = await session.execute(
                statement,
                parameters,
            )

            hits: list[VectorHit] = []

            for row in result:

                distance = float(row.distance)

                if not isfinite(distance):
                    raise ValueError(
                        "database returned nonfinite vector distance"
                    )

                hits.append(
                    VectorHit(
                        chunk_id=row.chunk_id,
                        document_id=row.document_id,
                        text=row.text,
                        model=row.embedding_model,
                        distance=distance,
                        section=row.section,
                        page_numbers=tuple(
                            row.page_numbers or ()
                        ),
                        source_ids=tuple(
                            row.source_ids or ()
                        ),
                    )
                )

            return tuple(hits)
