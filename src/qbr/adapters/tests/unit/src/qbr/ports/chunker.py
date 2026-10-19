"""Provider-independent contracts for provenance-preserving text chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChunkSource:
    document_id: str
    page_number: int
    text: str
    section: str | None = None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError('document_id is required')
        if self.page_number < 1:
            raise ValueError('page_number must be positive')


@dataclass(frozen=True, slots=True)
class TextChunk:
    document_id: str
    chunk_index: int
    text: str
    section: str | None
    page_numbers: tuple[int, ...]
    source_ids: tuple[str, ...]
    estimated_tokens: int


class Chunker(Protocol):
    def chunk(self, sources: list[ChunkSource]) -> list[TextChunk]: ...
