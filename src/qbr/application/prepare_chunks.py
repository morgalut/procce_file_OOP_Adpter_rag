## need build """Chunker adapter; no chunker is created until first use.""""""Prepare provenance-preserving chunks without invoking embedding providers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class Chunker(Protocol):
    def chunk(self, sources: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class PrepareChunks:
    chunker: Chunker

    def execute(self, sources: Iterable[Any], *, document_id: str) -> tuple[Any, ...]:
        if not document_id.strip():
            raise ValueError("document_id must not be blank")
        source_list = tuple(sources)
        if any(getattr(s, "document_id", None) != document_id for s in source_list):
            raise ValueError("chunk source document provenance mismatch")
        chunks = tuple(self.chunker.chunk(source_list))
        if any(getattr(c, "document_id", None) != document_id for c in chunks):
            raise ValueError("chunk result document provenance mismatch")
        return chunks
