"""Deterministic section/page-aware chunking without provider SDK dependencies.

A token counter may be injected for exact model-specific budgeting. The default
regex counter is only an estimate, NOT a tokenizer for any embedding model.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from qbr.ports.chunker import ChunkSource, TextChunk


_TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def approximate_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


class SectionChunker:
    def __init__(
        self,
        *,
        max_tokens: int = 700,
        overlap_tokens: int = 70,
        token_counter: Callable[[str], int] = approximate_tokens,
    ) -> None:
        if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError('require max_tokens > overlap_tokens >= 0')
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._count = token_counter

    def _split(self, text: str) -> list[str]:
        """Split on whitespace, retaining punctuation and source text exactly."""
        words = re.findall(r'\S+', text)
        if not words:
            return []
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            if self._count(word) > self.max_tokens:
                # A single huge unbroken word is not safe to embed; reject it.
                raise ValueError('source contains an unsplittable token exceeding max_tokens')
            candidate = ' '.join([*current, word])
            if current and self._count(candidate) > self.max_tokens:
                pieces.append(' '.join(current))
                overlap: list[str] = []
                if self.overlap_tokens:
                    for previous in reversed(current):
                        trial = [previous, *overlap]
                        if self._count(' '.join(trial)) > self.overlap_tokens:
                            break
                        overlap = trial
                # Never let overlap prevent forward progress.
                while overlap and self._count(' '.join([*overlap, word])) > self.max_tokens:
                    overlap.pop(0)
                current = [*overlap, word]
            else:
                current.append(word)
        if current:
            pieces.append(' '.join(current))
        return pieces

    def chunk(self, sources: list[ChunkSource]) -> list[TextChunk]:
        result: list[TextChunk] = []
        if sources and len({source.document_id for source in sources}) != 1:
            raise ValueError('cannot chunk sources from different documents')
        for source in sources:
            for text in self._split(source.text):
                result.append(TextChunk(
                    document_id=source.document_id,
                    chunk_index=len(result),
                    text=text,
                    section=source.section,
                    page_numbers=(source.page_number,),
                    source_ids=source.source_ids,
                    estimated_tokens=self._count(text),
                ))
        return result
