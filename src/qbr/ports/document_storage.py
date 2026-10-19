"""Storage contract: no FastAPI or provider-specific implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterable, Protocol

@dataclass(frozen=True)
class StoredPDF:
    uri: str
    sha256: str

class UploadRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

class DocumentStorage(Protocol):
    async def save_pdf(self, document_id: str, content: AsyncIterable[bytes], *, max_bytes: int) -> StoredPDF: ...
    async def delete(self, uri: str) -> None: ...
