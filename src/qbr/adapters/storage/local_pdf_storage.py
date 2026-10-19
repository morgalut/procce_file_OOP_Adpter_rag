"""Local private PDF storage; validates bytes while streaming to a staged file."""
from __future__ import annotations
import asyncio
import hashlib
import os
from pathlib import Path
import tempfile
from typing import AsyncIterable
from qbr.ports.document_storage import StoredPDF, UploadRejected

class LocalPDFStorage:
    def __init__(self, directory: Path):
        self._directory = directory.resolve()

    async def save_pdf(self, document_id: str, content: AsyncIterable[bytes], *, max_bytes: int) -> StoredPDF:
        if max_bytes < 5:
            raise ValueError('max_bytes must be at least 5')
        await asyncio.to_thread(self._directory.mkdir, parents=True, exist_ok=True, mode=0o700)
        fd, tmp = await asyncio.to_thread(tempfile.mkstemp, prefix='.upload-', dir=str(self._directory))
        staged = Path(tmp)
        destination = self._directory / f'{document_id}.pdf'
        digest = hashlib.sha256()
        total = 0
        prefix = bytearray()
        try:
            with os.fdopen(fd, 'wb') as handle:
                async for chunk in content:
                    if not isinstance(chunk, bytes):
                        raise TypeError('Upload stream must yield bytes')
                    total += len(chunk)
                    if total > max_bytes:
                        raise UploadRejected('too_large', 'PDF exceeds configured size limit')
                    if len(prefix) < 5:
                        prefix.extend(chunk[:5-len(prefix)])
                        if len(prefix) == 5 and prefix != b'%PDF-':
                            raise UploadRejected('invalid_pdf', 'Invalid PDF signature')
                    digest.update(chunk)
                    await asyncio.to_thread(handle.write, chunk)
            if prefix != b'%PDF-':
                raise UploadRejected('invalid_pdf', 'Invalid PDF signature')
            await asyncio.to_thread(os.replace, staged, destination)
            return StoredPDF(uri=str(destination), sha256=digest.hexdigest())
        finally:
            await asyncio.to_thread(staged.unlink, missing_ok=True)

    async def delete(self, uri: str) -> None:
        target = Path(uri).resolve()
        if target.parent != self._directory:
            raise ValueError('Storage URI outside configured directory')
        await asyncio.to_thread(target.unlink, missing_ok=True)
