"""Provider-independent full-document reviewer contract."""
from __future__ import annotations
from typing import Protocol
from qbr.domain.models import QBRMetadata
from qbr.domain.reconciliation_models import DocumentReview

class DocumentReviewer(Protocol):
    async def review(self, *, document_text: str, metadata: QBRMetadata) -> DocumentReview: ...
