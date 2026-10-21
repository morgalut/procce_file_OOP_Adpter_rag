"""Typed, audit-friendly document review proposals. No automatic metadata mutations."""
from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field

class ReviewAction(StrEnum):
    ADD_MISSING = "ADD_MISSING"
    CORRECT_VALUE = "CORRECT_VALUE"
    RECLASSIFY = "RECLASSIFY"
    FLAG_CONFLICT = "FLAG_CONFLICT"
    NO_CHANGE = "NO_CHANGE"

class CorrectionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: ReviewAction
    field_path: str = Field(min_length=1)
    original_value: str | None = None
    proposed_value: str | None = None
    reason: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    source_excerpt: str = Field(min_length=1)
    source_id: str | None = None

class DocumentReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposals: list[CorrectionProposal] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

class VerifiedProposal(BaseModel):
    proposal: CorrectionProposal
    evidence_verified: bool
    status: str
    rejection_reason: str | None = None
    semantic_status: str = "NOT_EVALUATED"

class ReviewResult(BaseModel):
    document_id: str
    page_count: int
    candidate_metadata: dict
    proposals: list[VerifiedProposal]
    unresolved_questions: list[str]
    # The original metadata is preserved. Applying changes requires a separate approval path.
    validated_metadata: dict
    review_id: str | None = None
    review_batches: int = 1
