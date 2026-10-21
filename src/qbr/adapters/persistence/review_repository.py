"""Postgres-backed immutable review snapshots and auditable human decisions."""
from __future__ import annotations
from uuid import UUID, uuid4
import json
from sqlalchemy import text
from qbr.domain.reconciliation_models import ReviewResult
from qbr.domain.models import QBRMetadata
from qbr.domain.reconciliation_models import ReviewAction

class PostgresReviewRepository:
    def __init__(self, sessions):
        self.sessions = sessions

    async def save(self, result: ReviewResult) -> ReviewResult:
        review_id = uuid4()
        snapshot = result.model_copy(update={"review_id": str(review_id)})
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(text("""INSERT INTO qbr_document_reviews
                    (id, document_id, status, review_data) VALUES
                    (:id, :document_id, 'PENDING', CAST(:payload AS jsonb))"""),
                    {"id": review_id, "document_id": UUID(result.document_id),
                     "payload": json.dumps(snapshot.model_dump(mode="json"))})
        return snapshot

    async def get(self, review_id: UUID) -> ReviewResult | None:
        async with self.sessions() as session:
            data = (await session.execute(text(
                "SELECT review_data FROM qbr_document_reviews WHERE id=:id"), {"id": review_id})).scalar_one_or_none()
        return ReviewResult.model_validate(data) if data is not None else None

    async def decide(
        self,
        review_id: UUID,
        index: int,
        decision: str,
        reviewer_id: str,
        reason: str,
    ):
        if decision not in {"APPROVE", "REJECT"}:
            raise ValueError("Invalid decision")

        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("Reviewer identity required")

        if not reason or not reason.strip():
            raise ValueError("Decision reason required")

        if len(reason) > 2000:
            raise ValueError("Decision reason too long")

        allowed_fields = {
            "customer_name",
            "reporting_period",
            "executive_summary",
        }

        metadata_changed = False

        async with self.sessions() as session:
            async with session.begin():

                # Lock the review to prevent concurrent decisions.
                review_row = (
                    await session.execute(
                        text("""
                            SELECT document_id, review_data
                            FROM qbr_document_reviews
                            WHERE id = :review_id
                            FOR UPDATE
                        """),
                        {"review_id": review_id},
                    )
                ).mappings().one_or_none()

                if review_row is None:
                    raise LookupError("Review not found")

                result = ReviewResult.model_validate(
                    review_row["review_data"]
                )

                if not 0 <= index < len(result.proposals):
                    raise IndexError("Proposal index out of range")

                # Prevent deciding the same proposal twice.
                existing = (
                    await session.execute(
                        text("""
                            SELECT id
                            FROM qbr_review_decisions
                            WHERE review_id = :review_id
                            AND proposal_index = :proposal_index
                        """),
                        {
                            "review_id": review_id,
                            "proposal_index": index,
                        },
                    )
                ).scalar_one_or_none()

                if existing is not None:
                    raise ValueError("Proposal already decided")

                verified = result.proposals[index]
                proposal = verified.proposal

                if decision == "APPROVE":

                    if verified.status != "PENDING_APPROVAL":
                        raise ValueError(
                            "Proposal is not pending approval"
                        )

                    if not verified.evidence_verified:
                        raise ValueError(
                            "Proposal evidence is not verified"
                        )

                    if proposal.action not in {
                        ReviewAction.ADD_MISSING,
                        ReviewAction.CORRECT_VALUE,
                    }:
                        raise ValueError(
                            "This proposal action cannot be "
                            "automatically applied"
                        )

                    if proposal.field_path not in allowed_fields:
                        raise ValueError(
                            "Unsupported metadata field"
                        )

                    if not proposal.proposed_value:
                        raise ValueError(
                            "Proposed value is missing"
                        )

                    # Lock the document during metadata modification.
                    document_row = (
                        await session.execute(
                            text("""
                                SELECT metadata
                                FROM qbr_documents
                                WHERE id = :document_id
                                FOR UPDATE
                            """),
                            {
                                "document_id": review_row["document_id"]
                            },
                        )
                    ).mappings().one_or_none()

                    if document_row is None:
                        raise LookupError("Document not found")

                    metadata = QBRMetadata.model_validate(
                        document_row["metadata"]
                    )

                    current_value = getattr(
                        metadata,
                        proposal.field_path,
                    )

                    # Avoid applying a proposal generated from
                    # an outdated document version.
                    if current_value != proposal.original_value:
                        raise ValueError(
                            "Metadata has changed since review generation"
                        )

                    if (
                        proposal.action == ReviewAction.ADD_MISSING
                        and current_value is not None
                    ):
                        raise ValueError(
                            "Field is no longer missing"
                        )

                    updated_metadata = metadata.model_copy(
                        update={
                            proposal.field_path:
                                proposal.proposed_value
                        }
                    )

                    # model_copy does not validate updates.
                    updated_metadata = QBRMetadata.model_validate(
                        updated_metadata.model_dump()
                    )

                    await session.execute(
                        text("""
                            UPDATE qbr_documents
                            SET metadata = CAST(:metadata AS jsonb),
                                updated_at = now()
                            WHERE id = :document_id
                        """),
                        {
                            "document_id": review_row["document_id"],
                            "metadata": json.dumps(
                                updated_metadata.model_dump(
                                    mode="json"
                                )
                            ),
                        },
                    )

                    metadata_changed = True

                # Save the reviewer decision in the same transaction.
                await session.execute(
                    text("""
                        INSERT INTO qbr_review_decisions (
                            id,
                            review_id,
                            proposal_index,
                            decision,
                            reviewer_id,
                            reason
                        )
                        VALUES (
                            :id,
                            :review_id,
                            :proposal_index,
                            :decision,
                            :reviewer_id,
                            :reason
                        )
                    """),
                    {
                        "id": uuid4(),
                        "review_id": review_id,
                        "proposal_index": index,
                        "decision": decision,
                        "reviewer_id": reviewer_id,
                        "reason": reason,
                    },
                )

        return {
            "review_id": str(review_id),
            "proposal_index": index,
            "decision": decision,
            "metadata_changed": metadata_changed,
        }
