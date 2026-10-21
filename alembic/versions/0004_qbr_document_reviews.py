"""Persistent review proposals and human decisions.

Revision ID: 0004_qbr_document_reviews
Revises: 0003_qbr_event_tracking
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
revision = "0004_qbr_document_reviews"
down_revision = "0003_qbr_event_tracking"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("qbr_document_reviews",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("qbr_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("review_data", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")))
    op.create_index("ix_qbr_document_reviews_document", "qbr_document_reviews", ["document_id"])
    op.create_table("qbr_review_decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("review_id", UUID(as_uuid=True), sa.ForeignKey("qbr_document_reviews.id", ondelete="CASCADE"), nullable=False),
        sa.Column("proposal_index", sa.Integer, nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("reviewer_id", sa.String(200), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("review_id", "proposal_index", name="uq_review_decision_proposal"))

def downgrade():
    op.drop_table("qbr_review_decisions")
    op.drop_index("ix_qbr_document_reviews_document", table_name="qbr_document_reviews")
    op.drop_table("qbr_document_reviews")
