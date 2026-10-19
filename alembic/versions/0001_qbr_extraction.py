"""Initial QBR extraction tables.

Revision ID: 0001_qbr_extraction
Revises:
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
revision = '0001_qbr_extraction'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('qbr_documents',
        sa.Column('id', UUID(as_uuid=False), primary_key=True),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('sha256', sa.String(64), nullable=False),
        sa.Column('storage_uri', sa.Text()),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('customer_name', sa.Text()),
        sa.Column('reporting_period', sa.Text()),
        sa.Column('metadata', JSONB()),
        sa.Column('validation', JSONB()),
        sa.Column('page_count', sa.Integer()),
        sa.Column('fragment_count', sa.Integer()),
        sa.Column('incomplete', sa.Boolean(), nullable=False),
        sa.Column('parser_version', sa.String(64)),
        sa.Column('preprocessor_version', sa.String(64)),
        sa.Column('prompt_version', sa.String(64)),
        sa.Column('schema_version', sa.String(64)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_qbr_customer_period', 'qbr_documents', ['customer_name', 'reporting_period'])
    op.create_index('ix_qbr_status', 'qbr_documents', ['status'])
    op.create_index('ix_qbr_sha256', 'qbr_documents', ['sha256'])
    op.create_table('qbr_fragment_outcomes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('document_id', UUID(as_uuid=False), sa.ForeignKey('qbr_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('fragment_id', sa.Text(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('fragment_index', sa.Integer(), nullable=False),
        sa.Column('warnings', JSONB(), nullable=False),
        sa.Column('validation', JSONB()),
        sa.Column('error', sa.Text()),
        sa.Column('failure_code', sa.String(80)),
        sa.Column('fallback_used', sa.Boolean(), nullable=False),
        sa.UniqueConstraint('document_id', 'fragment_id', name='uq_qbr_fragment'),
    )
    op.create_table('qbr_evidence',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('document_id', UUID(as_uuid=False), sa.ForeignKey('qbr_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('field_path', sa.Text(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Text()),
        sa.Column('source_excerpt', sa.Text(), nullable=False),
        sa.Column('validation_status', sa.String(40), nullable=False),
    )
    op.create_index('ix_qbr_evidence_document_field', 'qbr_evidence', ['document_id', 'field_path'])

def downgrade():
    op.drop_index('ix_qbr_evidence_document_field', table_name='qbr_evidence')
    op.drop_table('qbr_evidence')
    op.drop_table('qbr_fragment_outcomes')
    op.drop_index('ix_qbr_sha256', table_name='qbr_documents')
    op.drop_index('ix_qbr_status', table_name='qbr_documents')
    op.drop_index('ix_qbr_customer_period', table_name='qbr_documents')
    op.drop_table('qbr_documents')
