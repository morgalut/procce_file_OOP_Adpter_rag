"""Add QBR chunk embeddings (1536-dimensional cosine search).

Revision ID: 0002_qbr_pgvector
Revises: 0001_qbr_extraction
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = '0002_qbr_pgvector'
down_revision = '0001_qbr_extraction'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.create_table(
        'qbr_chunk_embeddings',
        sa.Column('chunk_id', sa.Text(), primary_key=True),
        sa.Column('document_id', UUID(as_uuid=False), sa.ForeignKey('qbr_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding_model', sa.String(128), nullable=False),
        sa.Column('embedding', sa.Text(), nullable=False),  # converted below to vector(1536)
        sa.Column('section', sa.Text()),
        sa.Column('page_numbers', JSONB(), nullable=False),
        sa.Column('source_ids', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('document_id', 'chunk_index', name='uq_qbr_chunk_document_index'),
    )
    op.execute('ALTER TABLE qbr_chunk_embeddings ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)')
    op.create_index('ix_qbr_chunk_document', 'qbr_chunk_embeddings', ['document_id'])
    op.execute('CREATE INDEX ix_qbr_chunk_embedding_hnsw ON qbr_chunk_embeddings USING hnsw (embedding vector_cosine_ops)')


def downgrade():
    op.execute('DROP INDEX IF EXISTS ix_qbr_chunk_embedding_hnsw')
    op.drop_index('ix_qbr_chunk_document', table_name='qbr_chunk_embeddings')
    op.drop_table('qbr_chunk_embeddings')
    # Do not drop extension: other tables may depend on it.
