"""Persistent pipeline runs, stage attempts, and append-only events.

Revision ID: 0003_qbr_event_tracking
Revises: 0002_qbr_pgvector
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
revision = '0003_qbr_event_tracking'
down_revision = '0002_qbr_pgvector'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('qbr_pipeline_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', sa.Text(), nullable=True),
        sa.Column('pipeline_name', sa.Text(), nullable=False),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True)))
    op.create_index('ix_qbr_pipeline_runs_document', 'qbr_pipeline_runs', ['document_id'])
    op.create_table('qbr_pipeline_stages',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('run_id', UUID(as_uuid=True), sa.ForeignKey('qbr_pipeline_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('parent_stage_id', UUID(as_uuid=True), sa.ForeignKey('qbr_pipeline_stages.id'), nullable=True),
        sa.Column('stage_name', sa.Text(), nullable=False),
        sa.Column('fragment_id', sa.Text()),
        sa.Column('attempt_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True)),
        sa.Column('duration_ms', sa.Float()),
        sa.Column('input_summary', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('output_summary', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('error_code', sa.String(100)))
    op.create_index('ix_qbr_pipeline_stages_run', 'qbr_pipeline_stages', ['run_id', 'started_at'])
    op.create_table('qbr_pipeline_events',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('run_id', UUID(as_uuid=True), sa.ForeignKey('qbr_pipeline_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('stage_id', UUID(as_uuid=True), sa.ForeignKey('qbr_pipeline_stages.id', ondelete='SET NULL')),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_data', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index('ix_qbr_pipeline_events_run', 'qbr_pipeline_events', ['run_id', 'id'])

def downgrade():
    op.drop_table('qbr_pipeline_events')
    op.drop_table('qbr_pipeline_stages')
    op.drop_table('qbr_pipeline_runs')
