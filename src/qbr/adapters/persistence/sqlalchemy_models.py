"""SQLAlchemy 2 persistence models. Domain objects remain independent of the ORM."""
from __future__ import annotations
from datetime import datetime
from uuid import uuid4
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class QBRDocumentRow(Base):
    __tablename__ = 'qbr_documents'
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    customer_name: Mapped[str | None] = mapped_column(Text)
    reporting_period: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column('metadata', JSONB)
    validation_json: Mapped[dict | None] = mapped_column('validation', JSONB)
    page_count: Mapped[int | None] = mapped_column(Integer)
    fragment_count: Mapped[int | None] = mapped_column(Integer)
    incomplete: Mapped[bool] = mapped_column(nullable=False, default=False)
    parser_version: Mapped[str | None] = mapped_column(String(64))
    preprocessor_version: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    schema_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    fragments: Mapped[list[QBRFragmentRow]] = relationship(back_populates='document', cascade='all, delete-orphan')
    evidence: Mapped[list[QBREvidenceRow]] = relationship(back_populates='document', cascade='all, delete-orphan')
    __table_args__ = (Index('ix_qbr_customer_period', 'customer_name', 'reporting_period'), Index('ix_qbr_status', 'status'), Index('ix_qbr_sha256', 'sha256'))

class QBRFragmentRow(Base):
    __tablename__ = 'qbr_fragment_outcomes'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey('qbr_documents.id', ondelete='CASCADE'), nullable=False)
    fragment_id: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    fragment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    validation_json: Mapped[dict | None] = mapped_column('validation', JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    failure_code: Mapped[str | None] = mapped_column(String(80))
    fallback_used: Mapped[bool] = mapped_column(nullable=False, default=False)
    document: Mapped[QBRDocumentRow] = relationship(back_populates='fragments')
    __table_args__ = (UniqueConstraint('document_id', 'fragment_id', name='uq_qbr_fragment'),)

class QBREvidenceRow(Base):
    __tablename__ = 'qbr_evidence'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey('qbr_documents.id', ondelete='CASCADE'), nullable=False)
    field_path: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_id: Mapped[str | None] = mapped_column(Text)
    source_excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    validation_status: Mapped[str] = mapped_column(String(40), nullable=False)
    document: Mapped[QBRDocumentRow] = relationship(back_populates='evidence')
    __table_args__ = (Index('ix_qbr_evidence_document_field', 'document_id', 'field_path'),)
