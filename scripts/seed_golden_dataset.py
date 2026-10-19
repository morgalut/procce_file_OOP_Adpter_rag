"""Transactional PostgreSQL adapter; replaces child snapshots atomically."""
from __future__ import annotations
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from qbr.ports.extraction_store import ExtractionRecord
from qbr.adapters.persistence.sqlalchemy_models import QBRDocumentRow, QBRFragmentRow, QBREvidenceRow

class PostgresExtractionStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, record: ExtractionRecord) -> None:
        document_id = str(UUID(record.document_id))
        if len(record.sha256) != 64 or any(c not in '0123456789abcdefABCDEF' for c in record.sha256):
            raise ValueError('sha256 must be a 64-character hexadecimal digest')
        async with self._sessions() as session:
            async with session.begin():
                row = await session.get(QBRDocumentRow, document_id, with_for_update=True)
                if row is None:
                    row = QBRDocumentRow(id=document_id, filename=record.filename, sha256=record.sha256, status=record.status)
                    session.add(row)
                    await session.flush()
                for name, value in {
                    'filename': record.filename, 'sha256': record.sha256, 'storage_uri': record.storage_uri,
                    'status': record.status, 'metadata_json': record.metadata, 'validation_json': record.validation,
                    'customer_name': (record.metadata or {}).get('customer_name'),
                    'reporting_period': (record.metadata or {}).get('reporting_period'),
                    'page_count': record.page_count, 'fragment_count': record.fragment_count, 'incomplete': record.incomplete,
                    'parser_version': record.parser_version, 'preprocessor_version': record.preprocessor_version,
                    'prompt_version': record.prompt_version, 'schema_version': record.schema_version,
                }.items():
                    setattr(row, name, value)
                # Flush removals before inserting replacements with the same unique keys.
                await session.refresh(row, ['fragments', 'evidence'])
                for child in (*row.fragments, *row.evidence):
                    await session.delete(child)
                await session.flush()
                for fragment in record.fragments:
                    session.add(QBRFragmentRow(document_id=document_id, **fragment))
                for evidence in record.evidence:
                    session.add(QBREvidenceRow(document_id=document_id, **evidence))

    async def get(self, document_id: str) -> ExtractionRecord | None:
        document_id = str(UUID(document_id))
        async with self._sessions() as session:
            row = await session.get(QBRDocumentRow, document_id)
            if row is None:
                return None
            fragments = (await session.scalars(select(QBRFragmentRow).where(QBRFragmentRow.document_id == document_id).order_by(QBRFragmentRow.id))).all()
            evidence = (await session.scalars(select(QBREvidenceRow).where(QBREvidenceRow.document_id == document_id).order_by(QBREvidenceRow.id))).all()
            return ExtractionRecord(
                document_id=row.id, filename=row.filename, sha256=row.sha256, status=row.status,
                metadata=row.metadata_json, validation=row.validation_json, page_count=row.page_count,
                fragment_count=row.fragment_count, incomplete=row.incomplete, storage_uri=row.storage_uri,
                parser_version=row.parser_version, preprocessor_version=row.preprocessor_version,
                prompt_version=row.prompt_version, schema_version=row.schema_version,
                fragments=tuple(dict(fragment_id=f.fragment_id, page_number=f.page_number, fragment_index=f.fragment_index,
                    warnings=f.warnings, validation_json=f.validation_json, error=f.error, failure_code=f.failure_code,
                    fallback_used=f.fallback_used) for f in fragments),
                evidence=tuple(dict(field_path=e.field_path, page_number=e.page_number, source_id=e.source_id,
                    source_excerpt=e.source_excerpt, validation_status=e.validation_status) for e in evidence),
            )
