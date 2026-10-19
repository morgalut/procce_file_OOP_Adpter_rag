"""Translate V9 extraction results into the persistence port contract."""
from __future__ import annotations
from uuid import UUID
from qbr.ports.extraction_store import ExtractionRecord, ExtractionStore

class PersistExtraction:
    def __init__(self, store: ExtractionStore) -> None:
        self._store = store

    async def execute(self, result, *, filename: str, sha256: str, storage_uri: str | None = None,
                      parser_version: str | None = None, preprocessor_version: str | None = None,
                      prompt_version: str | None = None, schema_version: str | None = None) -> ExtractionRecord:
        # V9 document IDs may be human-readable; persistence requires a stable UUID.
        document_id = str(UUID(result.document_id))
        metadata = result.metadata.model_dump(mode='json')
        validation = result.validation.model_dump(mode='json')
        evidence: list[dict] = []
        for field in ('customer_evidence', 'period_evidence', 'summary_evidence'):
            for item in getattr(result.metadata, field):
                evidence.append(self._evidence(field, item))
        for field in ('goals', 'kpis', 'achievements', 'challenges', 'risks', 'next_steps'):
            for index, item in enumerate(getattr(result.metadata, field)):
                for entry in item.evidence:
                    evidence.append(self._evidence(f'{field}[{index}]', entry))
        record = ExtractionRecord(
            document_id=document_id, filename=filename, sha256=sha256, storage_uri=storage_uri,
            status=result.status.value, metadata=metadata, validation=validation,
            page_count=result.page_count, fragment_count=result.fragment_count, incomplete=result.incomplete,
            parser_version=parser_version, preprocessor_version=preprocessor_version,
            prompt_version=prompt_version, schema_version=schema_version,
            fragments=tuple(dict(fragment_id=f.fragment_id, page_number=f.page_number,
                fragment_index=f.fragment_index, warnings=list(f.warnings),
                validation_json=f.validation.model_dump(mode='json') if f.validation else None,
                error=f.error, failure_code=f.failure_code, fallback_used=f.fallback_used) for f in result.fragments),
            evidence=tuple(evidence),
        )
        await self._store.save(record)
        return record

    @staticmethod
    def _evidence(path, entry):
        return dict(field_path=path, page_number=entry.page_number,
                    source_id=getattr(entry, 'source_id', None), source_excerpt=entry.source_text,
                    validation_status=entry.status.value)
