from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
import pytest
from qbr.application.persist_extraction import PersistExtraction
from qbr.ports.extraction_store import ExtractionRecord

class StubModel:
    def __init__(self, **data):
        self.data = data
    def model_dump(self, mode='json'):
        return self.data

@dataclass
class Evidence:
    page_number: int = 1
    source_text: str = '875'
    source_id: str | None = 'p1-t0-r1-c1'
    status: object = None
    def __post_init__(self):
        self.status = type('Status', (), {'value': 'verified'})()

class Metadata(StubModel):
    customer_evidence = []
    period_evidence = []
    summary_evidence = []
    goals = []
    achievements = []
    challenges = []
    risks = []
    next_steps = []
    kpis = []

class Store:
    def __init__(self): self.record = None
    async def save(self, record): self.record = record
    async def get(self, document_id): return self.record

@pytest.mark.asyncio
async def test_review_required_persists_failed_fragment_without_inventing_evidence():
    store = Store()
    metadata = Metadata(customer_name='Northwind Technologies', reporting_period='Q2 2026')
    metadata.kpis = [type('KPI', (), {'evidence': [Evidence()]})()]
    failure = type('Fragment', (), dict(fragment_id='page-000001-fragment-000002', page_number=1,
        fragment_index=2, warnings=('missing_or_invalid_evidence',), validation=None,
        error='missing source', failure_code='missing_or_invalid_evidence', fallback_used=True))()
    result = type('Result', (), dict(document_id=str(uuid4()), metadata=metadata,
        validation=StubModel(status='failed'), status=type('Status', (), {'value':'review_required'})(),
        page_count=1, fragment_count=3, incomplete=True, fragments=(failure,)))()
    record = await PersistExtraction(store).execute(result, filename='northwind.pdf', sha256='a'*64)
    assert record.status == 'review_required'
    assert record.incomplete
    assert record.fragments[0]['failure_code'] == 'missing_or_invalid_evidence'
    assert record.evidence[0]['source_id'] == 'p1-t0-r1-c1'
    assert store.record is record

@pytest.mark.asyncio
async def test_reject_non_uuid_document_id():
    store = Store()
    result = type('Result', (), {'document_id': 'northwind-q2-2026'})()
    with pytest.raises(ValueError):
        await PersistExtraction(store).execute(result, filename='x.pdf', sha256='a'*64)
    assert store.record is None
