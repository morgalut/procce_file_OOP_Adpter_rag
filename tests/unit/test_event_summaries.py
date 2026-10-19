"""Regression tests for redacted event summaries and nested vector stages."""
from __future__ import annotations
from uuid import uuid4
import pytest
from qbr.infrastructure.event_tracking import TrackedService, _safe, run_scope, stage
from qbr.application.vector_search import VectorSearch

class MemoryTracker:
    def __init__(self):
        self.runs=[];self.stages=[];self.events=[];self.completed={}
    async def start_run(self, name, document_id=None):
        rid=uuid4();self.runs.append((rid,name,document_id));return rid
    async def finish_run(self, rid, status):
        self.completed[rid]=status
    async def start_stage(self, rid, name, **kwargs):
        sid=uuid4();self.stages.append((sid,rid,name,kwargs));return sid
    async def complete_stage(self, sid, output_summary=None):
        self.completed[sid]=output_summary
    async def fail_stage(self, sid, error_code, error_message):
        self.completed[sid]=error_code
    async def record_event(self, rid, sid, kind, data):
        self.events.append((rid,sid,kind,data))

class Embedder:
    async def embed(self, texts):
        return [[0.1,0.2] for _ in texts]

class Store:
    async def search(self, vector, *, limit, document_id, model):
        return [object(),object()]

@pytest.mark.asyncio
async def test_search_captures_safe_summary_and_nested_stages():
    tracker=MemoryTracker()
    service=TrackedService(VectorSearch(Embedder(),Store(),'test-model'),'vector_search',tracker)
    results=await service.execute('private question',top_k=3)
    assert len(results)==2
    names=[entry[2] for entry in tracker.stages]
    assert names==['vector_search','embedding.query','vector_database.search']
    root=tracker.stages[0]
    assert root[3]['input_summary']=={'query_length':16,'top_k':3,'document_filter':None}
    assert tracker.completed[root[0]]=={'result_count':2,'model':'test-model'}
    assert tracker.stages[1][3]['parent_stage_id']==root[0]
    assert 'private question' not in str(tracker.events)

@pytest.mark.asyncio
async def test_stage_output_and_error_category():
    tracker=MemoryTracker()
    async with run_scope(tracker,'test'):
        async with stage('success',input_summary={'count':1}) as details:
            details.finish(result_count=3)
        with pytest.raises(ValueError):
            async with stage('failure'):
                raise ValueError('private payload')
    assert any(v=={'result_count':3} for v in tracker.completed.values())
    assert any(v=='ValueError' for v in tracker.completed.values())
    assert 'private payload' not in str(tracker.events)

def test_redaction():
    assert _safe({'text':'secret','prompt':'secret','query_length':12})=={'query_length':12}
