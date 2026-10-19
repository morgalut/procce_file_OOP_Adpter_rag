"""Integration of DocumentService and ProcessDocument without real LLM or DB."""
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
import pytest
from qbr.application.document_service import DocumentService
from qbr.application.process_document import ProcessDocument, DocumentProvenanceError

class Storage:
    def __init__(self, path):
        self.path = path
        self.deleted = []
    async def save_pdf(self, document_id, content, *, max_bytes):
        payload = b''.join([chunk async for chunk in content])
        self.path.mkdir(exist_ok=True, parents=True)
        path = self.path / f'{document_id}.pdf'
        path.write_bytes(payload)
        return SimpleNamespace(uri=str(path), sha256='abc123')
    async def delete(self, uri):
        self.deleted.append(uri)
        Path(uri).unlink()

class Extractor:
    def __init__(self, *, mismatch=False, failure=False):
        self.calls = 0
        self.mismatch = mismatch
        self.failure = failure
    async def execute(self, source):
        self.calls += 1
        assert source.path.read_bytes().startswith(b'%PDF-')
        if self.failure:
            raise RuntimeError('extractor failure')
        return SimpleNamespace(document_id='different' if self.mismatch else source.document_id)

class Persister:
    def __init__(self):
        self.calls = 0
    async def execute(self, result, **kwargs):
        self.calls += 1
        return SimpleNamespace(document_id=result.document_id, **kwargs)

class Store:
    async def get(self, document_id):
        return None

async def chunks():
    yield b'%PDF-1.7\nhello'

@pytest.mark.asyncio
async def test_process_document_called_once(tmp_path):
    extractor, persister = Extractor(), Persister()
    storage = Storage(tmp_path/'storage')
    processor = ProcessDocument(extractor, persister)
    service = DocumentService(pipeline=extractor, persistence=persister, store=Store(), storage=storage, processor=processor)
    record = await service.upload(filename='test.pdf', content_type='application/pdf', content=chunks())
    UUID(record.document_id)
    assert extractor.calls == 1
    assert persister.calls == 1
    assert Path(record.storage_uri).exists()

@pytest.mark.asyncio
async def test_legacy_wiring_still_uses_processor_once(tmp_path):
    extractor, persister = Extractor(), Persister()
    service = DocumentService(pipeline=extractor, persistence=persister, store=Store(), storage=Storage(tmp_path/'storage'))
    await service.upload(filename='test.pdf', content_type='application/pdf', content=chunks())
    assert (extractor.calls, persister.calls) == (1, 1)

@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch,failure,exception', [(True,False,DocumentProvenanceError),(False,True,RuntimeError)])
async def test_failed_processing_removes_pdf(tmp_path, mismatch, failure, exception):
    extractor, persister = Extractor(mismatch=mismatch, failure=failure), Persister()
    storage = Storage(tmp_path/'storage')
    service = DocumentService(pipeline=extractor, persistence=persister, store=Store(), storage=storage)
    with pytest.raises(exception):
        await service.upload(filename='test.pdf', content_type='application/pdf', content=chunks())
    assert persister.calls == 0
    assert len(storage.deleted) == 1
    assert not list((tmp_path/'storage').iterdir())
