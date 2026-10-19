from dataclasses import dataclass
from uuid import uuid4
import httpx
import pytest
from fastapi import FastAPI
from qbr.api.qbr_vectors import create_qbr_vectors_router
from qbr.application.vector_search import VectorSearch
from qbr.application.index_document import IndexDocument


class FakeEmbedder:
    def __init__(self): self.calls = []
    async def embed(self, texts):
        self.calls.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    text: str = 'QBR risks'
    model: str = 'fake-model'
    distance: float = 0.25
    section: str | None = None
    page_numbers: tuple = (2,)
    source_ids: tuple = ('pdf-page-2',)


class FakeStore:
    def __init__(self, doc): self.doc = doc; self.calls = []
    async def search(self, vector, **kwargs):
        self.calls.append(kwargs)
        return [Hit('chunk-1', self.doc)]


class FakeLoader:
    def __init__(self): self.calls = []
    async def load(self, document_id):
        self.calls.append(document_id)
        return ['source']


class FakePipeline:
    def __init__(self): self.calls = []
    async def execute(self, document_id, sources):
        self.calls.append((document_id, sources))
        return [type('V', (), {'model': 'fake-model', 'embedding': (0.1, 0.2, 0.3)})()]


@pytest.mark.asyncio
async def test_index_and_search_http():
    doc = str(uuid4())
    loader, pipeline, embedder = FakeLoader(), FakePipeline(), FakeEmbedder()
    store = FakeStore(doc)
    app = FastAPI()
    app.include_router(create_qbr_vectors_router(
        index_document=IndexDocument(loader, pipeline),
        vector_search=VectorSearch(embedder, store, 'fake-model')))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(f'/api/v1/qbr/documents/{doc}/prepare-vectors')
        assert response.status_code == 200, response.text
        assert response.json()['chunks_created'] == 1
        response = await client.post('/api/v1/qbr/search', json={'query': 'risks', 'document_id': doc})
        assert response.status_code == 200, response.text
        hit = response.json()['results'][0]
        assert hit['page_numbers'] == [2] and hit['similarity'] == 0.75
        assert store.calls[0]['model'] == 'fake-model'
        assert loader.calls == [doc] and pipeline.calls == [(doc, ['source'])]


@pytest.mark.asyncio
async def test_invalid_inputs():
    app = FastAPI()
    app.include_router(create_qbr_vectors_router(
        index_document=IndexDocument(FakeLoader(), FakePipeline()),
        vector_search=VectorSearch(FakeEmbedder(), FakeStore(str(uuid4())), 'fake-model')))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.post('/api/v1/qbr/documents/not-a-uuid/prepare-vectors')).status_code == 422
        assert (await client.post('/api/v1/qbr/search', json={'query': ' ', 'top_k': 5})).status_code == 422
        assert (await client.post('/api/v1/qbr/search', json={'query': 'valid', 'top_k': 101})).status_code == 422
