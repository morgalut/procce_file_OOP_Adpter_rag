from dataclasses import dataclass
from uuid import uuid4
import pytest
from qbr.adapters.persistence.pgvector_store import _vector_literal, PgVectorStore
from qbr.application.prepare_and_vectorize import PrepareAndVectorize
from qbr.application.prepare_chunks import PrepareChunks
from qbr.application.vectorize_document import VectorizeDocument
from qbr.ports.chunker import ChunkSource, TextChunk


def test_vector_dimension_and_finiteness():
    assert _vector_literal([1, 0], 2) == '[1.0,0.0]'
    with pytest.raises(ValueError): _vector_literal([1], 2)
    with pytest.raises(ValueError): _vector_literal([float('nan'), 1], 2)
    with pytest.raises(ValueError): _vector_literal([True, 1], 2)


class FakeChunker:
    def chunk(self, sources):
        return [TextChunk(sources[0].document_id, 0, sources[0].text, 'summary', (1,), ('p1',), 3)]


class FakeEmbedder:
    def __init__(self): self.calls = 0
    async def embed(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self): self.calls = []
    async def save_vectors(self, document_id, vectors): self.calls.append((document_id, vectors))


@pytest.mark.asyncio
async def test_chunk_to_vector_persistence_preserves_provenance():
    doc = str(uuid4())
    embedder, store = FakeEmbedder(), FakeStore()
    use_case = PrepareAndVectorize(PrepareChunks(FakeChunker()), VectorizeDocument(embedder, 'test-model'), store)
    result = await use_case.execute(doc, [ChunkSource(doc, 1, 'Revenue increased', 'summary', ('p1',))])
    assert len(result) == 1
    assert result[0].source_ids == ('p1',)
    assert result[0].page_numbers == (1,)
    assert result[0].chunk_index == 0
    assert embedder.calls == 1
    assert len(store.calls) == 1
    assert result[0].chunk_id == (await use_case.execute(doc, [ChunkSource(doc, 1, 'Revenue increased')]))[0].chunk_id


@pytest.mark.asyncio
async def test_rejects_cross_document_sources():
    doc = str(uuid4())
    use_case = PrepareAndVectorize(PrepareChunks(FakeChunker()), VectorizeDocument(FakeEmbedder(), 'test-model'), FakeStore())
    with pytest.raises(ValueError, match='provenance mismatch'):
        await use_case.execute(doc, [ChunkSource(str(uuid4()), 1, 'wrong')])
