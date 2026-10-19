import pytest
from qbr.ports.chunker import ChunkSource
from qbr.adapters.chunking.section_chunker import SectionChunker


def src(text, page=1, section='KPI'):
    return ChunkSource('doc-1', page, text, section, ('source-1',))


def test_empty():
    assert SectionChunker().chunk([]) == []
    assert SectionChunker().chunk([src('  ')]) == []


def test_provenance_and_section_boundaries():
    chunks = SectionChunker(max_tokens=5, overlap_tokens=0).chunk([
        src('Monthly Active Users 640 875', page=1),
        src('Risks and next steps', page=2, section='Risks'),
    ])
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.page_numbers for c in chunks] == [(1,), (2,)]
    assert chunks[0].source_ids == ('source-1',)
    assert chunks[1].section == 'Risks'


def test_budget_and_overlap():
    chunks = SectionChunker(max_tokens=5, overlap_tokens=2).chunk([src('one two three four five six seven eight nine')])
    assert len(chunks) > 1
    assert all(c.estimated_tokens <= 5 for c in chunks)
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert ' '.join(chunks[0].text.split()) == 'one two three four five'


def test_mixed_documents_rejected():
    with pytest.raises(ValueError, match='different documents'):
        SectionChunker().chunk([src('a'), ChunkSource('doc-2', 1, 'b')])


def test_invalid_config():
    with pytest.raises(ValueError):
        SectionChunker(max_tokens=5, overlap_tokens=5)


def test_oversize_unsplittable():
    with pytest.raises(ValueError, match='unsplittable'):
        SectionChunker(max_tokens=2, overlap_tokens=0).chunk([src('abc!!!')])
