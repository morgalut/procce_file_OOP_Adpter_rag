from types import SimpleNamespace
import pytest
from qbr.api.contracts.search_answer import AnswerDraft, Fact
from qbr.application.answer_search import AnswerSearch, validate_citations, hit_to_source

def fake_hit():
    return SimpleNamespace(chunk_id='c1', document_id='doc1', text='Q2 KPI: 95%', model='text-embedding-3-small', distance=0.2, section='KPIs', page_numbers=(2,), source_ids=('p2-f1',))

def test_citation_rejection():
    source = hit_to_source(fake_hit())
    draft = AnswerDraft(answer='a', summary='s', facts=[Fact(statement='claim', chunk_ids=['unknown'])], insufficient_evidence=False)
    with pytest.raises(ValueError, match='citations'):
        validate_citations(draft, [source])

@pytest.mark.asyncio
async def test_no_results_skips_llm():
    class Search:
        async def execute(self, *args, **kwargs): return ()
    result = await AnswerSearch(Search(), client=None).execute('What changed?')
    assert result.insufficient_evidence
    assert result.facts == []
    assert result.sources == []
