"""LLM V5: enforce table relationships without changing rows contract."""
import json
from unittest.mock import AsyncMock
import pytest
from qbr.adapters.llm.instructor_adapter import InstructorMetadataAdapter, LLMExtractionSettings, EvidenceVerificationError
from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment, CleanTable
from qbr.domain.models import QBRMetadata, KPIMetric, ActionItem, Evidence


def fragment(rows, header=(), section='performance'):
    t = CleanTable(page_number=1, table_index=0, rows=tuple(tuple(r) for r in rows),
        original_rows=tuple(tuple(r) for r in rows), table_id='p1-t0',
        header_rows=(tuple(header),) if header else ())
    return CleanDocumentFragment(document_id='doc', page_number=1, fragment_index=0,
        text='', tables=(t,), section=section, content_type='table')


def adapter(metadata):
    client = type('Client', (), {})()
    client.create = AsyncMock(return_value=metadata)
    return InstructorMetadataAdapter(client, LLMExtractionSettings(model='test-model')), client


@pytest.mark.asyncio
async def test_kpi_valid_same_row_and_payload_stays_2d():
    f = fragment([('Monthly Active Users', '640', '875')], ('Metric', 'Q1', 'Q2'))
    m = QBRMetadata(kpis=[KPIMetric(name='Monthly Active Users', previous_value='640', current_value='875',
         evidence=[Evidence(page_number=1, source_text='Monthly Active Users')])])
    a, client = adapter(m)
    result = await a.generate_clean_fragment(f)
    assert result.metadata.kpis[0].current_value == '875'
    payload = json.loads(client.create.await_args.kwargs['messages'][1]['content'])
    assert payload['tables'][0]['rows'] == [['Monthly Active Users', '640', '875']]
    assert payload['tables'][0]['row_refs'][0]['row_id'] == 'p1-t0-r0'


@pytest.mark.asyncio
async def test_kpi_rejects_cross_row_value():
    f = fragment([('Users', '640', '875'), ('Adoption', '61%', '74%')], ('Metric', 'Q1', 'Q2'))
    m = QBRMetadata(kpis=[KPIMetric(name='Users', previous_value='640', current_value='74%',
        evidence=[Evidence(page_number=1, source_text='Users')])])
    a, _ = adapter(m)
    with pytest.raises(EvidenceVerificationError, match='one source row'):
        await a.generate_clean_fragment(f)


@pytest.mark.asyncio
async def test_kpi_rejects_swapped_quarters():
    f = fragment([('Users', '640', '875')], ('Metric', 'Q1', 'Q2'))
    m = QBRMetadata(kpis=[KPIMetric(name='Users', previous_value='875', current_value='640',
        evidence=[Evidence(page_number=1, source_text='Users')])])
    a, _ = adapter(m)
    with pytest.raises(EvidenceVerificationError):
        await a.generate_clean_fragment(f)


@pytest.mark.asyncio
async def test_action_rejects_owner_from_different_row():
    f = fragment([('Onboard Operations', 'Sarah', 'July 2026'),
                  ('Prepare rollout', 'Daniel', 'August 2026')], ('Action', 'Owner', 'Target'), 'next_steps')
    m = QBRMetadata(next_steps=[ActionItem(description='Onboard Operations', owner='Daniel',
        evidence=[Evidence(page_number=1, source_text='Onboard Operations')])])
    a, _ = adapter(m)
    with pytest.raises(EvidenceVerificationError, match='unsupported owner'):
        await a.generate_clean_fragment(f)


@pytest.mark.asyncio
async def test_action_correct_owner_accepted():
    f = fragment([('Onboard Operations', 'Sarah', 'July 2026')], ('Action', 'Owner', 'Target'), 'next_steps')
    m = QBRMetadata(next_steps=[ActionItem(description='Onboard Operations', owner='Sarah', target_date_text='July 2026',
        evidence=[Evidence(page_number=1, source_text='Onboard Operations')])])
    a, _ = adapter(m)
    assert len((await a.generate_clean_fragment(f)).metadata.next_steps) == 1
