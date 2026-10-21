import pytest
from unittest.mock import AsyncMock
from qbr.adapters.llm.category_agents import CategoryAgent, CategoryFacts, Candidate, CandidateAgentFallback
from qbr.adapters.llm.instructor_adapter import InstructorMetadataAdapter, LLMExtractionSettings, LLMProviderError
from qbr.domain.models import QBRMetadata
from qbr.adapters.pdf.pdf_processing_adapter import CleanDocumentFragment


def fragment():
    return CleanDocumentFragment(document_id="doc", page_number=1, fragment_index=0,
                                 text="Risk: renewal delay", source_text="Risk: renewal delay")


@pytest.mark.asyncio
async def test_category_agent_builds_grounded_risk():
    completion = type("Completion", (), {})()
    completion.create = AsyncMock(return_value=CategoryFacts(facts=[Candidate(
        description="renewal delay", page_number=1,
        source_excerpt="Risk: renewal delay", source_id="p1-f0")]))
    fallback = CandidateAgentFallback((CategoryAgent("risks", completion, "model"),))
    metadata = await fallback.extract(fragment(), "Risk: renewal delay")
    assert len(metadata.risks) == 1
    assert metadata.risks[0].evidence[0].source_text == "Risk: renewal delay"
    assert metadata.goals == []


@pytest.mark.asyncio
async def test_wakeup_requires_explicit_probe():
    completion = type("Completion", (), {})()
    completion.create = AsyncMock(return_value=QBRMetadata())
    adapter = InstructorMetadataAdapter(completion, LLMExtractionSettings(
        model="model", wake_up_enabled=True))
    with pytest.raises(LLMProviderError, match="no provider readiness probe"):
        await adapter.generate_clean_fragment(fragment())
    completion.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_wakeup_runs_before_extraction():
    completion = type("Completion", (), {})()
    completion.create = AsyncMock(return_value=QBRMetadata())
    probe = AsyncMock(return_value=None)
    adapter = InstructorMetadataAdapter(completion, LLMExtractionSettings(
        model="model", wake_up_enabled=True), wake_up_probe=probe)
    await adapter.generate_clean_fragment(fragment())
    probe.assert_awaited_once()
    completion.create.assert_awaited_once()


def test_settings_validate_wakeup_timeout():
    with pytest.raises(ValueError, match="wake_up_timeout_seconds"):
        LLMExtractionSettings(model="model", wake_up_timeout_seconds=0)

@pytest.mark.asyncio
async def test_wakeup_only_once_per_adapter():
    completion = type("Completion", (), {})()
    completion.create = AsyncMock(return_value=QBRMetadata())
    probe = AsyncMock(return_value=None)
    adapter = InstructorMetadataAdapter(completion, LLMExtractionSettings(
        model="model", wake_up_enabled=True), wake_up_probe=probe)
    await adapter.generate_clean_fragment(fragment())
    await adapter.generate_clean_fragment(fragment())
    probe.assert_awaited_once()


@pytest.mark.asyncio
async def test_schema_error_recovers_with_grounded_fallback():
    from pydantic import ValidationError
    completion = type("Completion", (), {})()
    try:
        QBRMetadata.model_validate({"risks": ["not a finding"]})
    except ValidationError as exc:
        failure = exc
    completion.create = AsyncMock(side_effect=failure)
    fallback = type("Fallback", (), {})()
    fallback.extract = AsyncMock(return_value=QBRMetadata())
    adapter = InstructorMetadataAdapter(completion, LLMExtractionSettings(
        model="model", candidate_fallback_enabled=True), fallback_extractor=fallback)
    result = await adapter.generate_clean_fragment(fragment())
    assert result.metadata == QBRMetadata()
    fallback.extract.assert_awaited_once()
