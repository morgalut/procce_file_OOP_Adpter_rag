"""Unit tests for the QBR application's configuration layer."""

import pytest
from pydantic import ValidationError

from qbr.infrastructure.config import (
    AppSettings,
    VALIDATION_RULES,
    check_chunk_overlap,
    check_langfuse,
    check_llm_provider,
    check_production_debug,
    get_settings,
)


@pytest.fixture
def settings_factory(monkeypatch):
    """Create settings independently of local .env and QBR_* environment values."""
    def build(**overrides):
        return AppSettings(_env_file=None, **overrides)
    return build


def test_defaults(settings_factory):
    settings = settings_factory()
    assert settings.app_env == "development"
    assert settings.llm_provider == "openai"
    assert settings.pdf_max_pages == 100
    assert settings.chunk_size_tokens == 700
    assert settings.chunk_overlap_tokens == 70


@pytest.mark.parametrize("url", [
    "sqlite:///local.db",
    "postgresql://localhost/qbr",
    "mysql://localhost/qbr",
])
def test_database_url_rejects_non_async_postgres(settings_factory, url):
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        settings_factory(database_url=url)


def test_valid_database_url(settings_factory):
    settings = settings_factory(database_url="postgresql+asyncpg://user:pass@localhost:5432/db")
    assert settings.database_url.get_secret_value().startswith("postgresql+asyncpg://")


@pytest.mark.parametrize("field", ["llm_model", "embedding_model"])
@pytest.mark.parametrize("value", ["", "  "])
def test_empty_model_name_rejected(settings_factory, field, value):
    with pytest.raises(ValidationError, match="Model name cannot be empty"):
        settings_factory(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("database_pool_size", 0),
    ("llm_temperature", -0.1),
    ("pdf_max_pages", 0),
    ("deepeval_threshold", 1.1),
    ("validation_min_confidence", -0.01),
])
def test_field_bounds(settings_factory, field, value):
    with pytest.raises(ValidationError):
        settings_factory(**{field: value})


@pytest.mark.parametrize("size,overlap", [(700, 700), (700, 701)])
def test_invalid_chunk_overlap(settings_factory, size, overlap):
    with pytest.raises(ValidationError, match="chunk_overlap_tokens"):
        settings_factory(chunk_size_tokens=size, chunk_overlap_tokens=overlap)


def test_valid_chunk_overlap_rule(settings_factory):
    assert check_chunk_overlap(settings_factory()) is None


def test_ollama_requires_base_url(settings_factory):
    with pytest.raises(ValidationError, match="LLM_BASE_URL is required"):
        settings_factory(llm_provider="ollama")


def test_ollama_accepts_base_url(settings_factory):
    settings = settings_factory(llm_provider="ollama", llm_base_url="http://localhost:11434")
    assert check_llm_provider(settings) is None


def test_openai_rejects_custom_base_url(settings_factory):
    with pytest.raises(ValidationError, match="reserved for the Ollama adapter"):
        settings_factory(llm_provider="openai", llm_base_url="http://localhost:11434")


def test_langfuse_disabled_requires_no_keys(settings_factory):
    assert check_langfuse(settings_factory(langfuse_enabled=False)) is None

