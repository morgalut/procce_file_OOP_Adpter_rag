
"""Centralized application configuration for QBR Intelligence.

Responsibilities:
- Load configuration from the project-root .env file.
- Validate individual settings using Pydantic.
- Validate relationships between settings.
- Support PostgreSQL, FastAPI, LLM extraction, PDF processing,
  vector search, evaluation, and observability.
- Provide cached settings through get_settings().

Important:
- Secrets are stored using SecretStr.
- Environment variables override values in .env.
- Application services should use get_settings() instead of
  reading os.environ directly.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# PROJECT PATHS
# ============================================================

# config.py:
# project_root/src/qbr/infrastructure/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# VALIDATION RULES
# ============================================================

ValidationRule = Callable[["AppSettings"], str | None]


def check_chunk_overlap(
    settings: "AppSettings",
) -> str | None:
    """Ensure chunk overlap is smaller than chunk size."""

    if (
        settings.chunk_overlap_tokens
        >= settings.chunk_size_tokens
    ):
        return (
            "chunk_overlap_tokens must be smaller "
            "than chunk_size_tokens"
        )

    if (
        settings.qbr_chunk_overlap_tokens
        >= settings.qbr_chunk_max_tokens
    ):
        return (
            "qbr_chunk_overlap_tokens must be smaller "
            "than qbr_chunk_max_tokens"
        )

    return None


def check_llm_provider(
    settings: "AppSettings",
) -> str | None:
    """Validate LLM provider configuration."""

    if settings.llm_provider == "ollama":
        if not settings.llm_base_url:
            return (
                "LLM_BASE_URL is required "
                "when using Ollama"
            )

    if settings.llm_provider == "openai":
        if settings.llm_base_url is not None:
            return (
                "LLM_BASE_URL is reserved "
                "for the Ollama adapter"
            )

    return None


def check_embedding_provider(
    settings: "AppSettings",
) -> str | None:
    """Validate embedding configuration."""

    if settings.embedding_provider == "ollama":
        if not settings.embedding_base_url:
            return (
                "EMBEDDING_BASE_URL is required "
                "when using Ollama"
            )

    return None


def check_langfuse(
    settings: "AppSettings",
) -> str | None:
    """Require credentials when Langfuse is enabled."""

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key:
        return (
            "LANGFUSE_PUBLIC_KEY is required "
            "when Langfuse is enabled"
        )

    if not settings.langfuse_secret_key:
        return (
            "LANGFUSE_SECRET_KEY is required "
            "when Langfuse is enabled"
        )

    if not settings.langfuse_secret_key.get_secret_value():
        return (
            "LANGFUSE_SECRET_KEY cannot be empty"
        )

    return None


def check_production_debug(
    settings: "AppSettings",
) -> str | None:
    """Prevent debug mode in production."""

    if (
        settings.app_env == "production"
        and settings.debug
    ):
        return "DEBUG must be disabled in production"

    return None


def check_vector_dimensions(
    settings: "AppSettings",
) -> str | None:
    """Validate pgvector dimensions against V21 schema."""

    if not settings.qbr_vector_enabled:
        return None

    # Current Alembic V21 migration uses vector(1536).
    if settings.qbr_embedding_dimensions != 1536:
        return (
            "QBR_EMBEDDING_DIMENSIONS must be 1536 "
            "for the current pgvector schema"
        )

    return None


VALIDATION_RULES: tuple[ValidationRule, ...] = (
    check_chunk_overlap,
    check_llm_provider,
    check_embedding_provider,
    check_langfuse,
    check_production_debug,
    check_vector_dimensions,
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

class AppSettings(BaseSettings):
    """Centralized, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ========================================================
    # 1. APPLICATION
    # ========================================================

    app_name: str = "QBR Intelligence Pipeline"

    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = False

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ] = "INFO"

    # ========================================================
    # 2. FASTAPI / UVICORN
    # ========================================================

    qbr_host: str = "127.0.0.1"

    qbr_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    qbr_reload: bool = True

    # ========================================================
    # 3. POSTGRESQL
    # ========================================================

    database_url: SecretStr = Field(
        default=SecretStr(
            "postgresql+asyncpg://qbr:qbr@localhost:5432/qbr_db"
        ),
        repr=False,
    )

    database_echo: bool = False

    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    database_max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
    )

    # ========================================================
    # 4. EXTRACTION PIPELINE
    # ========================================================

    # Example:
    # qbr.bootstrap.pipeline_factory:create_pipeline
    #
    # Must point to a real factory returning a pipeline
    # with an execute(source) method.

    qbr_pipeline_factory: str | None = None

    # ========================================================
    # 5. LLM PROVIDER
    # ========================================================

    llm_provider: Literal[
        "openai",
        "ollama",
    ] = "openai"

    llm_model: str = "gpt-4o-mini"

    llm_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
    )

    # Standard OpenAI SDK environment variable.
    openai_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
    )

    llm_base_url: str | None = None

    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    llm_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=600,
    )

    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    # ========================================================
    # 6. PDF PROCESSING
    # ========================================================

    qbr_upload_dir: str = "data/uploads"

    pdf_max_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    pdf_max_pages: int = Field(
        default=100,
        ge=1,
        le=1000,
    )

    pdf_table_extraction: bool = True

    # ========================================================
    # 7. METADATA VALIDATION
    # ========================================================

    validation_min_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    validation_require_evidence: bool = True

    # ========================================================
    # 8. DEEPEVAL
    # ========================================================

    deepeval_enabled: bool = False

    deepeval_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # 9. LANGFUSE
    # ========================================================

    langfuse_enabled: bool = False

    langfuse_public_key: str | None = None

    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        repr=False,
    )

    langfuse_host: str = (
        "https://cloud.langfuse.com"
    )

    # ========================================================
    # 10. EXISTING VECTORIZATION SETTINGS
    # ========================================================

    # Preserved for compatibility with existing components.

    vectorization_enabled: bool = False

    embedding_provider: Literal[
        "openai",
        "ollama",
    ] = "openai"

    embedding_base_url: str | None = None

    embedding_model: str = (
        "text-embedding-3-small"
    )

    embedding_dimensions: int = Field(
        default=1536,
        ge=1,
    )

    chunk_size_tokens: int = Field(
        default=700,
        ge=100,
    )

    chunk_overlap_tokens: int = Field(
        default=70,
        ge=0,
    )

    # ========================================================
    # 11. V28 VECTOR RUNTIME SETTINGS
    # ========================================================

    # Matches the existing V28 runtime.py variable names.

    qbr_vector_enabled: bool = False

    qbr_embedding_model: str = (
        "text-embedding-3-small"
    )

    qbr_embedding_dimensions: int = Field(
        default=1536,
        ge=1,
    )

    qbr_chunk_max_tokens: int = Field(
        default=700,
        ge=1,
    )

    qbr_chunk_overlap_tokens: int = Field(
        default=70,
        ge=0,
    )

    # ========================================================
    # FIELD VALIDATORS
    # ========================================================

    @field_validator("database_url")
    @classmethod
    def validate_database_url(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        """Require async PostgreSQL driver."""

        database_url = value.get_secret_value()

        if not database_url.startswith(
            "postgresql+asyncpg://"
        ):
            raise ValueError(
                "DATABASE_URL must use "
                "postgresql+asyncpg://"
            )

        return value

    @field_validator(
        "llm_model",
        "embedding_model",
        "qbr_embedding_model",
    )
    @classmethod
    def validate_model_name(
        cls,
        value: str,
    ) -> str:
        """Reject empty model names."""

        value = value.strip()

        if not value:
            raise ValueError(
                "Model name cannot be empty"
            )

        return value

    @field_validator("qbr_pipeline_factory")
    @classmethod
    def validate_pipeline_factory(
        cls,
        value: str | None,
    ) -> str | None:
        """Validate configured factory path syntax."""

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        module, separator, function = (
            value.partition(":")
        )

        if (
            not separator
            or not module
            or not function
        ):
            raise ValueError(
                "QBR_PIPELINE_FACTORY must use "
                "module:function"
            )

        return value

    # ========================================================
    # CROSS-FIELD VALIDATION
    # ========================================================

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Apply all configuration rules."""

        errors: list[str] = []

        for rule in VALIDATION_RULES:
            message = rule(self)

            if message:
                errors.append(message)

        if errors:
            raise ValueError(
                "Invalid configuration:\\n- "
                + "\\n- ".join(errors)
            )

        return self

    # ========================================================
    # HELPER PROPERTIES
    # ========================================================

    @property
    def database_dsn(self) -> str:
        """Return database URL for SQLAlchemy."""

        return self.database_url.get_secret_value()

    @property
    def effective_openai_api_key(
        self,
    ) -> str | None:
        """Return configured OpenAI key, if available."""

        secret = (
            self.openai_api_key
            or self.llm_api_key
        )

        if secret is None:
            return None

        return (
            secret.get_secret_value()
            or None
        )

    @property
    def upload_directory(self) -> Path:
        """Return an absolute upload directory."""

        path = Path(self.qbr_upload_dir)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path


# ============================================================
# SETTINGS FACTORY
# ============================================================

@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached, validated application settings."""

    return AppSettings()
