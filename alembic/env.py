
"""Alembic migration environment using centralized application settings."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from qbr.adapters.persistence.sqlalchemy_models import Base
from qbr.infrastructure.config import get_settings


config = context.config

# Load DATABASE_URL from Pydantic Settings, including .env.
settings = get_settings()
database_url = settings.database_url.get_secret_value()

# Alembic uses ConfigParser, so literal % must be escaped.
config.set_main_option(
    "sqlalchemy.url",
    database_url.replace("%", "%%"),
)

target_metadata = Base.metadata

# Configure logging only when alembic.ini defines
# the complete Python logging configuration.
if (
    config.config_file_name is not None
    and config.file_config.has_section("loggers")
    and config.file_config.has_section("handlers")
    and config.file_config.has_section("formatters")
):
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Generate SQL without connecting to PostgreSQL."""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations through a synchronous Alembic connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Connect to PostgreSQL using SQLAlchemy async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
