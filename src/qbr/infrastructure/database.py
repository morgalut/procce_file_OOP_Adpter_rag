"""Explicit engine/session construction. Caller owns engine.dispose()."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

def make_database(database_url: str, *, echo: bool = False, pool_size: int = 5, max_overflow: int = 10):
    if not database_url.startswith('postgresql+asyncpg://'):
        raise ValueError('Expected DATABASE_URL using postgresql+asyncpg')
    engine = create_async_engine(database_url, echo=echo, pool_size=pool_size, max_overflow=max_overflow, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
