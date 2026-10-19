"""Provider-independent text embedding contract."""
from __future__ import annotations
from typing import Protocol

class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
