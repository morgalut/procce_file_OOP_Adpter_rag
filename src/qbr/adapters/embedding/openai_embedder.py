## need build """OpenAI embedding adapter. No credentials or network activity at import time."""
from __future__ import annotations
import math
from typing import Any


class EmbeddingError(RuntimeError):
    """Embedding provider returned invalid output or failed."""


class OpenAIEmbedder:
    def __init__(self, *, client: Any | None = None, api_key: str | None = None,
                 model: str = 'text-embedding-3-small', dimensions: int | None = 1536,
                 batch_size: int = 64) -> None:
        if not model or batch_size < 1 or (dimensions is not None and dimensions < 1):
            raise ValueError('Invalid embedding configuration')
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError('Install openai to use OpenAIEmbedder') from exc
            client = AsyncOpenAI(api_key=api_key)
        self._client = client
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Each text must be a nonempty string')
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            kwargs: dict[str, Any] = {'model': self.model, 'input': batch}
            if self.dimensions is not None:
                kwargs['dimensions'] = self.dimensions
            try:
                response = await self._client.embeddings.create(**kwargs)
            except Exception as exc:
                raise EmbeddingError(f'Embedding request failed: {type(exc).__name__}') from exc
            items = response.data
            if len(items) != len(batch):
                raise EmbeddingError('Embedding count does not match input count')
            indices = [item.index for item in items]
            if sorted(indices) != list(range(len(batch))):
                raise EmbeddingError('Embedding indices are missing or duplicated')
            for item in sorted(items, key=lambda item: item.index):
                vector = item.embedding
                if not vector or (self.dimensions is not None and len(vector) != self.dimensions):
                    raise EmbeddingError('Embedding dimensions do not match configuration')
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                    raise EmbeddingError('Embedding contains invalid numeric values')
                output.append([float(v) for v in vector])
        return output
