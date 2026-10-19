"""Langfuse event adapter with optional injection and disabled no-op mode."""
from __future__ import annotations
from typing import Any


class LangfuseAdapter:
    def __init__(self, *, enabled: bool = False, client: Any | None = None,
                 public_key: str | None = None, secret_key: str | None = None,
                 host: str | None = None) -> None:
        self.enabled = enabled
        if not enabled:
            self._client = None
            return
        if client is None:
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                raise RuntimeError('Install langfuse to enable tracing') from exc
            if not public_key or not secret_key:
                raise ValueError('Langfuse keys are required when enabled')
            options = {'public_key': public_key, 'secret_key': secret_key}
            if host:
                options['host'] = host
            client = Langfuse(**options)
        self._client = client

    def record_event(self, name: str, *, attributes: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        if not name.strip():
            raise ValueError('Event name must not be blank')
        # Langfuse SDK v3 supports start_observation; it associates the event
        # with the active observation context when one exists.
        try:
            observation = self._client.start_observation(name=name, as_type='event',
                                                         metadata=attributes or {})
            observation.end()
        except Exception as exc:
            raise RuntimeError(f'Langfuse event failed: {type(exc).__name__}') from exc

    def flush(self) -> None:
        if self.enabled:
            self._client.flush()
