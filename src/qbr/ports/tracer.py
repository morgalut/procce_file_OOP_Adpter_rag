"""Observability port; callers must redact sensitive attributes."""
from __future__ import annotations
from typing import Any, Protocol

class Tracer(Protocol):
    def record_event(self, name: str, *, attributes: dict[str, Any] | None = None) -> None: ...
    def flush(self) -> None: ...
