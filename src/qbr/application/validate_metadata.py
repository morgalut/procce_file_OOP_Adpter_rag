"""Deterministic metadata validation use case; independent of HTTP and LLM SDKs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any


class Validator(Protocol):
    def validate(self, metadata: Any) -> Any: ...
    def validate_fragment(self, metadata: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ValidateMetadata:
    validator: Validator

    def execute(self, metadata: Any, *, fragment: bool = False) -> Any:
        if metadata is None:
            raise ValueError("metadata is required")
        return (self.validator.validate_fragment(metadata) if fragment
                else self.validator.validate(metadata))
