"""Source-grounded text evaluation contracts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class EvaluationScore:
    metric: str
    score: float
    passed: bool
    reason: str | None = None

class GroundingEvaluator(Protocol):
    async def evaluate(self, *, claim: str, context: list[str], question: str = 'Summarize the QBR evidence.') -> EvaluationScore: ...
