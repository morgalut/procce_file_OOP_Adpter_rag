## need build """DeepEval faithfulness adapter; no LLM judge is created until evaluation."""
from __future__ import annotations
import asyncio
import math
from typing import Any, Callable
from qbr.ports.evaluator import EvaluationScore


class EvaluationError(RuntimeError):
    pass


class DeepEvalAdapter:
    def __init__(self, *, threshold: float = 0.8, metric_factory: Callable[[], Any] | None = None) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be in [0, 1]')
        self.threshold = threshold
        self._metric_factory = metric_factory

    def _run(self, claim: str, context: list[str], question: str) -> EvaluationScore:
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric
        except ImportError as exc:
            raise EvaluationError('Install deepeval to run faithfulness evaluation') from exc
        metric = self._metric_factory() if self._metric_factory else FaithfulnessMetric(threshold=self.threshold)
        case = LLMTestCase(input=question, actual_output=claim, retrieval_context=context)
        try:
            metric.measure(case)
            score = float(metric.score)
        except Exception as exc:
            raise EvaluationError(f'Faithfulness evaluation failed: {type(exc).__name__}') from exc
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise EvaluationError('Evaluator returned invalid score')
        return EvaluationScore(metric='faithfulness', score=score,
                               passed=score >= self.threshold,
                               reason=getattr(metric, 'reason', None))

    async def evaluate(self, *, claim: str, context: list[str],
                       question: str = 'Summarize the QBR evidence.') -> EvaluationScore:
        if not claim.strip() or not context or any(not text.strip() for text in context):
            raise ValueError('Claim and supporting context are required')
        return await asyncio.to_thread(self._run, claim, context, question)
