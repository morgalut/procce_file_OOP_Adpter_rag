"""Durable, fail-safe QBR stage events; no document text or secrets stored."""
from __future__ import annotations

import contextvars
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

logger = logging.getLogger(__name__)
_active: contextvars.ContextVar[tuple[Any, UUID, UUID | None] | None] = contextvars.ContextVar("qbr_tracking", default=None)


def _safe(value: Any) -> Any:
    """Only explicitly supplied non-sensitive scalar summaries may be stored."""
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, dict):
        blocked = ("text", "quote", "prompt", "content", "secret", "password", "token", "api_key", "authorization", "query")
        return {str(k)[:64]: _safe(v) for k, v in value.items()
                if not any(word in str(k).lower() for word in blocked) or str(k).lower() == "query_length"}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value[:30]]
    return type(value).__name__


class PostgresPipelineTracker:
    def __init__(self, sessions: Any) -> None:
        self.sessions = sessions

    async def _write(self, statement: str, values: dict[str, Any]) -> None:
        # Independent transaction: business rollbacks cannot erase diagnostics.
        async with self.sessions() as session:
            async with session.begin():
                await session.execute(text(statement), values)

    async def start_run(self, name: str, document_id: str | None = None) -> UUID:
        run_id = uuid4()
        await self._write("INSERT INTO qbr_pipeline_runs (id, pipeline_name, document_id, status) VALUES (:id,:name,:doc,'running')", {"id": run_id, "name": name, "doc": document_id})
        return run_id

    async def finish_run(self, run_id: UUID, status: str) -> None:
        await self._write("UPDATE qbr_pipeline_runs SET status=:status, finished_at=now() WHERE id=:id", {"id": run_id, "status": status})

    async def start_stage(self, run_id: UUID, stage_name: str, *, fragment_id: str | None = None, parent_stage_id: UUID | None = None, attempt_number: int = 1, input_summary: dict[str, Any] | None = None) -> UUID:
        from sqlalchemy.dialects.postgresql import JSONB
        import json
        stage_id = uuid4()
        await self._write("INSERT INTO qbr_pipeline_stages (id, run_id, parent_stage_id, stage_name, fragment_id, attempt_number, status, input_summary) VALUES (:id,:run,:parent,:name,:fragment,:attempt,'running',CAST(:summary AS jsonb))", {"id":stage_id,"run":run_id,"parent":parent_stage_id,"name":stage_name,"fragment":fragment_id,"attempt":attempt_number,"summary":json.dumps(_safe(input_summary or {}))})
        await self.record_event(run_id, stage_id, 'stage.started', {"stage":stage_name})
        return stage_id

    async def complete_stage(self, stage_id: UUID, output_summary: dict[str, Any] | None = None) -> None:
        import json
        await self._write("UPDATE qbr_pipeline_stages SET status='completed', finished_at=now(), duration_ms=EXTRACT(EPOCH FROM (now()-started_at))*1000, output_summary=CAST(:summary AS jsonb) WHERE id=:id", {"id":stage_id,"summary":json.dumps(_safe(output_summary or {}))})

    async def fail_stage(self, stage_id: UUID, error_code: str, error_message: str) -> None:
        # Error messages can contain source excerpts; persist error category only.
        await self._write("UPDATE qbr_pipeline_stages SET status='failed', finished_at=now(), duration_ms=EXTRACT(EPOCH FROM (now()-started_at))*1000, error_code=:code WHERE id=:id", {"id":stage_id,"code":error_code[:100]})

    async def record_event(self, run_id: UUID, stage_id: UUID | None, event_type: str, data: dict[str, Any]) -> None:
        import json
        await self._write("INSERT INTO qbr_pipeline_events (run_id, stage_id, event_type, event_data) VALUES (:run,:stage,:type,CAST(:data AS jsonb))", {"run":run_id,"stage":stage_id,"type":event_type[:100],"data":json.dumps(_safe(data))})


async def _best_effort(coro: Any) -> Any:
    try:
        return await coro
    except Exception:
        logger.exception('QBR event tracking unavailable')
        return None


@asynccontextmanager
async def run_scope(tracker: Any, name: str, document_id: str | None = None):
    run_id = await _best_effort(tracker.start_run(name, document_id))
    if run_id is None:
        yield None
        return
    token = _active.set((tracker, run_id, None))
    try:
        yield run_id
    except BaseException:
        await _best_effort(tracker.finish_run(run_id, 'failed'))
        raise
    else:
        await _best_effort(tracker.finish_run(run_id, 'completed'))
    finally:
        _active.reset(token)


@dataclass(slots=True)
class StageDetails:
    """Explicit, redacted stage details; no automatic prompt/response capture."""
    output_summary: dict[str, Any] = field(default_factory=dict)

    def finish(self, **values: Any) -> None:
        self.output_summary.update(values)


@asynccontextmanager
async def stage(name: str, *, fragment_id: str | None = None, input_summary: dict[str, Any] | None = None):
    details = StageDetails()
    active = _active.get()
    if active is None:
        yield details
        return
    tracker, run_id, parent_id = active
    stage_id = await _best_effort(tracker.start_stage(run_id, name, fragment_id=fragment_id, parent_stage_id=parent_id, input_summary=input_summary))
    if stage_id is None:
        yield details
        return
    token = _active.set((tracker, run_id, stage_id))
    try:
        yield details
    except BaseException as exc:
        await _best_effort(tracker.fail_stage(stage_id, type(exc).__name__, 'redacted'))
        await _best_effort(tracker.record_event(run_id, stage_id, 'stage.failed', {"error_code":type(exc).__name__}))
        raise
    else:
        await _best_effort(tracker.complete_stage(stage_id, details.output_summary))
        await _best_effort(tracker.record_event(run_id, stage_id, 'stage.completed', details.output_summary))
    finally:
        _active.reset(token)


async def emit(event_type: str, data: dict[str, Any]) -> None:
    active = _active.get()
    if active:
        tracker, run_id, stage_id = active
        await _best_effort(tracker.record_event(run_id, stage_id, event_type, data))


class TrackedService:
    """Wrap existing async execute() services without modifying their contracts."""
    def __init__(self, delegate: Any, stage_name: str, tracker: Any, *, document_arg: bool = False):
        self._delegate = delegate
        self._stage_name = stage_name
        self._tracker = tracker
        self._document_arg = document_arg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        document_id = (
            str(args[0]) if self._document_arg and args
            else str(kwargs["document_id"]) if self._document_arg and kwargs.get("document_id")
            else None
        )
        if self._stage_name == "vector_search":
            query = args[0] if args else kwargs.get("query", "")
            summary = {
                "query_length": len(query) if isinstance(query, str) else 0,
                "top_k": kwargs.get("top_k", 5),
                "document_filter": str(kwargs["document_id"]) if kwargs.get("document_id") else None,
            }
        else:
            summary = {"document_id": document_id} if document_id else {}
        async with run_scope(self._tracker, self._stage_name, document_id):
            async with stage(self._stage_name, input_summary=summary) as details:
                result = await self._delegate.execute(*args, **kwargs)
                if self._stage_name == "vector_search":
                    details.finish(result_count=len(result), model=getattr(self._delegate, "model", None))
                elif self._stage_name == "vector_index" and isinstance(result, dict):
                    details.finish(**{key: result[key] for key in
                        ("status", "chunks_created", "embedding_model", "dimensions") if key in result})
                return result
