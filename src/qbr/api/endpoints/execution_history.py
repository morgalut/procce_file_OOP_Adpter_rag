"""Read-only durable pipeline execution history."""
from __future__ import annotations
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

router = APIRouter(prefix="/api/v1/qbr/runs", tags=["QBR execution history"])

@router.get("/{run_id}")
async def get_run(run_id: UUID, request: Request):
    tracker = getattr(request.app.state, "pipeline_tracker", None)
    if tracker is None:
        raise HTTPException(503, "Event tracking is disabled")
    async with tracker.sessions() as session:
        row = (await session.execute(text("SELECT id, document_id, pipeline_name, status, started_at, finished_at FROM qbr_pipeline_runs WHERE id=:id"), {"id":run_id})).mappings().first()
        if row is None:
            raise HTTPException(404, "Run not found")
        return dict(row)

@router.get("/{run_id}/stages")
async def get_stages(run_id: UUID, request: Request):
    tracker = getattr(request.app.state, "pipeline_tracker", None)
    if tracker is None:
        raise HTTPException(503, "Event tracking is disabled")
    async with tracker.sessions() as session:
        rows = (await session.execute(text("SELECT id, parent_stage_id, stage_name, fragment_id, attempt_number, status, started_at, finished_at, duration_ms, input_summary, output_summary, error_code FROM qbr_pipeline_stages WHERE run_id=:id ORDER BY started_at, id"), {"id":run_id})).mappings().all()
        return [dict(row) for row in rows]

@router.get("/{run_id}/events")
async def get_events(run_id: UUID, request: Request, limit: int = 100, after_id: int = 0):
    if not 1 <= limit <= 500 or after_id < 0:
        raise HTTPException(422, "Invalid pagination")
    tracker = getattr(request.app.state, "pipeline_tracker", None)
    if tracker is None:
        raise HTTPException(503, "Event tracking is disabled")
    async with tracker.sessions() as session:
        rows = (await session.execute(text("SELECT id, stage_id, event_type, event_data, created_at FROM qbr_pipeline_events WHERE run_id=:id AND id>:after ORDER BY id LIMIT :limit"), {"id":run_id,"after":after_id,"limit":limit})).mappings().all()
        return [dict(row) for row in rows]


@router.get("")
async def list_runs(
    request: Request,
    document_id: UUID | None = None,
    limit: int = 25,
):
    """List recent pipeline runs, optionally filtered by document UUID."""
    if not 1 <= limit <= 100:
        raise HTTPException(422, "limit must be between 1 and 100")
    tracker = getattr(request.app.state, "pipeline_tracker", None)
    if tracker is None:
        raise HTTPException(503, "Event tracking is disabled")
    async with tracker.sessions() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, document_id, pipeline_name, status, started_at, finished_at "
                    "FROM qbr_pipeline_runs "
                    + ("WHERE document_id = :document_id " if document_id else "")
                    + "ORDER BY started_at DESC, id DESC LIMIT :limit"
                ),
                {"document_id": str(document_id), "limit": limit} if document_id else {"limit": limit},
            )
        ).mappings().all()
        return [dict(row) for row in rows]
