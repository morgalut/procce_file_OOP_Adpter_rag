from typing import Any, Protocol
from uuid import UUID


class PipelineTracker(Protocol):

    async def start_stage(
        self,
        run_id: UUID,
        stage_name: str,
        *,
        fragment_id: str | None = None,
        parent_stage_id: UUID | None = None,
        attempt_number: int = 1,
        input_summary: dict[str, Any] | None = None,
    ) -> UUID:
        ...

    async def complete_stage(
        self,
        stage_id: UUID,
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        ...

    async def fail_stage(
        self,
        stage_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        ...

    async def record_event(
        self,
        run_id: UUID,
        stage_id: UUID | None,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        ...