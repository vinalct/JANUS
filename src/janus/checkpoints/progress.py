from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.lineage.persistence import MetadataZonePaths, read_json_mapping, write_json_atomic
from janus.models import ExecutionPlan


class ProgressRecordError(ValueError):
    """Raised when persisted extraction progress is unsafe to resume."""

    def __init__(
        self,
        plan: ExecutionPlan,
        *,
        field_name: str,
        value: Any,
        reason: str,
    ) -> None:
        path = _progress_path(plan)
        source_id = plan.source.source_id
        super().__init__(
            f"extraction_progress.json for {source_id} carries an unusable "
            f"{field_name} {value!r}: {reason}. "
            f"Inspect or remove {path} before resuming."
        )


@dataclass(slots=True)
class ExtractionProgressStore:
    """Track per-page extraction progress so failed runs can resume where they stopped."""

    def load(self, plan: ExecutionPlan) -> dict[str, Any] | None:
        """Return stored progress for this source, or None if none exists."""
        path = _progress_path(plan)
        payload = read_json_mapping(path)
        if payload is None:
            return None
        if payload.get("source_id") != plan.source.source_id:
            return None
        return payload

    def save(
        self,
        plan: ExecutionPlan,
        *,
        page_number: int | None = None,
        offset: int | None = None,
        cursor: str | None = None,
        request_index: int = 0,
        artifact_count: int = 0,
        completed_inputs: list[tuple[str, int]] | None = None,
        current_input_key: str | None = None,
        current_input_index: int = 1,
        request_input_count: int = 1,
        raw_path_prefix: str | None = None,
    ) -> Path:
        """Atomically persist the last successfully processed pagination position."""
        payload = _progress_payload(
            plan,
            request_index=request_index,
            artifact_count=artifact_count,
            completed_inputs=completed_inputs,
            current_input_key=current_input_key,
            current_input_index=current_input_index,
            request_input_count=request_input_count,
            raw_path_prefix=raw_path_prefix,
        )
        if page_number is not None:
            payload["last_page_number"] = page_number
        if offset is not None:
            payload["last_offset"] = offset
        if cursor is not None:
            payload["last_cursor"] = cursor
        return write_json_atomic(_progress_path(plan), payload)

    def save_between_inputs(
        self,
        plan: ExecutionPlan,
        *,
        completed_inputs: list[tuple[str, int]],
        artifact_count: int,
        request_input_count: int,
        raw_path_prefix: str | None = None,
    ) -> Path:
        """Atomically persist that a request input finished and no other has started.

        ``save`` runs once per page, so on its own it records a finished input only with the
        next input's first page. A next input that failed before that page left the finished
        one named as the input in progress, and a resume asked for the page after its last.
        This record names no input in progress and no page: every finished input is
        rehydrated, never requested again.
        """
        payload = _progress_payload(
            plan,
            request_index=0,
            artifact_count=artifact_count,
            completed_inputs=completed_inputs,
            current_input_key=None,
            current_input_index=None,
            request_input_count=request_input_count,
            raw_path_prefix=raw_path_prefix,
        )
        return write_json_atomic(_progress_path(plan), payload)

    def clear(self, plan: ExecutionPlan) -> None:
        """Remove the progress file — called after successful extraction."""
        path = _progress_path(plan)
        if path.exists():
            path.unlink()


def _progress_payload(
    plan: ExecutionPlan,
    *,
    request_index: int,
    artifact_count: int,
    completed_inputs: list[tuple[str, int]] | None,
    current_input_key: str | None,
    current_input_index: int | None,
    request_input_count: int,
    raw_path_prefix: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_id": plan.source.source_id,
        "request_index": request_index,
        "completed_inputs": [{"key": k, "index": i} for k, i in (completed_inputs or [])],
        "current_input_key": current_input_key,
        "current_input_index": current_input_index,
        "request_input_count": request_input_count,
        "artifact_count": artifact_count,
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }
    if raw_path_prefix is not None and raw_path_prefix.strip():
        payload["raw_path_prefix"] = raw_path_prefix.strip()
    return payload


def _progress_path(plan: ExecutionPlan) -> Path:
    return MetadataZonePaths.from_plan(plan).base_dir / "extraction_progress.json"
