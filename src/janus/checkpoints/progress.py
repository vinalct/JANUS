from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.lineage.persistence import MetadataZonePaths, read_json_mapping, write_json_atomic
from janus.models import ExecutionPlan


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
        payload: dict[str, Any] = {
            "source_id": plan.source.source_id,
            "request_index": request_index,
            "completed_inputs": [
                {"key": k, "index": i} for k, i in (completed_inputs or [])
            ],
            "current_input_key": current_input_key,
            "current_input_index": current_input_index,
            "request_input_count": request_input_count,
            "artifact_count": artifact_count,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        if raw_path_prefix is not None and raw_path_prefix.strip():
            payload["raw_path_prefix"] = raw_path_prefix.strip()
        if page_number is not None:
            payload["last_page_number"] = page_number
        if offset is not None:
            payload["last_offset"] = offset
        if cursor is not None:
            payload["last_cursor"] = cursor
        return write_json_atomic(_progress_path(plan), payload)

    def clear(self, plan: ExecutionPlan) -> None:
        """Remove the progress file — called after successful extraction."""
        path = _progress_path(plan)
        if path.exists():
            path.unlink()


def _progress_path(plan: ExecutionPlan) -> Path:
    return MetadataZonePaths.from_plan(plan).base_dir / "extraction_progress.json"
