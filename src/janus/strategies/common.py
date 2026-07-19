from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan
from janus.utils.environment import load_environment_config, prepare_runtime
from janus.utils.storage import StorageLayout

RAW_PROGRESS_PATH_PREFIX_FIELD = "raw_path_prefix"

_RAW_PATH_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9._=-]+")


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _format_datetime(value: datetime) -> str:
    normalized = value.astimezone(UTC).isoformat()
    return normalized.replace("+00:00", "Z")


def _normalize_checkpoint_value(value: str) -> tuple[str, Any]:
    parsed_datetime = _parse_datetime(value)
    if parsed_datetime is not None:
        return ("datetime", parsed_datetime.astimezone(UTC))

    try:
        return ("decimal", Decimal(value))
    except InvalidOperation:
        return ("text", value)


def _compare_checkpoint_values(left: str, right: str) -> int:
    normalized_left = _normalize_checkpoint_value(left)
    normalized_right = _normalize_checkpoint_value(right)
    if normalized_left[0] == normalized_right[0]:
        left_value = normalized_left[1]
        right_value = normalized_right[1]
        if left_value < right_value:
            return -1
        if left_value > right_value:
            return 1
        return 0

    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _max_checkpoint_value(current_value: str | None, candidate_value: str) -> str:
    if current_value is None:
        return candidate_value
    if _compare_checkpoint_values(candidate_value, current_value) > 0:
        return candidate_value
    return current_value


def _retry_delay_seconds(
    plan: ExecutionPlan,
    attempt: int,
    response: Any,
) -> float:
    retry_config = plan.source_config.extraction.retry
    rate_limit_backoff = plan.source_config.access.rate_limit.backoff_seconds
    if retry_config.backoff_strategy == "exponential":
        delay = retry_config.backoff_seconds * (2 ** (attempt - 1))
    else:
        delay = retry_config.backoff_seconds

    maximum_delay = rate_limit_backoff or max(
        retry_config.backoff_seconds,
        retry_config.backoff_seconds * retry_config.max_attempts,
    )

    retry_after_header = None
    if response is not None:
        retry_after_header = response.headers_as_dict().get("Retry-After")
    if retry_after_header is not None:
        with contextlib.suppress(ValueError):
            delay = max(delay, float(retry_after_header))
    return float(min(delay, maximum_delay))


def _freeze_string_mapping(values: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()

    frozen_items: list[tuple[str, str]] = []
    for key, value in values.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if not normalized_key:
            raise ValueError("mapping keys must be non-empty strings")
        if not normalized_value:
            raise ValueError("mapping values must be non-empty strings")
        frozen_items.append((normalized_key, normalized_value))
    return tuple(sorted(frozen_items))


def _stringify_mapping(values: Mapping[str, Any]) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for key, value in values.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("mapping keys must be non-empty strings")
        if value is None:
            continue
        normalized_value = str(value).strip()
        if not normalized_value:
            continue
        rendered[normalized_key] = normalized_value
    return rendered


def _request_input_key(request_input: dict[str, Any] | None) -> str:
    """Stable, content-based fingerprint for a request input context."""
    if request_input is None:
        return "__none__"
    return "|".join(sorted(f"{k}={v}" for k, v in request_input.items()))


def _raw_page_path(
    pagination_state: Any,
    suffix: str,
    *,
    request_input_index: int,
    request_input_count: int,
) -> Path:
    """Return the relative raw artifact path for a paginated response."""
    if pagination_state.page_number is not None:
        filename = f"page-{pagination_state.page_number:04d}{suffix}"
    elif pagination_state.offset is not None:
        filename = f"offset-{pagination_state.offset:08d}{suffix}"
    elif pagination_state.cursor is not None:
        filename = f"cursor-{pagination_state.request_index:04d}{suffix}"
    else:
        filename = f"response-{pagination_state.request_index:04d}{suffix}"
    if request_input_count > 1:
        return Path(f"request-input-{request_input_index:06d}") / filename
    return Path("pages") / filename


def _raw_run_path_prefix(
    plan: ExecutionPlan,
    progress: Mapping[str, Any] | None = None,
) -> Path | None:
    if progress is not None:
        raw_path_prefix = progress.get(RAW_PROGRESS_PATH_PREFIX_FIELD)
        if isinstance(raw_path_prefix, str) and raw_path_prefix.strip():
            return Path(raw_path_prefix)
        return None

    ingestion_date = plan.run_context.started_at.astimezone(UTC).date().isoformat()
    return (
        Path("runs")
        / f"ingestion_date={ingestion_date}"
        / f"run_id={_safe_raw_path_segment(plan.run_context.run_id)}"
    )


def _safe_raw_path_segment(value: str) -> str:
    normalized = _RAW_PATH_SEGMENT_PATTERN.sub("-", value.strip()).strip("-")
    return normalized or "run"


def _default_storage_layout(plan: ExecutionPlan) -> StorageLayout:
    environment_config = load_environment_config(
        plan.run_context.environment,
        plan.run_context.project_root,
    )
    prepare_runtime(environment_config, plan.run_context.project_root)
    return StorageLayout.from_environment_config(
        environment_config,
        plan.run_context.project_root,
    )
