"""Pipeline identity and per-source attempt identity, derived once and safe in a path."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256

from janus.orchestration.errors import PipelineIdentityError
from janus.planner import normalize_run_id_segment

PIPELINE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_PIPELINE_RUN_ID_LENGTH = 96
MAX_SOURCE_SEGMENT_LENGTH = 48
ATTEMPT_DIGEST_LENGTH = 10
_DIGEST_SEPARATOR = "\x1f"


def validate_pipeline_run_id(pipeline_run_id: str) -> str:
    """Return ``pipeline_run_id`` unchanged, or refuse it before it reaches a path."""
    if not pipeline_run_id or not PIPELINE_RUN_ID_PATTERN.match(pipeline_run_id):
        raise PipelineIdentityError(
            f"pipeline_run_id {pipeline_run_id!r} is not usable as a path component: it must "
            "start with a letter or a digit and hold only letters, digits, '.', '-' and '_'"
        )
    if len(pipeline_run_id) > MAX_PIPELINE_RUN_ID_LENGTH:
        raise PipelineIdentityError(
            f"pipeline_run_id {pipeline_run_id!r} is {len(pipeline_run_id)} characters; the "
            f"maximum is {MAX_PIPELINE_RUN_ID_LENGTH}, because every source run id in the "
            "batch extends it"
        )
    return pipeline_run_id


def default_pipeline_run_id(environment: str, planned_at: datetime) -> str:
    """Derive a pipeline id from the environment and the logical planning timestamp."""
    timestamp = planned_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return validate_pipeline_run_id(
        f"pipeline-{normalize_run_id_segment(environment)}-{timestamp}"
    )


def source_attempt_run_id(*, pipeline_run_id: str, source_id: str, attempt: int) -> str:
    """Derive one source attempt's run id from the pipeline, the source and the attempt."""
    validate_pipeline_run_id(pipeline_run_id)
    if not source_id.strip():
        raise PipelineIdentityError("source_id must not be empty")
    if attempt < 1:
        raise PipelineIdentityError(f"attempt must be 1 or greater, got {attempt!r}")

    digest = sha256(
        _DIGEST_SEPARATOR.join((pipeline_run_id, source_id, str(attempt))).encode("utf-8")
    ).hexdigest()[:ATTEMPT_DIGEST_LENGTH]
    segment = normalize_run_id_segment(source_id)[:MAX_SOURCE_SEGMENT_LENGTH].strip("-")
    return f"{pipeline_run_id}-{segment or 'source'}-a{attempt}-{digest}"
