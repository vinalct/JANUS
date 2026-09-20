"""Serializable JANUS evidence carried by native Dagster step events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dagster import AssetObservation

from janus.orchestration import SourceAttempt
from janus.orchestration.plans import PlannedSource

JANUS_EVENT_METADATA_KEY = "janus/event"
JANUS_EVENT_SCHEMA_VERSION = 1


def attempt_event(pipeline_run_id: str, attempt: SourceAttempt) -> AssetObservation:
    """Return the durable event emitted before an op succeeds or raises Failure."""
    return AssetObservation(
        asset_key=("janus", "sources", attempt.source_id),
        description=f"JANUS source attempt {attempt.source_id} #{attempt.attempt}",
        metadata={
            JANUS_EVENT_METADATA_KEY: {
                "schema_version": JANUS_EVENT_SCHEMA_VERSION,
                "kind": "attempt",
                "pipeline_run_id": pipeline_run_id,
                "source_id": attempt.source_id,
                "attempt": attempt.to_summary(),
            }
        },
    )


def planning_failure_event(
    pipeline_run_id: str,
    planned_source: PlannedSource,
) -> AssetObservation:
    """Return a non-attempted planning failure event for terminal aggregation."""
    failure = planned_source.failure
    if failure is None:
        raise ValueError(f"Source {planned_source.source_id!r} has no planning failure")
    return AssetObservation(
        asset_key=("janus", "sources", planned_source.source_id),
        description=f"JANUS source planning failed for {planned_source.source_id}",
        metadata={
            JANUS_EVENT_METADATA_KEY: {
                "schema_version": JANUS_EVENT_SCHEMA_VERSION,
                "kind": "planning_failure",
                "pipeline_run_id": pipeline_run_id,
                "source_id": planned_source.source_id,
                "run_id": planned_source.run_id,
                "attempt": planned_source.attempt,
                "failure": failure.to_summary(),
            }
        },
    )


def payload_from_observation(observation: AssetObservation) -> Mapping[str, Any] | None:
    """Extract one adapter payload while ignoring unrelated asset observations."""
    value = observation.metadata.get(JANUS_EVENT_METADATA_KEY)
    if value is None:
        return None
    payload = getattr(value, "data", None)
    if not isinstance(payload, Mapping):
        raise ValueError("JANUS Dagster event metadata is not a JSON object")
    if payload.get("schema_version") != JANUS_EVENT_SCHEMA_VERSION:
        raise ValueError("JANUS Dagster event has an unsupported schema version")
    return payload
