"""Dagster definitions for the self-contained orchestration example."""

from __future__ import annotations

from pathlib import Path

from dagster import DefaultScheduleStatus, Definitions, RetryPolicy, ScheduleDefinition

from janus.adapters.dagster import DagsterAdapter, build_dagster_adapter

PROJECT_ROOT = Path(__file__).resolve().parent
SCHEDULE_TIMEZONE = "America/Sao_Paulo"


def build_example_adapter(*, max_retries: int = 0) -> DagsterAdapter:
    """Build the example adapter; whole-source retries are explicit and bounded."""
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    retry_policy = RetryPolicy(max_retries=max_retries) if max_retries else None
    return build_dagster_adapter(
        PROJECT_ROOT,
        environment="example",
        retry_policy=retry_policy,
    )


dagster_adapter = build_example_adapter()

daily_example_schedule = ScheduleDefinition(
    name="janus_orchestration_example_daily",
    job=dagster_adapter.job,
    cron_schedule="0 6 * * *",
    execution_timezone=SCHEDULE_TIMEZONE,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Disabled-by-default daily execution of the local A -> B and C example.",
)

defs = Definitions(
    jobs=[dagster_adapter.job],
    schedules=[daily_example_schedule],
    sensors=list(dagster_adapter.definitions.sensors),
    metadata={
        "janus/example": "orchestration",
        "janus/snapshot_id": dagster_adapter.manifest.snapshot_id,
        "janus/source_count": len(dagster_adapter.manifest.source_order),
    },
)
