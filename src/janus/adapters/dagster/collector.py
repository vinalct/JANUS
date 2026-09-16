"""Translate a terminal native Dagster run into the shared JANUS summary schema."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from dagster import DagsterEventType, DagsterInstance, DagsterRunStatus

from janus.orchestration import (
    ExecutionTiming,
    FailureDetails,
    PipelineOutcome,
    PipelineSummaryStore,
    SkipExplanation,
    SourceAttempt,
    SourceOutcome,
    source_attempt_run_id,
)
from janus.utils.storage import StorageLayout

from .errors import DagsterRunCollectionError
from .events import payload_from_observation
from .manifest import ADAPTER_NAME, ADAPTER_TAG, MANIFEST_TAG, SNAPSHOT_TAG, DagsterRunManifest

_TERMINAL_RUN_STATUSES = frozenset(
    {DagsterRunStatus.SUCCESS, DagsterRunStatus.FAILURE, DagsterRunStatus.CANCELED}
)
SummaryStoreFactory = Callable[[StorageLayout], PipelineSummaryStore]


def collect_dagster_run(
    instance: DagsterInstance,
    run_id: str,
    *,
    summary_store_factory: SummaryStoreFactory = PipelineSummaryStore,
) -> PipelineOutcome:
    """Collect and idempotently persist one terminal Dagster run.

    The run-carried manifest makes this independent of success-gated graph nodes and of
    the current code-location snapshot. Repeated status delivery replaces the same
    deterministic summary rather than adding attempts or changing pipeline identity.
    """
    dagster_run = instance.get_run_by_id(run_id)
    if dagster_run is None:
        raise DagsterRunCollectionError(f"Dagster run {run_id!r} was not found")
    if dagster_run.status not in _TERMINAL_RUN_STATUSES:
        raise DagsterRunCollectionError(
            f"Dagster run {run_id!r} is {dagster_run.status.value}, not terminal"
        )
    manifest = _manifest_from_run(dagster_run.tags, dagster_run.job_name)
    run_stats = instance.get_run_stats(run_id)
    timing = _timing_from_epoch(run_stats.start_time, run_stats.end_time, "pipeline")
    step_stats = {stats.step_key: stats for stats in instance.get_run_step_stats(run_id)}
    failure_messages = _failure_messages(instance, run_id)
    event_payloads = _event_payloads(instance, run_id)
    planned_at = timing.started_at
    assert planned_at is not None

    outcomes: dict[str, SourceOutcome] = {}
    for source_id in manifest.source_order:
        op_name = dict(manifest.op_names)[source_id]
        stats = step_stats.get(op_name)
        payloads = _step_payloads(event_payloads.get(op_name, ()), run_id, source_id)
        attempts = tuple(
            sorted(
                (
                    _attempt_from_payload(payload)
                    for payload in payloads
                    if payload["kind"] == "attempt"
                ),
                key=lambda attempt: attempt.attempt,
            )
        )
        planning_failures = tuple(
            payload for payload in payloads if payload["kind"] == "planning_failure"
        )
        outcomes[source_id] = _source_outcome(
            manifest,
            run_id,
            source_id,
            stats,
            attempts,
            planning_failures,
            outcomes,
            failure_messages.get(op_name),
        )

    outcome = PipelineOutcome(
        pipeline_run_id=run_id,
        attempt=1,
        trigger=ADAPTER_NAME,
        environment=manifest.environment,
        planned_at=planned_at,
        requested_tags=manifest.requested_tags,
        requested_domains=manifest.requested_domains,
        root_ids=manifest.root_ids,
        included_upstream_ids=manifest.included_upstream_ids,
        source_order=manifest.source_order,
        edges=manifest.edges,
        config_versions=manifest.config_versions,
        timing=timing,
        sources=tuple(outcomes[source_id] for source_id in manifest.source_order),
    )
    return summary_store_factory(manifest.storage_layout()).persist(
        outcome,
        allow_existing=True,
    )


def _manifest_from_run(tags: Mapping[str, str], job_name: str) -> DagsterRunManifest:
    if tags.get(ADAPTER_TAG) != ADAPTER_NAME:
        raise DagsterRunCollectionError("Dagster run is not tagged as a JANUS adapter run")
    encoded = tags.get(MANIFEST_TAG)
    if encoded is None:
        raise DagsterRunCollectionError("Dagster run does not carry a JANUS run manifest")
    try:
        manifest = DagsterRunManifest.from_json(encoded)
    except ValueError as exc:
        raise DagsterRunCollectionError(str(exc)) from exc
    if manifest.snapshot_id != tags.get(SNAPSHOT_TAG):
        raise DagsterRunCollectionError("Dagster run manifest does not match its snapshot id")
    if manifest.job_name != job_name:
        raise DagsterRunCollectionError(
            f"Dagster run job {job_name!r} does not match manifest job {manifest.job_name!r}"
        )
    return manifest


def _step_payloads(
    candidates: Sequence[Mapping[str, Any]],
    run_id: str,
    source_id: str,
) -> tuple[Mapping[str, Any], ...]:
    payloads: list[Mapping[str, Any]] = []
    for payload in candidates:
        if payload.get("pipeline_run_id") != run_id or payload.get("source_id") != source_id:
            raise DagsterRunCollectionError(
                f"Terminal evidence for source {source_id!r} has mismatched identity"
            )
        kind = payload.get("kind")
        if kind not in {"attempt", "planning_failure"}:
            raise DagsterRunCollectionError(
                f"Terminal evidence for source {source_id!r} has unknown kind {kind!r}"
            )
        payloads.append(payload)
    return tuple(payloads)


def _event_payloads(
    instance: DagsterInstance,
    run_id: str,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    by_step: dict[str, list[Mapping[str, Any]]] = {}
    entries = instance.all_logs(run_id, of_type=DagsterEventType.ASSET_OBSERVATION)
    for entry in entries:
        event = entry.dagster_event
        data = event.event_specific_data if event is not None else None
        observation = getattr(data, "asset_observation", None)
        if observation is None or entry.step_key is None:
            continue
        try:
            payload = payload_from_observation(observation)
        except ValueError as exc:
            raise DagsterRunCollectionError(
                f"Could not read terminal evidence for step {entry.step_key!r}: {exc}"
            ) from exc
        if payload is not None:
            by_step.setdefault(entry.step_key, []).append(payload)
    return {step_key: tuple(payloads) for step_key, payloads in by_step.items()}


def _attempt_from_payload(payload: Mapping[str, Any]) -> SourceAttempt:
    value = payload.get("attempt")
    if not isinstance(value, Mapping):
        raise DagsterRunCollectionError("JANUS attempt event has no attempt object")
    timing = _timing_from_summary(value.get("timing"))
    failure_value = value.get("failure")
    failure = _failure_from_summary(failure_value) if failure_value is not None else None
    evidence = value.get("evidence", {})
    if not isinstance(evidence, Mapping):
        raise DagsterRunCollectionError("JANUS attempt evidence is not an object")
    try:
        return SourceAttempt(
            source_id=_required_string(value, "source_id"),
            run_id=_required_string(value, "run_id"),
            attempt=int(value["attempt"]),
            status=_required_string(value, "status"),
            timing=timing,
            evidence=evidence,
            failure=failure,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DagsterRunCollectionError(f"Invalid JANUS attempt event: {exc}") from exc


def _source_outcome(
    manifest: DagsterRunManifest,
    run_id: str,
    source_id: str,
    stats: Any,
    attempts: tuple[SourceAttempt, ...],
    planning_failures: Sequence[Mapping[str, Any]],
    completed: Mapping[str, SourceOutcome],
    native_failure: str | None,
) -> SourceOutcome:
    upstream_ids = manifest.upstreams_of(source_id)
    blockers = tuple(
        upstream_id for upstream_id in upstream_ids if completed[upstream_id].status != "succeeded"
    )
    selected_directly = source_id in manifest.root_ids
    config_version = dict(manifest.config_versions)[source_id]

    if attempts:
        attempts = _reconcile_native_failure(attempts, stats, native_failure)
        final = attempts[-1]
        return SourceOutcome(
            source_id=source_id,
            run_id=final.run_id,
            attempt=final.attempt,
            selected_directly=selected_directly,
            upstream_ids=upstream_ids,
            config_version=config_version,
            status=final.status,
            attempts=attempts,
            failure=final.failure,
        )
    if planning_failures:
        payload = planning_failures[-1]
        return SourceOutcome(
            source_id=source_id,
            run_id=_required_string(payload, "run_id"),
            attempt=int(payload["attempt"]),
            selected_directly=selected_directly,
            upstream_ids=upstream_ids,
            config_version=config_version,
            status="failed",
            failure=_failure_from_summary(payload.get("failure")),
        )
    if blockers:
        return SourceOutcome(
            source_id=source_id,
            run_id=source_attempt_run_id(
                pipeline_run_id=run_id,
                source_id=source_id,
                attempt=1,
            ),
            attempt=1,
            selected_directly=selected_directly,
            upstream_ids=upstream_ids,
            config_version=config_version,
            status="skipped",
            skip=SkipExplanation.create(
                direct_blocking_upstream_ids=blockers,
                root_failed_source_ids=_root_failures(blockers, completed),
            ),
        )

    failure = FailureDetails(
        phase="orchestration",
        error_type="DagsterStepFailure",
        reason=native_failure or f"Dagster produced no JANUS terminal evidence for {source_id}",
    )
    step_timing = _optional_step_timing(stats)
    if step_timing is not None:
        attempt_number = max(int(stats.attempts or 1), 1)
        attempt = SourceAttempt(
            source_id=source_id,
            run_id=source_attempt_run_id(
                pipeline_run_id=run_id,
                source_id=source_id,
                attempt=attempt_number,
            ),
            attempt=attempt_number,
            status="failed",
            timing=step_timing,
            failure=failure,
        )
        return SourceOutcome(
            source_id=source_id,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
            selected_directly=selected_directly,
            upstream_ids=upstream_ids,
            config_version=config_version,
            status="failed",
            attempts=(attempt,),
            failure=failure,
        )
    return SourceOutcome(
        source_id=source_id,
        run_id=source_attempt_run_id(
            pipeline_run_id=run_id,
            source_id=source_id,
            attempt=1,
        ),
        attempt=1,
        selected_directly=selected_directly,
        upstream_ids=upstream_ids,
        config_version=config_version,
        status="failed",
        failure=failure,
    )


def _reconcile_native_failure(
    attempts: tuple[SourceAttempt, ...],
    stats: Any,
    native_failure: str | None,
) -> tuple[SourceAttempt, ...]:
    status = getattr(getattr(stats, "status", None), "value", None)
    final = attempts[-1]
    if status != "FAILURE" or final.status == "failed":
        return attempts
    failure = FailureDetails(
        phase="orchestration",
        error_type="DagsterStepFailure",
        reason=native_failure or "Dagster failed the step after JANUS execution succeeded",
    )
    return (*attempts[:-1], replace(final, status="failed", failure=failure))


def _root_failures(
    blockers: Sequence[str],
    outcomes: Mapping[str, SourceOutcome],
) -> tuple[str, ...]:
    roots: set[str] = set()
    for source_id in blockers:
        outcome = outcomes[source_id]
        if outcome.status == "failed":
            roots.add(source_id)
        elif outcome.skip is not None:
            roots.update(outcome.skip.root_failed_source_ids)
    return tuple(sorted(roots))


def _failure_messages(instance: DagsterInstance, run_id: str) -> dict[str, str]:
    messages: dict[str, str] = {}
    for entry in instance.all_logs(run_id, of_type=DagsterEventType.STEP_FAILURE):
        if entry.step_key:
            messages[entry.step_key] = entry.message
    return messages


def _timing_from_summary(value: Any) -> ExecutionTiming:
    if not isinstance(value, Mapping):
        raise DagsterRunCollectionError("JANUS attempt timing is not an object")
    try:
        started = datetime.fromisoformat(_required_string(value, "started_at"))
        ended = datetime.fromisoformat(_required_string(value, "ended_at"))
        return ExecutionTiming(
            started_at=started,
            ended_at=ended,
            duration_seconds=float(value["duration_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DagsterRunCollectionError(f"Invalid JANUS attempt timing: {exc}") from exc


def _timing_from_epoch(start: float | None, end: float | None, label: str) -> ExecutionTiming:
    if start is None or end is None:
        raise DagsterRunCollectionError(f"Dagster {label} has no complete terminal timing")
    return ExecutionTiming(
        started_at=datetime.fromtimestamp(start, tz=UTC),
        ended_at=datetime.fromtimestamp(end, tz=UTC),
        duration_seconds=round(max(end - start, 0.0), 6),
    )


def _optional_step_timing(stats: Any) -> ExecutionTiming | None:
    if stats is None or stats.start_time is None or stats.end_time is None:
        return None
    return _timing_from_epoch(stats.start_time, stats.end_time, "step")


def _failure_from_summary(value: Any) -> FailureDetails:
    if not isinstance(value, Mapping):
        raise DagsterRunCollectionError("JANUS failure evidence is not an object")
    try:
        return FailureDetails(
            phase=_required_string(value, "phase"),
            error_type=_required_string(value, "error_type"),
            reason=_required_string(value, "reason"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DagsterRunCollectionError(f"Invalid JANUS failure evidence: {exc}") from exc


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item
