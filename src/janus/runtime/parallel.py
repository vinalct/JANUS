"""Spawn-safe, session-free bootstrap for one source attempt per worker."""

from __future__ import annotations

import importlib
import json
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Any, Protocol, cast

from janus.orchestration import BatchPlanner
from janus.orchestration.plans import BatchPlanRequest, PlannedSource
from janus.orchestration.results import SourceAttempt
from janus.orchestration.timing import PipelineClock
from janus.registry import load_registry
from janus.runtime.batch import SourceExecution, SourceExecutionService, execute_source_attempt
from janus.runtime.executor import SourceExecutor
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.utils.environment import RuntimeLocation, read_spark_write_defaults
from janus.utils.logging import StructuredLogger, build_structured_logger

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

SNAPSHOT_DRIFT_PHASE = "snapshot_drift"
DEFAULT_EXECUTION_FACTORY = "janus.runtime.parallel:default_execution_factory"
_DIGEST_DISPLAY_LENGTH = 12


class SnapshotDriftError(RuntimeError):
    """A worker's reloaded node disagrees with the parent's pinned snapshot."""


@dataclass(frozen=True, slots=True)
class WorkerTask:
    """Transport one attempt's identity and prepared runtime as picklable values."""

    request: BatchPlanRequest
    source_id: str
    run_id: str
    attempt: int
    expected_config_version: str
    expected_schema_version: str | None
    expected_upstream_ids: tuple[str, ...]
    environment_config: Mapping[str, Any]
    resolved_paths: Mapping[str, RuntimeLocation]
    log_level: str
    log_bindings: tuple[tuple[str, str], ...]
    lock_wait_seconds: float
    execution_factory: str = DEFAULT_EXECUTION_FACTORY


@dataclass(slots=True)
class _ProviderRecorder:
    """Observe the executor's provider without acquiring a session."""

    provider: SparkSessionProvider | None = None

    def factory(
        self,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
        logger: StructuredLogger | None,
    ) -> SparkSessionProvider:
        self.provider = SparkSessionProvider(environment_config, resolved_paths, logger)
        return self.provider

    @property
    def was_started(self) -> bool:
        return bool(self.provider and self.provider.was_started)

    @property
    def application_id(self) -> str | None:
        return getattr(self.provider, "application_id", None)


class _ExecutionFactory(Protocol):
    def __call__(
        self,
        logger: StructuredLogger,
        recorder: _ProviderRecorder,
        lock_wait_seconds: float,
    ) -> SourceExecution: ...


def default_execution_factory(
    logger: StructuredLogger,
    recorder: _ProviderRecorder,
    lock_wait_seconds: float,
) -> SourceExecutionService:
    """Build the established execution service with one lazy provider lifetime."""
    return SourceExecutionService(
        executor=SourceExecutor(logger=logger, lock_wait_seconds=lock_wait_seconds),
        provider_factory=recorder.factory,
    )


def _resolve_execution_factory(reference: str) -> _ExecutionFactory:
    """Resolve only explicit JANUS or test factory references, never caller code."""
    module_name, separator, function_name = reference.partition(":")
    if (
        not separator
        or not module_name.startswith(("janus.", "tests."))
        or not all(part.isidentifier() for part in module_name.split("."))
        or not function_name.isidentifier()
    ):
        raise ValueError(f"Unsupported worker execution factory: {reference!r}")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"Worker execution factory {reference!r} is not callable")
    return cast(_ExecutionFactory, factory)


def run_worker(task: WorkerTask, connection: Connection) -> None:
    """Run one spawned attempt, send one UTF-8 JSON envelope, and close the pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        logger = _worker_logger(task)
        logger.info("parallel_worker_started")
        recorder = _ProviderRecorder()
        try:
            attempt = _attempt(task, logger, recorder)
        except KeyboardInterrupt:
            logger.warning("parallel_worker_interrupted")
            return
        envelope = {
            "attempt": attempt.to_summary(),
            "worker": {
                "pid": os.getpid(),
                "spark_started": recorder.was_started,
                "spark_application_id": recorder.application_id,
            },
        }
        connection.send_bytes(json.dumps(envelope, sort_keys=True).encode("utf-8"))
    finally:
        connection.close()


def _worker_logger(task: WorkerTask) -> StructuredLogger:
    bindings: dict[str, Any] = dict(task.log_bindings)
    bindings.update(
        environment=task.request.environment,
        project_root=str(task.request.project_root),
        pipeline_run_id=task.request.pipeline_run_id,
        source_id=task.source_id,
        run_id=task.run_id,
        worker_pid=os.getpid(),
    )
    return build_structured_logger("janus.parallel.worker", level=task.log_level).bind(**bindings)


def _attempt(
    task: WorkerTask, logger: StructuredLogger, recorder: _ProviderRecorder
) -> SourceAttempt:
    """Reload and verify a node before constructing its execution service."""
    clock = PipelineClock()
    started = clock.start()
    try:
        snapshot = load_registry(
            task.request.project_root,
            write_defaults=read_spark_write_defaults(task.environment_config),
        )
        plan = BatchPlanner().plan(task.request, registry=snapshot)
        planned_source = plan.source(task.source_id)
        _verify_source(task, planned_source)
    except Exception as error:
        drift = (
            error
            if isinstance(error, SnapshotDriftError)
            else SnapshotDriftError(
                f"Source {task.source_id!r} could not verify the parent's snapshot: "
                f"{type(error).__name__}: {error}; nothing was executed"
            )
        )
        return SourceAttempt.failed_from_exception(
            source_id=task.source_id,
            run_id=task.run_id,
            attempt=task.attempt,
            timing=clock.finish(started),
            error=drift,
            phase=SNAPSHOT_DRIFT_PHASE,
        )

    execution = _resolve_execution_factory(task.execution_factory)(
        logger, recorder, task.lock_wait_seconds
    )
    return execute_source_attempt(
        planned_source,
        task.environment_config,
        task.resolved_paths,
        source_execution=execution,
        request=task.request,
    )


def _verify_source(task: WorkerTask, source: PlannedSource) -> None:
    differences: list[str] = []
    if source.config_version != task.expected_config_version:
        differences.append(
            f"config_version {_display_digest(task.expected_config_version)} → "
            f"{_display_digest(source.config_version)}"
        )
    if source.failure is not None:
        differences.append(
            f"worker planning failed and disagreed with the parent's executable plan: "
            f"{source.failure.error_type}: {source.failure.reason}"
        )
    else:
        contract = source.require_planned_run().plan.data_contract
        schema_version = contract.schema_version if contract is not None else None
        if schema_version != task.expected_schema_version:
            differences.append(
                f"schema_version {_display_digest(task.expected_schema_version)} → "
                f"{_display_digest(schema_version)}"
            )
    for field_name, expected, actual in (
        ("run_id", task.run_id, source.run_id),
        ("attempt", task.attempt, source.attempt),
        ("upstream_ids", task.expected_upstream_ids, source.upstream_ids),
    ):
        if expected != actual:
            differences.append(f"{field_name} {expected!r} → {actual!r}")
    if differences:
        raise SnapshotDriftError(
            f"Source {task.source_id!r} changed after the batch was planned: "
            + "; ".join(differences)
            + "; nothing was executed"
        )


def _display_digest(version: str | None) -> str:
    return version[:_DIGEST_DISPLAY_LENGTH] if version is not None else "None"


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt
