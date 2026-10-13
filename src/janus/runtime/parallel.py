"""Spawned batch execution with a session-free bootstrap per source attempt."""

from __future__ import annotations

import importlib
import json
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from types import FrameType
from typing import TYPE_CHECKING, Any, Protocol, cast

from janus.observability.metrics.pipeline import emit_pipeline_metrics
from janus.orchestration import BatchPlanner
from janus.orchestration.persistence import PipelineSummaryStore
from janus.orchestration.plans import BatchPlan, BatchPlanRequest, PlannedSource
from janus.orchestration.results import PipelineOutcome, SourceAttempt, SourceOutcome
from janus.orchestration.scheduling import ReadyQueue
from janus.orchestration.timing import ClockSample, PipelineClock
from janus.registry import SourceRegistry, load_registry
from janus.runtime.batch import (
    BatchExecutionInterrupted,
    BatchExecutor,
    SourceExecution,
    SourceExecutionService,
    _apply_budget_policy,
    _completed_outcomes,
    _partial_execution,
    _preflight,
    execute_source_attempt,
)
from janus.runtime.executor import SourceExecutor
from janus.runtime.locking import source_state_lock_path
from janus.runtime.parallel_results import (
    MAX_ENVELOPE_BYTES,  # noqa: F401 — preserve the worker transport import surface
    WORKER_CRASH_PHASE,  # noqa: F401 — public failure phase
    WorkerCrashed,  # noqa: F401 — preserve the worker error import surface
    _attempt_from_worker,
    _read_envelope,
    _WorkerHandle,
)
from janus.runtime.parallel_signals import parent_signal_handlers
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.utils.environment import RuntimeLocation, read_spark_write_defaults
from janus.utils.logging import StructuredLogger, build_structured_logger
from janus.utils.storage import StorageLayout

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
        # The parent applies batch budget policy after decoding the envelope.
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


MIN_PARALLEL_WORKERS = 2


def _spawn_worker(task: WorkerTask) -> _WorkerHandle:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(task, writer))
    try:
        process.start()
    except BaseException:
        reader.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join()
        process.close()
        raise
    finally:
        writer.close()
    return _WorkerHandle(task, process, reader)


def _wait_any(handles: list[_WorkerHandle], *, timeout: float | None = None) -> list[_WorkerHandle]:
    from multiprocessing.connection import wait

    ready = wait(
        [h.sentinel for h in handles] + [h.reader for h in handles if not h.result_received],
        timeout,
    )
    return [h for h in handles if h.sentinel in ready or h.reader in ready]


@dataclass(slots=True)
class ParallelBatchExecutor:
    """Run ready nodes in spawned processes through the shared attempt seam.

    Parent signal handlers cover dispatch and shutdown only. When embedded off the
    main thread, existing handlers remain installed and one diagnostic is logged.
    """

    max_parallel: int
    clock: PipelineClock = field(default_factory=PipelineClock)
    summary_store_factory: Callable[[StorageLayout], PipelineSummaryStore] = PipelineSummaryStore
    execution_factory: str = DEFAULT_EXECUTION_FACTORY
    lock_wait_seconds: float = 0.0
    interrupt_grace_seconds: float = 30.0
    logger: StructuredLogger | None = None
    process_starter: Callable[[WorkerTask], _WorkerHandle] = _spawn_worker

    def __post_init__(self) -> None:
        if self.max_parallel < MIN_PARALLEL_WORKERS:
            raise ValueError(
                "ParallelBatchExecutor needs max_parallel >= 2; use BatchExecutor for 1"
            )
        if not 0 < self.interrupt_grace_seconds < float("inf"):
            raise ValueError("interrupt_grace_seconds must be finite and greater than zero")

    def execute(
        self,
        plan: BatchPlan,
        registry: SourceRegistry,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
        *,
        summary_store: PipelineSummaryStore | None = None,
    ) -> PipelineOutcome:
        import multiprocessing  # noqa: F401 — never imported by sequential composition

        layout = StorageLayout.from_environment_config(
            environment_config, plan.request.project_root
        )
        store = summary_store or self.summary_store_factory(layout)
        _preflight(plan, registry)
        store.assert_available(plan.request.pipeline_run_id)
        lock_keys = {
            source.source_id: str(
                Path(
                    os.path.realpath(
                        source_state_lock_path(source.require_planned_run().plan, layout)
                    )
                )
            )
            for source in plan.sources
            if source.is_planned
        }
        queue = ReadyQueue(plan, self.max_parallel, lock_keys)
        outcomes: dict[str, SourceOutcome | None] = dict.fromkeys(plan.source_ids)
        running: dict[str, _WorkerHandle] = {}
        started: dict[str, ClockSample] = {}
        pipeline_started = self.clock.start()
        with parent_signal_handlers(self._log):
            try:
                while not queue.is_finished(outcomes):
                    self._dispatch(
                        queue, outcomes, running, started, environment_config, resolved_paths
                    )
                    if running:
                        self._collect(plan, outcomes, running, started)
                    elif not queue.is_finished(outcomes):
                        raise RuntimeError(
                            "Batch scheduler has unfinished nodes but no runnable work"
                        )
            except KeyboardInterrupt as interruption:
                self._log(
                    "parallel_batch_interrupted",
                    running=sorted(running),
                    grace_seconds=self.interrupt_grace_seconds,
                )
                self._stop_workers(
                    running,
                    collect=lambda timeout: self._collect(
                        plan, outcomes, running, started, timeout=timeout, interrupted=True
                    ),
                )
                partial = _partial_execution(plan, outcomes, self.clock.finish(pipeline_started))
                self._log(
                    "parallel_batch_partial",
                    completed_source_ids=[s.source_id for s in partial.source_outcomes],
                    pending_source_ids=partial.pending_source_ids,
                )
                raise BatchExecutionInterrupted(partial) from interruption
            except BaseException:
                self._stop_workers(running)
                raise
        outcome = PipelineOutcome.from_plan(
            plan,
            timing=self.clock.finish(pipeline_started),
            sources=_completed_outcomes(plan, outcomes),
            max_parallel=self.max_parallel,
        )
        persisted = store.persist(outcome)
        emit_pipeline_metrics(persisted, environment_config, resolved_paths, logger=self.logger)
        return persisted

    def _dispatch(
        self,
        queue: ReadyQueue,
        outcomes: dict[str, SourceOutcome | None],
        running: dict[str, _WorkerHandle],
        started: dict[str, ClockSample],
        config: Mapping[str, Any],
        paths: Mapping[str, RuntimeLocation],
    ) -> None:
        while (decision := queue.next_decision(outcomes, running)) is not None:
            source = decision.planned_source
            if decision.kind == "skip":
                outcomes[source.source_id] = SourceOutcome.skipped(
                    source,
                    direct_blocking_upstream_ids=decision.blockers,
                    root_failed_source_ids=decision.roots,
                )
            elif decision.kind == "planning_failure":
                outcomes[source.source_id] = SourceOutcome.from_planning_failure(source)
            else:
                task = self._task(queue.plan.request, source, config, paths)
                started[source.source_id] = self.clock.start()
                handle = self.process_starter(task)
                running[source.source_id] = handle
                self._log(
                    "parallel_worker_dispatched",
                    source_id=source.source_id,
                    run_id=source.run_id,
                    worker_pid=handle.pid,
                    running=len(running),
                )

    def _task(
        self,
        request: BatchPlanRequest,
        source: PlannedSource,
        config: Mapping[str, Any],
        paths: Mapping[str, RuntimeLocation],
    ) -> WorkerTask:
        contract = source.require_planned_run().plan.data_contract
        bindings = self.logger.context if self.logger else {}
        return WorkerTask(
            request=request,
            source_id=source.source_id,
            run_id=source.run_id,
            attempt=source.attempt,
            expected_config_version=source.config_version,
            expected_schema_version=contract.schema_version if contract else None,
            expected_upstream_ids=source.upstream_ids,
            environment_config=config,
            resolved_paths=paths,
            log_level=str(config.get("runtime", {}).get("log_level", "INFO")),
            log_bindings=tuple((key, str(value)) for key, value in bindings.items()),
            lock_wait_seconds=self.lock_wait_seconds,
            execution_factory=self.execution_factory,
        )

    def _collect(
        self,
        plan: BatchPlan,
        outcomes: dict[str, SourceOutcome | None],
        running: dict[str, _WorkerHandle],
        started: dict[str, ClockSample],
        *,
        timeout: float | None = None,
        interrupted: bool = False,
    ) -> None:
        handles = list(running.values())
        ready = _wait_any(handles) if timeout is None else _wait_any(handles, timeout=timeout)
        for handle in ready:
            if handle.envelope is None and not handle.result_received:
                _read_envelope(handle)
            if interrupted:
                handle.join(timeout=0)
            else:
                handle.join()
            source_id = handle.request.source_id
            if outcomes[source_id] is None and (handle.envelope is not None or not interrupted):
                source = plan.source(source_id)
                attempt = _attempt_from_worker(handle, self.clock.finish(started.pop(source_id)))
                attempt = _apply_budget_policy(attempt, source, plan.request)
                outcomes[source_id] = SourceOutcome.from_attempts(source, (attempt,))
            if interrupted and handle.is_alive():
                continue
            worker = handle.envelope["worker"] if handle.envelope else {}
            self._log(
                "parallel_worker_finished",
                source_id=source_id,
                worker_pid=handle.pid,
                exit_code=handle.exitcode,
                spark_started=worker.get("spark_started"),
                spark_application_id=worker.get("spark_application_id"),
            )
            handle.close()
            del running[source_id]

    def _stop_workers(
        self,
        running: dict[str, _WorkerHandle],
        *,
        collect: Callable[[float], None] | None = None,
    ) -> None:
        try:
            for handle in running.values():
                if handle.is_alive():
                    handle.terminate()
            deadline = monotonic() + self.interrupt_grace_seconds
            if collect is not None:
                while running and (remaining := deadline - monotonic()) > 0:
                    collect(remaining)
            else:
                for handle in running.values():
                    handle.join(timeout=max(0.0, deadline - monotonic()))
        except KeyboardInterrupt:
            self._log("parallel_batch_interrupted_again")
        finally:
            # Signal every survivor before reaping any of them, even if collection fails.
            for handle in running.values():
                if handle.is_alive():
                    handle.kill()
            for source_id, handle in tuple(running.items()):
                handle.join(timeout=5)
                handle.close()
                del running[source_id]

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger:
            self.logger.info(event, **fields)


def build_batch_executor(
    *,
    max_parallel: int,
    logger: StructuredLogger | None,
    lock_wait_seconds: float = 0.0,
) -> BatchExecutor | ParallelBatchExecutor:
    """Choose the runner once; one slot preserves the established sequential path."""
    if max_parallel == 1:
        return BatchExecutor(
            source_execution=SourceExecutionService(
                executor=SourceExecutor(logger=logger, lock_wait_seconds=lock_wait_seconds)
            ),
            logger=logger,
        )
    return ParallelBatchExecutor(
        max_parallel=max_parallel, logger=logger, lock_wait_seconds=lock_wait_seconds
    )
