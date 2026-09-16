"""Sequential batch execution with dependency-scoped failure isolation.

Planning owns identity, selection, and ordering. This module starts only after a
validated batch plan exists, delegates every runnable node to the existing source
executor, and lets a failed node block only its descendants.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from janus.orchestration.persistence import PipelineSummaryStore
from janus.orchestration.plans import BatchPlan, PlannedSource
from janus.orchestration.results import (
    FailureDetails,
    PipelineOutcome,
    SourceAttempt,
    SourceOutcome,
)
from janus.orchestration.timing import ExecutionTiming, PipelineClock
from janus.planner import PlannedRun
from janus.registry import SourceRegistry
from janus.runtime.executor import ExecutedRun, SourceExecutor
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.utils.environment import RuntimeLocation
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout


class SparkProviderFactory(Protocol):
    """Build one lazy compute provider for one source attempt."""

    def __call__(
        self,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
        logger: StructuredLogger | None,
    ) -> SparkSessionProvider: ...


class SourceExecution(Protocol):
    """The shared source-attempt seam used by batch and external adapters."""

    def execute(
        self,
        planned_run: PlannedRun,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
    ) -> ExecutedRun: ...


def _new_spark_provider(
    environment_config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    logger: StructuredLogger | None,
) -> SparkSessionProvider:
    return SparkSessionProvider(environment_config, resolved_paths, logger)


@dataclass(slots=True)
class SourceExecutionService:
    """Delegate one source to SourceExecutor with an isolated provider lifetime."""

    executor: SourceExecutor = field(default_factory=SourceExecutor)
    provider_factory: SparkProviderFactory = _new_spark_provider

    def execute(
        self,
        planned_run: PlannedRun,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
    ) -> ExecutedRun:
        """Execute once and retain execution and cleanup failures when both occur."""
        provider = self.provider_factory(
            environment_config,
            resolved_paths,
            self.executor.logger,
        )
        cleanup_finished = False
        try:
            try:
                executed_run = self.executor.execute(
                    planned_run,
                    provider,
                    environment_config,
                )
            except Exception as execution_error:
                cleanup_error = _stop_provider(provider)
                cleanup_finished = True
                if cleanup_error is not None:
                    raise SourceExecutionAndCleanupError(
                        execution_error,
                        cleanup_error,
                    ) from execution_error
                raise

            cleanup_error = _stop_provider(provider)
            cleanup_finished = True
            if cleanup_error is not None:
                raise SourceCleanupError(cleanup_error, executed_run) from cleanup_error
            return executed_run
        finally:
            if not cleanup_finished:
                # On interruption, cleanup still runs, but an ordinary teardown error
                # must not replace KeyboardInterrupt or SystemExit.
                cleanup_error = _stop_provider(provider)
                active_error = sys.exception()
                if cleanup_error is not None and active_error is not None:
                    active_error.add_note(
                        "Source cleanup also failed with "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )


class SourceCleanupError(RuntimeError):
    """Cleanup failed after SourceExecutor returned a source result."""

    def __init__(self, cleanup_error: Exception, executed_run: ExecutedRun) -> None:
        self.cleanup_error = cleanup_error
        self.executed_run = executed_run
        super().__init__(
            f"Source cleanup failed with {type(cleanup_error).__name__}: {cleanup_error}"
        )


class SourceExecutionAndCleanupError(RuntimeError):
    """Execution raised and its cleanup backstop independently failed."""

    def __init__(
        self,
        execution_error: Exception,
        cleanup_error: Exception,
    ) -> None:
        self.execution_error = execution_error
        self.cleanup_error = cleanup_error
        super().__init__(
            f"Source execution failed with {type(execution_error).__name__}: "
            f"{execution_error}; cleanup also failed with "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


@dataclass(frozen=True, slots=True)
class PartialBatchExecution:
    """Evidence completed before an interrupted batch stopped scheduling sources."""

    pipeline_run_id: str
    timing: ExecutionTiming
    source_outcomes: tuple[SourceOutcome, ...]
    pending_source_ids: tuple[str, ...]


class BatchExecutionInterrupted(KeyboardInterrupt):
    """A user interruption carrying partial evidence, never a completed summary."""

    def __init__(self, partial: PartialBatchExecution) -> None:
        self.partial = partial
        super().__init__(
            f"Pipeline {partial.pipeline_run_id!r} was interrupted after "
            f"{len(partial.source_outcomes)} source outcome(s)"
        )


class BatchExecutionPreflightError(ValueError):
    """The supplied plan and registry snapshot do not describe the same batch."""


def execute_source_attempt(
    planned_source: PlannedSource,
    environment_config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    source_execution: SourceExecution | None = None,
    clock: PipelineClock | None = None,
) -> SourceAttempt:
    """Execute and translate one planned source through the shared runtime seam.

    Batch runners and external orchestration adapters use this function so cleanup
    handling, returned failures, raised exceptions, timing, and retained evidence have
    one interpretation. Scheduling and retry decisions remain the caller's concern.
    """
    execution = source_execution or SourceExecutionService()
    execution_clock = clock or PipelineClock()
    started = execution_clock.start()
    try:
        executed_run = execution.execute(
            planned_source.require_planned_run(),
            environment_config,
            resolved_paths,
        )
    except Exception as exc:
        return _attempt_from_exception(
            planned_source,
            execution_clock.finish(started),
            exc,
        )

    timing = execution_clock.finish(started)
    try:
        return SourceAttempt.from_executed_run(
            executed_run,
            timing=timing,
            source_id=planned_source.source_id,
            run_id=planned_source.run_id,
            attempt=planned_source.attempt,
        )
    except Exception as exc:
        return SourceAttempt(
            source_id=planned_source.source_id,
            run_id=planned_source.run_id,
            attempt=planned_source.attempt,
            status="failed",
            timing=timing,
            failure=FailureDetails.from_exception(exc, phase="result_translation"),
        )


@dataclass(slots=True)
class BatchExecutor:
    """Execute a validated batch once, sequentially and in its existing DAG order."""

    source_execution: SourceExecution = field(default_factory=SourceExecutionService)
    clock: PipelineClock = field(default_factory=PipelineClock)
    summary_store_factory: Callable[[StorageLayout], PipelineSummaryStore] = PipelineSummaryStore

    def execute(
        self,
        plan: BatchPlan,
        registry: SourceRegistry,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
        *,
        summary_store: PipelineSummaryStore | None = None,
    ) -> PipelineOutcome:
        """Run every eligible source and persist the complete aggregate outcome."""
        store = summary_store or self.summary_store_factory(
            StorageLayout.from_environment_config(
                environment_config,
                plan.request.project_root,
            )
        )
        _preflight(plan, registry)
        store.assert_available(plan.request.pipeline_run_id)

        outcomes: dict[str, SourceOutcome | None] = dict.fromkeys(plan.source_ids)
        pipeline_started = self.clock.start()
        try:
            for planned_source in plan.sources:
                blockers = _blocking_upstreams(plan, planned_source, outcomes)
                if blockers:
                    outcomes[planned_source.source_id] = SourceOutcome.skipped(
                        planned_source,
                        direct_blocking_upstream_ids=blockers,
                        root_failed_source_ids=_root_failures(blockers, outcomes),
                    )
                    continue

                if planned_source.failure is not None:
                    outcomes[planned_source.source_id] = SourceOutcome.from_planning_failure(
                        planned_source
                    )
                    continue

                attempt = self._execute_source(
                    planned_source,
                    environment_config,
                    resolved_paths,
                )
                outcomes[planned_source.source_id] = SourceOutcome.from_attempts(
                    planned_source,
                    (attempt,),
                )
        except KeyboardInterrupt as interruption:
            partial = _partial_execution(
                plan,
                outcomes,
                self.clock.finish(pipeline_started),
            )
            raise BatchExecutionInterrupted(partial) from interruption

        outcome = PipelineOutcome.from_plan(
            plan,
            timing=self.clock.finish(pipeline_started),
            sources=_completed_outcomes(plan, outcomes),
        )
        return store.persist(outcome)

    def _execute_source(
        self,
        planned_source: PlannedSource,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
    ) -> SourceAttempt:
        """Measure and translate exactly one call through the shared source service."""
        return execute_source_attempt(
            planned_source,
            environment_config,
            resolved_paths,
            source_execution=self.source_execution,
            clock=self.clock,
        )


def _preflight(plan: BatchPlan, registry: SourceRegistry) -> None:
    """Prove the plan still belongs to the supplied validated snapshot before writes."""
    issues: list[str] = []
    if plan.request.project_root != registry.project_root:
        issues.append(
            f"plan project root {plan.request.project_root} does not match registry root "
            f"{registry.project_root}"
        )

    try:
        snapshot_graph = registry.graph.subgraph(plan.source_ids)
    except LookupError as exc:
        issues.append(str(exc))
    else:
        if snapshot_graph != plan.selection.graph:
            issues.append("the plan dependency graph differs from the registry snapshot")
        if plan.selection.graph.topological_order() != plan.source_ids:
            issues.append("the plan source order is not the graph's deterministic order")

    for planned_source in plan.sources:
        try:
            source_config = registry.get_source(planned_source.source_id)
        except LookupError as exc:
            issues.append(str(exc))
            continue
        if source_config.config_path != planned_source.config_path:
            issues.append(
                f"source {planned_source.source_id!r} was planned from "
                f"{planned_source.config_path}, not snapshot path {source_config.config_path}"
            )
        if planned_source.upstream_ids != plan.upstreams_of(planned_source.source_id):
            issues.append(
                f"source {planned_source.source_id!r} carries upstreams "
                f"{list(planned_source.upstream_ids)}, not the plan graph's "
                f"{list(plan.upstreams_of(planned_source.source_id))}"
            )

    if issues:
        rendered = "\n".join(f"- {issue}" for issue in issues)
        raise BatchExecutionPreflightError(
            f"Batch plan and registry snapshot do not match; nothing was executed:\n{rendered}"
        )


def _blocking_upstreams(
    plan: BatchPlan,
    planned_source: PlannedSource,
    outcomes: Mapping[str, SourceOutcome | None],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for upstream_id in plan.upstreams_of(planned_source.source_id):
        upstream = outcomes[upstream_id]
        if upstream is None:
            raise BatchExecutionPreflightError(
                f"Source {planned_source.source_id!r} was reached before upstream "
                f"{upstream_id!r}; nothing further was scheduled"
            )
        if upstream.status in {"failed", "skipped"}:
            blockers.append(upstream_id)
    return tuple(blockers)


def _root_failures(
    blockers: tuple[str, ...],
    outcomes: Mapping[str, SourceOutcome | None],
) -> tuple[str, ...]:
    root_ids: set[str] = set()
    for source_id in blockers:
        outcome = outcomes[source_id]
        assert outcome is not None
        if outcome.status == "failed":
            root_ids.add(source_id)
            continue
        assert outcome.skip is not None
        root_ids.update(outcome.skip.root_failed_source_ids)
    return tuple(sorted(root_ids))


def _attempt_from_exception(
    planned_source: PlannedSource,
    timing: ExecutionTiming,
    error: Exception,
) -> SourceAttempt:
    if not isinstance(error, SourceCleanupError):
        return SourceAttempt.failed_from_exception(
            source_id=planned_source.source_id,
            run_id=planned_source.run_id,
            attempt=planned_source.attempt,
            timing=timing,
            error=error,
        )

    try:
        returned = SourceAttempt.from_executed_run(
            error.executed_run,
            timing=timing,
            source_id=planned_source.source_id,
            run_id=planned_source.run_id,
            attempt=planned_source.attempt,
        )
        evidence = returned.evidence
        returned_failure = returned.failure
    except Exception as translation_error:
        evidence = {}
        returned_failure = FailureDetails.from_exception(
            translation_error,
            phase="result_translation",
        )

    cleanup = FailureDetails.from_exception(error.cleanup_error, phase="cleanup")
    reason = cleanup.reason
    if returned_failure is not None:
        reason = (
            f"Source execution also failed ({returned_failure.error_type}): "
            f"{returned_failure.reason}; cleanup failed: {cleanup.reason}"
        )
    return SourceAttempt(
        source_id=planned_source.source_id,
        run_id=planned_source.run_id,
        attempt=planned_source.attempt,
        status="failed",
        timing=timing,
        evidence=evidence,
        failure=FailureDetails(
            phase="cleanup",
            error_type=cleanup.error_type,
            reason=reason,
        ),
    )


def _stop_provider(provider: SparkSessionProvider) -> Exception | None:
    failures: list[Exception] = []
    try:
        provider.stop()
    except Exception as exc:
        failures.append(exc)
    failures.extend(provider.take_cleanup_failures())
    if not failures:
        return None
    if len(failures) == 1:
        return failures[0]
    details = "; ".join(f"{type(error).__name__}: {error}" for error in failures)
    return RuntimeError(f"Multiple source cleanup failures occurred: {details}")


def _completed_outcomes(
    plan: BatchPlan,
    outcomes: Mapping[str, SourceOutcome | None],
) -> tuple[SourceOutcome, ...]:
    completed: list[SourceOutcome] = []
    for source_id in plan.source_ids:
        outcome = outcomes[source_id]
        if outcome is None:
            raise RuntimeError(f"Source {source_id!r} has no terminal batch outcome")
        completed.append(outcome)
    return tuple(completed)


def _partial_execution(
    plan: BatchPlan,
    outcomes: Mapping[str, SourceOutcome | None],
    timing: ExecutionTiming,
) -> PartialBatchExecution:
    completed = tuple(
        outcome for source_id in plan.source_ids if (outcome := outcomes[source_id]) is not None
    )
    completed_ids = {outcome.source_id for outcome in completed}
    return PartialBatchExecution(
        pipeline_run_id=plan.request.pipeline_run_id,
        timing=timing,
        source_outcomes=completed,
        pending_source_ids=tuple(
            source_id for source_id in plan.source_ids if source_id not in completed_ids
        ),
    )
