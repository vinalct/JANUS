"""Deterministic batch execution and dependency failure isolation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from janus.orchestration import (
    BatchPlanner,
    BatchPlanRequest,
    DuplicatePipelineRunError,
    SourcePlanFailure,
)
from janus.registry import load_registry
from janus.runtime import (
    BatchExecutionInterrupted,
    BatchExecutionPreflightError,
    BatchExecutor,
    SourceCleanupError,
    SourceExecutionService,
    SparkSessionProvider,
)
from tests.support.orchestration import build_graph_project

PLANNED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
PIPELINE_ID = "batch"


class FakeExecutedRun:
    """The established ExecutedRun surface without invoking an ingestion strategy."""

    def __init__(
        self,
        planned_run: Any,
        *,
        status: str = "succeeded",
        is_successful: bool | None = None,
        failure_reason: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.planned_run = planned_run
        self.status = status
        self._is_successful = status == "succeeded" if is_successful is None else is_successful
        self.failure_reason = failure_reason
        self.error_type = error_type

    @property
    def is_successful(self) -> bool:
        return self._is_successful

    def to_summary(self) -> dict[str, Any]:
        source_id = self.planned_run.plan.source.source_id
        return {
            "status": self.status,
            "strategy_metadata": {"source": source_id},
            "materialized_outputs": [],
            "metadata_outputs": {
                "run_metadata_path": f"/metadata/runs/{source_id}.json",
                "lineage_path": f"/metadata/lineage/{source_id}.json",
                "checkpoint_state_path": None,
                "checkpoint_history_path": None,
                "validation_report_path": None,
            },
            **({"failure_reason": self.failure_reason} if self.failure_reason is not None else {}),
            **({"error_type": self.error_type} if self.error_type is not None else {}),
        }


class ProviderSpy:
    def __init__(self, *, cleanup_error: Exception | None = None) -> None:
        self.cleanup_error = cleanup_error
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error

    def take_cleanup_failures(self) -> tuple[Exception, ...]:
        return ()


class ProviderFactorySpy:
    def __init__(self, *, cleanup_failures: dict[int, Exception] | None = None) -> None:
        self.cleanup_failures = cleanup_failures or {}
        self.providers: list[ProviderSpy] = []

    def __call__(self, _config, _paths, _logger):
        provider = ProviderSpy(cleanup_error=self.cleanup_failures.get(len(self.providers)))
        self.providers.append(provider)
        return provider


class ExecutorSpy:
    logger = None

    def __init__(self, behavior: dict[str, object] | None = None) -> None:
        self.behavior = behavior or {}
        self.calls: list[str] = []
        self.providers: list[ProviderSpy] = []

    def execute(self, planned_run, provider, _environment_config):
        source_id = planned_run.plan.source.source_id
        self.calls.append(source_id)
        self.providers.append(provider)
        action = self.behavior.get(source_id, "succeeded")
        if isinstance(action, BaseException):
            raise action
        if action == "failed":
            return FakeExecutedRun(
                planned_run,
                status="failed",
                failure_reason=f"{source_id} returned a failed extraction",
                error_type="ExtractionFailure",
            )
        if action == "status_only_success":
            return FakeExecutedRun(
                planned_run,
                status="succeeded",
                is_successful=False,
                failure_reason=f"{source_id} reported unsuccessful",
                error_type="QualityFailure",
            )
        return FakeExecutedRun(planned_run)


def _batch(tmp_path: Path, graph: str = "chain_and_peer"):
    project_root = build_graph_project(tmp_path, graph)
    registry = load_registry(project_root)
    plan = BatchPlanner().plan(
        BatchPlanRequest.create(
            environment="local",
            project_root=project_root,
            pipeline_run_id=PIPELINE_ID,
            planned_at=PLANNED_AT,
        ),
        registry=registry,
    )
    config = {
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        }
    }
    resolved_paths = {name: project_root / value for name, value in config["storage"].items()}
    return plan, registry, config, resolved_paths


def _run(
    tmp_path: Path,
    *,
    graph: str = "chain_and_peer",
    behavior: dict[str, object] | None = None,
    cleanup_failures: dict[int, Exception] | None = None,
    plan_transform=None,
):
    plan, registry, config, paths = _batch(tmp_path, graph)
    if plan_transform is not None:
        plan = plan_transform(plan)
    executor = ExecutorSpy(behavior)
    providers = ProviderFactorySpy(cleanup_failures=cleanup_failures)
    service = SourceExecutionService(
        executor=executor,
        provider_factory=providers,
    )
    outcome = BatchExecutor(source_execution=service).execute(
        plan,
        registry,
        config,
        paths,
    )
    return outcome, executor, providers


def _sources(outcome) -> dict[str, Any]:
    return {source.source_id: source for source in outcome.sources}


@pytest.mark.parametrize(
    "failure",
    ("failed", RuntimeError("A raised during extraction")),
)
def test_a_failure_skips_only_b_and_d_while_independent_c_runs(tmp_path, failure):
    outcome, executor, providers = _run(
        tmp_path,
        behavior={"A": failure},
    )
    sources = _sources(outcome)

    assert executor.calls == ["A", "C"]
    assert [source.status for source in outcome.sources] == [
        "failed",
        "skipped",
        "succeeded",
        "skipped",
    ]
    assert sources["B"].skip.direct_blocking_upstream_ids == ("A",)
    assert sources["B"].skip.root_failed_source_ids == ("A",)
    assert sources["D"].skip.direct_blocking_upstream_ids == ("B",)
    assert sources["D"].skip.root_failed_source_ids == ("A",)
    assert all(provider.stop_calls == 1 for provider in providers.providers)
    assert len(providers.providers) == 2
    assert outcome.totals() | {"duration_seconds": 0} == {
        "selected": 4,
        "expanded": 4,
        "attempted": 2,
        "succeeded": 1,
        "failed": 1,
        "skipped": 2,
        "status": "failed",
        "duration_seconds": 0,
    }
    assert outcome.summary_persistence.path is not None
    assert outcome.summary_persistence.path.exists()


def test_b_failure_after_a_success_skips_d_and_leaves_c_eligible(tmp_path):
    outcome, executor, providers = _run(
        tmp_path,
        behavior={"B": "failed"},
    )
    sources = _sources(outcome)

    assert executor.calls == ["A", "B", "C"]
    assert {source_id: source.status for source_id, source in sources.items()} == {
        "A": "succeeded",
        "B": "failed",
        "C": "succeeded",
        "D": "skipped",
    }
    assert sources["D"].skip.direct_blocking_upstream_ids == ("B",)
    assert sources["D"].skip.root_failed_source_ids == ("B",)
    assert len({id(provider) for provider in executor.providers}) == 3
    assert all(provider.stop_calls == 1 for provider in providers.providers)


def test_empty_success_stays_success_and_a_downstream_lookup_error_is_b_failure(tmp_path):
    outcome, executor, _providers = _run(
        tmp_path,
        behavior={"B": RuntimeError("upstream bronze table is absent")},
    )
    sources = _sources(outcome)

    assert executor.calls == ["A", "B", "C"]
    assert sources["A"].status == "succeeded"
    assert sources["A"].attempts[0].evidence["materialized_outputs"] == ()
    assert sources["B"].status == "failed"
    assert sources["B"].failure.error_type == "RuntimeError"
    assert sources["B"].failure.reason == "upstream bronze table is absent"
    assert sources["D"].skip.root_failed_source_ids == ("B",)


def test_fan_in_with_one_failed_and_one_successful_upstream_never_executes_consumer(
    tmp_path,
):
    outcome, executor, providers = _run(
        tmp_path,
        graph="fan_in",
        behavior={"A": "failed"},
    )
    sources = _sources(outcome)

    assert executor.calls == ["A", "B"]
    assert sources["B"].status == "succeeded"
    assert sources["D"].status == "skipped"
    assert sources["D"].skip.direct_blocking_upstream_ids == ("A",)
    assert sources["D"].skip.root_failed_source_ids == ("A",)
    assert len(providers.providers) == 2


def test_diamond_skip_deduplicates_one_root_failure_across_two_blockers(tmp_path):
    outcome, executor, providers = _run(
        tmp_path,
        graph="diamond",
        behavior={"A": "failed"},
    )
    sources = _sources(outcome)

    assert executor.calls == ["A"]
    assert sources["D"].skip.direct_blocking_upstream_ids == ("B", "C")
    assert sources["D"].skip.root_failed_source_ids == ("A",)
    assert len(providers.providers) == 1


def test_planning_failure_is_not_attempted_and_blocks_only_descendants(tmp_path):
    def fail_a(plan):
        sources = tuple(
            replace(
                source,
                planned_run=None,
                failure=SourcePlanFailure(
                    error_type="HookResolutionError",
                    reason="hook did not load",
                ),
            )
            if source.source_id == "A"
            else source
            for source in plan.sources
        )
        return replace(plan, sources=sources)

    outcome, executor, providers = _run(tmp_path, plan_transform=fail_a)
    sources = _sources(outcome)

    assert executor.calls == ["C"]
    assert sources["A"].status == "failed"
    assert sources["A"].attempted is False
    assert sources["A"].failure.phase == "planning"
    assert sources["B"].skip.root_failed_source_ids == ("A",)
    assert sources["D"].skip.root_failed_source_ids == ("A",)
    assert len(providers.providers) == 1


def test_is_successful_not_status_text_determines_a_returned_result(tmp_path):
    outcome, executor, _providers = _run(
        tmp_path,
        behavior={"A": "status_only_success"},
    )

    assert executor.calls == ["A", "C"]
    assert _sources(outcome)["A"].status == "failed"
    assert _sources(outcome)["A"].failure.error_type == "QualityFailure"


def test_cleanup_failure_becomes_a_failed_attempt_with_returned_evidence(tmp_path):
    outcome, executor, providers = _run(
        tmp_path,
        cleanup_failures={0: OSError("session stop failed")},
    )
    failed = _sources(outcome)["A"]
    attempt = failed.attempts[0]

    assert executor.calls == ["A", "C"]
    assert failed.status == "failed"
    assert attempt.failure.phase == "cleanup"
    assert attempt.failure.error_type == "OSError"
    assert attempt.evidence["status"] == "succeeded"
    assert providers.providers[0].stop_calls == 1
    assert providers.providers[1].stop_calls == 1


def test_standard_provider_reports_a_stop_error_swallowed_inside_the_executor(tmp_path):
    plan, _registry, config, paths = _batch(tmp_path)

    class BrokenSession:
        class sparkContext:
            appName = "task-06"
            master = "local"

        def stop(self) -> None:
            raise OSError("real provider stop failed")

    class InternallyStoppingExecutor:
        logger = None

        def execute(self, planned_run, provider, _environment_config):
            provider.get()
            provider.stop()
            return FakeExecutedRun(planned_run)

    session = BrokenSession()
    service = SourceExecutionService(
        executor=InternallyStoppingExecutor(),
        provider_factory=lambda _config, _paths, _logger: SparkSessionProvider(
            {},
            {},
            session_factory=lambda: session,
        ),
    )

    with pytest.raises(SourceCleanupError) as exc_info:
        service.execute(plan.source("A").require_planned_run(), config, paths)

    assert isinstance(exc_info.value.cleanup_error, OSError)
    assert exc_info.value.executed_run.is_successful


def test_execution_and_cleanup_failures_retain_both_contexts_and_continue(tmp_path):
    outcome, executor, providers = _run(
        tmp_path,
        behavior={"A": RuntimeError("extraction exploded")},
        cleanup_failures={0: OSError("session stop exploded")},
    )
    failure = _sources(outcome)["A"].failure

    assert executor.calls == ["A", "C"]
    assert failure.error_type == "SourceExecutionAndCleanupError"
    assert "RuntimeError: extraction exploded" in failure.reason
    assert "OSError: session stop exploded" in failure.reason
    assert all(provider.stop_calls == 1 for provider in providers.providers)


def test_returned_failure_and_cleanup_failure_keep_source_evidence_and_both_reasons(
    tmp_path,
):
    outcome, _executor, _providers = _run(
        tmp_path,
        behavior={"A": "failed"},
        cleanup_failures={0: OSError("cleanup unavailable")},
    )
    attempt = _sources(outcome)["A"].attempts[0]

    assert attempt.evidence["failure_reason"] == "A returned a failed extraction"
    assert "A returned a failed extraction" in attempt.failure.reason
    assert "cleanup unavailable" in attempt.failure.reason


def test_duplicate_summary_id_is_refused_before_any_source_or_provider_call(tmp_path):
    plan, registry, config, paths = _batch(tmp_path)
    first_executor = ExecutorSpy()
    first_factory = ProviderFactorySpy()
    first_runner = BatchExecutor(
        source_execution=SourceExecutionService(
            executor=first_executor,
            provider_factory=first_factory,
        )
    )
    first_runner.execute(plan, registry, config, paths)
    second_executor = ExecutorSpy()
    second_factory = ProviderFactorySpy()
    second_runner = BatchExecutor(
        source_execution=SourceExecutionService(
            executor=second_executor,
            provider_factory=second_factory,
        )
    )

    with pytest.raises(DuplicatePipelineRunError):
        second_runner.execute(plan, registry, config, paths)

    assert second_executor.calls == []
    assert second_factory.providers == []


def test_mismatched_registry_snapshot_is_refused_before_execution(tmp_path):
    planned_parent = tmp_path / "planned"
    planned_parent.mkdir()
    plan, _registry, config, paths = _batch(planned_parent)
    snapshot_parent = tmp_path / "snapshot"
    snapshot_parent.mkdir()
    other_root = build_graph_project(snapshot_parent, "independent")
    other_registry = load_registry(other_root)
    executor = ExecutorSpy()
    providers = ProviderFactorySpy()
    runner = BatchExecutor(
        source_execution=SourceExecutionService(
            executor=executor,
            provider_factory=providers,
        )
    )

    with pytest.raises(BatchExecutionPreflightError, match="nothing was executed"):
        runner.execute(plan, other_registry, config, paths)

    assert executor.calls == []
    assert providers.providers == []


def test_keyboard_interrupt_stops_scheduling_and_exposes_partial_evidence(tmp_path):
    plan, registry, config, paths = _batch(tmp_path)
    executor = ExecutorSpy({"B": KeyboardInterrupt()})
    providers = ProviderFactorySpy()
    runner = BatchExecutor(
        source_execution=SourceExecutionService(
            executor=executor,
            provider_factory=providers,
        )
    )

    with pytest.raises(BatchExecutionInterrupted) as exc_info:
        runner.execute(plan, registry, config, paths)

    partial = exc_info.value.partial
    assert executor.calls == ["A", "B"]
    assert tuple(source.source_id for source in partial.source_outcomes) == ("A",)
    assert partial.pending_source_ids == ("B", "C", "D")
    assert all(provider.stop_calls == 1 for provider in providers.providers)
    assert not (
        plan.request.project_root / "data" / "metadata" / "pipelines" / PIPELINE_ID / "summary.json"
    ).exists()
