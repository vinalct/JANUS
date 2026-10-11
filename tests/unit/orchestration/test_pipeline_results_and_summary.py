"""Versioned pipeline outcomes and durable aggregate summaries."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

import pytest

import janus.observability.metrics.pipeline as pipeline_metrics_module
from janus.observability.bounded import bounded_emission
from janus.observability.metrics.sink import MetricsRunSink
from janus.observability.metrics.transport import MetricsEmissionOutcome, MetricsEmissionResult
from janus.orchestration import (
    BatchPlanner,
    BatchPlanRequest,
    BatchSelection,
    DuplicatePipelineRunError,
    ExecutionTiming,
    PipelineClock,
    PipelineOutcome,
    PipelineSummaryPersistenceError,
    PipelineSummaryStore,
    SourceAttempt,
    SourceOutcome,
    SourcePlanFailure,
    SummaryPersistence,
    source_attempt_run_id,
)
from janus.runtime import BatchExecutor
from janus.utils.logging import REDACTED_VALUE
from janus.utils.storage import StorageLayout
from tests.support.metrics_order23 import WarningSpy
from tests.support.orchestration import build_graph_project
from tests.unit.runtime.test_batch_executor import (
    ExecutorSpy,
    ProviderFactorySpy,
    SourceExecutionService,
    _batch,
)

PLANNED_AT = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
STARTED_AT = datetime(2026, 9, 14, 3, 0, tzinfo=UTC)
ENDED_AT = datetime(2026, 9, 14, 3, 0, 5, tzinfo=UTC)
PIPELINE_ID = "order14-summary"
PIPELINE_TIMING = ExecutionTiming(STARTED_AT, ENDED_AT, 5.0)
ATTEMPT_TIMING = ExecutionTiming(STARTED_AT, ENDED_AT, 4.25)


class FakeExecutedRun:
    """Established per-source result surface, without importing runtime code here."""

    def __init__(
        self,
        planned_run: Any,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        failure_reason: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.planned_run = planned_run
        self.status = status
        self.failure_reason = failure_reason
        self.error_type = error_type
        self.summary = summary or {
            "status": status,
            "strategy_metadata": {},
            "materialized_outputs": [],
            "metadata_outputs": {
                "run_metadata_path": "/metadata/runs/source.json",
                "lineage_path": "/metadata/lineage/source.json",
                "checkpoint_state_path": "/metadata/checkpoints/current.json",
                "checkpoint_history_path": "/metadata/checkpoints/history/source.json",
                "validation_report_path": "/metadata/quality/source.json",
            },
        }

    @property
    def is_successful(self) -> bool:
        return self.status == "succeeded"

    def to_summary(self) -> dict[str, Any]:
        return self.summary


def _plan(tmp_path: Path, graph: str, *, selection: BatchSelection | None = None):
    project_root = build_graph_project(tmp_path, graph)
    request = BatchPlanRequest.create(
        environment="local",
        project_root=project_root,
        pipeline_run_id=PIPELINE_ID,
        planned_at=PLANNED_AT,
        selection=selection,
    )
    return BatchPlanner().plan(request)


def _attempt(planned_source, *, status: str = "succeeded", **kwargs: Any) -> SourceAttempt:
    result = FakeExecutedRun(
        planned_source.require_planned_run(),
        status=status,
        failure_reason=kwargs.pop("failure_reason", None),
        error_type=kwargs.pop("error_type", None),
        summary=kwargs.pop("summary", None),
    )
    assert not kwargs
    return SourceAttempt.from_executed_run(result, timing=ATTEMPT_TIMING)


def _successful_outcomes(plan) -> tuple[SourceOutcome, ...]:
    return tuple(
        SourceOutcome.from_attempts(source, (_attempt(source),)) for source in plan.sources
    )


def _outcome(
    plan,
    sources: tuple[SourceOutcome, ...],
    *,
    persisted_path: Path | None = None,
) -> PipelineOutcome:
    persistence = (
        SummaryPersistence.succeeded(persisted_path)
        if persisted_path is not None
        else SummaryPersistence()
    )
    return PipelineOutcome.from_plan(
        plan,
        timing=PIPELINE_TIMING,
        sources=sources,
        summary_persistence=persistence,
    )


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout(
        project_root=tmp_path.resolve(),
        root_dir=(tmp_path / "data").resolve(),
        raw_dir=(tmp_path / "data" / "raw").resolve(),
        bronze_dir=(tmp_path / "data" / "bronze").resolve(),
        metadata_dir=(tmp_path / "data" / "metadata").resolve(),
    )


def test_a_successful_pipeline_serializes_the_versioned_contract_and_exact_totals(tmp_path):
    plan = _plan(
        tmp_path,
        "chain",
        selection=BatchSelection.create(tags=("report",)),
    )
    path = tmp_path / "metadata" / "pipelines" / PIPELINE_ID / "summary.json"
    outcome = _outcome(plan, _successful_outcomes(plan), persisted_path=path)

    summary = json.loads(json.dumps(outcome.to_summary()))

    assert list(summary) == [
        "schema_version",
        "pipeline",
        "selection",
        "graph",
        "config_versions",
        "sources",
        "totals",
        "summary_persistence",
    ]
    assert summary["schema_version"] == 1
    assert summary["pipeline"] == {
        "pipeline_run_id": PIPELINE_ID,
        "attempt": 1,
        "trigger": "run-all",
        "environment": "local",
        "planned_at": PLANNED_AT.isoformat(),
        "started_at": STARTED_AT.isoformat(),
        "ended_at": ENDED_AT.isoformat(),
    }
    assert summary["selection"] == {
        "requested": {"tags": ["report"], "domains": []},
        "root_ids": ["B"],
        "included_upstream_ids": ["A"],
        "source_ids": ["A", "B"],
    }
    assert summary["totals"] == {
        "selected": 1,
        "expanded": 2,
        "attempted": 2,
        "succeeded": 2,
        "failed": 0,
        "skipped": 0,
        "status": "succeeded",
        "duration_seconds": 5.0,
    }
    assert outcome.is_successful
    assert all(source["attempted"] for source in summary["sources"])


def test_mixed_failure_and_skip_outcome_reports_two_blockers_and_one_root(tmp_path):
    plan = _plan(tmp_path, "diamond")
    failed_a = SourceOutcome.from_attempts(
        plan.source("A"),
        (
            _attempt(
                plan.source("A"),
                status="failed",
                failure_reason="upstream unavailable",
                error_type="ConnectionError",
            ),
        ),
    )
    skipped_b = SourceOutcome.skipped(
        plan.source("B"),
        direct_blocking_upstream_ids=("A",),
        root_failed_source_ids=("A",),
    )
    skipped_c = SourceOutcome.skipped(
        plan.source("C"),
        direct_blocking_upstream_ids=("A",),
        root_failed_source_ids=("A",),
    )
    skipped_d = SourceOutcome.skipped(
        plan.source("D"),
        direct_blocking_upstream_ids=("C", "B"),
        root_failed_source_ids=("A", "A"),
    )
    path = tmp_path / "summary.json"
    outcome = _outcome(
        plan,
        (failed_a, skipped_b, skipped_c, skipped_d),
        persisted_path=path,
    )

    summary = outcome.to_summary()
    skipped = summary["sources"][-1]

    assert summary["totals"] == {
        "selected": 4,
        "expanded": 4,
        "attempted": 1,
        "succeeded": 0,
        "failed": 1,
        "skipped": 3,
        "status": "failed",
        "duration_seconds": 5.0,
    }
    assert skipped["skip"] == {
        "reason_code": "upstream_failed",
        "direct_blocking_upstream_ids": ["B", "C"],
        "root_failed_source_ids": ["A"],
    }
    assert skipped["attempted"] is False
    assert skipped["attempts"] == []
    assert skipped["timing"] == {
        "started_at": None,
        "ended_at": None,
        "duration_seconds": 0.0,
    }
    assert "evidence" not in skipped


def test_a_source_planning_failure_is_failed_without_becoming_an_attempt(tmp_path):
    plan = _plan(tmp_path, "chain")
    planned_a = replace(
        plan.source("A"),
        planned_run=None,
        failure=SourcePlanFailure(
            error_type="HookResolutionError",
            reason="hook binding is absent",
        ),
    )
    failed_a = SourceOutcome.from_planning_failure(planned_a)
    skipped_b = SourceOutcome.skipped(
        plan.source("B"),
        direct_blocking_upstream_ids=("A",),
        root_failed_source_ids=("A",),
    )
    outcome = _outcome(
        plan,
        (failed_a, skipped_b),
        persisted_path=tmp_path / "summary.json",
    )

    source = outcome.to_summary()["sources"][0]

    assert source["status"] == "failed"
    assert source["attempted"] is False
    assert source["attempts"] == []
    assert source["failure"] == {
        "phase": "planning",
        "error_type": "HookResolutionError",
        "reason": "hook binding is absent",
    }
    assert outcome.totals()["attempted"] == 0


def test_attempt_history_preserves_a_failure_before_eventual_success(tmp_path):
    plan = _plan(tmp_path, "independent")
    source = plan.source("A")
    first = _attempt(
        source,
        status="failed",
        failure_reason="temporary outage",
        error_type="TimeoutError",
    )
    second = SourceAttempt(
        source_id="A",
        run_id=source_attempt_run_id(
            pipeline_run_id=PIPELINE_ID,
            source_id="A",
            attempt=2,
        ),
        attempt=2,
        status="succeeded",
        timing=ExecutionTiming(
            STARTED_AT + timedelta(seconds=10),
            ENDED_AT + timedelta(seconds=10),
            3.0,
        ),
        evidence={"metadata_outputs": {"run_metadata_path": "/metadata/runs/retry.json"}},
    )

    final = SourceOutcome.from_attempts(source, (first, second))
    summary = final.to_summary()

    assert final.status == "succeeded"
    assert final.attempted
    assert [attempt["status"] for attempt in summary["attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert summary["attempt"] == 2
    assert summary["run_id"] == second.run_id
    assert summary["timing"]["duration_seconds"] == 7.25


def test_fixed_clocks_use_monotonic_elapsed_time_when_wall_time_moves_backwards():
    wall_times = iter(
        (
            datetime(2026, 9, 14, 3, 0, tzinfo=UTC),
            datetime(2026, 9, 14, 2, 50, tzinfo=UTC),
        )
    )
    monotonic_times = iter((100.0, 700.0))
    clock = PipelineClock(
        wall_clock=lambda: next(wall_times),
        monotonic_clock=lambda: next(monotonic_times),
    )

    timing = clock.finish(clock.start())

    assert timing.ended_at < timing.started_at
    assert timing.duration_seconds == 600.0


def test_attempt_evidence_is_frozen_json_redacted_and_reuses_the_source_summary(tmp_path):
    plan = _plan(tmp_path, "independent")
    source = plan.source("A")
    evidence = {
        "metadata_outputs": {
            "run_metadata_path": "/metadata/runs/A.json",
            "lineage_path": "/metadata/lineage/A.json",
            "checkpoint_state_path": "/metadata/checkpoints/current.json",
            "checkpoint_history_path": "/metadata/checkpoints/history/A.json",
            "validation_report_path": "/metadata/quality/A.json",
        },
        "strategy_metadata": {
            "api_token": "secret-value",
            "credentials": {"user": "operator", "password": "secret-value"},
            "response_body": "unbounded raw response",
            "url": "https://example.invalid/data?token=secret-value&page=1",
        },
    }
    attempt = _attempt(source, summary=evidence)
    evidence["metadata_outputs"]["run_metadata_path"] = "mutated"

    summary = attempt.to_summary()["evidence"]

    assert summary["metadata_outputs"]["run_metadata_path"] == "/metadata/runs/A.json"
    assert summary["strategy_metadata"]["api_token"] == REDACTED_VALUE
    assert summary["strategy_metadata"]["credentials"] == REDACTED_VALUE
    assert summary["strategy_metadata"]["response_body"] == REDACTED_VALUE
    assert "secret-value" not in json.dumps(summary)
    with pytest.raises(TypeError):
        attempt.evidence["new"] = "value"


def test_failure_reasons_are_redacted_and_bounded(tmp_path):
    plan = _plan(tmp_path, "independent")
    reason = "https://example.invalid/data?token=secret-value&x=1 " + ("x" * 3000)

    attempt = _attempt(
        plan.source("A"),
        status="failed",
        failure_reason=reason,
        error_type="RequestError",
    )

    failure = attempt.to_summary()["failure"]
    assert "secret-value" not in failure["reason"]
    assert failure["reason"].endswith("… (truncated)")
    assert len(failure["reason"]) < 2050


def test_result_records_are_frozen(tmp_path):
    plan = _plan(tmp_path, "independent")
    outcome = _outcome(plan, _successful_outcomes(plan))

    with pytest.raises(FrozenInstanceError):
        outcome.environment = "production"


@pytest.mark.parametrize(
    "pipeline_id",
    ("../escape", "nested/id", "..", "with space", "/absolute"),
)
def test_pipeline_summary_paths_reject_unsafe_id_components(tmp_path, pipeline_id):
    store = PipelineSummaryStore(_storage_layout(tmp_path))

    with pytest.raises(Exception, match="path component"):
        store.summary_path(pipeline_id)


def test_a_summary_path_cannot_escape_through_an_existing_symlink(tmp_path):
    layout = _storage_layout(tmp_path)
    pipelines = layout.metadata_dir / "pipelines"
    pipelines.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (pipelines / PIPELINE_ID).symlink_to(outside, target_is_directory=True)
    store = PipelineSummaryStore(layout)

    with pytest.raises(Exception, match="escapes metadata root"):
        store.summary_path(PIPELINE_ID)


def test_persist_uses_the_predictable_path_and_rejects_a_completed_id(tmp_path):
    plan = _plan(tmp_path, "independent")
    pending = _outcome(plan, _successful_outcomes(plan))
    store = PipelineSummaryStore(_storage_layout(tmp_path))

    assert store.assert_available(PIPELINE_ID) == (
        store.storage_layout.metadata_dir
        / "pipelines"
        / PIPELINE_ID
        / "summary.json"
    ).resolve()

    persisted = store.persist(pending)
    path = persisted.summary_persistence.path

    assert path is not None
    assert json.loads(path.read_text(encoding="utf-8")) == persisted.to_summary()
    assert persisted.is_successful
    with pytest.raises(DuplicatePipelineRunError, match="new pipeline identity"):
        store.assert_available(PIPELINE_ID)
    with pytest.raises(DuplicatePipelineRunError):
        store.persist(pending)


def test_failed_atomic_replace_keeps_previous_summary_and_returns_diagnostic_outcome(tmp_path):
    plan = _plan(tmp_path, "independent")
    pending = _outcome(plan, _successful_outcomes(plan))
    layout = _storage_layout(tmp_path)
    initial = PipelineSummaryStore(layout).persist(pending)
    path = initial.summary_persistence.path
    assert path is not None
    previous = path.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("disk unavailable")

    failing_store = PipelineSummaryStore(layout, atomic_replace=fail_replace)
    with pytest.raises(PipelineSummaryPersistenceError) as exc_info:
        failing_store.persist(pending, allow_existing=True)

    diagnostic = exc_info.value.pipeline_outcome
    assert path.read_bytes() == previous
    assert diagnostic.summary_persistence.status == "failed"
    assert diagnostic.status == "failed"
    assert diagnostic.totals()["succeeded"] == 2
    assert diagnostic.totals()["failed"] == 0
    assert diagnostic.to_summary()["summary_persistence"]["failure"]["phase"] == (
        "summary_persistence"
    )
    assert not list(path.parent.glob(".*.tmp"))


def _metrics_runner(*, behavior=None, logger=None):
    service = SourceExecutionService(
        executor=ExecutorSpy(behavior), provider_factory=ProviderFactorySpy()
    )
    return BatchExecutor(
        source_execution=service,
        clock=PipelineClock(wall_clock=lambda: STARTED_AT, monotonic_clock=lambda: 0.0),
        logger=logger,
    )


@pytest.mark.parametrize("failed", [False, True])
def test_pipeline_metrics_emit_once_after_summary_persistence_even_for_failure(
    tmp_path, monkeypatch, failed, record_property
):
    plan, registry, config, paths = _batch(tmp_path)
    events = []
    persisted_outcomes = []
    captured = []

    class OrderedStore(PipelineSummaryStore):
        def persist(self, outcome, **kwargs):
            result = super().persist(outcome, **kwargs)
            events.append("persist")
            persisted_outcomes.append(result)
            return result

    class CaptureTransport:
        kind = "file"

        def send(self, samples, *, budget_seconds):
            assert 0 < budget_seconds <= 5
            persisted = persisted_outcomes[0]
            assert (
                json.loads(persisted.summary_persistence.path.read_text()) == persisted.to_summary()
            )
            events.append("emit")
            captured.extend(samples)
            return MetricsEmissionResult(MetricsEmissionOutcome.EMITTED, self.kind, sample_count=4)

    monkeypatch.setattr(
        pipeline_metrics_module,
        "build_metrics_sink",
        lambda *_args, **_kwargs: MetricsRunSink(CaptureTransport()),
    )
    runner = _metrics_runner(behavior={"A": "failed"} if failed else {})
    outcome = runner.execute(
        plan,
        registry,
        config,
        paths,
        summary_store=OrderedStore(
            StorageLayout.from_environment_config(config, plan.request.project_root)
        ),
    )
    assert outcome is persisted_outcomes[0]
    assert events == ["persist", "emit"]
    assert len(captured) == 4
    assert [sample.value for sample in captured[1:]] == ([1, 1, 2] if failed else [4, 0, 0])
    assert outcome.status == ("failed" if failed else "succeeded")
    assert all(
        sample.scope == "pipeline" and "source_id" not in dict(sample.labels) for sample in captured
    )
    record_property("ordered_spy", json.dumps(events))
    record_property(
        "pipeline_samples",
        json.dumps(
            [
                {"name": sample.name, "labels": dict(sample.labels), "value": sample.value}
                for sample in captured
            ]
        ),
    )


@pytest.mark.parametrize("failure", ["builder", "raising", "returned", "logger"])
def test_metrics_failure_leaves_outcome_and_authoritative_summary_bytes_unchanged(
    tmp_path, monkeypatch, failure, record_property
):
    plan, registry, config, paths = _batch(tmp_path)
    baseline = _metrics_runner().execute(plan, registry, config, paths)
    path = baseline.summary_persistence.path
    before = path.read_bytes()
    path.unlink()
    logger = WarningSpy()
    calls = []

    class FailingTransport:
        kind = "file"

        def send(self, _samples, *, budget_seconds):
            assert budget_seconds > 0
            calls.append("send")
            if failure == "returned":
                return MetricsEmissionResult(
                    MetricsEmissionOutcome.FAILED,
                    self.kind,
                    reason="send_failed",
                    exception_type="OSError",
                )
            raise OSError("secret=do-not-log-this-exception")

    def build(*_args, **_kwargs):
        if failure == "builder":
            calls.append("build")
            raise OSError("secret=do-not-log-this-exception")
        return MetricsRunSink(FailingTransport())

    if failure == "logger":

        def broken_warning(event, **fields):
            logger.warnings.append((event, fields))
            raise OSError("log sink failed")

        monkeypatch.setattr(logger, "warning", broken_warning)
    monkeypatch.setattr(pipeline_metrics_module, "build_metrics_sink", build)
    outcome = _metrics_runner(logger=logger).execute(plan, registry, config, paths)
    assert outcome == baseline
    assert json.dumps(outcome.to_summary(), sort_keys=True) == json.dumps(
        baseline.to_summary(), sort_keys=True
    )
    assert path.read_bytes() == before
    assert outcome.is_successful
    assert calls == (["build"] if failure == "builder" else ["send"])
    assert len(logger.warnings) == 1
    assert logger.warnings[0][0] == "metrics_emission_degraded"
    assert logger.warnings[0][1]["exception_type"] == "OSError"
    assert "do-not-log" not in str(logger.warnings)
    record_property(
        "failure_evidence",
        json.dumps(
            {
                "mode": failure,
                "unchanged_outcome": outcome == baseline,
                "summary_sha256_before": sha256(before).hexdigest(),
                "summary_sha256_after": sha256(path.read_bytes()).hexdigest(),
                "warnings": logger.warnings,
            }
        ),
    )


@pytest.mark.parametrize("stage", ["builder", "send"])
def test_hanging_pipeline_metrics_are_bounded_and_leave_the_summary_unchanged(
    tmp_path, monkeypatch, stage, record_property
):
    plan, registry, config, paths = _batch(tmp_path)
    baseline = _metrics_runner().execute(plan, registry, config, paths)
    path = baseline.summary_persistence.path
    before = path.read_bytes()
    path.unlink()
    release, finished = Event(), Event()
    logger = WarningSpy()

    class HangingTransport:
        kind = "file"

        def send(self, samples, *, budget_seconds):
            assert samples and budget_seconds > 0
            try:
                release.wait(1)
                return MetricsEmissionResult(MetricsEmissionOutcome.EMITTED, self.kind)
            finally:
                finished.set()

    def build(*_args, **_kwargs):
        if stage == "builder":
            try:
                release.wait(1)
            finally:
                finished.set()
        return MetricsRunSink(HangingTransport())

    def short_guard(worker, **kwargs):
        return bounded_emission(worker, **{**kwargs, "timeout_seconds": 0.03})

    monkeypatch.setattr(pipeline_metrics_module, "build_metrics_sink", build)
    monkeypatch.setattr(pipeline_metrics_module, "bounded_emission", short_guard)
    started = monotonic()
    try:
        outcome = _metrics_runner(logger=logger).execute(plan, registry, config, paths)
        elapsed = monotonic() - started
        assert 0.03 <= elapsed < 0.5
        assert outcome == baseline
        assert path.read_bytes() == before
        assert len(logger.warnings) == 1
        assert logger.warnings[0][1]["step"] == "budget"
        assert logger.warnings[0][1]["exception_type"] == "EmissionTimeoutError"
        record_property(
            "timeout_evidence",
            json.dumps(
                {
                    "stage": stage,
                    "budget_seconds": 0.03,
                    "elapsed_seconds": elapsed,
                    "unchanged_outcome": outcome == baseline,
                    "summary_sha256_before": sha256(before).hexdigest(),
                    "summary_sha256_after": sha256(path.read_bytes()).hexdigest(),
                    "warnings": logger.warnings,
                }
            ),
        )
    finally:
        release.set()
        assert finished.wait(1)


def test_pipeline_metrics_do_not_emit_when_summary_persistence_fails(tmp_path, monkeypatch):
    plan, registry, config, paths = _batch(tmp_path)
    calls = []
    monkeypatch.setattr(
        pipeline_metrics_module,
        "build_metrics_sink",
        lambda *_args, **_kwargs: calls.append("emit"),
    )
    store = PipelineSummaryStore(
        StorageLayout.from_environment_config(config, plan.request.project_root),
        atomic_replace=lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(PipelineSummaryPersistenceError):
        _metrics_runner().execute(plan, registry, config, paths, summary_store=store)
    assert calls == []
    assert not store.summary_path(plan.request.pipeline_run_id).exists()
