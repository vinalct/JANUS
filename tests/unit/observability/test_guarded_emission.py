"""Guarded observer emission and the runtime wiring seam."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import janus.observability.emission as emission
from janus.checkpoints import CheckpointStore
from janus.lineage import (
    LineageStore,
    NullRunEventEmitter,
    PersistedArtifacts,
    RunMetadataStore,
    RunObserver,
)
from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    WriteResult,
)
from janus.observability import (
    GuardedRunEventEmitter,
    IcebergAppendOutcome,
    IcebergAppendResult,
    RunEmissionOutcome,
    RunEmissionResult,
    build_run_event_emitter,
    latest_run_emission,
    wire_run_event_emitter,
)
from janus.planner import Planner, PlanningRequest
from janus.registry import load_registry
from janus.runtime.executor import ExecutedRun
from tests.support import observability_baseline as baseline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STARTED_AT = datetime(2026, 9, 15, 12, tzinfo=UTC)
FINISHED_AT = datetime(2026, 9, 15, 12, 0, 5, tzinfo=UTC)


class SpyEmitter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.persisted: list[PersistedArtifacts] = []

    def emit_started(self, plan, persisted) -> None:
        del plan
        assert persisted.run_metadata_path.is_file()
        self.events.append("emit_started")
        self.persisted.append(persisted)

    def emit_succeeded(self, plan, persisted) -> None:
        del plan
        assert persisted.run_metadata_path.is_file()
        assert persisted.lineage_path is not None and persisted.lineage_path.is_file()
        assert persisted.checkpoint_result is not None
        assert persisted.checkpoint_result.history_path.is_file()
        self.events.append("emit_succeeded")
        self.persisted.append(persisted)

    def emit_failed(self, plan, persisted) -> None:
        del plan
        assert persisted.run_metadata_path.is_file()
        assert persisted.lineage_path is not None and persisted.lineage_path.is_file()
        assert persisted.checkpoint_result is None
        self.events.append("emit_failed")
        self.persisted.append(persisted)


class RaisingEmitter:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def emit_started(self, plan, persisted) -> None:
        del plan, persisted
        raise self.error

    def emit_succeeded(self, plan, persisted) -> None:
        del plan, persisted
        raise self.error

    def emit_failed(self, plan, persisted) -> None:
        del plan, persisted
        raise self.error


def test_observer_calls_each_emitter_hook_after_authoritative_persistence(tmp_path):
    events: list[str] = []

    class OrderedRunStore:
        def write(self, plan, record):
            path = RunMetadataStore().write(plan, record)
            events.append("run_metadata")
            return path

    class OrderedLineageStore:
        def write(self, plan, record):
            path = LineageStore().write(plan, record)
            events.append("lineage")
            return path

    class OrderedCheckpointStore:
        def save(self, *args, **kwargs):
            result = CheckpointStore().save(*args, **kwargs)
            events.append("checkpoint")
            return result

    emitter = SpyEmitter(events)
    observer = RunObserver(
        run_metadata_store=OrderedRunStore(),
        lineage_store=OrderedLineageStore(),
        checkpoint_store=OrderedCheckpointStore(),
        emitter=emitter,
    )
    plan = _plan(tmp_path / "success", "task06-order-success")

    observer.start_run(plan)
    assert events == ["run_metadata", "emit_started"]

    events.clear()
    observer.record_success(
        plan,
        _extraction(plan),
        _writes(plan),
        finished_at=FINISHED_AT,
    )
    assert events == ["run_metadata", "lineage", "checkpoint", "emit_succeeded"]

    events.clear()
    failure_plan = _plan(tmp_path / "failure", "task06-order-failure")
    observer.record_failure(
        failure_plan,
        RuntimeError("scripted failure"),
        _extraction(failure_plan),
        _writes(failure_plan),
        finished_at=FINISHED_AT,
    )
    assert events == ["run_metadata", "lineage", "emit_failed"]


@pytest.mark.parametrize("terminal", ["succeeded", "failed"])
def test_injected_emitter_exception_cannot_change_terminal_artifacts(tmp_path, terminal):
    plan = _plan(tmp_path, f"task06-guard-{terminal}")
    observer = RunObserver(emitter=RaisingEmitter(RuntimeError("emitter bug")))

    if terminal == "succeeded":
        persisted = observer.record_success(
            plan,
            _extraction(plan),
            _writes(plan),
            finished_at=FINISHED_AT,
        )
    else:
        persisted = observer.record_failure(
            plan,
            RuntimeError("run failure"),
            _extraction(plan),
            _writes(plan),
            finished_at=FINISHED_AT,
        )

    assert persisted.run_metadata.status == terminal
    assert persisted.run_metadata_path.is_file()
    assert persisted.lineage_path is not None and persisted.lineage_path.is_file()
    assert persisted.lineage_record is not None
    assert persisted.lineage_record.status == terminal


def test_observer_does_not_catch_keyboard_interrupt(tmp_path):
    plan = _plan(tmp_path, "task06-interrupt")
    observer = RunObserver(emitter=RaisingEmitter(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        observer.start_run(plan)

    assert (tmp_path / plan.metadata_output.path / "runs" / "task06-interrupt.json").is_file()


@pytest.mark.parametrize("stage", ["projection", "runs_table"])
def test_guarded_fanout_propagates_keyboard_interrupt_from_worker(tmp_path, stage):
    persisted = _terminal_artifacts(tmp_path, status="succeeded")

    def interrupting_projector(artifacts):
        del artifacts
        raise KeyboardInterrupt()

    def interrupting_sink(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt()

    def successful_sink(*args, **kwargs):
        del args, kwargs
        return IcebergAppendResult(
            IcebergAppendOutcome.EMITTED,
            "metadata.runs",
        )

    emitter = build_run_event_emitter(
        {},
        {},
        projector=(
            interrupting_projector if stage == "projection" else emission._project_run_record
        ),
        runs_table_sink=interrupting_sink if stage == "runs_table" else successful_sink,
    )

    with pytest.raises(KeyboardInterrupt):
        emitter.emit_succeeded(_plan(tmp_path, "unused"), persisted)

    assert emitter.last_result is None


def test_bare_observer_owns_a_silent_null_emitter(tmp_path):
    observer = RunObserver()
    assert isinstance(observer.emitter, NullRunEventEmitter)

    persisted = observer.start_run(_plan(tmp_path, "task06-null"))

    assert persisted.run_metadata.status == "running"


def test_lineage_package_has_no_observability_dependency():
    lineage_root = PROJECT_ROOT / "src" / "janus" / "lineage"
    offenders = [
        path.relative_to(PROJECT_ROOT)
        for path in lineage_root.glob("*.py")
        if "janus.observability" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_terminal_fanout_projects_once_and_passes_the_same_record_to_sink(tmp_path):
    persisted = _terminal_artifacts(tmp_path, status="succeeded")
    projections: list[Any] = []
    sink_records: list[Any] = []
    sink_budgets: list[float] = []

    def projector(artifacts):
        record = emission._project_run_record(artifacts)
        projections.append(record)
        return record

    def sink(record, config, resolved_paths, *, logger, timeout_seconds):
        del config, resolved_paths, logger
        sink_records.append(record)
        sink_budgets.append(timeout_seconds)
        return IcebergAppendResult(
            IcebergAppendOutcome.EMITTED,
            "metadata.runs",
        )

    emitter = build_run_event_emitter(
        {},
        {},
        projector=projector,
        runs_table_sink=sink,
        timeout_seconds=0.5,
    )
    emitter.emit_succeeded(_plan(tmp_path, "unused"), persisted)

    assert len(projections) == 1
    assert sink_records == projections
    assert 0 < sink_budgets[0] <= 0.5
    assert emitter.last_result is not None
    assert emitter.last_result.outcome is RunEmissionOutcome.EMITTED
    assert emitter.last_result.lifecycle == "succeeded"


@pytest.mark.parametrize(
    ("projector", "sink", "expected_stage", "expected_type"),
    [
        (
            lambda persisted: (_ for _ in ()).throw(ValueError("bad projection")),
            lambda *args, **kwargs: pytest.fail("sink must not run"),
            "projection",
            "ValueError",
        ),
        (
            emission._project_run_record,
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("catalog down")),
            "runs_table",
            "OSError",
        ),
    ],
)
def test_fanout_exceptions_become_failed_results(
    tmp_path,
    projector,
    sink,
    expected_stage,
    expected_type,
):
    persisted = _terminal_artifacts(tmp_path, status="failed")
    emitter = build_run_event_emitter(
        {},
        {},
        projector=projector,
        runs_table_sink=sink,
    )

    emitter.emit_failed(_plan(tmp_path, "unused"), persisted)

    assert emitter.last_result is not None
    assert emitter.last_result.outcome is RunEmissionOutcome.FAILED
    assert emitter.last_result.stage == expected_stage
    assert emitter.last_result.exception_type == expected_type


@pytest.mark.parametrize(
    ("sink_outcome", "expected_outcome"),
    [
        (IcebergAppendOutcome.SKIPPED, RunEmissionOutcome.SKIPPED),
        (IcebergAppendOutcome.FAILED, RunEmissionOutcome.FAILED),
    ],
)
def test_sink_degradation_is_reported_truthfully(tmp_path, sink_outcome, expected_outcome):
    persisted = _terminal_artifacts(tmp_path, status="succeeded")

    def sink(*args, **kwargs):
        del args, kwargs
        return IcebergAppendResult(
            sink_outcome,
            "audit.runs",
            reason="append_degraded",
            step="append",
            exception_type="CatalogError",
        )

    emitter = build_run_event_emitter({}, {}, runs_table_sink=sink)
    emitter.emit_succeeded(_plan(tmp_path, "unused"), persisted)

    result = emitter.last_result
    assert result is not None
    assert result.outcome is expected_outcome
    assert result.table_identifier == "audit.runs"
    assert result.reason == "append_degraded"
    assert result.stage == "append"
    assert result.to_summary()["outcome"] == expected_outcome.value


def test_total_budget_joins_worker_once_and_records_the_timeout(tmp_path, monkeypatch):
    persisted = _terminal_artifacts(tmp_path, status="succeeded")
    started = []
    joined_with = []

    class BudgetExhaustingThread:
        def __init__(self, *, target, name, daemon):
            del target
            assert name == "janus-run-event-emission"
            assert daemon is True

        def start(self):
            started.append(True)

        def join(self, timeout):
            joined_with.append(timeout)

        def is_alive(self):
            return True

    monkeypatch.setattr(emission.threading, "Thread", BudgetExhaustingThread)

    emitter = build_run_event_emitter(
        {},
        {},
        timeout_seconds=0.01,
    )

    emitter.emit_succeeded(_plan(tmp_path, "unused"), persisted)

    assert started == [True]
    assert joined_with == [0.01]
    assert emitter.last_result is not None
    assert emitter.last_result.outcome is RunEmissionOutcome.FAILED
    assert emitter.last_result.stage == "budget"
    assert emitter.last_result.exception_type == "EmissionTimeoutError"


def test_wiring_builds_isolated_emitters_without_overriding_explicit_observers():
    first = RunObserver()
    second = RunObserver()

    wired_first = wire_run_event_emitter(first, {"name": "first"}, {}, None)
    wired_second = wire_run_event_emitter(second, {"name": "second"}, {}, None)

    assert wired_first is not first
    assert wired_second is not second
    assert isinstance(wired_first.emitter, GuardedRunEventEmitter)
    assert isinstance(wired_second.emitter, GuardedRunEventEmitter)
    assert wired_first.emitter is not wired_second.emitter
    assert isinstance(first.emitter, NullRunEventEmitter)

    class FixedObserver(RunObserver):
        pass

    custom = FixedObserver()
    assert wire_run_event_emitter(custom, {}, {}, None) is custom

    explicit = RunObserver(emitter=SpyEmitter([]))
    assert wire_run_event_emitter(explicit, {}, {}, None) is explicit


def test_live_replay_and_empty_handoff_use_production_wiring(monkeypatch, tmp_path):
    records = []

    def sink(record, config, resolved_paths, *, logger, timeout_seconds):
        del config, resolved_paths, logger, timeout_seconds
        records.append(record)
        return IcebergAppendResult(
            IcebergAppendOutcome.EMITTED,
            "metadata.runs",
        )

    monkeypatch.setattr(baseline, "FixedObserver", RunObserver)
    monkeypatch.setattr(emission, "append_run_record", sink)
    manifests = {
        case: baseline.capture_case(tmp_path / case, case)
        for case in ("api_success", "empty_handoff", "replay")
    }

    assert len(records) == 3
    for case in ("api_success", "empty_handoff"):
        summary = manifests[case]["summary"]["executed_run"]
        assert summary["run_event_emission"]["outcome"] == "emitted"
    replay_summary = manifests["replay"]["summary"]["executed_run"]
    assert replay_summary["run_event_emission"]["outcome"] == "emitted"

    empty_handoff = manifests["empty_handoff"]
    assert empty_handoff["spark_session_started"] is False
    assert "session_start" not in empty_handoff["events"]

    live = records[0].to_dict()
    replay = records[2].to_dict()
    difference_set = {key for key in live if live[key] != replay[key]}
    assert difference_set == {
        "bronze_table_identifier",
        "checkpoint_advanced",
        "checkpoint_decision",
        "checkpoint_history_path",
        "checkpoint_value",
        "duration_seconds",
        "emitted_at",
        "ended_at",
        "lineage_path",
        "records_extracted",
        "run_id",
        "run_metadata_path",
        "source_config_path",
        "validation_report_path",
    }


def test_emission_summary_is_present_only_for_a_terminal_attempt():
    planned_run = Planner().plan(
        PlanningRequest.create(
            source_id="federal_open_data_example",
            environment="local",
            project_root=PROJECT_ROOT,
            run_id="task06-summary",
            started_at=STARTED_AT,
        )
    )
    result = RunEmissionResult(
        lifecycle="succeeded",
        outcome=RunEmissionOutcome.SKIPPED,
        table_identifier="metadata.runs",
        reason="pyiceberg_or_pyarrow_unavailable",
        stage="dependency_import",
        exception_type="ImportError",
    )

    absent = ExecutedRun(planned_run=planned_run, status="succeeded").to_summary()
    present = ExecutedRun(
        planned_run=planned_run,
        status="succeeded",
        run_event_emission=result,
    ).to_summary()

    assert "run_event_emission" not in absent
    assert present["run_event_emission"]["outcome"] == "skipped"
    assert present["run_event_emission"]["table_identifier"] == "metadata.runs"


def test_latest_result_is_absent_for_null_and_present_for_guarded_emitter(tmp_path):
    assert latest_run_emission(RunObserver()) is None

    persisted = _terminal_artifacts(tmp_path, status="succeeded")
    emitter = build_run_event_emitter(
        {},
        {},
        runs_table_sink=lambda *args, **kwargs: IcebergAppendResult(
            IcebergAppendOutcome.EMITTED,
            "metadata.runs",
        ),
    )
    observer = RunObserver(emitter=emitter)
    emitter.emit_succeeded(_plan(tmp_path, "unused"), persisted)

    assert latest_run_emission(observer) is emitter.last_result


def _terminal_artifacts(tmp_path: Path, *, status: str) -> PersistedArtifacts:
    plan = _plan(tmp_path / status, f"task06-{status}")
    observer = RunObserver()
    if status == "succeeded":
        return observer.record_success(
            plan,
            _extraction(plan),
            _writes(plan),
            finished_at=FINISHED_AT,
        )
    return observer.record_failure(
        plan,
        RuntimeError("scripted failure"),
        _extraction(plan),
        _writes(plan),
        finished_at=FINISHED_AT,
    )


def _plan(root: Path, run_id: str) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=root,
            started_at=STARTED_AT,
        ),
    )


def _extraction(plan: ExecutionPlan) -> ExtractionResult:
    return ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(
                path=f"{plan.raw_output.path}/page-0001.json",
                format="json",
            ),
        ),
        records_extracted=3,
        checkpoint_value="2026-09-15T12:00:00Z",
    )


def _writes(plan: ExecutionPlan) -> tuple[WriteResult, ...]:
    return (
        WriteResult.from_plan(
            plan,
            "bronze",
            path="bronze.task",
            format_name="iceberg",
            mode="append",
            records_written=3,
        ),
    )
