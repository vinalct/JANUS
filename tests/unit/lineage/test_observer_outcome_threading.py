"""The quality and checkpoint outcomes reach the observer."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext, WriteResult
from janus.quality import PersistedValidationReport, ValidationCheck, ValidationReport
from janus.registry import load_registry
from tests.support import observability_baseline as baseline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDENS = PROJECT_ROOT / "tests" / "fixtures" / "observability" / "baseline"

STARTED_AT = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 4, 12, 5, tzinfo=UTC)

# Which observer method each baseline case ends on, and whether validation ran before it.
TERMINAL_CALLS = {
    "api_success": ("record_success", True),
    "catalog_success": ("record_success", True),
    "empty_handoff": ("record_success", True),
    "replay": ("record_success", True),
    "quality_failure": ("record_failure", True),
    "extraction_failure": ("record_failure", False),
}


class SpyObserver(baseline.FixedObserver):
    """The shipped observer, clock pinned, recording what each call was handed."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def record_success(self, *args, **kwargs):
        persisted = super().record_success(*args, **kwargs)
        type(self).calls.append(
            {"method": "record_success", "kwargs": kwargs, "persisted": persisted}
        )
        return persisted

    def record_failure(self, *args, **kwargs):
        persisted = super().record_failure(*args, **kwargs)
        type(self).calls.append(
            {"method": "record_failure", "kwargs": kwargs, "persisted": persisted}
        )
        return persisted


# --------------------------------------------------------------------------------------
# The signature stays backwards compatible
# --------------------------------------------------------------------------------------


def test_existing_positional_call_signatures_still_work_unchanged(tmp_path):
    """A bare `RunObserver()` called positionally — how the integration suites call it."""

    plan = _build_plan(tmp_path, run_id="threading-compat-001")
    observer = RunObserver()

    started = observer.start_run(plan)
    assert started.validation_report is None

    succeeded = observer.record_success(
        plan,
        _extraction_result(plan),
        _write_results(plan),
    )
    failed = observer.record_failure(
        plan,
        RuntimeError("upstream returned HTTP 500"),
        _extraction_result(plan),
        _write_results(plan)[:1],
    )

    assert succeeded.validation_report is None
    assert failed.validation_report is None


# --------------------------------------------------------------------------------------
# Routed, never persisted (AC-4 / NFR-1)
# --------------------------------------------------------------------------------------


def test_record_success_json_is_unchanged_by_the_validation_report(tmp_path):
    plan = _build_plan(tmp_path, run_id="threading-additive-001")
    report = _validation_report(plan, failing=False)

    without = _record_success_payloads(plan, validation_report=None)
    with_report = _record_success_payloads(plan, validation_report=report)

    assert with_report == without


def test_record_failure_json_is_unchanged_by_the_validation_report(tmp_path):
    plan = _build_plan(tmp_path, run_id="threading-additive-002")
    report = _validation_report(plan, failing=True)

    without = _record_failure_payloads(plan, validation_report=None)
    with_report = _record_failure_payloads(plan, validation_report=report)

    assert with_report == without


# --------------------------------------------------------------------------------------
# "Validation failed" and "validation never ran" are different facts
# --------------------------------------------------------------------------------------


def test_record_failure_carries_a_failed_report_with_its_failed_checks(tmp_path):
    """The quality-gate branch: the run fails *because* of the report it now carries."""

    plan = _build_plan(tmp_path, run_id="threading-quality-001")
    report = _validation_report(plan, failing=True)

    persisted = RunObserver().record_failure(
        plan,
        RuntimeError("quality gate failed"),
        _extraction_result(plan),
        _write_results(plan),
        finished_at=FINISHED_AT,
        validation_report=report,
    )

    assert persisted.validation_report is report
    assert persisted.validation_report.report.is_successful is False
    assert [check.name for check in persisted.validation_report.report.failed_checks] == [
        "required_fields"
    ]
    assert persisted.validation_report.path.exists() is False  # a path, not a second write


def test_record_failure_before_validation_reports_absence_rather_than_failure(tmp_path):
    """An extraction failure never reaches the gate; `None` says so and must not raise."""

    plan = _build_plan(tmp_path, run_id="threading-quality-002")

    persisted = RunObserver().record_failure(
        plan,
        RuntimeError("scripted extraction failure"),
        None,
        (),
        finished_at=FINISHED_AT,
        validation_report=None,
    )

    assert persisted.validation_report is None
    assert persisted.run_metadata.status == "failed"


# --------------------------------------------------------------------------------------
# The checkpoint decision survives to the caller, and "not attempted" stays distinct
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("decision", ["advanced", "retained", "reused", "skipped"])
def test_record_success_returns_the_checkpoint_decision_the_store_made(tmp_path, decision):
    root = tmp_path / decision
    plan = _build_plan(root, run_id=f"threading-checkpoint-{decision}")

    if decision == "skipped":
        # No candidate value at all: the store declines to write and says so.
        persisted = _record_success(plan, checkpoint_value=None)
    elif decision == "advanced":
        persisted = _record_success(plan, checkpoint_value="2026-04-08T12:00:00Z")
    else:
        _record_success(
            _build_plan(root, run_id=f"threading-checkpoint-{decision}-seed"),
            checkpoint_value="2026-04-09T12:00:00Z",
        )
        persisted = _record_success(
            plan,
            checkpoint_value=(
                "2026-04-09T12:00:00Z" if decision == "reused" else "2026-04-08T12:00:00Z"
            ),
        )

    assert persisted.checkpoint_result is not None
    assert persisted.checkpoint_result.decision == decision
    assert persisted.checkpoint_result.advanced is (decision == "advanced")


def test_record_failure_reports_no_checkpoint_write_rather_than_a_skipped_one(tmp_path):
    """`None` is "never attempted" — the failure path does not call the store at all."""

    plan = _build_plan(tmp_path, run_id="threading-checkpoint-none")

    persisted = RunObserver().record_failure(
        plan,
        RuntimeError("upstream returned HTTP 500"),
        _extraction_result(plan, checkpoint_value="2026-04-08T12:00:00Z"),
        (),
        finished_at=FINISHED_AT,
        validation_report=_validation_report(plan, failing=True),
    )

    assert persisted.checkpoint_result is None
    metadata_root = Path(plan.metadata_output.path)
    assert not (metadata_root / "checkpoints").exists()


@pytest.mark.parametrize("case", sorted(TERMINAL_CALLS))
def test_live_and_replay_paths_thread_the_outcomes_on_every_terminal_path(
    tmp_path, monkeypatch, case
):
    expected_method, validation_ran = TERMINAL_CALLS[case]
    SpyObserver.calls = []
    monkeypatch.setattr(baseline, "FixedObserver", SpyObserver)

    manifest = baseline.capture_case(tmp_path, case)

    assert [call["method"] for call in SpyObserver.calls] == [expected_method]
    call = SpyObserver.calls[0]
    report = call["kwargs"]["validation_report"]

    if validation_ran:
        assert isinstance(report, PersistedValidationReport)
        assert report.report.run_id == manifest["run_id"]
        assert report.report.is_successful is (case != "quality_failure")
    else:
        assert report is None

    persisted = call["persisted"]
    assert persisted.validation_report is report
    # The checkpoint outcome is present exactly where a write was attempted.
    assert (persisted.checkpoint_result is not None) is (expected_method == "record_success")


def test_replay_threads_the_same_values_the_live_path_does(tmp_path, monkeypatch):
    """A replay's evidence must not be poorer than a live run's."""

    captured = {}
    for case in ("api_success", "replay"):
        SpyObserver.calls = []
        monkeypatch.setattr(baseline, "FixedObserver", SpyObserver)
        baseline.capture_case(tmp_path / case, case)
        call = SpyObserver.calls[0]
        captured[case] = (
            call["method"],
            sorted(call["kwargs"]),
            type(call["kwargs"]["validation_report"]),
            type(call["persisted"].checkpoint_result),
        )

    assert captured["replay"] == captured["api_success"]


def test_metadata_zone_is_byte_identical_to_the_goldens(tmp_path):
    destination = tmp_path / "regenerated"
    baseline.capture_all(destination)

    regenerated = _tree(destination)
    goldens = _tree(GOLDENS)

    assert sorted(regenerated) == sorted(goldens), "the captured file set changed"
    differing = [name for name in goldens if regenerated[name] != goldens[name]]
    assert differing == []


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _build_plan(root: Path, *, run_id: str) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=root,
        started_at=STARTED_AT,
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _extraction_result(
    plan: ExecutionPlan,
    *,
    checkpoint_value: str | None = "2026-04-08T12:00:00Z",
) -> ExtractionResult:
    return ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(path=f"{plan.raw_output.path}/page-0001.json", format="json"),
        ),
        records_extracted=100,
        checkpoint_value=checkpoint_value,
        metadata={"http_status": "200"},
    )


def _write_results(plan: ExecutionPlan) -> tuple[WriteResult, ...]:
    return (
        WriteResult.from_plan(
            plan,
            "raw",
            path=f"{plan.raw_output.path}/page-0001.json",
            format_name="json",
            mode="append",
            records_written=1,
        ),
        WriteResult.from_plan(
            plan,
            "bronze",
            path=f"{plan.bronze_output.path}/ingestion_date=2026-04-08",
            format_name="parquet",
            mode="append",
            records_written=100,
            partition_by=("ingestion_date",),
        ),
    )


def _validation_report(plan: ExecutionPlan, *, failing: bool) -> PersistedValidationReport:
    check = (
        ValidationCheck.failed("data", "required_fields", "missing required field: id")
        if failing
        else ValidationCheck.passed("data", "required_fields", "all required fields present")
    )
    report = ValidationReport.from_plan(plan, (check,), emitted_at=FINISHED_AT)
    return PersistedValidationReport(report, Path(plan.metadata_output.path) / "validations" / "x")


def _record_success(plan: ExecutionPlan, *, checkpoint_value: str | None):
    return RunObserver().record_success(
        plan,
        _extraction_result(plan, checkpoint_value=checkpoint_value),
        _write_results(plan),
        finished_at=FINISHED_AT,
    )


def _record_success_payloads(plan, *, validation_report) -> tuple[bytes, bytes]:
    persisted = RunObserver().record_success(
        plan,
        _extraction_result(plan),
        _write_results(plan),
        finished_at=FINISHED_AT,
        strategy_metadata={"pages": 3},
        validation_report=validation_report,
    )
    return _payload_bytes(persisted)


def _record_failure_payloads(plan, *, validation_report) -> tuple[bytes, bytes]:
    persisted = RunObserver().record_failure(
        plan,
        RuntimeError("quality gate failed"),
        _extraction_result(plan),
        _write_results(plan),
        finished_at=FINISHED_AT,
        strategy_metadata={"pages": 3},
        validation_report=validation_report,
    )
    return _payload_bytes(persisted)


def _payload_bytes(persisted) -> tuple[bytes, bytes]:
    assert persisted.lineage_path is not None
    run_bytes = persisted.run_metadata_path.read_bytes()
    lineage_bytes = persisted.lineage_path.read_bytes()
    assert json.loads(run_bytes) == persisted.run_metadata.to_dict()
    assert json.loads(lineage_bytes) == persisted.lineage_record.to_dict()
    return run_bytes, lineage_bytes
