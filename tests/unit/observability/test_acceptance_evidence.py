"""Acceptance evidence that composes the focused observability contracts end to end."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from janus.lineage import compute_config_version
from janus.observability import (
    IcebergAppendOutcome,
    IcebergAppendResult,
    build_run_event_emitter,
)
from tests.support import observability_baseline as baseline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "observability" / "baseline"
OPENLINEAGE_SCHEMA = PROJECT_ROOT / "tests" / "fixtures" / "openlineage" / "OpenLineage-2-0-2.json"
JANUS_FACET_SCHEMA = PROJECT_ROOT / "docs" / "schemas" / "openlineage" / "JanusRunFacet.json"
AUTHORITATIVE_DIRECTORIES = frozenset({"runs", "lineage", "checkpoints", "validations"})


class _NeverSparkProvider:
    """A provider-shaped tripwire: observability may inspect paths, never acquire Spark."""

    def __init__(self, resolved_paths: dict[str, Any]) -> None:
        self.resolved_paths = resolved_paths
        self.was_started = False

    def get(self) -> None:
        raise AssertionError("run observability must not acquire a Spark session")

    def stop(self) -> None:
        pass


def _validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(
        json.loads(path.read_text(encoding="utf-8")),
        format_checker=FormatChecker(),
    )


def _emitter_config() -> dict[str, Any]:
    return {
        "spark": {"iceberg": {"catalog_name": "janus", "warehouse_dir": "/warehouse"}},
        "observability": {
            "openlineage": {
                "transport": "file",
                "path": "lineage/openlineage",
            }
        },
    }


def _authoritative_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
        if AUTHORITATIVE_DIRECTORIES.intersection(path.relative_to(root).parts)
    }


@pytest.mark.parametrize("case", baseline.BASELINE_CASES)
def test_contract_identity_is_additive_in_run_and_lineage_artifacts(case, tmp_path):
    root = tmp_path / "project"
    manifest = baseline.capture_case(root, case)
    contract = manifest["summary"]["planned_run"]["contract"]
    metadata_root = root / "data" / "metadata" / manifest["source_id"]
    expected = {
        "schema_version": contract["schema_version"],
        "contract_id": contract["id"],
        "contract_version": contract["version"],
    }

    for directory in ("runs", "lineage"):
        records = sorted((metadata_root / directory).glob("*.json"))
        assert len(records) == 1
        payload = json.loads(records[0].read_text(encoding="utf-8"))
        assert {key: payload[key] for key in expected} == expected


@pytest.mark.parametrize("case", baseline.BASELINE_CASES)
def test_enabled_emission_preserves_goldens_and_emits_valid_lifecycle_events(
    case,
    tmp_path,
    monkeypatch,
):
    """AC-3/AC-4: enable both destinations without changing one authoritative byte."""
    root = tmp_path / "project"
    normalized = tmp_path / "normalized"
    appended = []

    def append(record, config, resolved_paths, *, logger, timeout_seconds):
        del config, resolved_paths, logger, timeout_seconds
        appended.append(record)
        return IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")

    resolved_paths = {
        "metadata_dir": root / "data" / "metadata",
        "iceberg_warehouse_dir": root / "data" / "bronze" / "iceberg",
    }
    provider = _NeverSparkProvider(resolved_paths)
    fixed_observer = baseline.FixedObserver

    class EmittingFixedObserver(fixed_observer):
        def __init__(self) -> None:
            super().__init__(
                emitter=build_run_event_emitter(
                    _emitter_config(),
                    provider.resolved_paths,
                    runs_table_sink=append,
                )
            )

    monkeypatch.setattr(baseline, "FixedObserver", EmittingFixedObserver)
    manifest = baseline.capture_case(root, case)
    baseline._publish(root, normalized, manifest)

    expected_root = BASELINE_FIXTURES / case / "metadata"
    actual_root = normalized / "metadata"
    assert _authoritative_files(actual_root) == _authoritative_files(expected_root)

    assert len(appended) == 1
    record = appended[0]
    assert record.status == manifest["status"]
    assert record.source_id == manifest["source_id"]
    assert record.started_at == baseline.STARTED_AT
    assert record.ended_at == baseline.FINISHED_AT
    assert record.duration_seconds == 5.0
    assert record.config_version == compute_config_version(Path(record.source_config_path))

    expected_terminal = "FAIL" if manifest["status"] == "failed" else "COMPLETE"
    event_files = sorted((root / "data" / "metadata" / "lineage" / "openlineage").glob("*.ndjson"))
    lines = [line for path in event_files for line in path.read_text().splitlines()]
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    for event in events:
        _validator(OPENLINEAGE_SCHEMA).validate(event)
        _validator(JANUS_FACET_SCHEMA).validate(event["run"]["facets"]["janusRun"])
    assert [event["eventType"] for event in events] == ["START", expected_terminal]
    assert events[0]["run"]["runId"] == events[1]["run"]["runId"]
    assert provider.was_started is False


def test_every_captured_terminal_shape_carries_the_ac1_fields(tmp_path, monkeypatch):
    """AC-1's named fields are asserted together, not only in projection-unit isolation."""
    records = {}
    current_case = ""

    def append(record, config, resolved_paths, *, logger, timeout_seconds):
        del config, resolved_paths, logger, timeout_seconds
        records[current_case] = record
        return IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")

    fixed_observer = baseline.FixedObserver

    for case in baseline.BASELINE_CASES:
        current_case = case
        root = tmp_path / case
        paths = {
            "metadata_dir": root / "data" / "metadata",
            "iceberg_warehouse_dir": root / "data" / "bronze" / "iceberg",
        }

        class EmittingFixedObserver(fixed_observer):
            def __init__(self) -> None:
                super().__init__(
                    emitter=build_run_event_emitter(
                        _emitter_config(),
                        paths,  # noqa: B023
                        runs_table_sink=append,
                    )
                )

        monkeypatch.setattr(baseline, "FixedObserver", EmittingFixedObserver)
        manifest = baseline.capture_case(root, case)
        contract = manifest["summary"]["planned_run"]["contract"]
        assert records[case].schema_version == contract["schema_version"]
        assert records[case].contract_id == contract["id"]
        expected_contract_version = contract["version"]
        assert records[case].contract_version == expected_contract_version

    assert set(records) == set(baseline.BASELINE_CASES)
    assert {record.source_id for record in records.values()} == {
        "baseline_api",
        "baseline_catalog",
    }
    assert {case: record.status for case, record in records.items()} == {
        "api_success": "succeeded",
        "catalog_success": "succeeded",
        "extraction_failure": "failed",
        "quality_failure": "failed",
        "empty_handoff": "succeeded",
        "replay": "succeeded",
    }
    assert {case: record.records_written for case, record in records.items()} == {
        "api_success": 2,
        "catalog_success": 2,
        "extraction_failure": None,
        "quality_failure": 2,
        "empty_handoff": None,
        "replay": 2,
    }
    assert {case: record.checkpoint_decision for case, record in records.items()} == {
        "api_success": "advanced",
        "catalog_success": "advanced",
        "extraction_failure": None,
        "quality_failure": None,
        "empty_handoff": "advanced",
        "replay": "skipped",
    }
    assert {case: record.quality_outcome for case, record in records.items()} == {
        "api_success": "passed",
        "catalog_success": "passed",
        "extraction_failure": "not_run",
        "quality_failure": "failed",
        "empty_handoff": "passed",
        "replay": "passed",
    }

    assert {case: record.contract_preflight_outcome for case, record in records.items()} == (
        dict.fromkeys(baseline.BASELINE_CASES, "catalog_unavailable")
    )
    assert {record.schema_evolution for record in records.values()} == {None}
    assert {record.malformed_rows for record in records.values()} == {None}
    for record in records.values():
        assert record.started_at == baseline.STARTED_AT
        assert record.ended_at == baseline.FINISHED_AT
        assert record.config_version == compute_config_version(Path(record.source_config_path))
        assert record.record_schema_version == 3


@pytest.mark.parametrize("case", baseline.BASELINE_CASES)
@pytest.mark.parametrize(
    ("budget", "outcome"), [(None, "undeclared"), (5, "within"), (4, "exceeded")]
)
def test_budget_outcome_reaches_terminal_summary_and_source_attempt(
    case, budget, outcome, tmp_path, monkeypatch
):
    from janus.models.config.types import OperationsConfig
    from janus.orchestration import ExecutionTiming, SourceAttempt

    source_payload = baseline.source_payload
    evaluate = OperationsConfig.evaluate_duration
    calls, executed = [], []

    def budget_source(case):
        payload = source_payload(case)
        if budget is not None:
            payload["operations"] = {"max_duration_seconds": budget}
        return payload

    def count_evaluation(self, duration_seconds):
        calls.append(duration_seconds)
        return evaluate(self, duration_seconds)

    owner = baseline.RawToBronzeLoader if case == "replay" else baseline.SourceExecutor
    method = "ingest" if case == "replay" else "execute"
    original = getattr(owner, method)

    def capture_result(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        executed.append(result)
        return result

    monkeypatch.setattr(baseline, "source_payload", budget_source)
    monkeypatch.setattr(OperationsConfig, "evaluate_duration", count_evaluation)
    monkeypatch.setattr(owner, method, capture_result)
    manifest = baseline.capture_case(tmp_path, case)
    assert calls == [5.0]
    assert len(executed) == 1
    result = executed[0]
    assert result.duration_budget_outcome == outcome
    assert manifest["summary"]["executed_run"]["duration_budget_outcome"] == outcome
    assert result.status == ("failed" if case.endswith("failure") else "succeeded")
    attempt = SourceAttempt.from_executed_run(
        result,
        timing=ExecutionTiming(baseline.STARTED_AT, baseline.FINISHED_AT, 5.0),
    )
    assert attempt.evidence["duration_budget_outcome"] == outcome
    assert attempt.to_summary()["evidence"]["duration_budget_outcome"] == outcome
    assert calls == [5.0], "summaries and pipeline evidence must not evaluate again"


METRICS_PROFILE = {
    "transport": "${JANUS_METRICS_TRANSPORT:-disabled}",
    "path": "${JANUS_METRICS_DIR:-metrics}",
}


def _capture_metrics_case(root, case, enabled, monkeypatch, capsys):
    from janus.main import main
    from janus.observability.metrics.transport import MetricsEmissionOutcome
    from janus.utils.environment import load_environment_config
    from tests.support.metrics_order23 import exit_code, write_profile

    write_profile(root, METRICS_PROFILE if enabled else None)
    monkeypatch.setenv("JANUS_METRICS_TRANSPORT", "file")
    monkeypatch.setenv("JANUS_METRICS_DIR", "acceptance-metrics")
    config = load_environment_config("local", root)
    emitter = build_run_event_emitter(
        config,
        {"metadata_dir": root / "data/metadata"},
        runs_table_sink=lambda *args, **kwargs: IcebergAppendResult(
            IcebergAppendOutcome.EMITTED, "metadata.runs"
        ),
    )
    fixed = baseline.FixedObserver

    class MetricsObserver(fixed):
        def __init__(self):
            super().__init__(emitter=emitter)

    captured = []
    owner = baseline.RawToBronzeLoader if case == "replay" else baseline.SourceExecutor
    method = "ingest" if case == "replay" else "execute"
    original = getattr(owner, method)

    def capture_result(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(owner, method, capture_result)
    monkeypatch.setattr(baseline, "FixedObserver", MetricsObserver)
    manifest = baseline.capture_case(root, case)
    assert len(captured) == 1
    result = emitter.last_result
    assert result is not None
    metrics_files = list((root / "data/metadata").rglob("*.ndjson"))
    if enabled:
        assert result.metrics is not None
        assert result.metrics.outcome is MetricsEmissionOutcome.EMITTED
        assert result.metrics.sample_count > 0
        assert metrics_files and all(path.stat().st_size > 0 for path in metrics_files)
        batches = [
            json.loads(line) for path in metrics_files for line in path.read_bytes().splitlines()
        ]
        assert len(batches) == 1
        assert batches[0]["samples"]
    else:
        assert result.metrics is None
        assert not metrics_files

    files = _authoritative_files(root / "data/metadata")
    assert files, "the differential must compare authoritative bytes"
    monkeypatch.setattr(owner, method, lambda *args, **kwargs: captured[0])
    monkeypatch.setattr(
        "janus.cli.run.load_registry", lambda *args, **kwargs: SimpleNamespace(sources=())
    )
    monkeypatch.setattr(
        "janus.cli.run.Planner.plan", lambda *args, **kwargs: captured[0].planned_run
    )
    invocation = ["run", "--project-root", str(root), "--source-id", manifest["source_id"]]
    invocation += (
        ["--ingest-raw-to-bronze", "--bronze-table", "bronze.baseline_replay"]
        if case == "replay"
        else ["--execute"]
    )
    code = exit_code(lambda: main(invocation))
    assert code == (0 if manifest["status"] == "succeeded" else 1)
    assert _authoritative_files(root / "data/metadata") == files
    capsys.readouterr()
    return files, captured[0].status, code


@pytest.mark.parametrize("case", baseline.BASELINE_CASES)
def test_metrics_off_and_on_preserve_exact_bytes_status_and_cli_exit(
    case, tmp_path, monkeypatch, capsys
):
    root = tmp_path / "project"
    with monkeypatch.context() as off_patch:
        disabled = _capture_metrics_case(root, case, False, off_patch, capsys)
    shutil.rmtree(root)
    with monkeypatch.context() as on_patch:
        enabled = _capture_metrics_case(root, case, True, on_patch, capsys)
    assert enabled == disabled


@pytest.mark.skipif(
    Path("/workspace") != PROJECT_ROOT,
    reason="frozen absolute-path goldens require the /workspace reference container",
)
class TestReferenceContainerGoldens:
    @staticmethod
    @pytest.mark.parametrize("case", baseline.BASELINE_CASES)
    def test_metrics_preserve_frozen_task07_golden_bytes_status_and_cli_exit(
        case, monkeypatch, capsys
    ):
        goldens = PROJECT_ROOT / "tests/fixtures/observability/goldens" / case / "metadata"
        expected = _authoritative_files(goldens)
        assert expected, "exact-byte acceptance cannot compare two empty trees"
        run_json = next(goldens.rglob("runs/*.json"))
        expected_run = json.loads(run_json.read_bytes())
        root = Path(expected_run["source_config_path"]).parents[2]
        assert root == PROJECT_ROOT / "data/metadata/evidence/golden-project" / case
        backup = root.with_name(f".{case}.saved-{uuid4().hex}")
        if root.exists():
            root.rename(backup)
        try:
            captures = []
            for enabled in (False, True):
                with monkeypatch.context() as patch:
                    captured = _capture_metrics_case(root, case, enabled, patch, capsys)
                assert captured[0] == expected
                assert captured[1] == expected_run["status"]
                captures.append(captured)
                shutil.rmtree(root)
            assert captures[0] == captures[1]
        finally:
            if root.exists():
                shutil.rmtree(root)
            if backup.exists():
                backup.rename(root)
