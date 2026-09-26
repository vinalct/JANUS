"""Acceptance evidence that composes the focused observability contracts end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        baseline.capture_case(root, case)

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
    for record in records.values():
        assert record.started_at == baseline.STARTED_AT
        assert record.ended_at == baseline.FINISHED_AT
        assert record.config_version == compute_config_version(Path(record.source_config_path))
