"""FR-4 acceptance matrix: catalog degradation cannot alter a successful run."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

import janus.observability.emission as emission
import janus.observability.iceberg_sink as iceberg_sink
from janus.observability import (
    RunEmissionOutcome,
    append_run_record,
    build_run_event_emitter,
)
from janus.utils.catalog_options import (
    JDBC_CATALOG_TYPE,
    REST_CATALOG_TYPE,
    SUPPORTED_CATALOG_TYPES,
)
from tests.support import observability_baseline as baseline
from tests.unit.observability import test_pyiceberg_append_sink as sink_fakes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "observability" / "baseline" / "api_success"
)
UNREPRESENTABLE_CATALOG_TYPE = next(
    iter(SUPPORTED_CATALOG_TYPES - {JDBC_CATALOG_TYPE, REST_CATALOG_TYPE})
)
AUTHORITATIVE_DIRECTORIES = frozenset({"runs", "lineage", "checkpoints", "validations"})

RUNS_TABLE_FAILURES = (
    "dependency_import",
    "catalog_load",
    "namespace_create",
    "table_create",
    "schema_validation",
    "append",
    "budget",
    "catalog_properties",
    "unrepresentable_catalog",
    "projection",
)


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.events.append(("info", event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self.events.append(("warning", event, fields))

    @property
    def warning_events(self) -> list[tuple[str, dict[str, Any]]]:
        return [(event, fields) for level, event, fields in self.events if level == "warning"]

    @property
    def text(self) -> str:
        return json.dumps(self.events, default=str)


def _authoritative_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
        if AUTHORITATIVE_DIRECTORIES.intersection(path.relative_to(root).parts)
    }


def _sink_failure(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], str, threading.Event | None]:
    config = sink_fakes._config()
    paths = sink_fakes._paths(tmp_path)
    catalog = sink_fakes.FakeCatalog()
    release = None

    if failure == "dependency_import":
        monkeypatch.setattr(
            iceberg_sink,
            "_load_engine_dependencies",
            lambda: (_ for _ in ()).throw(ImportError("secret import detail")),
        )
        return config, paths, "dependency_import", release

    if failure == "catalog_properties":
        config["observability"] = "secret malformed block"
        return config, paths, "catalog_properties", release

    if failure == "unrepresentable_catalog":
        return (
            sink_fakes._config(UNREPRESENTABLE_CATALOG_TYPE),
            paths,
            "catalog_properties",
            release,
        )

    load_error = RuntimeError("secret credential detail") if failure == "catalog_load" else None
    dependencies, _captured = sink_fakes._dependencies(catalog, load_error=load_error)
    if failure == "namespace_create":
        catalog.namespace_error = RuntimeError("secret namespace detail")
    if failure == "table_create":
        catalog.table_error = RuntimeError("secret table detail")
    if failure == "schema_validation":
        catalog.tables["metadata.runs"] = sink_fakes.FakeTable(
            sink_fakes.FakeSchema(("unexpected",))
        )
    if failure == "append":
        original_create_table = catalog.create_table

        def create_failing_table(*args, **kwargs):
            table = original_create_table(*args, **kwargs)
            table.append_error = RuntimeError("secret append detail")
            return table

        catalog.create_table = create_failing_table
    monkeypatch.setattr(iceberg_sink, "_load_engine_dependencies", lambda: dependencies)

    if failure == "budget":
        release = threading.Event()

        def hang(record, request):
            del record, request
            release.wait(timeout=1)
            return iceberg_sink.IcebergAppendResult(
                iceberg_sink.IcebergAppendOutcome.EMITTED, "metadata.runs"
            )

        monkeypatch.setattr(iceberg_sink, "_append_unbounded", hang)
    return config, paths, failure, release


@pytest.mark.parametrize("failure", RUNS_TABLE_FAILURES)
def test_every_runs_table_failure_preserves_artifacts_exit_status_and_redacted_warning(
    failure,
    tmp_path,
    monkeypatch,
):
    """Exercise all catalog stages through the observer, after authoritative persistence."""
    root = tmp_path / "project"
    normalized = tmp_path / "normalized"
    logger = _Logger()
    persisted = []
    emitters = []
    config, paths, expected_stage, release = _sink_failure(failure, monkeypatch, tmp_path)

    def projector(artifacts):
        persisted.append(artifacts)
        if failure == "projection":
            raise RuntimeError("secret projection detail")
        return emission._project_run_record(artifacts)

    fixed_observer = baseline.FixedObserver

    class FailingEmissionObserver(fixed_observer):
        def __init__(self) -> None:
            emitter = build_run_event_emitter(
                config,
                paths,
                logger=logger,
                timeout_seconds=0.02 if failure == "budget" else 5.0,
                projector=projector,
                runs_table_sink=append_run_record,
            )
            emitters.append(emitter)
            super().__init__(emitter=emitter)

    monkeypatch.setattr(baseline, "FixedObserver", FailingEmissionObserver)
    try:
        manifest = baseline.capture_case(root, "api_success")
    finally:
        if release is not None:
            release.set()
    baseline._publish(root, normalized, manifest)

    assert manifest["status"] == "succeeded"
    assert (0 if manifest["status"] == "succeeded" else 1) == 0
    assert len(persisted) == 1
    artifacts = persisted[0]
    assert artifacts.run_metadata.status == "succeeded"
    assert artifacts.lineage_record is not None
    assert artifacts.lineage_record.status == "succeeded"
    assert artifacts.checkpoint_result is not None
    assert artifacts.validation_report is not None
    assert artifacts.run_metadata_path.is_file()
    assert artifacts.lineage_path is not None and artifacts.lineage_path.is_file()

    expected = _authoritative_files(BASELINE_FIXTURE / "metadata")
    actual = _authoritative_files(normalized / "metadata")
    assert actual == expected

    assert len(emitters) == 1
    result = emitters[0].last_result
    assert result is not None
    assert result.outcome in {RunEmissionOutcome.FAILED, RunEmissionOutcome.SKIPPED}
    assert result.stage == expected_stage

    warnings = logger.warning_events
    assert warnings
    assert any(
        fields.get("stage") == expected_stage or fields.get("step") == expected_stage
        for _event, fields in warnings
    )
    assert "secret" not in logger.text
