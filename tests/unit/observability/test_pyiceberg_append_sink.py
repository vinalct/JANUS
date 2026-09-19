"""The guarded, engine-lazy append path for ``metadata.runs``."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import janus
import janus.observability.iceberg_sink as sink
import janus.observability.runs_table as runs_table
from janus.observability import IcebergAppendOutcome, RunRecord, append_run_record
from janus.observability.runs_table import IcebergType


class FakeNamespaceAlreadyExistsError(Exception):
    pass


class FakeTableAlreadyExistsError(Exception):
    pass


@dataclass(frozen=True)
class FakeValue:
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> FakeValue:
        return cls(args, tuple(sorted(kwargs.items())))


@dataclass(frozen=True)
class FakeSchema:
    fields: tuple[Any, ...]

    def as_arrow(self) -> str:
        return "arrow-schema"


class FakeArrowTable:
    rows: list[dict[str, Any]]
    schema: Any

    @classmethod
    def from_pylist(cls, rows, *, schema):
        cls.rows = rows
        cls.schema = schema
        return ("arrow-table", rows, schema)


class FakeTable:
    def __init__(self, schema: Any):
        self.live_schema = schema
        self.appended: list[Any] = []
        self.append_error: Exception | None = None

    def schema(self):
        return self.live_schema

    def append(self, arrow_table):
        if self.append_error is not None:
            raise self.append_error
        self.appended.append(arrow_table)


class FakeCatalog:
    def __init__(self):
        self.namespaces: set[str] = set()
        self.tables: dict[str, FakeTable] = {}
        self.namespace_error: Exception | None = None
        self.table_error: Exception | None = None
        self.created_partition_spec: Any = None

    def create_namespace(self, namespace: str):
        if self.namespace_error is not None:
            raise self.namespace_error
        if namespace in self.namespaces:
            raise FakeNamespaceAlreadyExistsError(namespace)
        self.namespaces.add(namespace)

    def create_table(self, identifier: str, *, schema: Any, partition_spec: Any):
        if self.table_error is not None:
            raise self.table_error
        if identifier in self.tables:
            raise FakeTableAlreadyExistsError(identifier)
        table = FakeTable(schema)
        self.tables[identifier] = table
        self.created_partition_spec = partition_spec
        return table

    def load_table(self, identifier: str):
        return self.tables[identifier]


class CapturingLogger:
    def __init__(self):
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))


def _record(run_id: str = "run-001") -> RunRecord:
    instant = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    return RunRecord(
        run_id=run_id,
        source_id="source",
        source_name="Source",
        environment="local",
        strategy_family="api",
        strategy_variant="paginated",
        extraction_mode="incremental",
        status="succeeded",
        started_at=instant,
        emitted_at=instant,
        config_version="sha256:config",
        source_config_path="conf/sources/source.yaml",
        artifact_count=1,
        quality_outcome="not_run",
    )


def _config(catalog_type: str = "jdbc") -> dict[str, Any]:
    return {
        "spark": {
            "iceberg": {
                "catalog_name": "janus",
                "catalog_type": catalog_type,
                "uri": "jdbc:sqlite:data/catalog.sqlite",
                "warehouse_dir": "data/warehouse",
            }
        }
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "iceberg_catalog_db": tmp_path / "catalog.sqlite",
        "iceberg_warehouse_dir": tmp_path / "warehouse",
    }


def _dependencies(catalog: FakeCatalog, *, load_error: Exception | None = None):
    captured: list[tuple[str, dict[str, str]]] = []

    def load_catalog(name: str, **properties: str):
        captured.append((name, properties))
        if load_error is not None:
            raise load_error
        return catalog

    primitive_types = {
        iceberg_type: type(
            f"Fake{iceberg_type.value.title()}",
            (),
            {"__eq__": lambda self, other: type(self) is type(other)},
        )
        for iceberg_type in IcebergType
        if iceberg_type is not IcebergType.STRING_LIST
    }
    dependencies = sink._EngineDependencies(
        load_catalog=load_catalog,
        pyarrow=SimpleNamespace(Table=FakeArrowTable),
        schema_type=lambda *fields: FakeSchema(fields),
        nested_field_type=FakeValue.build,
        partition_field_type=FakeValue.build,
        partition_spec_type=FakeValue.build,
        day_transform_type=lambda: "day",
        primitive_types=primitive_types,
        list_type=FakeValue.build,
        namespace_already_exists=FakeNamespaceAlreadyExistsError,
        table_already_exists=FakeTableAlreadyExistsError,
    )
    return dependencies, captured


def test_importing_observability_and_lineage_loads_neither_engine():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(janus.__file__).parents[1]), env.get("PYTHONPATH")) if part
    )
    program = (
        "import sys, janus.lineage, janus.observability\n"
        "roots = ('pyiceberg', 'pyarrow')\n"
        "print(sorted(m for m in sys.modules "
        "if any(m == r or m.startswith(r + '.') for r in roots)))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_missing_engines_skip_and_log_instead_of_raising(monkeypatch, tmp_path):
    logger = CapturingLogger()
    monkeypatch.setattr(
        sink,
        "_load_engine_dependencies",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )

    result = append_run_record(_record(), _config(), _paths(tmp_path), logger=logger)

    assert result.outcome is IcebergAppendOutcome.SKIPPED
    assert result.reason == "pyiceberg_or_pyarrow_unavailable"
    assert result.step == "dependency_import"
    assert result.table_identifier == "metadata.runs"
    assert logger.warnings == [
        (
            "runs_table_append_degraded",
            {
                "outcome": IcebergAppendOutcome.SKIPPED,
                "reason": "pyiceberg_or_pyarrow_unavailable",
                "step": "dependency_import",
                "exception_type": "ImportError",
                "table_identifier": "metadata.runs",
            },
        )
    ]


def test_derived_catalog_properties_pass_through_unchanged(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, captured = _dependencies(catalog)
    properties = {"type": "sql", "uri": "sentinel", "warehouse": "sentinel-warehouse"}
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)
    monkeypatch.setattr(
        runs_table,
        "derive_pyiceberg_catalog_name",
        lambda config: "derived_catalog",
    )
    monkeypatch.setattr(
        sink,
        "derive_pyiceberg_catalog_properties",
        lambda config, paths: properties,
    )

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.emitted
    assert captured == [("derived_catalog", properties)]


def test_bootstrap_is_idempotent_and_each_call_appends_one_row(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    first = append_run_record(_record("run-001"), _config(), _paths(tmp_path))
    second = append_run_record(_record("run-002"), _config(), _paths(tmp_path))

    table = catalog.tables["metadata.runs"]
    assert first.emitted and second.emitted
    assert catalog.namespaces == {"metadata"}
    assert len(catalog.tables) == 1
    assert len(table.appended) == 2
    assert FakeArrowTable.rows[0]["run_id"] == "run-002"
    assert catalog.created_partition_spec.args[0].kwargs == (
        ("field_id", 1000),
        ("name", "emitted_at_day"),
        ("source_id", 15),
        ("transform", "day"),
    )


@pytest.mark.parametrize(
    ("failure", "expected_step"),
    (
        ("catalog_load", "catalog_load"),
        ("namespace_create", "namespace_create"),
        ("table_create", "table_create"),
        ("arrow_conversion", "arrow_conversion"),
        ("append", "append"),
    ),
)
def test_each_guarded_step_returns_failed(monkeypatch, tmp_path, failure, expected_step):
    catalog = FakeCatalog()
    load_error = RuntimeError("secret catalog detail") if failure == "catalog_load" else None
    dependencies, _captured = _dependencies(catalog, load_error=load_error)
    if failure == "namespace_create":
        catalog.namespace_error = RuntimeError("namespace detail")
    if failure == "table_create":
        catalog.table_error = RuntimeError("table detail")
    if failure == "arrow_conversion":
        dependencies.pyarrow.Table = SimpleNamespace(
            from_pylist=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("arrow detail")
            )
        )
    if failure == "append":
        original_create_table = catalog.create_table

        def create_failing_table(*args, **kwargs):
            table = original_create_table(*args, **kwargs)
            table.append_error = RuntimeError("append detail")
            return table

        catalog.create_table = create_failing_table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.FAILED
    assert result.step == expected_step
    assert result.reason == f"{expected_step}_failed"
    assert result.exception_type == "RuntimeError"


def test_hadoop_profile_is_a_named_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sink,
        "_load_engine_dependencies",
        lambda: pytest.fail("the hadoop degradation must happen before engine import"),
    )

    result = append_run_record(_record(), _config("hadoop"), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.SKIPPED
    assert result.reason == "hadoop_catalog_unrepresentable"
    assert result.exception_type == "HadoopCatalogUnrepresentableError"


def test_schema_mismatch_skips_without_altering_or_appending(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    mismatched = FakeTable(FakeSchema(("unexpected",)))
    catalog.tables["metadata.runs"] = mismatched
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.SKIPPED
    assert result.reason == "live_schema_does_not_match_declaration"
    assert mismatched.live_schema == FakeSchema(("unexpected",))
    assert mismatched.appended == []


def test_hung_catalog_work_is_bounded(monkeypatch, tmp_path):
    release = threading.Event()

    def hang(_record, _request):
        release.wait(timeout=1)
        return sink.IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")

    monkeypatch.setattr(sink, "_append_unbounded", hang)
    started_at = time.monotonic()

    result = append_run_record(
        _record(),
        _config(),
        _paths(tmp_path),
        timeout_seconds=0.02,
    )
    elapsed = time.monotonic() - started_at
    release.set()

    assert result.outcome is IcebergAppendOutcome.FAILED
    assert result.step == "budget"
    assert result.exception_type == "AppendTimeoutError"
    assert elapsed < 0.2
