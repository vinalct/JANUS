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
from janus.observability.runs_table import RUNS_TABLE_SCHEMA, IcebergType


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
class FakePrimitive:
    type_name: str

    def __str__(self) -> str:
        return self.type_name


@dataclass(frozen=True)
class FakeListType:
    element_id: int
    element: FakePrimitive
    element_required: bool = True

    def __str__(self) -> str:
        return f"list<{self.element}>"


@dataclass(frozen=True)
class FakeField:
    field_id: int
    name: str
    field_type: Any
    required: bool


@dataclass(frozen=True)
class FakeSchema:
    fields: tuple[FakeField, ...]

    def as_arrow(self) -> str:
        return "arrow-schema"


@dataclass(frozen=True)
class FakeSchemaView:
    fields: tuple[FakeField, ...]

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


class FakeSchemaUpdate:
    def __init__(self, table: FakeTable):
        self.table = table
        self.added: list[FakeField] = []

    def __enter__(self):
        self.table.schema_update_count += 1
        if self.table.schema_update_wait is not None:
            release = self.table.schema_update_wait
            release.wait(timeout=1)
        if self.table.schema_update_error is not None:
            raise self.table.schema_update_error
        return self

    def add_column(self, name: str, field_type: Any, *, required: bool):
        existing_ids = [field.field_id for field in self.table.live_schema.fields]
        existing_ids.extend(field.field_id for field in self.added)
        existing_ids.extend(
            field.field_type.element_id
            for field in self.table.live_schema.fields
            if isinstance(field.field_type, FakeListType)
        )
        field_id = max(existing_ids, default=0) + 1
        if not self.added:
            field_id += self.table.assigned_id_offset
        self.added.append(FakeField(field_id, name, field_type, required))

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None:
            self.table.live_schema = FakeSchema((*self.table.live_schema.fields, *self.added))
        return False


class FakeTable:
    def __init__(self, schema: Any):
        self.live_schema = schema
        self.appended: list[Any] = []
        self.append_error: Exception | None = None
        self.schema_update_error: Exception | None = None
        self.schema_update_wait: threading.Event | None = None
        self.schema_update_count = 0
        self.assigned_id_offset = 0

    def schema(self):
        return self.live_schema

    def update_schema(self):
        return FakeSchemaUpdate(self)

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
        self.loaded_tables: list[str] = []

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
        self.loaded_tables.append(identifier)
        return self.tables[identifier]


class CapturingLogger:
    def __init__(self):
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))


V1_COLUMN_COUNT = 42
V2_COLUMN_COUNT = 45
V2_COLUMNS = ("schema_version", "contract_id", "contract_version")
V3_COLUMNS = ("contract_preflight_outcome", "schema_evolution", "malformed_rows")


def _v1_schema(dependencies):
    return sink._declared_schema(dependencies, RUNS_TABLE_SCHEMA[:V1_COLUMN_COUNT])


def _v2_schema(dependencies):
    return sink._declared_schema(dependencies, RUNS_TABLE_SCHEMA[:V2_COLUMN_COUNT])


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
        iceberg_type: (lambda type_name=iceberg_type.value: FakePrimitive(type_name))
        for iceberg_type in IcebergType
        if iceberg_type is not IcebergType.STRING_LIST
    }
    dependencies = sink._EngineDependencies(
        load_catalog=load_catalog,
        pyarrow=SimpleNamespace(Table=FakeArrowTable),
        schema_type=lambda *fields: FakeSchema(fields),
        nested_field_type=lambda **values: FakeField(**values),
        partition_field_type=FakeValue.build,
        partition_spec_type=FakeValue.build,
        day_transform_type=lambda: "day",
        primitive_types=primitive_types,
        list_type=lambda **values: FakeListType(**values),
        namespace_already_exists=FakeNamespaceAlreadyExistsError,
        table_already_exists=FakeTableAlreadyExistsError,
    )
    return dependencies, captured


def test_additive_plan_contains_only_missing_nullable_fields():
    declared = (
        (1, "run_id", "string", True),
        (44, "schema_version", "string", False),
        (45, "contract_id", "string", False),
    )

    plan = sink._plan_additive_evolution(declared[:1], declared)

    assert plan is not None
    assert plan.columns == tuple((*field, None, None) for field in declared[1:])


def test_a_fresh_table_is_created_at_the_prefix_pyiceberg_numbers_as_declared():
    pytest.importorskip("pyarrow")
    schema_module = pytest.importorskip("pyiceberg.schema")
    dependencies = sink._load_engine_dependencies()
    creation = sink._creation_columns()
    created = schema_module.assign_fresh_schema_ids(
        sink._declared_schema(dependencies, creation)
    )
    declared = sink._schema_field_signatures(sink._declared_schema(dependencies))

    assert sink._schema_field_signatures(created) == declared[: len(creation)]
    plan = sink._plan_additive_evolution(sink._schema_field_signatures(created), declared)
    assert plan is not None
    assert [field[1] for field in plan.columns] == [*V2_COLUMNS, *V3_COLUMNS]


def test_additive_plan_for_an_empty_difference_has_no_columns():
    fields = ((1, "run_id", "string", True),)

    plan = sink._plan_additive_evolution(fields, fields)

    assert plan is not None
    assert plan.columns == ()


@pytest.mark.parametrize(
    "live,declared",
    (
        (
            ((1, "run_id", "string", True),),
            ((1, "run_id", "string", True), (2, "required", "string", True)),
        ),
        (((1, "run_id", "int", True),), ((1, "run_id", "string", True),)),
        (((1, "renamed", "string", True),), ((1, "run_id", "string", True),)),
        (
            ((1, "run_id", "string", True), (2, "extra", "string", False)),
            ((1, "run_id", "string", True),),
        ),
        (((9, "run_id", "string", True),), ((1, "run_id", "string", True),)),
        (
            ((1, "run_id", "string", True), (3, "later", "string", True)),
            (
                (1, "run_id", "string", True),
                (2, "dropped_nullable", "string", False),
                (3, "later", "string", True),
            ),
        ),
        (
            ((33, "quality_failed_checks", "list<string>", False, 88),),
            ((33, "quality_failed_checks", "list<string>", False, 43),),
        ),
        (
            ((33, "quality_failed_checks", "list<string>", False, 43, False),),
            ((33, "quality_failed_checks", "list<string>", False, 43, True),),
        ),
    ),
    ids=(
        "missing-required",
        "type-change",
        "rename",
        "extra-live",
        "id-mismatch",
        "dropped-old-nullable",
        "element-id-mismatch",
        "element-requiredness",
    ),
)
def test_additive_plan_refuses_every_conflicting_schema(live, declared):
    assert sink._plan_additive_evolution(live, declared) is None


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
            from_pylist=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("arrow detail"))
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


def test_empty_evolution_plan_treats_the_declared_fields_as_equal(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    declared = sink._declared_schema(dependencies)
    table = FakeTable(FakeSchemaView(tuple(reversed(declared.fields))))
    catalog.tables["metadata.runs"] = table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.EMITTED
    assert table.schema_update_count == 0
    assert len(table.appended) == 1


def test_v1_schema_evolves_once_reloads_and_appends(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    table = FakeTable(_v1_schema(dependencies))
    catalog.tables["metadata.runs"] = table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.EMITTED
    assert table.schema_update_count == 1, "v2 and v3 columns land in one transaction"
    assert catalog.loaded_tables == ["metadata.runs", "metadata.runs"]
    assert [field.field_id for field in table.schema().fields[-6:]] == [*range(44, 50)]
    assert [field.name for field in table.schema().fields[-6:]] == [*V2_COLUMNS, *V3_COLUMNS]
    assert len(table.appended) == 1


def test_v2_schema_evolves_to_v3_once_reloads_and_appends(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    table = FakeTable(_v2_schema(dependencies))
    catalog.tables["metadata.runs"] = table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.EMITTED
    assert table.schema_update_count == 1
    assert catalog.loaded_tables == ["metadata.runs", "metadata.runs"]
    assert len(table.schema().fields) == len(RUNS_TABLE_SCHEMA)
    assert [field.field_id for field in table.schema().fields[-3:]] == [47, 48, 49]
    assert [field.name for field in table.schema().fields[-3:]] == [*V3_COLUMNS]
    assert all(not field.required for field in table.schema().fields[-3:])
    assert len(table.appended) == 1


def test_schema_evolution_failure_is_reported_at_its_own_step(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    table = FakeTable(_v1_schema(dependencies))
    table.schema_update_error = RuntimeError("commit conflict")
    catalog.tables["metadata.runs"] = table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.FAILED
    assert result.reason == "schema_evolution_failed"
    assert result.step == "schema_evolution"
    assert result.exception_type == "RuntimeError"
    assert table.appended == []


def test_schema_evolution_refuses_unexpected_assigned_field_ids(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    table = FakeTable(_v1_schema(dependencies))
    table.assigned_id_offset = 1
    catalog.tables["metadata.runs"] = table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.SKIPPED
    assert result.reason == "live_schema_does_not_match_declaration_after_evolution"
    assert result.step == "schema_validation"
    assert [field.field_id for field in table.schema().fields[-6:]] == [*range(45, 51)]
    assert table.appended == []


def test_schema_evolution_work_remains_inside_the_append_budget(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    table = FakeTable(_v1_schema(dependencies))
    release = threading.Event()
    table.schema_update_wait = release
    catalog.tables["metadata.runs"] = table
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path), timeout_seconds=0.02)
    release.set()

    assert result.outcome is IcebergAppendOutcome.FAILED
    assert result.step == "budget"
    assert result.exception_type == "AppendTimeoutError"


def test_schema_mismatch_skips_without_altering_or_appending(monkeypatch, tmp_path):
    catalog = FakeCatalog()
    dependencies, _captured = _dependencies(catalog)
    mismatched = FakeTable(
        FakeSchema((FakeField(999, "unexpected", FakePrimitive("string"), False),))
    )
    catalog.tables["metadata.runs"] = mismatched
    monkeypatch.setattr(sink, "_load_engine_dependencies", lambda: dependencies)

    result = append_run_record(_record(), _config(), _paths(tmp_path))

    assert result.outcome is IcebergAppendOutcome.SKIPPED
    assert result.reason == "live_schema_does_not_match_declaration"
    assert mismatched.live_schema == FakeSchema(
        (FakeField(999, "unexpected", FakePrimitive("string"), False),)
    )
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
