"""Engine neutrality: a second engine reads *and* writes the tables Spark writes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import ExecutionPlan, RunContext, resolve_bronze_write_intent
from janus.registry import load_registry
from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
    derive_pyiceberg_default_namespace,
)
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.support.spark_sessions import CatalogTarget, require_pyiceberg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PROJECT_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "catalog_commits"
SOURCE_ID = "catalog_commits_round_trip"

SPARK_SCHEMA = "id BIGINT, name STRING"
SPARK_ROWS = [(1, "written by spark"), (2, "written by spark")]
PYICEBERG_ROW = {"id": 3, "name": "written by pyiceberg"}

#: What both engines must agree the table looks like. `pyiceberg` spells Iceberg's own
#: types, which is the vocabulary the table metadata is written in.
EXPECTED_FIELDS = (("id", "long"), ("name", "string"))

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "runtime",
        "raw_dir": "runtime/raw",
        "bronze_dir": "runtime/bronze",
        "metadata_dir": "runtime/metadata",
    }
}


@pytest.fixture(scope="module")
def spark_written_table(shared_catalog_session, tmp_path_factory) -> str:
    """Two rows, written by Spark through the writer every bronze run uses."""

    project_root = tmp_path_factory.mktemp("janus-round-trip-run")
    source_config = load_registry(FIXTURE_PROJECT_ROOT).get_source(SOURCE_ID)
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-order13-round-trip-001",
            environment="local",
            project_root=project_root,
            started_at=datetime(2026, 7, 11, 9, 0, tzinfo=UTC),
        ),
    )
    writer = SparkDatasetWriter(
        StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, project_root)
    )

    result = writer.write(
        shared_catalog_session.createDataFrame(SPARK_ROWS, SPARK_SCHEMA),
        plan,
        "bronze",
        intent=resolve_bronze_write_intent(plan),
        apply_repartition=False,
    )
    return result.path


@pytest.fixture(scope="module")
def pyiceberg_table(catalog_target: CatalogTarget, spark_written_table: str):
    """The same table, reached by the second engine through the derived properties only."""

    catalog_module = require_pyiceberg()

    config = catalog_target.environment_config()
    catalog = catalog_module.load_catalog(
        derive_pyiceberg_catalog_name(config),
        **derive_pyiceberg_catalog_properties(config, catalog_target.resolved_paths),
    )
    return catalog.load_table(spark_written_table)


def test_the_identifier_pyiceberg_needs_is_the_one_the_profile_derives(
    catalog_target: CatalogTarget, spark_written_table: str
):
    """`pyiceberg` has no `default-namespace` property, so it must spell what Spark resolves.

    The fixture source writes into the profile's own default namespace precisely so this
    comparison is meaningful: the namespace Spark would have filled in implicitly is the
    namespace the derivation hands a `pyiceberg` caller.
    """

    config = catalog_target.environment_config()
    namespace = derive_pyiceberg_default_namespace(config)
    table_name = load_registry(FIXTURE_PROJECT_ROOT).get_source(SOURCE_ID).outputs.bronze

    assert namespace is not None
    assert f"{namespace}.{table_name.table_name}" == spark_written_table


@pytest.fixture(scope="module")
def pyiceberg_initial_read(pyiceberg_table) -> tuple[tuple[tuple[str, str], ...], list[dict]]:
    """What the second engine saw before it wrote anything of its own.

    Captured once, and the append below is ordered behind it, so the two read assertions
    cannot be perturbed by the write — however the tests are ordered or selected.
    """

    fields = tuple(
        (field.name, str(field.field_type)) for field in pyiceberg_table.schema().fields
    )
    return fields, _rows(pyiceberg_table)


def test_pyiceberg_reads_the_schema_spark_wrote(pyiceberg_initial_read):
    fields, _rows_read = pyiceberg_initial_read

    assert fields == EXPECTED_FIELDS


def test_pyiceberg_reads_the_rows_spark_wrote(pyiceberg_initial_read):
    _fields, rows = pyiceberg_initial_read

    assert rows == [{"id": id_, "name": name} for id_, name in SPARK_ROWS]


@pytest.fixture(scope="module")
def pyiceberg_commit(pyiceberg_table, pyiceberg_initial_read) -> int:
    """The second engine's own commit. Returns the snapshot id it produced."""

    import pyarrow as pa

    pyiceberg_table.append(
        pa.Table.from_pylist([PYICEBERG_ROW], schema=pyiceberg_table.schema().as_arrow())
    )
    snapshot = pyiceberg_table.current_snapshot()
    assert snapshot is not None
    return snapshot.snapshot_id


def test_spark_reads_the_row_pyiceberg_committed(
    shared_catalog_session, spark_written_table, pyiceberg_commit
):
    """A fresh session, because a stale one would prove the cache works, not the catalog."""

    fresh = shared_catalog_session.newSession()
    rows = fresh.table(spark_written_table).orderBy("id").collect()

    assert [(row["id"], row["name"]) for row in rows] == [
        *SPARK_ROWS,
        (PYICEBERG_ROW["id"], PYICEBERG_ROW["name"]),
    ]


def test_spark_sees_pyiceberg_in_the_tables_history(
    shared_catalog_session, spark_written_table, pyiceberg_commit
):
    """The commit is in the shared log, not merely visible in a read."""

    snapshots = shared_catalog_session.sql(
        f"SELECT snapshot_id FROM {spark_written_table}.snapshots"
    ).collect()

    assert pyiceberg_commit in {row["snapshot_id"] for row in snapshots}


def _rows(table) -> list[dict[str, object]]:
    return sorted(table.scan().to_arrow().to_pylist(), key=lambda row: row["id"])
