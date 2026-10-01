"""The runs sink commits through PyIceberg into the catalog Spark reads."""

from __future__ import annotations

import multiprocessing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import Any
from uuid import uuid4

import pytest

import janus.observability.iceberg_sink as iceberg_sink
from janus.observability import IcebergAppendOutcome, RunRecord, append_run_record
from janus.observability.runs_table import IcebergType, RunsTableColumn
from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
)
from tests.support.spark_sessions import (
    CatalogTarget,
    catalog_acceptance_prerequisites_available,
    require_pyiceberg,
)

PROCESS_RESULT_TIMEOUT_SECONDS = 15
PROCESS_EXIT_TIMEOUT_SECONDS = 20
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_TABLE_SCHEMA_V1 = (
    RunsTableColumn(1, "run_id", IcebergType.STRING, False),
    RunsTableColumn(2, "source_id", IcebergType.STRING, False),
    RunsTableColumn(3, "source_name", IcebergType.STRING, False),
    RunsTableColumn(4, "environment", IcebergType.STRING, False),
    RunsTableColumn(5, "strategy_family", IcebergType.STRING, False),
    RunsTableColumn(6, "strategy_variant", IcebergType.STRING, False),
    RunsTableColumn(7, "extraction_mode", IcebergType.STRING, False),
    RunsTableColumn(8, "source_hook", IcebergType.STRING, True),
    RunsTableColumn(9, "pipeline_run_id", IcebergType.STRING, True),
    RunsTableColumn(10, "pipeline_attempt", IcebergType.INTEGER, True),
    RunsTableColumn(11, "trigger", IcebergType.STRING, True),
    RunsTableColumn(12, "status", IcebergType.STRING, False),
    RunsTableColumn(13, "started_at", IcebergType.TIMESTAMPTZ, False),
    RunsTableColumn(14, "ended_at", IcebergType.TIMESTAMPTZ, True),
    RunsTableColumn(15, "emitted_at", IcebergType.TIMESTAMPTZ, False),
    RunsTableColumn(16, "duration_seconds", IcebergType.DOUBLE, True),
    RunsTableColumn(17, "config_version", IcebergType.STRING, False),
    RunsTableColumn(18, "source_config_path", IcebergType.STRING, False),
    RunsTableColumn(19, "records_extracted", IcebergType.LONG, True),
    RunsTableColumn(20, "artifact_count", IcebergType.INTEGER, False),
    RunsTableColumn(21, "records_written", IcebergType.LONG, True),
    RunsTableColumn(22, "bronze_table_identifier", IcebergType.STRING, True),
    RunsTableColumn(23, "bronze_write_mode", IcebergType.STRING, True),
    RunsTableColumn(24, "checkpoint_field", IcebergType.STRING, True),
    RunsTableColumn(25, "checkpoint_strategy", IcebergType.STRING, True),
    RunsTableColumn(26, "checkpoint_value", IcebergType.STRING, True),
    RunsTableColumn(27, "checkpoint_decision", IcebergType.STRING, True),
    RunsTableColumn(28, "checkpoint_advanced", IcebergType.BOOLEAN, True),
    RunsTableColumn(29, "quality_outcome", IcebergType.STRING, False),
    RunsTableColumn(30, "quality_checks_passed", IcebergType.INTEGER, True),
    RunsTableColumn(31, "quality_checks_failed", IcebergType.INTEGER, True),
    RunsTableColumn(32, "quality_checks_skipped", IcebergType.INTEGER, True),
    RunsTableColumn(33, "quality_failed_checks", IcebergType.STRING_LIST, True, element_id=43),
    RunsTableColumn(34, "failure_reason", IcebergType.STRING, True),
    RunsTableColumn(35, "failure_reason_truncated", IcebergType.BOOLEAN, True),
    RunsTableColumn(36, "failure_reason_length", IcebergType.INTEGER, True),
    RunsTableColumn(37, "error_type", IcebergType.STRING, True),
    RunsTableColumn(38, "run_metadata_path", IcebergType.STRING, True),
    RunsTableColumn(39, "lineage_path", IcebergType.STRING, True),
    RunsTableColumn(40, "checkpoint_history_path", IcebergType.STRING, True),
    RunsTableColumn(41, "validation_report_path", IcebergType.STRING, True),
    RunsTableColumn(42, "record_schema_version", IcebergType.INTEGER, False),
)
RUNS_TABLE_SCHEMA_V2 = (
    *RUNS_TABLE_SCHEMA_V1,
    RunsTableColumn(44, "schema_version", IcebergType.STRING, True),
    RunsTableColumn(45, "contract_id", IcebergType.STRING, True),
    RunsTableColumn(46, "contract_version", IcebergType.STRING, True),
)
V2_COLUMNS = ("schema_version", "contract_id", "contract_version")
V3_COLUMNS = ("contract_preflight_outcome", "schema_evolution", "malformed_rows")

RUNS_TABLE_SKIP_REASON = (
    "AC1_REAL_CATALOG_SUITE_UNAVAILABLE: catalog engines or seeded jars are missing"
)
pytestmark = pytest.mark.skipif(
    not catalog_acceptance_prerequisites_available(),
    reason=RUNS_TABLE_SKIP_REASON,
)


def _record(run_id: str) -> RunRecord:
    instant = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    return RunRecord(
        run_id=run_id,
        source_id="runs_sink_source",
        source_name="Runs sink source",
        environment="local",
        strategy_family="api",
        strategy_variant="paginated",
        extraction_mode="incremental",
        status="succeeded",
        started_at=instant,
        ended_at=instant,
        emitted_at=instant,
        duration_seconds=0.0,
        config_version="sha256:runs-sink",
        source_config_path="conf/sources/runs_sink.yaml",
        records_extracted=1,
        artifact_count=1,
        records_written=1,
        bronze_table_identifier="bronze.runs_sink_source",
        bronze_write_mode="append",
        quality_outcome="not_run",
    )


def _isolated_config(catalog_target: CatalogTarget) -> tuple[dict[str, Any], str]:
    identifier = f"metadata.runs_sink_{uuid4().hex}"
    config = catalog_target.environment_config()
    config["observability"] = {"runs_table": identifier}
    return config, identifier


def _append_in_process(
    run_id: str,
    config: dict[str, Any],
    resolved_paths: dict[str, Any],
    result_queue: Any,
) -> None:
    result = append_run_record(_record(run_id), config, resolved_paths)
    result_queue.put((result.outcome, result.reason, result.step, result.exception_type))


def _run_process_append(
    run_id: str,
    config: dict[str, Any],
    resolved_paths: dict[str, Any],
) -> tuple[str, str | None, str | None, str | None]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_append_in_process,
        args=(run_id, config, resolved_paths, result_queue),
    )
    process.start()
    try:
        result = result_queue.get(timeout=PROCESS_RESULT_TIMEOUT_SECONDS)
    except Empty:
        process.terminate()
        process.join(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
        raise AssertionError("runs sink subprocess produced no result") from None
    process.join(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
    assert process.exitcode == 0
    return result


def _load_table(config: dict[str, Any], catalog_target: CatalogTarget):
    catalog_module = require_pyiceberg()
    catalog = catalog_module.load_catalog(
        derive_pyiceberg_catalog_name(config),
        **derive_pyiceberg_catalog_properties(config, catalog_target.resolved_paths),
    )
    return catalog.load_table(config["observability"]["runs_table"])


def test_two_calls_bootstrap_once_append_twice_and_spark_reads_both(
    catalog_target: CatalogTarget,
    shared_catalog_session,
):
    config, identifier = _isolated_config(catalog_target)

    first = append_run_record(_record("same-process-001"), config, catalog_target.resolved_paths)
    second = append_run_record(_record("same-process-002"), config, catalog_target.resolved_paths)

    assert first.outcome is IcebergAppendOutcome.EMITTED
    assert second.outcome is IcebergAppendOutcome.EMITTED
    table = _load_table(config, catalog_target)
    assert len(table.metadata.snapshots) == 2
    assert [field.name for field in table.spec().fields] == ["emitted_at_day"]

    rows = shared_catalog_session.newSession().table(identifier).orderBy("run_id").collect()
    assert [row["run_id"] for row in rows] == ["same-process-001", "same-process-002"]


def _v3_record(run_id: str) -> RunRecord:
    """A v3 row whose contract identity and enforcement columns all carry a value."""
    return replace(
        _record(run_id),
        schema_version="c" * 64,
        contract_id="runs.sink_source",
        contract_version="1.1.0",
        contract_preflight_outcome="will_evolve",
        schema_evolution="added:note",
        malformed_rows=3,
        quality_outcome="passed",
        quality_checks_passed=2,
        quality_checks_failed=0,
        quality_checks_skipped=0,
        quality_failed_checks=(),
    )


def _older_row(record: RunRecord, version: int, columns) -> dict[str, Any]:
    """Project a record onto an older declaration, the way that writer would have."""
    payload = replace(record, record_schema_version=version).to_dict()
    return {column.name: payload[column.name] for column in columns}


def test_v1_table_evolves_additively_and_appends_a_v3_row(
    catalog_target: CatalogTarget,
    shared_catalog_session,
):
    if catalog_target.id != "sqlite":
        pytest.skip("the v1 evolution acceptance case uses the isolated SQLite catalog")

    import pyarrow as pa

    config = catalog_target.environment_config()
    identifier = "metadata.runs"
    dependencies = iceberg_sink._load_engine_dependencies()
    catalog = dependencies.load_catalog(
        derive_pyiceberg_catalog_name(config),
        **derive_pyiceberg_catalog_properties(config, catalog_target.resolved_paths),
    )
    catalog.create_namespace_if_not_exists("metadata")
    v1_schema = iceberg_sink._declared_schema(dependencies, RUNS_TABLE_SCHEMA_V1)
    table = catalog.create_table(
        identifier,
        schema=v1_schema,
        partition_spec=iceberg_sink._declared_partition_spec(dependencies),
    )

    old_row = _older_row(_record("v1-before-evolution"), 1, RUNS_TABLE_SCHEMA_V1)
    table.append(pa.Table.from_pylist([old_row], schema=table.schema().as_arrow()))

    result = append_run_record(
        _v3_record("v3-after-evolution"), config, catalog_target.resolved_paths
    )

    assert result.outcome is IcebergAppendOutcome.EMITTED
    table = catalog.load_table(identifier)
    fields = table.schema().fields
    assert len(fields) == 48
    assert [field.field_id for field in fields[-6:]] == [*range(44, 50)]
    assert [field.name for field in fields[-6:]] == [*V2_COLUMNS, *V3_COLUMNS]
    assert len(table.metadata.schemas) == 2, "all six columns land in one schema change"

    spark_identifier = f"janus.{identifier}"
    spark = shared_catalog_session.newSession()
    rows = spark.table(spark_identifier).orderBy("run_id").collect()
    assert [row["run_id"] for row in rows] == [
        "v1-before-evolution",
        "v3-after-evolution",
    ]
    old_row, new_row = rows
    assert {name: old_row[name] for name in (*V2_COLUMNS, *V3_COLUMNS)} == dict.fromkeys(
        (*V2_COLUMNS, *V3_COLUMNS)
    )
    assert old_row["record_schema_version"] == 1
    assert new_row["contract_version"] == "1.1.0"
    assert new_row["contract_preflight_outcome"] == "will_evolve"
    assert new_row["schema_evolution"] == "added:note"
    assert new_row["malformed_rows"] == 3
    assert new_row["record_schema_version"] == 3

    query_directory = PROJECT_ROOT / "docs" / "queries" / "observability"
    queries = sorted(query_directory.glob("*.sql"))
    assert len(queries) == 7
    for query in queries:
        spark.sql(query.read_text(encoding="utf-8")).collect()


def test_v2_table_evolves_additively_and_appends_a_v3_row(
    catalog_target: CatalogTarget,
    shared_catalog_session,
):
    import pyarrow as pa

    config, identifier = _isolated_config(catalog_target)
    dependencies = iceberg_sink._load_engine_dependencies()
    catalog = dependencies.load_catalog(
        derive_pyiceberg_catalog_name(config),
        **derive_pyiceberg_catalog_properties(config, catalog_target.resolved_paths),
    )
    catalog.create_namespace_if_not_exists("metadata")
    creation = iceberg_sink._creation_columns(RUNS_TABLE_SCHEMA_V2)
    table = catalog.create_table(
        identifier,
        schema=iceberg_sink._declared_schema(dependencies, creation),
        partition_spec=iceberg_sink._declared_partition_spec(dependencies),
    )
    with table.update_schema() as update:
        for column in RUNS_TABLE_SCHEMA_V2[len(creation) :]:
            update.add_column(
                column.name,
                iceberg_sink._iceberg_column_type(column, dependencies),
                required=column.required,
            )
    table = catalog.load_table(identifier)
    assert [field.field_id for field in table.schema().fields] == [
        column.field_id for column in RUNS_TABLE_SCHEMA_V2
    ]

    v2_row = _older_row(
        replace(
            _record("v2-before-evolution"),
            schema_version="b" * 64,
            contract_id="runs.sink_source",
            contract_version="1.0.0",
        ),
        2,
        RUNS_TABLE_SCHEMA_V2,
    )
    table.append(pa.Table.from_pylist([v2_row], schema=table.schema().as_arrow()))
    schemas_before = len(catalog.load_table(identifier).metadata.schemas)

    result = append_run_record(
        _v3_record("v3-after-evolution"), config, catalog_target.resolved_paths
    )

    assert result.outcome is IcebergAppendOutcome.EMITTED
    table = catalog.load_table(identifier)
    fields = table.schema().fields
    assert len(fields) == 48
    assert [(field.field_id, field.name) for field in fields[-3:]] == [
        (47, "contract_preflight_outcome"),
        (48, "schema_evolution"),
        (49, "malformed_rows"),
    ]
    assert [str(field.field_type) for field in fields[-3:]] == ["string", "string", "long"]
    assert not any(field.required for field in fields[-3:])
    assert len(table.metadata.schemas) == schemas_before + 1

    rows = shared_catalog_session.newSession().table(identifier).orderBy("run_id").collect()
    assert [row["run_id"] for row in rows] == ["v2-before-evolution", "v3-after-evolution"]
    old_row, new_row = rows
    assert {name: old_row[name] for name in V3_COLUMNS} == dict.fromkeys(V3_COLUMNS)
    assert old_row["contract_version"] == "1.0.0"
    assert old_row["record_schema_version"] == 2
    assert {name: new_row[name] for name in V3_COLUMNS} == {
        "contract_preflight_outcome": "will_evolve",
        "schema_evolution": "added:note",
        "malformed_rows": 3,
    }
    assert new_row["record_schema_version"] == 3


def test_bootstrap_and_append_are_idempotent_across_processes(
    catalog_target: CatalogTarget,
):
    config, _identifier = _isolated_config(catalog_target)

    first = _run_process_append("process-001", config, catalog_target.resolved_paths)
    second = _run_process_append("process-002", config, catalog_target.resolved_paths)

    assert first == (IcebergAppendOutcome.EMITTED, None, None, None)
    assert second == (IcebergAppendOutcome.EMITTED, None, None, None)
    table = _load_table(config, catalog_target)
    rows = sorted(row["run_id"] for row in table.scan().to_arrow().to_pylist())
    assert rows == ["process-001", "process-002"]
    assert len(table.metadata.snapshots) == 2
