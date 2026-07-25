from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig, WriteResult
from janus.normalizers import BaseNormalizer
from janus.quality import (
    QualityGate,
    QualityValidationError,
    ValidationReportStore,
    load_expected_fields_from_schema_path,
    validate_bronze_key_uniqueness,
)
from janus.registry import load_registry
from janus.utils.storage import bronze_table_identifier

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def spark():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-quality-tests")
        .master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_quality_gate_persists_successful_validation_report(spark: SparkSession, tmp_path):
    schema_path = tmp_path / "contracts" / "source_schema.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        json.dumps({"fields": [{"name": "id"}, {"name": "updated_at"}]}),
        encoding="utf-8",
    )
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-001",
        started_at=datetime(2026, 4, 9, 13, 0, tzinfo=UTC),
        source_config=_source_config_with_schema_path(schema_path, tmp_path),
    )
    dataframe = BaseNormalizer().normalize(
        spark.createDataFrame(
            [
                {"id": "1", "updated_at": "2026-04-09T13:00:00Z", "name": "alpha"},
                {"id": "2", "updated_at": "2026-04-09T13:01:00Z", "name": "beta"},
            ]
        ),
        plan,
    )
    write_result = _bronze_write_result(plan, records_written=2)

    persisted = QualityGate(ValidationReportStore()).validate_and_store(
        plan,
        dataframe=dataframe,
        write_results=(write_result,),
    )

    assert persisted.report.is_successful is True
    assert persisted.path == (
        tmp_path
        / "data"
        / "metadata"
        / "example"
        / "federal_open_data_example"
        / "validations"
        / "run-quality-001.json"
    )

    payload = json.loads(persisted.path.read_text(encoding="utf-8"))
    # bronze_key_uniqueness skips here: no committed bronze frame is supplied to the gate.
    assert payload["summary"] == {"failed": 0, "passed": 7, "skipped": 1}
    assert payload["checks"][4]["name"] == "schema_expectations"
    assert payload["checks"][7]["name"] == "bronze_key_uniqueness"
    assert payload["checks"][7]["outcome"] == "skipped"


def test_quality_gate_raises_with_actionable_dataset_errors(spark: SparkSession, tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-002",
        started_at=datetime(2026, 4, 9, 13, 15, tzinfo=UTC),
    )
    dataframe = BaseNormalizer().normalize(
        spark.createDataFrame(
            [
                {"id": "1", "updated_at": "2026-04-09T13:00:00Z"},
                {"id": "1", "updated_at": ""},
            ]
        ),
        plan,
    )

    with pytest.raises(QualityValidationError) as exc_info:
        QualityGate().validate(
            plan,
            dataframe=dataframe,
            expected_fields=("id", "updated_at"),
            raise_on_failure=True,
        )

    message = str(exc_info.value)
    assert "[data.required_fields]" in message
    assert "[data.unique_fields]" in message
    assert "null or blank values" in message


def test_quality_gate_detects_conflicting_quality_contract(tmp_path):
    source_config = _base_source_config()
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-003",
        started_at=datetime(2026, 4, 9, 13, 30, tzinfo=UTC),
        source_config=replace(
            source_config,
            quality=replace(
                source_config.quality,
                required_fields=("updated_at",),
                unique_fields=("id",),
            ),
        ),
    )

    report = QualityGate().validate(plan)

    assert report.is_successful is False
    assert report.failed_checks[0].name == "quality_contract"
    assert "unique_fields must also appear in required_fields" in report.failed_checks[0].message


def test_quality_gate_blocks_unexpected_columns_when_schema_evolution_is_disabled(
    spark: SparkSession,
    tmp_path,
):
    source_config = _base_source_config()
    strict_source_config = replace(
        source_config,
        quality=replace(source_config.quality, allow_schema_evolution=False),
    )
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-004",
        started_at=datetime(2026, 4, 9, 13, 45, tzinfo=UTC),
        source_config=strict_source_config,
    )
    dataframe = BaseNormalizer().normalize(
        spark.createDataFrame(
            [
                {"id": "1", "updated_at": "2026-04-09T13:00:00Z", "name": "alpha"},
            ]
        ),
        plan,
    )

    report = QualityGate().validate(
        plan,
        dataframe=dataframe,
        expected_fields=("id", "updated_at"),
    )

    assert report.is_successful is False
    assert report.failed_checks[0].name == "schema_expectations"
    assert "unexpected fields" in report.failed_checks[0].message


def test_quality_gate_detects_output_paths_outside_the_configured_zone(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-005",
        started_at=datetime(2026, 4, 9, 14, 0, tzinfo=UTC),
    )
    bad_write_result = _bronze_write_result(
        plan,
        records_written=10,
        path="bronze.outside__bronze_dataset",
    )

    report = QualityGate().validate(plan, write_results=(bad_write_result,))

    assert report.is_successful is False
    assert report.failed_checks[0].name == "materialized_outputs"
    assert "must match configured iceberg table" in report.failed_checks[0].message


def test_quality_gate_uses_configured_bronze_iceberg_namespace_and_table(tmp_path):
    source_config = replace(
        _base_source_config(),
        outputs=replace(
            _base_source_config().outputs,
            bronze=replace(
                _base_source_config().outputs.bronze,
                namespace="curated",
                table_name="named_bronze_table",
            ),
        ),
    )
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-005-named",
        started_at=datetime(2026, 4, 9, 14, 5, tzinfo=UTC),
        source_config=source_config,
    )
    write_result = _bronze_write_result(plan, records_written=10)

    report = QualityGate().validate(plan, write_results=(write_result,))

    assert report.is_successful is True


def test_schema_field_loader_supports_spark_style_schema_json(tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "struct",
                "fields": [
                    {"name": "id", "type": "string", "nullable": False},
                    {"name": "updated_at", "type": "string", "nullable": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_expected_fields_from_schema_path(schema_path) == ("id", "updated_at")


def test_bronze_key_uniqueness_skips_without_unique_fields(tmp_path):
    source_config = _base_source_config()
    plan = _build_plan(
        tmp_path,
        run_id="run-bronze-uniqueness-001",
        started_at=datetime(2026, 4, 9, 15, 0, tzinfo=UTC),
        source_config=replace(
            source_config,
            quality=replace(source_config.quality, unique_fields=()),
        ),
    )

    check = validate_bronze_key_uniqueness(plan, None, None)

    assert check.phase == "output"
    assert check.name == "bronze_key_uniqueness"
    assert check.outcome == "skipped"
    assert "unique_fields" in check.message


def test_bronze_key_uniqueness_skips_without_a_bronze_frame(tmp_path):
    # The base example source is incremental+append with a key, so its intent is an upsert;
    # the skip here is only because no committed bronze frame was handed to the check.
    plan = _build_plan(
        tmp_path,
        run_id="run-bronze-uniqueness-002",
        started_at=datetime(2026, 4, 9, 15, 15, tzinfo=UTC),
    )

    check = validate_bronze_key_uniqueness(plan, None, None)

    assert check.outcome == "skipped"
    assert "committed bronze" in check.message


def test_bronze_key_uniqueness_skips_for_a_non_upsert_run(tmp_path):
    source_config = _base_source_config()
    plan = _build_plan(
        tmp_path,
        run_id="run-bronze-uniqueness-003",
        started_at=datetime(2026, 4, 9, 15, 30, tzinfo=UTC),
        source_config=replace(
            source_config,
            extraction=replace(source_config.extraction, mode="full_refresh"),
        ),
    )

    check = validate_bronze_key_uniqueness(plan, None, None)

    assert check.outcome == "skipped"
    assert "not an upsert" in check.message
    assert check.details_as_dict()["write_strategy"] == "insert"


def _base_source_config() -> SourceConfig:
    return load_registry(PROJECT_ROOT).get_source("federal_open_data_example")


def _source_config_with_schema_path(schema_path: Path, project_root: Path) -> SourceConfig:
    source_config = _base_source_config()
    return replace(
        source_config,
        schema=replace(
            source_config.schema,
            path=str(schema_path.relative_to(project_root)),
        ),
    )


def _build_plan(
    tmp_path: Path,
    *,
    run_id: str,
    started_at: datetime,
    source_config: SourceConfig | None = None,
) -> ExecutionPlan:
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
    )
    return ExecutionPlan.from_source_config(source_config or _base_source_config(), run_context)


def _bronze_write_result(
    plan: ExecutionPlan,
    *,
    records_written: int,
    path: str | Path | None = None,
) -> WriteResult:
    target_path = (
        str(path)
        if path is not None
        else bronze_table_identifier(
            plan.bronze_output.path,
            fallback_name=plan.source.source_id,
            namespace=plan.bronze_output.namespace,
            table_name=plan.bronze_output.table_name,
        )
    )
    return WriteResult.from_plan(
        plan,
        "bronze",
        path=target_path,
        format_name=plan.bronze_output.format,
        mode="append",
        partition_by=("ingestion_date",),
        records_written=records_written,
    )
