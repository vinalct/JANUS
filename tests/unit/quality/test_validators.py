from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig, WriteResult
from janus.models.data_contracts import (
    DataContract,
    contract_from_legacy_schema_file,
    load_data_contract,
)
from janus.normalizers import BaseNormalizer
from janus.quality import (
    ContractCheck,
    ContractMismatch,
    PreWriteEvidence,
    QualityGate,
    QualityValidationError,
    ValidationReportStore,
    resolve_schema_expectation,
    validate_bronze_key_uniqueness,
)
from janus.registry import load_registry
from janus.utils.storage import bronze_table_identifier

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"


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
        data_contract=_legacy_contract(schema_path, tmp_path),
    )
    dataframe = BaseNormalizer().normalize(
        spark.createDataFrame(
            [
                {"id": "1", "updated_at": "2026-04-09T13:00:00Z"},
                {"id": "2", "updated_at": "2026-04-09T13:01:00Z"},
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
        QualityGate().validate(plan, dataframe=dataframe, raise_on_failure=True)

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


def test_quality_gate_reports_an_undeclared_column_whatever_the_compatibility(
    spark: SparkSession,
    tmp_path,
):
    """D-4: a column the contract does not declare is a mismatch; evolution is by declaration."""
    schema_path = tmp_path / "contracts" / "source_schema.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(json.dumps({"columns": ["id", "updated_at"]}), encoding="utf-8")
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-004",
        started_at=datetime(2026, 4, 9, 13, 45, tzinfo=UTC),
        data_contract=_legacy_contract(schema_path, tmp_path),
    )
    dataframe = BaseNormalizer().normalize(
        spark.createDataFrame(
            [
                {"id": "1", "updated_at": "2026-04-09T13:00:00Z", "name": "alpha"},
            ]
        ),
        plan,
    )

    report = QualityGate().validate(plan, dataframe=dataframe)

    assert report.is_successful is False
    assert [check.name for check in report.failed_checks] == ["schema_expectations"]
    assert "name: unexpected column (frame string)" in report.failed_checks[0].message


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

    report = QualityGate().validate(
        plan.with_data_contract(_example_contract()), write_results=(bad_write_result,)
    )

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

    report = QualityGate().validate(
        plan.with_data_contract(_example_contract()), write_results=(write_result,)
    )

    assert report.is_successful is True


def test_a_spark_style_schema_file_still_names_the_expected_columns(tmp_path):
    """The names the gate compares against now come from the converted contract."""
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

    contract = _legacy_contract(schema_path, tmp_path)

    assert contract.column_names == ("id", "updated_at")


def test_the_schema_expectation_is_the_contract_the_plan_carries(tmp_path):
    """Its source stays the declared file, so the persisted report reads as it always did."""
    schema_path = tmp_path / "contracts" / "source_schema.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        json.dumps({"fields": [{"name": "id"}, {"name": "updated_at"}]}),
        encoding="utf-8",
    )
    contract = _legacy_contract(schema_path, tmp_path)
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-expectation-001",
        started_at=datetime(2026, 4, 9, 13, 5, tzinfo=UTC),
        source_config=_source_config_with_schema_path(schema_path, tmp_path),
        data_contract=contract,
    )

    expectation = resolve_schema_expectation(plan)

    assert expectation.fields == ("id", "updated_at")
    assert expectation.source == str(schema_path)
    assert expectation.error is None


def test_a_plan_without_a_contract_expects_nothing(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-quality-expectation-002",
        started_at=datetime(2026, 4, 9, 13, 10, tzinfo=UTC),
    )

    expectation = resolve_schema_expectation(plan)

    assert expectation.fields == ()
    assert expectation.source is None


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


# ── the pre-write evidence is reported, never decided a second time (FR-1) ───


def test_a_strict_run_reports_required_fields_from_the_pre_write_count(tmp_path):
    """No frame is handed over, so the counts can only come from the evidence: no second pass."""
    plan = _enforced_plan(tmp_path, "base")

    report = QualityGate().validate(plan, pre_write_evidence=(_evidence(plan, counts={"id": 0}),))

    checks = {check.name: check for check in report.checks}
    assert checks["required_fields"].outcome == "passed"
    assert checks["required_fields"].details_as_dict() == {
        "batches": "1",
        "required_field_count": "1",
    }
    assert checks["schema_expectations"].outcome == "passed"
    assert checks["schema_expectations"].details_as_dict()["batches"] == "1"
    assert report.metadata_as_dict()["enforcement"] == "strict"
    assert report.metadata_as_dict()["compatibility"] == "additive"


def test_a_lenient_run_leaves_required_fields_to_the_post_write_check(tmp_path):
    plan = _enforced_plan(tmp_path, "base_lenient")

    report = QualityGate().validate(plan, pre_write_evidence=(_evidence(plan),))

    required = next(check for check in report.checks if check.name == "required_fields")
    assert "batches" not in required.details_as_dict()
    assert report.metadata_as_dict()["enforcement"] == "lenient"


def test_the_refused_batch_is_the_failed_check_the_report_carries(tmp_path):
    plan = _enforced_plan(tmp_path, "base")
    missing = ContractMismatch("missing_column", "amount", expected="long")

    report = QualityGate().validate(
        plan,
        pre_write_evidence=(
            _evidence(plan, counts={"id": 0}, batch_count=2),
            _evidence(plan, mismatches=(missing,), batch_index=2, batch_count=2),
        ),
    )

    failed = {check.name: check for check in report.failed_checks}
    assert set(failed) == {"schema_expectations"}
    assert failed["schema_expectations"].message.startswith("batch 2/2: Frame does not match")
    details = failed["schema_expectations"].details_as_dict()
    assert details["batches"] == "2"
    assert json.loads(details["mismatches"]) == ["amount: missing column (contract long)"]
    required = next(check for check in report.checks if check.name == "required_fields")
    assert required.outcome == "skipped"
    assert required.message.startswith("batch 2/2: Required fields were not counted")


def _enforced_plan(tmp_path: Path, contract_name: str) -> ExecutionPlan:
    return _build_plan(
        tmp_path,
        run_id=f"run-pre-write-{contract_name}",
        started_at=datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
        data_contract=load_data_contract(HOSTILE / f"{contract_name}.yaml"),
    )


def _evidence(
    plan: ExecutionPlan,
    *,
    counts: dict[str, int] | None = None,
    mismatches: tuple[ContractMismatch, ...] = (),
    batch_index: int = 1,
    batch_count: int = 1,
) -> PreWriteEvidence:
    contract = plan.data_contract
    assert contract is not None
    check = ContractCheck(
        contract_id=contract.id,
        contract_version=contract.version,
        schema_version=contract.schema_version,
        mismatches=mismatches,
        nullability_relaxed=(),
        checked_columns=len(contract.column_names),
    )
    return PreWriteEvidence(batch_index, batch_count, check, counts or {})


def _base_source_config() -> SourceConfig:
    return load_registry(PROJECT_ROOT).get_source("federal_open_data_example")


def _example_contract() -> DataContract:
    return load_data_contract(PROJECT_ROOT / _base_source_config().schema.contract)


def _source_config_with_schema_path(schema_path: Path, project_root: Path) -> SourceConfig:
    source_config = _base_source_config()
    return replace(
        source_config,
        schema=replace(
            source_config.schema,
            path=str(schema_path.relative_to(project_root)),
        ),
    )


def _legacy_contract(schema_path: Path, project_root: Path) -> DataContract:
    """What the registry snapshot builds for an entry that still declares a legacy file."""
    return contract_from_legacy_schema_file(
        schema_path,
        source_id="federal_open_data_example",
        bronze_table="bronze.federal_open_data_example",
        domain="example",
        project_root=project_root,
    )


def _build_plan(
    tmp_path: Path,
    *,
    run_id: str,
    started_at: datetime,
    source_config: SourceConfig | None = None,
    data_contract: DataContract | None = None,
) -> ExecutionPlan:
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
    )
    return ExecutionPlan.from_source_config(
        source_config or _base_source_config(),
        run_context,
        data_contract=data_contract,
    )


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
