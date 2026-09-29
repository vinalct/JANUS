from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from janus.models import (
    ExecutionPlan,
    QualityConfig,
    WriteResult,
    resolve_bronze_write_intent,
)
from janus.normalizers import NORMALIZATION_METADATA_COLUMNS
from janus.quality.contract_checks import check_frame_against_contract
from janus.quality.models import QualityValidationError, ValidationCheck, ValidationReport
from janus.quality.pre_write import (
    PreWriteEvidence,
    required_fields_check,
    required_null_aggregations,
    summarize_pre_write_evidence,
)
from janus.quality.schema_expectation import SchemaExpectation, resolve_schema_expectation
from janus.quality.store import PersistedValidationReport, ValidationReportStore
from janus.utils.environment import resolve_project_path
from janus.utils.storage import bronze_table_identifier

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass(slots=True)
class QualityGate:
    """Strategy-agnostic validation orchestrator for config, data, and outputs.

    The structural contract check is decided before the write, by the materializer; given that
    ``pre_write_evidence``, the gate reports it as ``data.schema_expectations`` (and, for a
    ``strict`` contract, ``data.required_fields``) and ``data.malformed_rows`` instead
    of deciding a second time.
    """

    report_store: ValidationReportStore | None = None

    def validate(
        self,
        plan: ExecutionPlan,
        *,
        dataframe: DataFrame | None = None,
        write_results: Sequence[WriteResult] = (),
        output_columns: Sequence[str] = NORMALIZATION_METADATA_COLUMNS,
        bronze_dataframe: DataFrame | None = None,
        run_keys: DataFrame | None = None,
        pre_write_evidence: Sequence[PreWriteEvidence] | None = None,
        raise_on_failure: bool = False,
    ) -> ValidationReport:
        schema_expectation = resolve_schema_expectation(plan)
        reported = _pre_write_report(plan, pre_write_evidence)
        checks = (
            validate_quality_contract(plan.source_config.quality),
            validate_schema_contract_mode(plan, schema_expectation),
            reported.get("required_fields") or validate_required_fields(plan, dataframe),
            validate_unique_fields(plan, dataframe),
            reported.get("schema_expectations") or validate_schema_expectations(plan, dataframe),
            *((reported["malformed_rows"],) if "malformed_rows" in reported else ()),
            validate_output_columns(dataframe, output_columns),
            validate_materialized_outputs(plan, write_results),
            validate_bronze_key_uniqueness(plan, bronze_dataframe, run_keys),
        )
        report = ValidationReport.from_plan(
            plan, checks, metadata=_report_metadata(plan, schema_expectation)
        )
        if raise_on_failure and not report.is_successful:
            raise QualityValidationError(report)
        return report

    def validate_and_store(
        self,
        plan: ExecutionPlan,
        *,
        dataframe: DataFrame | None = None,
        write_results: Sequence[WriteResult] = (),
        output_columns: Sequence[str] = NORMALIZATION_METADATA_COLUMNS,
        bronze_dataframe: DataFrame | None = None,
        run_keys: DataFrame | None = None,
        pre_write_evidence: Sequence[PreWriteEvidence] | None = None,
        raise_on_failure: bool = False,
    ) -> PersistedValidationReport:
        if self.report_store is None:
            raise ValueError("report_store must be configured to persist validation results")

        report = self.validate(
            plan,
            dataframe=dataframe,
            write_results=write_results,
            output_columns=output_columns,
            bronze_dataframe=bronze_dataframe,
            run_keys=run_keys,
            pre_write_evidence=pre_write_evidence,
            raise_on_failure=False,
        )
        persisted_path = self.report_store.write(plan, report)
        persisted = PersistedValidationReport(report=report, path=persisted_path)
        if raise_on_failure and not report.is_successful:
            raise QualityValidationError(report)
        return persisted


def _pre_write_report(
    plan: ExecutionPlan, evidence: Sequence[PreWriteEvidence] | None
) -> dict[str, ValidationCheck]:
    """The checks the materializer already decided, merged across batches; none without it."""
    contract = plan.data_contract
    if not evidence or contract is None:
        return {}
    return summarize_pre_write_evidence(evidence, enforcement=contract.janus.enforcement)


def _report_metadata(plan: ExecutionPlan, expectation: SchemaExpectation) -> dict[str, str]:
    metadata = {"schema_expectation_source": expectation.source or ""}
    if plan.data_contract is not None:
        metadata["compatibility"] = plan.data_contract.janus.compatibility
        metadata["enforcement"] = plan.data_contract.janus.enforcement
    return metadata


def validate_quality_contract(quality_config: QualityConfig) -> ValidationCheck:
    issues: list[str] = []
    duplicate_required = _duplicate_fields(quality_config.required_fields)
    duplicate_unique = _duplicate_fields(quality_config.unique_fields)
    if duplicate_required:
        issues.append("required_fields contains duplicates: " + ", ".join(duplicate_required))
    if duplicate_unique:
        issues.append("unique_fields contains duplicates: " + ", ".join(duplicate_unique))

    missing_from_required = [
        field
        for field in quality_config.unique_fields
        if field not in quality_config.required_fields
    ]
    if missing_from_required:
        issues.append(
            "unique_fields must also appear in required_fields: " + ", ".join(missing_from_required)
        )

    if issues:
        return ValidationCheck.failed(
            "config",
            "quality_contract",
            "; ".join(issues),
            details={
                "required_fields": ",".join(quality_config.required_fields),
                "unique_fields": ",".join(quality_config.unique_fields),
            },
        )

    return ValidationCheck.passed(
        "config",
        "quality_contract",
        "Quality rules are internally consistent.",
        details={
            "required_fields": ",".join(quality_config.required_fields),
            "unique_fields": ",".join(quality_config.unique_fields),
        },
    )


def validate_schema_contract_mode(
    plan: ExecutionPlan,
    schema_expectation: SchemaExpectation,
) -> ValidationCheck:
    contract = plan.data_contract
    if contract is None:
        # Not decided here: the materializer refuses to write without a contract.
        return ValidationCheck.skipped(
            "config",
            "schema_contract_mode",
            "No data contract is declared; bronze is materialized only under one.",
            details={"schema_mode": plan.source_config.schema.mode},
        )

    return ValidationCheck.passed(
        "config",
        "schema_contract_mode",
        "Schema validation will use an explicit field contract.",
        details={
            "schema_source": schema_expectation.source or "",
            "expected_field_count": len(schema_expectation.fields),
            "compatibility": contract.janus.compatibility,
            "enforcement": contract.janus.enforcement,
        },
    )


def validate_required_fields(
    plan: ExecutionPlan,
    dataframe: DataFrame | None,
) -> ValidationCheck:
    """The post-write count; a ``strict`` run's gate renders the pre-write count instead."""
    required_fields = plan.source_config.quality.required_fields
    if not required_fields:
        return ValidationCheck.skipped(
            "data",
            "required_fields",
            "No required_fields were configured.",
        )
    if dataframe is None:
        return ValidationCheck.skipped(
            "data",
            "required_fields",
            "No dataframe was provided for required field validation.",
        )

    missing_columns = [field for field in required_fields if field not in dataframe.columns]
    if missing_columns:
        return ValidationCheck.failed(
            "data",
            "required_fields",
            "Missing required fields: " + ", ".join(missing_columns),
            details={"missing_fields": ",".join(missing_columns)},
        )

    return required_fields_check(_required_field_violation_counts(dataframe, required_fields))


def validate_unique_fields(
    plan: ExecutionPlan,
    dataframe: DataFrame | None,
) -> ValidationCheck:
    unique_fields = plan.source_config.quality.unique_fields
    if not unique_fields:
        return ValidationCheck.skipped(
            "data",
            "unique_fields",
            "No unique_fields were configured.",
        )
    if dataframe is None:
        return ValidationCheck.skipped(
            "data",
            "unique_fields",
            "No dataframe was provided for uniqueness validation.",
        )

    missing_columns = [field for field in unique_fields if field not in dataframe.columns]
    if missing_columns:
        return ValidationCheck.failed(
            "data",
            "unique_fields",
            "Unique-field validation cannot run because fields are missing: "
            + ", ".join(missing_columns),
            details={"missing_fields": ",".join(missing_columns)},
        )

    duplicate_groups, sample_duplicates = _duplicate_key_groups(dataframe, unique_fields)
    if duplicate_groups:
        return ValidationCheck.failed(
            "data",
            "unique_fields",
            "Duplicate keys were found for unique_fields.",
            details={
                "duplicate_groups": duplicate_groups,
                "sample_duplicates": json.dumps(sample_duplicates, sort_keys=True),
            },
        )

    return ValidationCheck.passed(
        "data",
        "unique_fields",
        "Configured unique_fields are unique in the provided dataframe.",
        details={"unique_field_count": len(unique_fields)},
    )


def validate_bronze_key_uniqueness(
    plan: ExecutionPlan,
    bronze_dataframe: DataFrame | None,
    run_keys: DataFrame | None,
) -> ValidationCheck:
    """Assert the committed bronze table holds one row per key this run wrote."""

    unique_fields = plan.source_config.quality.unique_fields
    if not unique_fields:
        return ValidationCheck.skipped(
            "output",
            "bronze_key_uniqueness",
            "No unique_fields were configured.",
        )

    intent = resolve_bronze_write_intent(plan)
    if not intent.is_upsert:
        return ValidationCheck.skipped(
            "output",
            "bronze_key_uniqueness",
            f"Write strategy {intent.strategy!r} is not an upsert; the whole table was "
            "rewritten from the frame the in-flight uniqueness check already covered.",
            details={"write_strategy": intent.strategy},
        )

    if bronze_dataframe is None or run_keys is None:
        return ValidationCheck.skipped(
            "output",
            "bronze_key_uniqueness",
            "No committed bronze table frame was provided for uniqueness validation.",
        )

    missing_columns = [field for field in unique_fields if field not in bronze_dataframe.columns]
    if missing_columns:
        return ValidationCheck.failed(
            "output",
            "bronze_key_uniqueness",
            "Bronze uniqueness validation cannot run because fields are missing: "
            + ", ".join(missing_columns),
            details={"missing_fields": ",".join(missing_columns)},
        )

    duplicate_groups, sample_duplicates, keys_checked = _bronze_duplicate_key_groups(
        bronze_dataframe, run_keys, unique_fields
    )
    if duplicate_groups:
        return ValidationCheck.failed(
            "output",
            "bronze_key_uniqueness",
            "Committed bronze holds duplicate rows for keys this run wrote.",
            details={
                "duplicate_groups": duplicate_groups,
                "sample_duplicates": json.dumps(sample_duplicates, sort_keys=True),
                "scan_scope": "run_keys",
                "keys_checked": keys_checked,
            },
        )

    return ValidationCheck.passed(
        "output",
        "bronze_key_uniqueness",
        "Committed bronze holds one row per key among the keys this run wrote.",
        details={
            "keys_checked": keys_checked,
            "scan_scope": "run_keys",
        },
    )


def validate_schema_expectations(
    plan: ExecutionPlan,
    dataframe: DataFrame | None,
) -> ValidationCheck:
    """Report the structural contract check for a frame validated outside the materializer.

    It is the same pure check the pre-write gate runs, so the two cannot disagree; the
    normalization columns a normalized frame carries are ignored unless the contract declares them.
    """
    if dataframe is None:
        return ValidationCheck.skipped(
            "data",
            "schema_expectations",
            "No dataframe was provided for schema validation.",
        )
    if plan.data_contract is None:
        return ValidationCheck.skipped(
            "data",
            "schema_expectations",
            "No explicit schema expectation was available for comparison.",
        )
    from janus.schema_contracts import frame_columns_from_spark_schema

    contract = plan.data_contract
    declared = set(contract.column_names)
    columns = [
        column
        for column in frame_columns_from_spark_schema(dataframe.schema)
        if column.name in declared or column.name not in NORMALIZATION_METADATA_COLUMNS
    ]
    return check_frame_against_contract(columns, contract).to_validation_check()


def validate_output_columns(
    dataframe: DataFrame | None,
    output_columns: Sequence[str],
) -> ValidationCheck:
    if dataframe is None:
        return ValidationCheck.skipped(
            "output",
            "output_columns",
            "No dataframe was provided for bronze output sanity checks.",
        )
    if not output_columns:
        return ValidationCheck.skipped(
            "output",
            "output_columns",
            "No output columns were configured for sanity checks.",
        )

    missing_columns = [column for column in output_columns if column not in dataframe.columns]
    if missing_columns:
        return ValidationCheck.failed(
            "output",
            "output_columns",
            "Output dataframe is missing required contract columns: " + ", ".join(missing_columns),
            details={"missing_columns": ",".join(missing_columns)},
        )

    return ValidationCheck.passed(
        "output",
        "output_columns",
        "Output dataframe satisfies the configured sanity columns.",
        details={"output_column_count": len(output_columns)},
    )


def validate_materialized_outputs(
    plan: ExecutionPlan,
    write_results: Sequence[WriteResult],
) -> ValidationCheck:
    if not write_results:
        return ValidationCheck.skipped(
            "output",
            "materialized_outputs",
            "No write_results were provided for output contract validation.",
        )

    violations: list[str] = []
    for write_result in write_results:
        expected_format = _expected_zone_format(plan, write_result.zone)
        if write_result.zone == "bronze" and expected_format == "iceberg":
            expected_identifier = bronze_table_identifier(
                plan.bronze_output.path,
                fallback_name=plan.source.source_id,
                namespace=plan.bronze_output.namespace,
                table_name=plan.bronze_output.table_name,
            )
            if write_result.path != expected_identifier:
                violations.append(
                    f"{write_result.zone}: table {write_result.path!r} "
                    f"must match configured iceberg table {expected_identifier!r}"
                )
        else:
            expected_root = resolve_project_path(
                plan.run_context.project_root,
                _expected_zone_path(plan, write_result.zone),
            )
            materialized_path = resolve_project_path(
                plan.run_context.project_root, write_result.path
            )
            if not materialized_path.is_relative_to(expected_root):
                violations.append(
                    f"{write_result.zone}: path {materialized_path} must stay under {expected_root}"
                )

        if write_result.records_written is not None and write_result.records_written < 0:
            violations.append(f"{write_result.zone}: records_written must not be negative")

        if write_result.zone != "raw" and write_result.format != expected_format:
            violations.append(
                f"{write_result.zone}: format {write_result.format!r} "
                f"must match configured format {expected_format!r}"
            )

    if violations:
        return ValidationCheck.failed(
            "output",
            "materialized_outputs",
            "; ".join(violations),
            details={"checked_outputs": len(write_results)},
        )

    return ValidationCheck.passed(
        "output",
        "materialized_outputs",
        "All materialized outputs satisfy the configured zone contracts.",
        details={"checked_outputs": len(write_results)},
    )


def _required_field_violation_counts(
    dataframe: DataFrame,
    fields: Sequence[str],
) -> dict[str, int]:
    row = dataframe.agg(*required_null_aggregations(dataframe, fields)).first()
    if row is None:
        raise RuntimeError("Spark returned no row for the required-field aggregation")
    return {field: int(row[field] or 0) for field in fields}


def _duplicate_key_groups(
    dataframe: DataFrame,
    fields: Sequence[str],
) -> tuple[int, list[dict[str, Any]]]:
    from pyspark.sql.functions import col

    duplicates = dataframe.groupBy(*fields).count().where(col("count") > 1)
    duplicate_groups = duplicates.count()
    sample_duplicates = [row.asDict(recursive=True) for row in duplicates.limit(5).collect()]
    return duplicate_groups, sample_duplicates


def _bronze_duplicate_key_groups(
    bronze_dataframe: DataFrame,
    run_keys: DataFrame,
    fields: Sequence[str],
) -> tuple[int, list[dict[str, Any]], int]:
    """Find duplicate key groups in bronze, scoped to the keys this run wrote.

    A ``left_semi`` join keeps only bronze rows whose key the run touched, so the
    ``groupBy`` cost is bounded by the batch size rather than the whole table, and Iceberg
    can prune on the join keys when they align with partitioning. Returns the duplicate
    group count, up to five sample keys, and the number of distinct keys examined.
    """
    from pyspark.sql.functions import col

    key_columns = list(fields)
    scoped_keys = run_keys.select(*key_columns).distinct()
    suspects = bronze_dataframe.join(scoped_keys, on=key_columns, how="left_semi")
    duplicates = suspects.groupBy(*key_columns).count().where(col("count") > 1)
    duplicate_groups = duplicates.count()
    sample_duplicates = [row.asDict(recursive=True) for row in duplicates.limit(5).collect()]
    keys_checked = scoped_keys.count()
    return duplicate_groups, sample_duplicates, keys_checked


def _duplicate_fields(fields: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for field in fields:
        if field in seen and field not in duplicates:
            duplicates.append(field)
        seen.add(field)
    return duplicates


def _expected_zone_path(plan: ExecutionPlan, zone: str) -> str:
    if zone == "raw":
        return plan.raw_output.path
    if zone == "bronze":
        return plan.bronze_output.path
    if zone == "metadata":
        return plan.metadata_output.path
    raise ValueError(f"Unsupported output zone: {zone}")


def _expected_zone_format(plan: ExecutionPlan, zone: str) -> str:
    if zone == "raw":
        return plan.raw_output.format
    if zone == "bronze":
        return plan.bronze_output.format
    if zone == "metadata":
        return plan.metadata_output.format
    raise ValueError(f"Unsupported output zone: {zone}")
