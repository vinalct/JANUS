from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from janus.models import ExecutionPlan, QualityConfig, WriteResult
from janus.normalizers import NORMALIZATION_METADATA_COLUMNS
from janus.quality.models import QualityValidationError, ValidationCheck, ValidationReport
from janus.quality.store import PersistedValidationReport, ValidationReportStore
from janus.schema_contracts import (
    load_expected_fields_from_schema_path,
    resolve_schema_path_for_plan,
)
from janus.utils.environment import resolve_project_path
from janus.utils.storage import bronze_table_identifier

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass(frozen=True, slots=True)
class SchemaExpectation:
    """Resolved schema contract used by schema-related validations."""

    fields: tuple[str, ...] = ()
    source: str | None = None
    error: str | None = None


@dataclass(slots=True)
class QualityGate:
    """Strategy-agnostic validation orchestrator for config, data, and outputs."""

    report_store: ValidationReportStore | None = None

    def validate(
        self,
        plan: ExecutionPlan,
        *,
        dataframe: DataFrame | None = None,
        write_results: Sequence[WriteResult] = (),
        expected_fields: Sequence[str] | None = None,
        output_columns: Sequence[str] = NORMALIZATION_METADATA_COLUMNS,
        raise_on_failure: bool = False,
    ) -> ValidationReport:
        schema_expectation = resolve_schema_expectation(plan, expected_fields=expected_fields)
        checks = (
            validate_quality_contract(plan.source_config.quality),
            validate_schema_contract_mode(plan, schema_expectation),
            validate_required_fields(plan, dataframe),
            validate_unique_fields(plan, dataframe),
            validate_schema_expectations(plan, dataframe, schema_expectation),
            validate_output_columns(dataframe, output_columns),
            validate_materialized_outputs(plan, write_results),
        )
        report = ValidationReport.from_plan(
            plan,
            checks,
            metadata={
                "allow_schema_evolution": str(plan.source_config.quality.allow_schema_evolution),
                "schema_expectation_source": schema_expectation.source or "",
            },
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
        expected_fields: Sequence[str] | None = None,
        output_columns: Sequence[str] = NORMALIZATION_METADATA_COLUMNS,
        raise_on_failure: bool = False,
    ) -> PersistedValidationReport:
        if self.report_store is None:
            raise ValueError("report_store must be configured to persist validation results")

        report = self.validate(
            plan,
            dataframe=dataframe,
            write_results=write_results,
            expected_fields=expected_fields,
            output_columns=output_columns,
            raise_on_failure=False,
        )
        persisted_path = self.report_store.write(plan, report)
        persisted = PersistedValidationReport(report=report, path=persisted_path)
        if raise_on_failure and not report.is_successful:
            raise QualityValidationError(report)
        return persisted


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
            "unique_fields must also appear in required_fields: "
            + ", ".join(missing_from_required)
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
    if schema_expectation.error:
        return ValidationCheck.failed(
            "config",
            "schema_contract_mode",
            schema_expectation.error,
            details={"schema_source": schema_expectation.source or ""},
        )

    if (
        not plan.source_config.quality.allow_schema_evolution
        and plan.source_config.schema.mode != "explicit"
    ):
        return ValidationCheck.failed(
            "config",
            "schema_contract_mode",
            "allow_schema_evolution=false requires schema.mode='explicit'.",
            details={"schema_mode": plan.source_config.schema.mode},
        )

    if schema_expectation.fields:
        return ValidationCheck.passed(
            "config",
            "schema_contract_mode",
            "Schema validation will use an explicit field contract.",
            details={
                "schema_source": schema_expectation.source or "",
                "expected_field_count": len(schema_expectation.fields),
            },
        )

    if plan.source_config.quality.allow_schema_evolution:
        return ValidationCheck.passed(
            "config",
            "schema_contract_mode",
            "Schema evolution is enabled; unexpected columns will be tolerated.",
            details={"schema_mode": plan.source_config.schema.mode},
        )

    return ValidationCheck.passed(
        "config",
        "schema_contract_mode",
        "Strict schema mode is configured through an explicit schema contract.",
        details={"schema_mode": plan.source_config.schema.mode},
    )


def validate_required_fields(
    plan: ExecutionPlan,
    dataframe: DataFrame | None,
) -> ValidationCheck:
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

    invalid_counts = _required_field_violation_counts(dataframe, required_fields)
    failing_fields = {field: count for field, count in invalid_counts.items() if count > 0}
    if failing_fields:
        rendered_counts = ", ".join(f"{field} ({count})" for field, count in failing_fields.items())
        return ValidationCheck.failed(
            "data",
            "required_fields",
            "Required fields contain null or blank values: " + rendered_counts,
            details={field: count for field, count in failing_fields.items()},
        )

    return ValidationCheck.passed(
        "data",
        "required_fields",
        "All required fields are present and populated.",
        details={"required_field_count": len(required_fields)},
    )


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


def validate_schema_expectations(
    plan: ExecutionPlan,
    dataframe: DataFrame | None,
    schema_expectation: SchemaExpectation,
) -> ValidationCheck:
    if dataframe is None:
        return ValidationCheck.skipped(
            "data",
            "schema_expectations",
            "No dataframe was provided for schema validation.",
        )
    if schema_expectation.error:
        return ValidationCheck.failed(
            "data",
            "schema_expectations",
            schema_expectation.error,
            details={"schema_source": schema_expectation.source or ""},
        )
    if not schema_expectation.fields:
        return ValidationCheck.skipped(
            "data",
            "schema_expectations",
            "No explicit schema expectation was available for comparison.",
        )

    observed_fields = tuple(dataframe.columns)
    missing_fields = [
        field for field in schema_expectation.fields if field not in observed_fields
    ]
    unexpected_fields = [
        field for field in observed_fields if field not in schema_expectation.fields
    ]
    if missing_fields:
        return ValidationCheck.failed(
            "data",
            "schema_expectations",
            "Observed schema is missing expected fields: " + ", ".join(missing_fields),
            details={
                "schema_source": schema_expectation.source or "",
                "missing_fields": ",".join(missing_fields),
            },
        )

    if unexpected_fields and not plan.source_config.quality.allow_schema_evolution:
        return ValidationCheck.failed(
            "data",
            "schema_expectations",
            "Observed schema contains unexpected fields while schema evolution is disabled: "
            + ", ".join(unexpected_fields),
            details={
                "schema_source": schema_expectation.source or "",
                "unexpected_fields": ",".join(unexpected_fields),
            },
        )

    if unexpected_fields:
        return ValidationCheck.passed(
            "data",
            "schema_expectations",
            "Observed schema satisfies the expected contract and allowed extra fields.",
            details={
                "schema_source": schema_expectation.source or "",
                "unexpected_fields": ",".join(unexpected_fields),
            },
        )

    return ValidationCheck.passed(
        "data",
        "schema_expectations",
        "Observed schema matches the expected contract.",
        details={
            "schema_source": schema_expectation.source or "",
            "expected_field_count": len(schema_expectation.fields),
        },
    )


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


def resolve_schema_expectation(
    plan: ExecutionPlan,
    *,
    expected_fields: Sequence[str] | None = None,
) -> SchemaExpectation:
    if expected_fields is not None:
        return SchemaExpectation(
            fields=_normalize_field_names(expected_fields),
            source="provided",
        )

    if plan.source_config.schema.mode != "explicit" or not plan.source_config.schema.path:
        return SchemaExpectation()

    schema_path = resolve_schema_path_for_plan(plan)
    assert schema_path is not None
    if not schema_path.exists():
        return SchemaExpectation()

    try:
        return SchemaExpectation(
            fields=load_expected_fields_from_schema_path(schema_path),
            source=str(schema_path),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return SchemaExpectation(source=str(schema_path), error=str(exc))


def _required_field_violation_counts(
    dataframe: DataFrame,
    fields: Sequence[str],
) -> dict[str, int]:
    from pyspark.sql.functions import col, trim, when
    from pyspark.sql.functions import sum as spark_sum
    from pyspark.sql.types import StringType

    schema_by_name = {field.name: field.dataType for field in dataframe.schema.fields}
    aggregations = []
    for field in fields:
        invalid_condition = col(field).isNull()
        if isinstance(schema_by_name[field], StringType):
            invalid_condition = invalid_condition | (trim(col(field)) == "")
        aggregations.append(spark_sum(when(invalid_condition, 1).otherwise(0)).alias(field))

    row = dataframe.agg(*aggregations).first()
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


def _duplicate_fields(fields: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for field in fields:
        if field in seen and field not in duplicates:
            duplicates.append(field)
        seen.add(field)
    return duplicates


def _normalize_field_names(fields: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_field in fields:
        field = str(raw_field).strip()
        if field and field not in seen:
            normalized.append(field)
            seen.add(field)
    return tuple(normalized)


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
