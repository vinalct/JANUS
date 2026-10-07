"""Builders for the blocks that describe what a run produces.

``schema``, ``spark``, ``outputs`` and ``quality`` share one concern — the shape of the
written result — and none of them depends on how the data was fetched.
"""

from __future__ import annotations

from typing import Any

from janus.models.config.coercion import (
    _optional_int,
    _optional_string,
    _optional_string_list,
    _optional_string_mapping,
    _require_enum,
    _require_mapping,
    _require_string,
)
from janus.models.config.constants import (
    RETIRED_QUALITY_KEYS,
    RETIRED_SCHEMA_KEYS,
    SUPPORTED_DATA_FORMATS,
    SUPPORTED_WRITE_MODES,
)
from janus.models.config.issues import ValidationIssue
from janus.models.config.types import (
    BronzeRetentionConfig,
    OutputsConfig,
    OutputTarget,
    QualityConfig,
    SchemaConfig,
    SparkConfig,
)


def _build_schema_config(raw_value: Any, issues: list[ValidationIssue]) -> SchemaConfig:
    """Require one declared contract and reject the retired schema keys."""
    data = _require_mapping(raw_value, "schema", issues)
    for key in sorted(RETIRED_SCHEMA_KEYS):
        if key in data:
            issues.append(
                ValidationIssue(
                    f"schema.{key}",
                    "is no longer supported: declare schema.contract: "
                    "conf/contracts/<domain>/<table>.yaml — draft one with "
                    "`janus contract draft --source-id <id>` (see docs/data-contracts.md)",
                )
            )
    contract = _require_string(data, "contract", issues, "schema")
    return SchemaConfig(contract=contract)


def _build_spark_config(raw_value: Any, issues: list[ValidationIssue]) -> SparkConfig:
    """Validate the Spark-facing options that later tasks will consume."""
    data = _require_mapping(raw_value, "spark", issues)
    input_format = _require_enum(data, "input_format", SUPPORTED_DATA_FORMATS, issues, "spark")
    write_mode = _require_enum(data, "write_mode", SUPPORTED_WRITE_MODES, issues, "spark")
    repartition = _optional_int(data, "repartition", issues, "spark", minimum=1)
    partition_by = tuple(_optional_string_list(data, "partition_by", issues, "spark"))
    read_options = _optional_string_mapping(data, "read_options", issues, "spark")

    return SparkConfig(
        input_format=input_format,
        write_mode=write_mode,
        repartition=repartition,
        partition_by=partition_by,
        read_options=read_options,
    )


def _build_outputs_config(raw_value: Any, issues: list[ValidationIssue]) -> OutputsConfig:
    """Validate the output zone contract for raw, bronze, and metadata targets."""
    data = _require_mapping(raw_value, "outputs", issues)
    return OutputsConfig(
        raw=_build_output_target(data.get("raw"), "outputs.raw", issues),
        bronze=_build_output_target(data.get("bronze"), "outputs.bronze", issues),
        metadata=_build_output_target(data.get("metadata"), "outputs.metadata", issues),
    )


def _build_output_target(
    raw_value: Any, field_path: str, issues: list[ValidationIssue]
) -> OutputTarget:
    """Validate one concrete output target inside the outputs block."""
    data = _require_mapping(raw_value, field_path, issues)
    path = _require_string(data, "path", issues, field_path)
    format_name = _require_enum(data, "format", SUPPORTED_DATA_FORMATS, issues, field_path)
    namespace = _optional_string(data, "namespace", issues, field_path)
    table_name = _optional_string(data, "table_name", issues, field_path)
    shared_with = _build_shared_with(data, field_path, format_name, issues)
    retention = _build_bronze_retention(data, field_path, format_name, issues)

    if "table" in data and data["table"] is not None:
        issues.append(
            ValidationIssue(
                f"{field_path}.table",
                "is not supported; use table_name",
            )
        )

    if field_path != "outputs.bronze":
        if namespace is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.namespace",
                    "is only supported for outputs.bronze",
                )
            )
        if table_name is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.table_name",
                    "is only supported for outputs.bronze",
                )
            )

    if format_name != "iceberg":
        if namespace is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.namespace",
                    "requires format='iceberg'",
                )
            )
        if table_name is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.table_name",
                    "requires format='iceberg'",
                )
            )

    return OutputTarget(
        path=path,
        format=format_name,
        namespace=namespace,
        table_name=table_name,
        shared_with=shared_with,
        retention=retention,
    )


def _build_bronze_retention(
    data: Any,
    field_path: str,
    format_name: str,
    issues: list[ValidationIssue],
) -> BronzeRetentionConfig | None:
    """Validate one complete override; shared-table consistency belongs to maintenance."""
    if "retention" not in data:
        return None
    issue_count = len(issues)
    retention_path = f"{field_path}.retention"
    if field_path != "outputs.bronze":
        issues.append(ValidationIssue(retention_path, "is only supported for outputs.bronze"))
    elif format_name != "iceberg":
        issues.append(ValidationIssue(retention_path, "requires format='iceberg'"))

    if data["retention"] is None:
        issues.append(ValidationIssue(retention_path, "must be a mapping"))
        return None
    mapping_issue_count = len(issues)
    retention = _require_mapping(data["retention"], retention_path, issues)
    if len(issues) != mapping_issue_count:
        return None

    for key in sorted(retention, key=str):
        if key not in {"retain_last", "older_than_days"}:
            issues.append(
                ValidationIssue(
                    f"{retention_path}.{key}",
                    "is not supported; supported keys: older_than_days, retain_last",
                )
            )

    values: dict[str, int] = {}
    for key, minimum, message in (
        ("retain_last", 1, "must be a positive integer"),
        ("older_than_days", 0, "must be a non-negative integer"),
    ):
        integer_issues: list[ValidationIssue] = []
        value = _optional_int(retention, key, integer_issues, retention_path, minimum=minimum)
        if value is None or integer_issues:
            issues.append(ValidationIssue(f"{retention_path}.{key}", message))
        else:
            values[key] = value

    if len(issues) != issue_count:
        return None
    return BronzeRetentionConfig(
        retain_last=values["retain_last"], older_than_days=values["older_than_days"]
    )


def _build_shared_with(
    data: Any,
    field_path: str,
    format_name: str,
    issues: list[ValidationIssue],
) -> tuple[str, ...]:
    """Validate the declared co-writers of one bronze Iceberg table.

    Only the shape is decided here. Whether the named sources exist, write this same
    table, and name this source back is a whole-registry question, answered where the
    dependency graph is built.
    """
    peers = tuple(_optional_string_list(data, "shared_with", issues, field_path))
    if not peers:
        return ()

    if field_path != "outputs.bronze":
        issues.append(
            ValidationIssue(f"{field_path}.shared_with", "is only supported for outputs.bronze")
        )
    elif format_name != "iceberg":
        issues.append(ValidationIssue(f"{field_path}.shared_with", "requires format='iceberg'"))

    if len(set(peers)) != len(peers):
        issues.append(
            ValidationIssue(f"{field_path}.shared_with", "must not repeat a source id")
        )
        return tuple(dict.fromkeys(peers))
    return peers


def _build_quality_config(raw_value: Any, issues: list[ValidationIssue]) -> QualityConfig:
    """Validate the quality rules that travel with a source definition."""
    data = _require_mapping(raw_value, "quality", issues)
    required_fields = tuple(_optional_string_list(data, "required_fields", issues, "quality"))
    unique_fields = tuple(_optional_string_list(data, "unique_fields", issues, "quality"))
    for key in sorted(RETIRED_QUALITY_KEYS):
        if key in data:
            issues.append(
                ValidationIssue(
                    f"quality.{key}",
                    "is no longer supported: schema evolution is governed by "
                    "the contract's customProperties janus.compatibility "
                    "(additive | backward | frozen)",
                )
            )

    return QualityConfig(
        required_fields=required_fields,
        unique_fields=unique_fields,
    )
