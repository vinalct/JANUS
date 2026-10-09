"""Builders for the blocks that describe what a run produces.

``schema``, ``spark``, ``outputs`` and ``quality`` share one concern — the shape of the
written result — and none of them depends on how the data was fetched.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any

from janus.models.config.coercion import (
    _optional_enum,
    _optional_int,
    _optional_string,
    _optional_string_list,
    _optional_string_mapping,
    _require_enum,
    _require_mapping,
    _require_string,
)
from janus.models.config.constants import (
    DEFAULT_HANDOFF_MAX_ARTIFACT_MB,
    DEFAULT_HANDOFF_MODE,
    DEFAULT_TARGET_FILE_SIZE_MB,
    MAX_TARGET_FILE_SIZE_MB,
    RETIRED_QUALITY_KEYS,
    RETIRED_SCHEMA_KEYS,
    SUPPORTED_DATA_FORMATS,
    SUPPORTED_DISTRIBUTION_MODES,
    SUPPORTED_HANDOFF_MODES,
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
    SparkHandoffConfig,
    SparkWriteConfig,
    SparkWriteDefaults,
)
from janus.utils.catalog_options import non_empty_text


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


def _build_spark_config(
    raw_value: Any,
    issues: list[ValidationIssue],
    *,
    deprecations: list[ValidationIssue] | None = None,
    write_defaults: SparkWriteDefaults | None = None,
) -> SparkConfig:
    """Validate the Spark-facing options that later tasks will consume."""
    data = _require_mapping(raw_value, "spark", issues)
    input_format = _require_enum(data, "input_format", SUPPORTED_DATA_FORMATS, issues, "spark")
    write_mode = _require_enum(data, "write_mode", SUPPORTED_WRITE_MODES, issues, "spark")
    repartition = _optional_int(data, "repartition", issues, "spark", minimum=1)
    _collect_repartition_deprecation(data, issues, deprecations)
    partition_by = tuple(_optional_string_list(data, "partition_by", issues, "spark"))
    read_options = _optional_string_mapping(data, "read_options", issues, "spark")
    handoff = _build_spark_handoff_config(data.get("handoff"), issues)
    write = _build_spark_write_config(
        data.get("write"), issues, partition_by=partition_by, defaults=write_defaults
    )
    if repartition is not None and write.partitions is None:
        write = replace(write, partitions=repartition)

    return SparkConfig(
        input_format=input_format,
        write_mode=write_mode,
        repartition=repartition,
        partition_by=partition_by,
        read_options=read_options,
        handoff=handoff,
        write=write,
    )


def _collect_repartition_deprecation(
    data: Mapping[str, Any],
    issues: list[ValidationIssue],
    deprecations: list[ValidationIssue] | None,
) -> None:
    """Keep legacy warnings separate from the ordered, fatal issue collector."""
    if "repartition" not in data:
        return
    write = data.get("write")
    if isinstance(write, Mapping) and "partitions" in write:
        issues.append(
            ValidationIssue(
                "spark.write.partitions",
                "must not be declared together with the deprecated spark.repartition",
            )
        )
    elif deprecations is not None:
        deprecations.append(
            ValidationIssue(
                "spark.repartition",
                "is deprecated and will be removed in the next order; "
                "declare spark.write.partitions instead",
            )
        )


def _build_spark_write_config(
    raw_value: Any,
    issues: list[ValidationIssue],
    *,
    partition_by: tuple[str, ...],
    defaults: SparkWriteDefaults | None = None,
) -> SparkWriteConfig:
    """Source keys override profile values, then partition-aware built-ins.

    An absent block is valid. Malformed declarations append issues without raising.
    This declaration is consumed by later partitioning and table-property tasks.
    """
    data = {} if raw_value is None else _require_mapping(raw_value, "spark.write", issues)
    profile = defaults or SparkWriteDefaults()
    size = _optional_int(
        data, "target_file_size_mb", issues, "spark.write",
        default=profile.target_file_size_mb or DEFAULT_TARGET_FILE_SIZE_MB, minimum=1,
    )
    _check_target_file_size(size, issues, "spark.write")
    mode = _write_distribution_mode(
        data, issues, "spark.write",
        default=profile.distribution_mode or ("hash" if partition_by else "none"),
    )
    partitions = _optional_int(data, "partitions", issues, "spark.write", minimum=1)
    return SparkWriteConfig(target_file_size_mb=size, distribution_mode=mode, partitions=partitions)


def _build_spark_write_defaults(
    raw_value: Any, issues: list[ValidationIssue]
) -> SparkWriteDefaults:
    """Missing profile defaults are valid; wrong values collect structural issues.

    Environment expansion produces numeric strings. Accept those only at this profile
    boundary, and use the shared empty guard for an undeclared distribution mode.
    """
    data = dict(
        {} if raw_value is None else _require_mapping(raw_value, "spark.write_defaults", issues)
    )
    size_value = data.get("target_file_size_mb")
    if isinstance(size_value, str):
        # Leave invalid text for the integer validator to report.
        with suppress(ValueError):
            data["target_file_size_mb"] = int(size_value)
    size = _optional_int(data, "target_file_size_mb", issues, "spark.write_defaults", minimum=1)
    _check_target_file_size(size, issues, "spark.write_defaults")
    data["distribution_mode"] = non_empty_text(data.get("distribution_mode"))
    mode = _write_distribution_mode(data, issues, "spark.write_defaults")
    return SparkWriteDefaults(target_file_size_mb=size, distribution_mode=mode or None)


def resolve_spark_write_defaults(raw_value: Any) -> SparkWriteDefaults:
    """The profile boundary raises once, after its builder has collected every issue."""
    issues: list[ValidationIssue] = []
    defaults = _build_spark_write_defaults(raw_value, issues)
    if issues:
        raise ValueError(
            "Invalid Spark write defaults:\n" + "\n".join(issue.render() for issue in issues)
        )
    return defaults


def _check_target_file_size(
    size: int | None, issues: list[ValidationIssue], prefix: str
) -> None:
    if size is not None and size > MAX_TARGET_FILE_SIZE_MB:
        issues.append(
            ValidationIssue(
                f"{prefix}.target_file_size_mb", f"must be <= {MAX_TARGET_FILE_SIZE_MB}"
            )
        )


def _write_distribution_mode(
    data: Any, issues: list[ValidationIssue], prefix: str, *, default: str = ""
) -> str:
    mode_issues: list[ValidationIssue] = []
    mode = _optional_enum(
        data, "distribution_mode", SUPPORTED_DISTRIBUTION_MODES,
        mode_issues, prefix, default=default,
    )
    if mode_issues:
        issues.append(
            ValidationIssue(
                f"{prefix}.distribution_mode",
                f"must be one of: {', '.join(sorted(SUPPORTED_DISTRIBUTION_MODES))}",
            )
        )
        return default
    return mode


def _build_spark_handoff_config(
    raw_value: Any, issues: list[ValidationIssue]
) -> SparkHandoffConfig:
    """Validate spark.handoff. Absent means the default; present and wrong is an issue."""
    if raw_value is None:
        return SparkHandoffConfig()
    if isinstance(raw_value, str):
        raw_value = {"mode": raw_value}
    data = _require_mapping(raw_value, "spark.handoff", issues)
    mode = _optional_enum(
        data,
        "mode",
        SUPPORTED_HANDOFF_MODES,
        issues,
        "spark.handoff",
        default=DEFAULT_HANDOFF_MODE,
    )
    size_issues: list[ValidationIssue] = []
    max_artifact_mb = _optional_int(
        data,
        "max_artifact_mb",
        size_issues,
        "spark.handoff",
        default=DEFAULT_HANDOFF_MAX_ARTIFACT_MB,
        minimum=1,
    )
    if size_issues:
        issues.append(ValidationIssue("spark.handoff.max_artifact_mb", "must be >= 1"))
        max_artifact_mb = DEFAULT_HANDOFF_MAX_ARTIFACT_MB
    return SparkHandoffConfig(mode=mode, max_artifact_mb=max_artifact_mb)


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
