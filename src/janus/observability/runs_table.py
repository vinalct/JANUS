"""The engine-free contract for the append-only ``metadata.runs`` Iceberg table."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from janus.utils.catalog_properties import derive_pyiceberg_catalog_name
from janus.utils.storage import sanitize_identifier_segment

DEFAULT_RUNS_TABLE_NAMESPACE = "metadata"
DEFAULT_RUNS_TABLE_NAME = "runs"
DEFAULT_RUNS_TABLE_IDENTIFIER = f"{DEFAULT_RUNS_TABLE_NAMESPACE}.{DEFAULT_RUNS_TABLE_NAME}"
_QUALIFIED_IDENTIFIER_SEGMENT_COUNT = 2


class RunsTableContractError(ValueError):
    """The environment profile cannot identify a valid runs table."""


class IcebergType(StrEnum):
    """Iceberg logical types used by the declaration, without importing PyIceberg."""

    STRING = "string"
    INTEGER = "int"
    LONG = "long"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    TIMESTAMPTZ = "timestamptz"
    STRING_LIST = "list<string>"


@dataclass(frozen=True, slots=True)
class RunsTableColumn:
    """One stable Iceberg field in the runs-table schema."""

    field_id: int
    name: str
    iceberg_type: IcebergType
    nullable: bool
    element_id: int | None = None

    @property
    def required(self) -> bool:
        """The spelling PyIceberg uses for the inverse of nullability."""
        return not self.nullable


@dataclass(frozen=True, slots=True)
class RunsTablePartitionField:
    """One transform in the runs-table partition spec."""

    source_column: str
    transform: str
    name: str


@dataclass(frozen=True, slots=True)
class RunsTableDeclaration:
    """The environment-independent declaration a sink may execute."""

    schema: tuple[RunsTableColumn, ...]
    partition_spec: tuple[RunsTablePartitionField, ...]
    append_only: bool
    retention_days: int | None


@dataclass(frozen=True, slots=True)
class RunsTableTarget:
    """The catalog and two-part identifier resolved from one environment profile."""

    catalog_name: str
    namespace: str
    table_name: str

    @property
    def identifier(self) -> str:
        return f"{self.namespace}.{self.table_name}"


RUNS_TABLE_SCHEMA = (
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
    RunsTableColumn(44, "schema_version", IcebergType.STRING, True),
    RunsTableColumn(45, "contract_id", IcebergType.STRING, True),
    RunsTableColumn(46, "contract_version", IcebergType.STRING, True),
    RunsTableColumn(47, "contract_preflight_outcome", IcebergType.STRING, True),
    RunsTableColumn(48, "schema_evolution", IcebergType.STRING, True),
    RunsTableColumn(49, "malformed_rows", IcebergType.LONG, True),
)

RUNS_TABLE_PARTITION_SPEC = (
    RunsTablePartitionField(
        source_column="emitted_at",
        transform="day",
        name="emitted_at_day",
    ),
)

RUNS_TABLE = RunsTableDeclaration(
    schema=RUNS_TABLE_SCHEMA,
    partition_spec=RUNS_TABLE_PARTITION_SPEC,
    append_only=True,
    retention_days=None,
)


def resolve_runs_table(config: Mapping[str, Any]) -> RunsTableTarget:
    """Resolve the shared catalog and runs-table identifier from one profile."""
    catalog_name = derive_pyiceberg_catalog_name(dict(config))
    namespace, table_name = _configured_identifier(config)
    return RunsTableTarget(
        catalog_name=catalog_name,
        namespace=namespace,
        table_name=table_name,
    )


def resolve_runs_table_identifier(config: Mapping[str, Any]) -> str:
    """Return the sanitized ``<namespace>.<table>`` identifier for one profile."""
    return resolve_runs_table(config).identifier


def _configured_identifier(config: Mapping[str, Any]) -> tuple[str, str]:
    observability = config.get("observability")
    if observability is None:
        return DEFAULT_RUNS_TABLE_NAMESPACE, DEFAULT_RUNS_TABLE_NAME
    if not isinstance(observability, Mapping):
        raise RunsTableContractError("observability must be a mapping")
    if "runs_table" not in observability:
        return DEFAULT_RUNS_TABLE_NAMESPACE, DEFAULT_RUNS_TABLE_NAME

    configured = observability["runs_table"]
    if not isinstance(configured, str) or not configured.strip():
        raise RunsTableContractError(
            "observability.runs_table must be a non-empty '<namespace>.<table>' identifier"
        )

    segments = configured.split(".")
    if len(segments) != _QUALIFIED_IDENTIFIER_SEGMENT_COUNT or any(
        not segment.strip() for segment in segments
    ):
        raise RunsTableContractError(
            "observability.runs_table must be qualified as '<namespace>.<table>'"
        )

    namespace, table_name = (sanitize_identifier_segment(segment) for segment in segments)
    if not namespace or not table_name:
        raise RunsTableContractError(
            "observability.runs_table namespace and table must survive identifier sanitization"
        )
    return namespace, table_name
