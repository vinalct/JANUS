"""Read-only inventories for retention decisions.

Bronze identities come exclusively from the registry and the writer's derivation.
A namespace listing only probes existence; it never adds a table to the inventory.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

from janus.models import BronzeRetentionConfig

if typing.TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from janus.maintenance.settings import MaintenancePolicy
    from janus.models import SourceConfig
    from janus.registry import SourceRegistry
    from janus.utils.environment import RuntimeLocation

MAX_EXCEPTION_CAUSES = 8


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    snapshot_id: int
    committed_at: datetime
    is_current: bool
    parent_id: int | None = None


@dataclass(frozen=True, slots=True)
class BronzeTableInventory:
    table_identifier: str
    source_ids: tuple[str, ...]
    snapshots: tuple[SnapshotEntry, ...]
    override: BronzeRetentionConfig | None = None
    unavailable_reason: str | None = None
    selected_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunArtifactEntry:
    kind: str
    path: Path
    source_id: str | None
    run_id: str | None
    timestamp: datetime | None
    read_error: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataZoneInventory:
    artifacts: tuple[RunArtifactEntry, ...]
    protected_paths: frozenset[Path]
    live_raw_run_segments: typing.Mapping[str, str | None]


@dataclass(frozen=True, slots=True)
class EventFileEntry:
    path: Path
    day: date


@dataclass(frozen=True, slots=True)
class RunsTablePartitionEntry:
    emitted_at_day: date
    row_count: int | None = None


@dataclass(frozen=True, slots=True)
class RawRunPrefixEntry:
    path: Path
    source_id: str
    run_segment: str
    ingestion_date: date
    run_succeeded: bool | None


@dataclass(frozen=True, slots=True)
class MaintenanceInventory:
    bronze: tuple[BronzeTableInventory, ...] = ()
    metadata: MetadataZoneInventory | None = None
    lineage_events: tuple[EventFileEntry, ...] = ()
    runs_table: tuple[RunsTablePartitionEntry, ...] = ()
    raw: tuple[RawRunPrefixEntry, ...] = ()


def collect_inventory(
    registry: SourceRegistry,
    config: Mapping[str, typing.Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    policy: MaintenancePolicy,
    now: datetime,
    *,
    zones: frozenset[str],
    source_ids: frozenset[str] | None,
    session: SparkSession | None,
) -> MaintenanceInventory:
    """Collect through one seam using the command's already acquired session."""
    from janus.utils.catalog_properties import derive_pyiceberg_catalog_name

    bronze: tuple[BronzeTableInventory, ...] = ()
    if "bronze" in zones:
        if session is None:
            raise ValueError("bronze inventory requires a Spark session")
        bronze = collect_bronze_inventory(
            registry,
            catalog_name=derive_pyiceberg_catalog_name(dict(config)),
            source_ids=source_ids,
            session=session,
        )
    return MaintenanceInventory(bronze=bronze)


def collect_bronze_inventory(
    registry: SourceRegistry,
    *,
    catalog_name: str,
    source_ids: frozenset[str] | None,
    session: SparkSession,
) -> tuple[BronzeTableInventory, ...]:
    """Group every registry Iceberg target, then read only in-scope tables.

    Disabled sources and unselected co-writers still participate in grouping and
    override agreement. A failure in one namespace or table never aborts the walk.
    """
    from janus.registry.dependencies import producer_table_identifier

    groups: dict[str, list[SourceConfig]] = {}
    for source in registry.list_sources(enabled_only=False):
        identifier = producer_table_identifier(source)
        if identifier is not None:
            groups.setdefault(identifier, []).append(source)
    tables = []
    for identifier, sources in sorted(groups.items()):
        writers = tuple(sorted(source.source_id for source in sources))
        selected = tuple(
            writer for writer in writers if source_ids is not None and writer in source_ids
        )
        if source_ids is not None and not selected:
            continue
        override, conflict = _retention_override(sources)
        tables.append(BronzeTableInventory(identifier, writers, (), override, conflict, selected))

    namespaces: dict[str, frozenset[str] | str] = {}
    result = []
    for table in tables:
        if table.unavailable_reason is not None:
            result.append(table)
            continue
        namespace, name = table.table_identifier.rsplit(".", 1)
        if namespace not in namespaces:
            namespaces[namespace] = _existing_tables(session, f"{catalog_name}.{namespace}")
        existing = namespaces[namespace]
        if isinstance(existing, str):
            result.append(replace(table, unavailable_reason=existing))
        elif name not in existing:
            result.append(replace(table, unavailable_reason="absent_table"))
        else:
            result.append(_read_bronze_table(table, catalog_name, session))
    return tuple(result)


def _retention_override(
    sources: list[SourceConfig],
) -> tuple[BronzeRetentionConfig | None, str | None]:
    ordered = sorted(sources, key=lambda source: source.source_id)
    override = ordered[0].outputs.bronze.retention
    if all(source.outputs.bronze.retention == override for source in ordered):
        return override, None
    declarations = []
    for source in ordered:
        retention = source.outputs.bronze.retention
        value = (
            "none"
            if retention is None
            else f"retain_last={retention.retain_last}, older_than_days={retention.older_than_days}"
        )
        declarations.append(f"{source.source_id} declares {value}")
    return None, f"retention_conflict: {'; '.join(declarations)}"


def _existing_tables(session: SparkSession, namespace: str) -> frozenset[str] | str:
    from janus.writers.identifiers import quote_identifier

    try:
        # Existence probe only: the registry, never SHOW TABLES, defines ownership.
        rows = session.sql(f"SHOW TABLES IN {quote_identifier(namespace)}").collect()
        return frozenset(row.tableName for row in rows if not row.isTemporary)
    except Exception as exc:
        if _absent_namespace(exc):
            return "absent_table"
        return f"snapshot_read_failed: {type(exc).__name__}"


def _absent_namespace(exc: Exception) -> bool:
    """Recognize Spark conditions and Iceberg's unwrapped JDBC/REST Java error."""
    try:
        condition = getattr(exc, "getCondition", None) or getattr(exc, "getErrorClass", None)
        if condition is not None and condition() in {
            "SCHEMA_NOT_FOUND",
            "NAMESPACE_NOT_FOUND",
            "NO_SUCH_NAMESPACE",
        }:
            return True
        java = getattr(exc, "java_exception", None)
        for _ in range(MAX_EXCEPTION_CAUSES):
            if java is None:
                break
            if java.getClass().getName() in {
                "org.apache.iceberg.exceptions.NoSuchNamespaceException",
                "org.apache.spark.sql.catalyst.analysis.NoSuchNamespaceException",
            }:
                return True
            java = java.getCause()
    except Exception:
        # A failed JVM error inspection remains a read failure, never an absent table.
        pass
    return False


def _read_bronze_table(
    table: BronzeTableInventory, catalog_name: str, session: SparkSession
) -> BronzeTableInventory:
    from janus.writers.identifiers import quote_identifier

    qualified = f"{catalog_name}.{table.table_identifier}"
    try:
        columns = ", ".join(
            quote_identifier(column) for column in ("snapshot_id", "parent_id", "committed_at")
        )
        rows = session.sql(
            f"SELECT {columns} FROM {quote_identifier(f'{qualified}.snapshots')} "
            f"ORDER BY {quote_identifier('committed_at')}"
        ).collect()
        refs = session.sql(
            f"SELECT {quote_identifier('snapshot_id')} "
            f"FROM {quote_identifier(f'{qualified}.refs')} "
            f"WHERE {quote_identifier('name')} = 'main'"
        ).collect()
        if rows and len(refs) != 1:
            raise ValueError("a nonempty table must have exactly one main snapshot reference")
        current = refs[0].snapshot_id if refs else None
        snapshots = tuple(
            SnapshotEntry(
                row.snapshot_id,
                _utc_timestamp(row.committed_at),
                row.snapshot_id == current,
                row.parent_id,
            )
            for row in rows
        )
        if snapshots and not any(snapshot.is_current for snapshot in snapshots):
            raise ValueError("the main reference is absent from the snapshot inventory")
        return replace(table, snapshots=snapshots)
    except Exception as exc:
        return replace(table, unavailable_reason=f"snapshot_read_failed: {type(exc).__name__}")


def _utc_timestamp(value: datetime) -> datetime:
    """Spark returns naive local Python datetimes; restore UTC at the boundary."""
    return value.astimezone(UTC)
