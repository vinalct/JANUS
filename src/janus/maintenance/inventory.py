"""Read-only inventories for retention decisions.

Bronze identities come exclusively from the registry and the writer's derivation.
A namespace listing only probes existence; it never adds a table to the inventory.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

from janus.lineage.persistence import MetadataZonePaths, read_json_mapping
from janus.models import BronzeRetentionConfig, ExecutionPlan
from janus.observability.openlineage.settings import OpenLineageSettings
from janus.observability.openlineage.transport import (
    EVENTS_FILE_PREFIX,
    EVENTS_FILE_SUFFIX,
    FileOpenLineageTransport,
    resolve_openlineage_transport,
)

if typing.TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from janus.maintenance.settings import MaintenancePolicy
    from janus.models import SourceConfig
    from janus.registry import SourceRegistry
    from janus.utils.environment import RuntimeLocation
    from janus.utils.storage import StorageLayout

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
class RunArtifactKind:
    kind: str
    directory_attribute: str
    timestamp_field: str


RUN_ARTIFACT_KINDS = (
    RunArtifactKind("runs", "runs_dir", "started_at"),
    RunArtifactKind("lineage", "lineage_dir", "emitted_at"),
    RunArtifactKind("checkpoint_history", "checkpoint_history_dir", "recorded_at"),
    RunArtifactKind("validations", "validations_dir", "emitted_at"),
)


@dataclass(frozen=True, slots=True)
class MetadataZoneInventory:
    artifacts: tuple[RunArtifactEntry, ...]
    protected_paths: frozenset[Path]
    live_raw_run_segments: typing.Mapping[str, str | None]


@dataclass(frozen=True, slots=True)
class EventFileEntry:
    path: Path
    day: date | None
    skipped_reason: str | None = None


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
    metadata = None
    if "metadata" in zones:
        from janus.planner import Planner, PlanningRequest
        from janus.utils.storage import StorageLayout

        planner = Planner()
        plans = (
            planner.plan(
                PlanningRequest.create(
                    source_id=source.source_id,
                    environment=config["name"],
                    project_root=registry.project_root,
                    run_id="maintenance-inventory",
                    started_at=now,
                    include_disabled=True,
                ),
                registry=registry,
            ).plan
            for source in registry.list_sources(enabled_only=False)
            if source_ids is None or source.source_id in source_ids
        )
        metadata = collect_metadata_inventory(
            plans,
            StorageLayout.from_environment_config(config, registry.project_root),
            source_ids=source_ids,
        )
    events: tuple[EventFileEntry, ...] = ()
    if "lineage" in zones:
        from janus.observability.openlineage.settings import resolve_openlineage_settings

        events = collect_lineage_event_files(
            (),
            resolve_openlineage_settings(config),
            now=now,
            resolved_paths=resolved_paths,
        )
    return MaintenanceInventory(bronze=bronze, metadata=metadata, lineage_events=events)


def collect_lineage_event_files(
    plans: Iterable[ExecutionPlan],
    settings: OpenLineageSettings,
    *,
    now: datetime,
    resolved_paths: Mapping[str, RuntimeLocation] | None = None,
) -> tuple[EventFileEntry, ...]:
    """List day files without reading payloads or aging by mtime.

    The command supplies the shared runtime paths used by the emitting transport.
    Plan-owned metadata roots are also supported for independently scoped inventories.
    Invalid filenames remain skipped evidence; only the planner decides file age.
    """
    del now  # Collection is independent of the retention clock.
    if settings.file is None:
        return ()
    roots = (
        (resolved_paths,)
        if resolved_paths is not None
        else ({"metadata_dir": MetadataZonePaths.from_plan(plan).base_dir} for plan in plans)
    )
    directories = set()
    for paths in roots:
        transport = resolve_openlineage_transport(settings, paths)
        assert isinstance(transport, FileOpenLineageTransport)
        directories.add(transport.directory)
    entries = []
    for directory in sorted(directories):
        for path in sorted(directory.glob(f"{EVENTS_FILE_PREFIX}*{EVENTS_FILE_SUFFIX}")):
            if not path.is_file():
                continue
            filename_day = path.name.removeprefix(EVENTS_FILE_PREFIX).removesuffix(
                EVENTS_FILE_SUFFIX
            )
            try:
                day = date.fromisoformat(filename_day)
                if day.isoformat() != filename_day:
                    raise ValueError("event filenames require YYYY-MM-DD")
            except ValueError:
                entries.append(EventFileEntry(path, None, "invalid_event_filename"))
            else:
                entries.append(EventFileEntry(path, day))
    return tuple(entries)


def collect_metadata_inventory(
    plans: Iterable[ExecutionPlan],
    storage_layout: StorageLayout,
    *,
    source_ids: frozenset[str] | None,
) -> MetadataZoneInventory:
    """Read declared history roots, with state protected even when absent.

    Only timestamps inside each record determine its age. A malformed record
    remains inventory evidence for the planner to protect, never a candidate.
    """
    artifacts = []
    protected_paths: set[Path] = set()
    live: dict[str, str | None] = {}
    for plan in plans:
        source_id = plan.source.source_id
        if source_ids is not None and source_id not in source_ids:
            continue
        paths = MetadataZonePaths.from_plan(plan)
        progress_path = paths.base_dir / "extraction_progress.json"
        protected_paths.update(
            (paths.checkpoint_state_path, paths.dead_letter_state_path, progress_path)
        )

        progress = read_json_mapping(progress_path)
        if progress is not None:
            live[source_id] = _progress_run_segment(progress)
        for kind in RUN_ARTIFACT_KINDS:
            directory = getattr(paths, kind.directory_attribute)
            for path in sorted(directory.glob("*.json")):
                if path.is_file():
                    artifacts.append(_read_run_artifact(path, kind, source_id))

    for path in sorted((storage_layout.metadata_dir / "pipelines").glob("*/summary.json")):
        if path.is_file():
            artifacts.append(_read_pipeline_artifact(path))
    return MetadataZoneInventory(tuple(artifacts), frozenset(protected_paths), live)


def _progress_run_segment(progress: Mapping[str, typing.Any]) -> str | None:
    prefix = progress.get("raw_path_prefix")
    if not isinstance(prefix, str):
        return None
    return next(
        (
            part.removeprefix("run_id=")
            for part in Path(prefix).parts
            if part.startswith("run_id=") and part != "run_id="
        ),
        None,
    )


def _record_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return timestamp.astimezone(UTC) if timestamp.utcoffset() is not None else None


def _read_run_artifact(path: Path, kind: RunArtifactKind, source_id: str) -> RunArtifactEntry:
    entry = RunArtifactEntry(kind.kind, path, source_id, path.stem, None)
    try:
        payload = read_json_mapping(path)
        if payload is None:
            return replace(entry, read_error="already_absent")
        if payload.get("run_id") != path.stem:
            return replace(entry, read_error="run_id_mismatch")
        return replace(entry, timestamp=_record_timestamp(payload.get(kind.timestamp_field)))
    except (OSError, ValueError) as exc:
        return replace(entry, read_error=type(exc).__name__)


def _read_pipeline_artifact(path: Path) -> RunArtifactEntry:
    entry = RunArtifactEntry("pipelines", path, None, path.parent.name, None)
    try:
        payload = read_json_mapping(path)
        if payload is None:
            return replace(entry, read_error="already_absent")
        pipeline = payload.get("pipeline")
        timestamp = pipeline.get("started_at") if isinstance(pipeline, Mapping) else None
        return replace(entry, timestamp=_record_timestamp(timestamp))
    except (OSError, ValueError) as exc:
        return replace(entry, read_error=type(exc).__name__)


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
