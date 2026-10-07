"""Explicit retention maintenance: dry run by default, with evidence on every plan."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any

import yaml

from janus.cli.common import format_runtime_permission_error
from janus.maintenance.execute import execute_retention
from janus.maintenance.inventory import MaintenanceInventory, collect_inventory
from janus.maintenance.locking import (
    LOCKED_SOURCE_REASON,
    MaintenanceLock,
    NullMaintenanceLock,
    source_lock,
)
from janus.maintenance.planning import PlannedItem, RetentionPlan, plan_retention
from janus.maintenance.records import (
    ItemOutcome,
    MaintenanceRecord,
    MaintenanceRecordStore,
    RecordFailure,
)
from janus.maintenance.settings import MaintenancePolicy, resolve_maintenance_settings
from janus.registry import SourceNotFoundError, SourceRegistry, load_registry
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.utils.catalog_properties import derive_pyiceberg_catalog_name
from janus.utils.environment import RuntimeLocation, load_environment_config, prepare_runtime
from janus.utils.logging import StructuredLogger, build_structured_logger
from janus.utils.storage import StorageLayout

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

ARGUMENT_ERROR = 2
OPERATIONAL_ERROR = 1
ZONES = ("bronze", "metadata", "lineage", "runs-table", "raw")
COMPUTE_ZONES = frozenset({"bronze", "runs-table"})
SOURCE_SCOPED_ZONES = frozenset({"metadata", "raw"})
LOCK_WARNING = (
    "no source lock is held — do not run maintain while a run of the same source is in flight"
)
ORPHAN_WARNING = "do not remove orphan files while an extraction is in flight"

_UNSUPPORTED_OPTIONS = {
    "execute": True,
    "run-id": False,
    "ingest-raw-to-bronze": True,
    "with-spark": True,
    "bronze-table": False,
    "include-disabled": True,
    "tag": False,
    "domain": False,
    "max-parallel": False,
    "started-at": False,
}


def configure(parser: argparse.ArgumentParser) -> None:
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print and record the retention plan without deleting anything (the default).",
    )
    modes.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="Apply the declared retention plan and record each item's outcome.",
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument(
        "--zone",
        choices=ZONES,
        action="append",
        default=[],
        help="Limit maintenance to this zone; repeat to select several. Defaults to all "
        "declared zones, with raw included only when enabled.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Restrict bronze, metadata and raw to these sources; repeat to select several. "
        "Shared lineage files and runs-table partitions cannot be filtered by source.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format. JSON is exactly the persisted maintenance record.",
    )
    for option, boolean in _UNSUPPORTED_OPTIONS.items():
        kwargs: dict[str, Any] = {"default": argparse.SUPPRESS, "help": argparse.SUPPRESS}
        kwargs.update({"action": "store_true"} if boolean else {"nargs": "?", "const": ""})
        parser.add_argument(f"--{option}", **kwargs)


@dataclass(frozen=True, slots=True)
class _PreparedMaintenance:
    config: dict[str, Any]
    resolved_paths: dict[str, RuntimeLocation]
    registry: SourceRegistry
    policy: MaintenancePolicy
    zones: frozenset[str]
    source_ids: frozenset[str] | None
    store: MaintenanceRecordStore
    logger: StructuredLogger


@dataclass(slots=True)
class _Compute:
    prepared: _PreparedMaintenance
    provider: SparkSessionProvider | None = None
    session: SparkSession | None = None

    def get_session(self) -> SparkSession:
        if self.provider is None:
            self.provider = SparkSessionProvider(
                self.prepared.config, self.prepared.resolved_paths, self.prepared.logger
            )
            self.session = self.provider.get()
        assert self.session is not None
        return self.session

    def stop(self) -> tuple[RecordFailure, ...]:
        if self.provider is None:
            return ()
        self.provider.stop()
        return tuple(
            RecordFailure("spark_cleanup", type(exc).__name__, str(exc))
            for exc in self.provider.take_cleanup_failures()
        )


def maintain_command(
    args: argparse.Namespace,
    *,
    lock: MaintenanceLock = NullMaintenanceLock(),
) -> int:
    """Exit 2 before an attempt, 1 for operational failures, or 0 including empty plans."""
    now = datetime.now(tz=UTC)
    started = perf_counter()
    try:
        _reject_other_options(args)
        prepared = _prepare(args)
    except PermissionError as exc:
        return _refuse(format_runtime_permission_error(exc))
    except (OSError, KeyError, TypeError, ValueError, SourceNotFoundError, yaml.YAMLError) as exc:
        return _refuse(str(exc))

    try:
        plan, record = _run(prepared, args, now, started, lock)
    except ValueError as exc:
        return _refuse(str(exc))
    except Exception as exc:
        print(f"janus maintain: maintenance failed: {exc}", file=sys.stderr)
        return OPERATIONAL_ERROR

    if args.format == "json":
        if _warn_unlocked(record):
            print(f"warning: {LOCK_WARNING}", file=sys.stderr)
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    else:
        print(_render_text(plan, record))
    for item in record.items:
        item.log(prepared.logger, maintenance_run_id=record.maintenance_run_id)
    for failure in record.failures:
        prepared.logger.error(
            "maintenance_failed", maintenance_run_id=record.maintenance_run_id, **failure.to_dict()
        )
    try:
        prepared.store.persist(record)
    except OSError as exc:
        print(f"janus maintain: could not persist maintenance record: {exc}", file=sys.stderr)
        return OPERATIONAL_ERROR
    return OPERATIONAL_ERROR if record.has_failures else 0


def _refuse(message: str) -> int:
    print(f"janus maintain: {' '.join(message.split())}", file=sys.stderr)
    return ARGUMENT_ERROR


def _reject_other_options(args: argparse.Namespace) -> None:
    for option in _UNSUPPORTED_OPTIONS:
        if hasattr(args, option.replace("-", "_")):
            raise ValueError(f"--{option} is not a maintain option")


def _prepare(args: argparse.Namespace) -> _PreparedMaintenance:
    project_root = args.project_root.resolve()
    config = load_environment_config(args.environment, project_root)
    paths = prepare_runtime(config, project_root)
    policy = resolve_maintenance_settings(config)
    registry = load_registry(project_root)
    zones, source_ids = _selection(args, policy, registry)
    logger = build_structured_logger(
        "janus.maintenance",
        level=config.get("runtime", {}).get("log_level", "INFO"),
    ).bind(environment=args.environment, project_root=str(project_root))
    return _PreparedMaintenance(
        config,
        paths,
        registry,
        policy,
        zones,
        source_ids,
        MaintenanceRecordStore(StorageLayout.from_environment_config(config, project_root)),
        logger,
    )


def _selection(
    args: argparse.Namespace,
    policy: MaintenancePolicy,
    registry: SourceRegistry,
) -> tuple[frozenset[str], frozenset[str] | None]:
    zones = frozenset(args.zone or (zone for zone in ZONES if zone != "raw" or policy.raw.enabled))
    if "raw" in zones and not policy.raw.enabled:
        raise ValueError("--zone raw requires maintenance.raw.enabled: true")
    source_ids = frozenset(args.source_id) if args.source_id else None
    if source_ids is not None:
        if args.zone and "runs-table" in zones:
            raise ValueError(
                "--source-id cannot restrict --zone runs-table: these are shared artifacts "
                "retained as whole partitions"
            )
        for source_id in sorted(source_ids):
            registry.get_source(source_id, include_disabled=True)
    return zones, source_ids


def _run(
    prepared: _PreparedMaintenance,
    args: argparse.Namespace,
    now: datetime,
    started: float,
    lock: MaintenanceLock,
) -> tuple[RetentionPlan, MaintenanceRecord]:
    compute = _Compute(prepared)
    locks = ExitStack()
    refused: set[str] = set()
    failures = []
    plan = RetentionPlan((), (), now, prepared.policy.digest)
    items = None
    pending: list[ItemOutcome] = []
    interruption: KeyboardInterrupt | SystemExit | None = None
    try:
        if prepared.zones & SOURCE_SCOPED_ZONES:
            for source_id in sorted(_selected_sources(prepared)):
                if not locks.enter_context(source_lock(lock, source_id)):
                    refused.add(source_id)
        skipped = _locked_items(prepared.zones, refused)
        plan = replace(plan, items=skipped)
        pending = [ItemOutcome.pending_apply(item) for item in skipped]
        session = compute.get_session() if prepared.zones & COMPUTE_ZONES else None
        inventory = _collect_inventory(prepared, now, session, refused)
        plan = _plan_inventory(inventory, prepared, now, refused)
        plan = replace(plan, items=(*plan.items, *skipped))
        pending = [ItemOutcome.pending_apply(item) for item in plan.items]
        # Validate the generated identity and its destination before an apply can act.
        record = MaintenanceRecord.from_plan(
            plan,
            environment=args.environment,
            dry_run=True,
            zones=prepared.zones,
            source_ids=prepared.source_ids or (),
            started_at=now,
            ended_at=now,
        )
        prepared.store.record_path(record.maintenance_run_id)
        if not args.dry_run:
            plan, items = _apply(plan, prepared, compute, pending)
    except (KeyboardInterrupt, SystemExit) as exc:
        interruption = exc
        items = tuple(pending)
        failures.append(RecordFailure("interrupted", type(exc).__name__, str(exc)))
    except ValueError:
        raise
    except Exception as exc:
        failures.append(RecordFailure("maintenance", type(exc).__name__, str(exc)))
        if not args.dry_run:
            items = _failed_outcomes(plan, exc)
    finally:
        try:
            failures.extend(compute.stop())
        finally:
            locks.close()
    record = MaintenanceRecord.from_plan(
        plan,
        environment=args.environment,
        dry_run=args.dry_run or interruption is not None,
        zones=prepared.zones,
        source_ids=prepared.source_ids or (),
        started_at=now,
        ended_at=now + timedelta(seconds=perf_counter() - started),
        items=None if interruption is not None else items,
    )
    record = replace(record, lock=lock.name, failures=tuple(failures))
    if interruption is not None:
        record = record if args.dry_run else record.with_interrupted_outcomes(plan, tuple(pending))
        _persist_partial(prepared.store, record)
        raise interruption
    return plan, record


def _locked_items(zones: frozenset[str], refused: set[str]) -> tuple[PlannedItem, ...]:
    # Refused sources are never inspected, so evidence identifies the source and zone.
    return tuple(
        PlannedItem(
            zone,
            source_id,
            "delete_file" if zone == "metadata" else "delete_prefix",
            {"source_id": source_id},
            skipped_reason=LOCKED_SOURCE_REASON,
        )
        for zone in sorted(zones & SOURCE_SCOPED_ZONES)
        for source_id in sorted(refused)
    )


def _selected_sources(prepared: _PreparedMaintenance) -> frozenset[str]:
    return prepared.source_ids or frozenset(
        source.source_id for source in prepared.registry.list_sources(enabled_only=False)
    )


def _collect_inventory(
    prepared: _PreparedMaintenance,
    now: datetime,
    session: SparkSession | None,
    refused: set[str],
) -> MaintenanceInventory:
    def collect(zones: frozenset[str], source_ids: frozenset[str] | None) -> MaintenanceInventory:
        return collect_inventory(
            prepared.registry,
            prepared.config,
            prepared.resolved_paths,
            prepared.policy,
            now,
            zones=zones,
            source_ids=source_ids,
            session=session,
        )

    if not refused:
        return collect(prepared.zones, prepared.source_ids)
    allowed = _selected_sources(prepared) - refused
    scoped = (
        collect(prepared.zones & SOURCE_SCOPED_ZONES, allowed)
        if allowed
        else MaintenanceInventory()
    )
    shared_zones = prepared.zones - SOURCE_SCOPED_ZONES
    shared = collect(shared_zones, prepared.source_ids) if shared_zones else MaintenanceInventory()
    return replace(shared, metadata=scoped.metadata, raw=scoped.raw)


def _plan_inventory(
    inventory: MaintenanceInventory,
    prepared: _PreparedMaintenance,
    now: datetime,
    refused: set[str],
) -> RetentionPlan:
    def plan(zones: frozenset[str], source_ids: frozenset[str] | None) -> RetentionPlan:
        return plan_retention(inventory, prepared.policy, now, zones=zones, source_ids=source_ids)

    if not refused:
        return plan(prepared.zones, prepared.source_ids)

    scoped = plan(prepared.zones & SOURCE_SCOPED_ZONES, _selected_sources(prepared) - refused)
    shared = plan(prepared.zones - SOURCE_SCOPED_ZONES, prepared.source_ids)
    return replace(
        shared,
        items=(*shared.items, *scoped.items),
        protected=(*shared.protected, *scoped.protected),
    )


def _warn_unlocked(record: MaintenanceRecord) -> bool:
    return record.lock == "none" and bool(SOURCE_SCOPED_ZONES.intersection(record.zones))


def _persist_partial(store: MaintenanceRecordStore, record: MaintenanceRecord) -> None:
    # No completed summary is published after a process interruption.
    try:
        store.persist(record)
    except OSError as exc:
        print(
            f"janus maintain: could not persist partial maintenance record: {exc}", file=sys.stderr
        )


def _apply(
    plan: RetentionPlan,
    prepared: _PreparedMaintenance,
    compute: _Compute,
    pending: list[ItemOutcome],
) -> tuple[RetentionPlan, tuple[ItemOutcome, ...]]:
    if plan.is_empty:
        return plan, tuple(ItemOutcome.from_planned_item(item) for item in plan.items)
    try:
        return plan, execute_retention(
            plan,
            policy=prepared.policy,
            session=compute.session,
            catalog_name=derive_pyiceberg_catalog_name(prepared.config),
            outcomes=pending,
        )
    except Exception as exc:
        # A dispatcher failure still needs evidence for every planned item.
        # Unknown partial results remain unknown; do not invent removed counts.
        return plan, _failed_outcomes(plan, exc)


def _failed_outcomes(plan: RetentionPlan, failure: Exception) -> tuple[ItemOutcome, ...]:
    return tuple(
        ItemOutcome.from_planned_item(item)
        if item.skipped_reason is not None
        else ItemOutcome(
            zone=item.zone,
            target=item.target,
            action=item.action,
            status="failed",
            detail=item.detail,
            failure_type=type(failure).__name__,
            failure_message=str(failure),
        )
        for item in plan.items
    )


def _render_text(plan: RetentionPlan, record: MaintenanceRecord) -> str:
    mode = "DRY RUN (nothing will be deleted)" if record.dry_run else "APPLY"
    lines = [
        f"janus maintain — {mode}",
        f"environment: {record.environment}    policy: sha256:{record.policy_digest}    "
        f"lock: {record.lock}",
        f"zones: {', '.join(record.zones)}",
    ]
    if _warn_unlocked(record):
        lines.append(f"warning: {LOCK_WARNING}")
    if record.dry_run and any(item.action == "remove_orphan_files" for item in record.items):
        lines.append(f"⚠ {ORPHAN_WARNING}")
    lines.append("")
    for zone in record.zones:
        items = [item for item in record.items if item.zone == zone]
        if not items:
            verb = "would be" if record.dry_run else "was"
            lines.append(f"nothing {verb} deleted in: {zone} (no retention candidates)")
            continue
        lines.append(zone)
        for item in items:
            lines.append(f"  {item.target}")
            if "selected_source_ids" in item.detail:
                selected = ", ".join(json.loads(item.detail["selected_source_ids"]))
                writers = ", ".join(json.loads(item.detail["source_ids"]))
                lines.append(
                    f"    included via shared_with: selected {selected}; writers {writers}"
                )
            if item.status == "skipped":
                lines.append(f"    skipped: {item.detail['skipped_reason']}")
                continue
            arguments = "  ".join(
                f"{key}={value}"
                for key, value in sorted(item.detail.items())
                if key not in {"snapshot_ids", "source_ids", "selected_source_ids"}
            )
            caution = "⚠ " if item.action == "remove_orphan_files" else ""
            lines.append(f"    {caution}{item.action}  {arguments}".rstrip())
            if item.status == "failed":
                lines.append(f"      failed: {item.failure_type}: {item.failure_message}")
            elif record.dry_run and item.expired_snapshot_ids:
                ids = ", ".join(str(value) for value in item.expired_snapshot_ids)
                lines.append(
                    f"      would expire {len(item.expired_snapshot_ids)} snapshots: {ids}"
                )
            else:
                lines.append(
                    f"      {item.status}: count={item.removed_count} bytes={item.removed_bytes}"
                )
    if plan.protected:
        lines.extend(("", "protected (not candidates)"))
        lines.extend(
            f"  {item['zone']}  {item['target']}  {', '.join(item['reasons'])}"
            for item in record.to_dict()["protected"]
        )
    for failure in record.failures:
        lines.append(f"failed ({failure.stage}): {failure.failure_type}: {failure.failure_message}")
    return "\n".join(lines)
