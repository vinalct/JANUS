"""Explicit retention maintenance: dry run by default, with evidence on every plan."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any

import yaml

from janus.cli.common import format_runtime_permission_error
from janus.maintenance.execute import execute_retention
from janus.maintenance.inventory import collect_inventory
from janus.maintenance.locking import NullMaintenanceLock
from janus.maintenance.planning import RetentionPlan, plan_retention
from janus.maintenance.records import (
    ItemOutcome,
    MaintenanceRecord,
    MaintenanceRecordStore,
    RecordFailure,
)
from janus.maintenance.settings import MaintenancePolicy, resolve_maintenance_settings
from janus.registry import SourceNotFoundError, SourceRegistry, load_registry
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.utils.environment import RuntimeLocation, load_environment_config, prepare_runtime
from janus.utils.logging import StructuredLogger, build_structured_logger
from janus.utils.storage import StorageLayout

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

ARGUMENT_ERROR = 2
OPERATIONAL_ERROR = 1
ZONES = ("bronze", "metadata", "lineage", "runs-table", "raw")
COMPUTE_ZONES = frozenset({"bronze", "runs-table"})
LOCK_WARNING = "maintain must not overlap a run of the same source (no lock is held)"

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


def maintain_command(args: argparse.Namespace) -> int:
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
        plan, record = _run(prepared, args, now, started)
    except ValueError as exc:
        return _refuse(str(exc))
    except Exception as exc:
        print(f"janus maintain: maintenance failed: {exc}", file=sys.stderr)
        return OPERATIONAL_ERROR

    if args.format == "json":
        if record.lock == "none":
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
) -> tuple[RetentionPlan, MaintenanceRecord]:
    compute = _Compute(prepared)
    lock = NullMaintenanceLock()
    acquired = []
    failures = []
    plan = RetentionPlan((), (), now, prepared.policy.digest)
    items = None
    try:
        if prepared.zones & {"metadata", "raw"}:
            source_ids = prepared.source_ids or frozenset(
                source.source_id for source in prepared.registry.list_sources(enabled_only=False)
            )
            for source_id in sorted(source_ids):
                if lock.acquire(source_id):
                    acquired.append(source_id)
        session = compute.get_session() if prepared.zones & COMPUTE_ZONES else None
        inventory = collect_inventory(
            prepared.registry,
            prepared.config,
            prepared.resolved_paths,
            prepared.policy,
            now,
            zones=prepared.zones,
            source_ids=prepared.source_ids,
            session=session,
        )
        plan = plan_retention(
            inventory,
            prepared.policy,
            now,
            zones=prepared.zones,
            source_ids=prepared.source_ids,
        )
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
            plan, items = _apply(plan, prepared, compute)
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
            for source_id in reversed(acquired):
                lock.release(source_id)
    record = MaintenanceRecord.from_plan(
        plan,
        environment=args.environment,
        dry_run=args.dry_run,
        zones=prepared.zones,
        source_ids=prepared.source_ids or (),
        started_at=now,
        ended_at=now + timedelta(seconds=perf_counter() - started),
        items=items,
    )
    return plan, replace(record, lock=lock.name, failures=tuple(failures))


def _apply(
    plan: RetentionPlan,
    prepared: _PreparedMaintenance,
    compute: _Compute,
) -> tuple[RetentionPlan, tuple[ItemOutcome, ...]]:
    if plan.is_empty:
        return plan, tuple(ItemOutcome.from_planned_item(item) for item in plan.items)
    try:
        return plan, execute_retention(
            plan,
            policy=prepared.policy,
            session=compute.session,
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
    if record.lock == "none":
        lines.append(f"warning: {LOCK_WARNING}")
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
            lines.append(f"    {item.action}  {arguments}".rstrip())
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
        lines.extend(f"  {item.zone}  {item.target}  {item.reason}" for item in plan.protected)
    for failure in record.failures:
        lines.append(f"failed ({failure.stage}): {failure.failure_type}: {failure.failure_message}")
    return "\n".join(lines)
