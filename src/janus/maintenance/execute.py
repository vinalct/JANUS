"""Execute declared retention without deciding policy or opening compute.

Expiration precedes compaction in plan order. Compaction creates a snapshot, so an
enabled rewrite can leave retain_last + 1 snapshots; that ordering is deliberate.
rewrite_manifests is reserved by the deletion boundary and is not implemented here.
Timeout cancellation is best effort: a failed item may still be running in Spark.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Thread
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from janus.maintenance.errors import (
    MaintenanceExecutionUnavailable,
    MaintenanceInvariantError,
    MaintenanceItemTimeout,
)
from janus.maintenance.planning import PlannedItem, RetentionPlan
from janus.maintenance.records import ItemOutcome
from janus.maintenance.settings import MaintenancePolicy
from janus.writers.identifiers import quote_identifier

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_EXPIRATION_COUNTS = (
    "deleted_data_files_count",
    "deleted_position_delete_files_count",
    "deleted_equality_delete_files_count",
    "deleted_manifest_files_count",
    "deleted_manifest_lists_count",
    "deleted_statistics_files_count",
)
_COMPACTION_COUNTS = (
    "rewritten_data_files_count",
    "added_data_files_count",
    "rewritten_bytes_count",
    "failed_data_files_count",
    "removed_delete_files_count",
)


@dataclass(slots=True)
class _Measurements:
    detail: dict[str, str]
    removed_count: int | None = None
    expired_snapshot_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class _WorkerResult:
    failures: list[Exception | KeyboardInterrupt | SystemExit] = field(default_factory=list)


def execute_retention(
    plan: RetentionPlan,
    *,
    policy: MaintenancePolicy,
    session: SparkSession | None,
    catalog_name: str,
    clock: Callable[[], float] = perf_counter,
    outcomes: list[ItemOutcome] | None = None,
) -> tuple[ItemOutcome, ...]:
    """Execute in plan order, updating an optional interruption-evidence ledger."""
    if outcomes is None:
        outcomes = [ItemOutcome.pending_apply(item) for item in plan.items]
    protected_paths = frozenset(
        Path(item.target) for item in plan.protected if item.zone in {"metadata", "lineage"}
    )
    for index, item in enumerate(plan.items):
        outcomes[index] = execute_item(
            session,
            item,
            catalog_name=catalog_name,
            timeout_seconds=policy.item_timeout_seconds,
            clock=clock,
            protected_paths=protected_paths,
        )
    return tuple(outcomes)


def execute_item(
    session: SparkSession | None,
    item: PlannedItem,
    *,
    catalog_name: str,
    timeout_seconds: float,
    clock: Callable[[], float] = perf_counter,
    protected_paths: frozenset[Path] = frozenset(),
) -> ItemOutcome:
    """Record one failure and continue; process interruptions propagate unchanged."""
    if item.skipped_reason is not None:
        return ItemOutcome.from_planned_item(item)
    started = clock()
    try:
        if item.zone in {"metadata", "lineage"}:
            return execute_metadata_item(item, protected_paths=protected_paths, clock=clock)
        if item.zone != "bronze":
            raise MaintenanceExecutionUnavailable(
                f"No maintenance executor is available for {item.zone}: {item.action}"
            )
        if session is None:
            raise MaintenanceExecutionUnavailable("bronze execution requires a Spark session")
        return execute_bronze_item(
            session, item, catalog_name=catalog_name, timeout_seconds=timeout_seconds, clock=clock
        )
    except Exception as exc:
        outcome = _outcome(item, _Measurements(dict(item.detail)), exc, clock() - started)
        if isinstance(exc, MaintenanceExecutionUnavailable):
            return replace(outcome, removed_count=0, removed_bytes=0)
        return outcome


def execute_metadata_item(
    item: PlannedItem,
    *,
    protected_paths: frozenset[Path] = frozenset(),
    clock: Callable[[], float] = perf_counter,
) -> ItemOutcome:
    """Remove one history or event file, refusing protected paths before any I/O."""
    path = Path(item.target)
    state_file = path.name == "extraction_progress.json" or (
        path.name == "current.json" and path.parent.name in {"checkpoints", "dead_letters"}
    )
    if state_file or path in protected_paths:
        raise MaintenanceInvariantError(f"Refusing to delete protected metadata path: {path}")
    if item.zone not in {"metadata", "lineage"} or item.action != "delete_file":
        raise MaintenanceExecutionUnavailable(f"Unsupported metadata action: {item.action}")
    if item.skipped_reason is not None:
        return ItemOutcome.from_planned_item(item)
    started = clock()
    detail = dict(item.detail)
    status = "applied"
    removed_count = removed_bytes = 0
    failure: OSError | None = None
    try:
        size = path.stat().st_size
        path.unlink()
        removed_count, removed_bytes = 1, size
        # Leave empty directories: removing them races the next run's mkdir.
    except FileNotFoundError:
        status = "skipped"
        detail["skipped_reason"] = "already_absent"
    except OSError as exc:
        status, failure = "failed", exc
    return ItemOutcome(
        zone=item.zone,
        target=item.target,
        action=item.action,
        status=status,
        detail=detail,
        removed_count=removed_count,
        removed_bytes=removed_bytes,
        failure_type=type(failure).__name__ if failure is not None else None,
        failure_message=str(failure) if failure is not None else None,
        duration_seconds=clock() - started,
    )


def execute_bronze_item(
    session: SparkSession,
    item: PlannedItem,
    *,
    catalog_name: str,
    timeout_seconds: float,
    clock: Callable[[], float] = perf_counter,
) -> ItemOutcome:
    """Bound the CALL and its verification under one uniquely assigned job group."""
    started = clock()
    measured = _Measurements(dict(item.detail))
    result = _WorkerResult()
    group_id = f"janus-maintenance-{uuid4().hex}"
    worker = Thread(
        target=_run_worker,
        args=(session, item, catalog_name, group_id, measured, result),
        name=group_id,
        daemon=True,
    )
    failure: Exception | None = None
    try:
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            requested = _request_cancellation(session, group_id)
            measured.detail = {**measured.detail, "cancellation_requested": json.dumps(requested)}
            failure = MaintenanceItemTimeout(timeout_seconds, cancellation_requested=requested)
        elif result.failures:
            raise result.failures[0]
    except (KeyboardInterrupt, SystemExit):
        _request_cancellation(session, group_id)
        raise
    except Exception as exc:
        failure = exc
    return _outcome(item, measured, failure, clock() - started)


def _run_worker(
    session: SparkSession,
    item: PlannedItem,
    catalog_name: str,
    group_id: str,
    measured: _Measurements,
    result: _WorkerResult,
) -> None:
    try:
        session.sparkContext.setJobGroup(group_id, group_id, interruptOnCancel=True)
        actions = {
            "expire_snapshots": _expire,
            "remove_orphan_files": _orphans,
            "rewrite_data_files": _compact,
        }
        action = actions.get(item.action)
        if action is None:
            raise MaintenanceExecutionUnavailable(f"Unsupported bronze action: {item.action}")
        action(session, item, catalog_name, measured)
    except (KeyboardInterrupt, SystemExit) as exc:
        result.failures.append(exc)
    except Exception as exc:
        result.failures.append(exc)
    finally:
        # Spark local properties belong to the worker/JVM thread that set them.
        # Cleanup must not replace a completed deletion or a process interruption.
        try:
            for key in (
                "spark.jobGroup.id",
                "spark.job.description",
                "spark.job.interruptOnCancel",
            ):
                session.sparkContext.setLocalProperty(key, None)
        except (KeyboardInterrupt, SystemExit) as exc:
            result.failures.append(exc)
        except Exception as exc:
            measured.detail = {**measured.detail, "job_group_cleanup_failure": type(exc).__name__}


def _request_cancellation(session: SparkSession, group_id: str) -> bool:
    try:
        session.sparkContext.cancelJobGroup(group_id)
    except Exception:
        return False
    return True


def _outcome(
    item: PlannedItem, measured: _Measurements, failure: Exception | None, duration: float
) -> ItemOutcome:
    return ItemOutcome(
        zone=item.zone,
        target=item.target,
        action=item.action,
        status="failed" if failure is not None else "applied",
        detail=measured.detail,
        removed_count=measured.removed_count,
        expired_snapshot_ids=measured.expired_snapshot_ids,
        failure_type=type(failure).__name__ if failure is not None else None,
        failure_message=str(failure) if failure is not None else None,
        duration_seconds=duration,
    )


def _literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _table_argument(item: PlannedItem) -> str:
    return _literal(quote_identifier(item.target))


def _snapshot_ids(session: SparkSession, qualified_table: str) -> frozenset[int]:
    rows = session.sql(
        f"SELECT {quote_identifier('snapshot_id')} "
        f"FROM {quote_identifier(f'{qualified_table}.snapshots')}"
    ).collect()
    return frozenset(int(row[0]) for row in rows)


def _expire(
    session: SparkSession, item: PlannedItem, catalog_name: str, measured: _Measurements
) -> None:
    qualified = f"{catalog_name}.{item.target}"
    predicted = frozenset(json.loads(item.detail["snapshot_ids"]))
    older_than = item.detail["older_than"]
    retain_last = int(item.detail["retain_last"])
    before = _snapshot_ids(session, qualified)
    rows = session.sql(
        f"CALL {quote_identifier(catalog_name)}.system.expire_snapshots(\n"
        f"  table => {_table_argument(item)},\n"
        f"  older_than => TIMESTAMP {_literal(older_than)},\n"
        f"  retain_last => {retain_last}\n)"
    ).collect()
    counts = _capture_counts(rows, _EXPIRATION_COUNTS, measured)
    measured.removed_count = sum(counts.values()) if counts else None
    surviving = _snapshot_ids(session, qualified)
    expired = before - surviving
    measured.expired_snapshot_ids = tuple(sorted(expired))
    measured.detail = {
        **measured.detail,
        "surviving_snapshot_ids": json.dumps(sorted(surviving)),
        "predicted_but_retained_snapshot_ids": json.dumps(sorted(predicted - expired)),
        "unexpected_expired_snapshot_ids": json.dumps(sorted(expired - predicted)),
    }


def _orphans(
    session: SparkSession, item: PlannedItem, catalog_name: str, measured: _Measurements
) -> None:
    rows = session.sql(
        f"CALL {quote_identifier(catalog_name)}.system.remove_orphan_files(\n"
        f"  table => {_table_argument(item)},\n"
        f"  older_than => TIMESTAMP {_literal(item.detail['orphan_older_than'])}\n)"
    ).collect()
    measured.removed_count = len(rows)
    measured.detail = {**measured.detail, "orphan_files_count": str(len(rows))}


def _compact(
    session: SparkSession, item: PlannedItem, catalog_name: str, measured: _Measurements
) -> None:
    rows = session.sql(
        f"CALL {quote_identifier(catalog_name)}.system.rewrite_data_files(\n"
        f"  table => {_table_argument(item)},\n"
        "  options => map('target-file-size-bytes', "
        f"'{int(item.detail['target_file_size_bytes'])}')\n)"
    ).collect()
    counts = _capture_counts(rows, _COMPACTION_COUNTS, measured)
    rewritten = counts.get("rewritten_data_files_count")
    if rewritten is not None:
        measured.removed_count = rewritten + counts.get("removed_delete_files_count", 0)
    # Rewritten bytes describe I/O, not reclaimed disk space. Keep them in detail.


def _capture_counts(
    rows: Sequence[Any], expected: tuple[str, ...], measured: _Measurements
) -> dict[str, int]:
    if not rows:
        measured.detail = {**measured.detail, "raw_result": "[]"}
        return {}
    row = rows[0]
    fields = row if isinstance(row, Mapping) else row.asDict() if hasattr(row, "asDict") else {}
    positions = list(fields.values()) if isinstance(row, Mapping) else list(row)
    counts = {}
    for index, name in enumerate(expected):
        positional = None
        if index < len(positions) and (not fields or list(fields)[index] not in expected):
            positional = positions[index]
        value = fields.get(name, positional)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            counts[name] = value
    detail = {name: str(value) for name, value in counts.items()}
    if set(expected) - fields.keys():
        detail["raw_result"] = json.dumps(dict(fields) if fields else positions, default=str)
    measured.detail = {**measured.detail, **detail}
    return counts
