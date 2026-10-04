"""Pure retention decisions over an injected inventory and clock.

Progress references are compared in sanitized run-ID space using the extraction
helper. Normalization is lossy: collisions protect every matching run, deliberately
erring toward preservation. Collectors read and executors act; this module does neither.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from janus.maintenance.inventory import (
    BronzeTableInventory,
    MaintenanceInventory,
    MetadataZoneInventory,
    RawRunPrefixEntry,
    RunArtifactEntry,
)
from janus.maintenance.settings import MaintenancePolicy
from janus.strategies.common import raw_run_path_segment


@dataclass(frozen=True, slots=True)
class PlannedItem:
    zone: str
    target: str
    action: str
    detail: Mapping[str, str]
    estimated_bytes: int | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProtectedItem:
    zone: str
    target: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    items: tuple[PlannedItem, ...]
    protected: tuple[ProtectedItem, ...]
    now: datetime
    policy_digest: str

    @property
    def digest(self) -> str:
        """SHA-256 of sorted actions and arguments, independent of inventory order."""
        candidates = sorted(
            (item.zone, item.target, item.action, sorted(item.detail.items()))
            for item in self.items
        )
        encoded = json.dumps(candidates, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def is_empty(self) -> bool:
        """Skipped items are evidence; only actionable items make a nonempty plan."""
        return not any(item.skipped_reason is None for item in self.items)


@dataclass(slots=True)
class _Decisions:
    items: list[PlannedItem] = field(default_factory=list)
    protected: list[ProtectedItem] = field(default_factory=list)

    def protect(self, zone: str, target: str, reason: str) -> None:
        self.protected.append(ProtectedItem(zone, target, reason))


def plan_retention(
    inventory: MaintenanceInventory,
    policy: MaintenancePolicy,
    now: datetime,
    *,
    zones: frozenset[str],
    source_ids: frozenset[str] | None = None,
) -> RetentionPlan:
    """Decide every candidate. Reads nothing, writes nothing, imports no engine."""
    if not _aware(now):
        raise ValueError("retention now must be timezone-aware")
    now = now.astimezone(UTC)
    deciders = {
        "bronze": _plan_bronze,
        "metadata": _plan_metadata,
        "lineage": _plan_lineage,
        "runs-table": _plan_runs_table,
        "raw": _plan_raw,
    }
    unknown = zones - deciders.keys()
    if unknown:
        raise ValueError(f"unsupported maintenance zones: {', '.join(sorted(unknown))}")
    decisions = _Decisions()
    for zone in sorted(zones):
        result = deciders[zone](inventory, policy, now, source_ids)
        decisions.items.extend(result.items)
        decisions.protected.extend(result.protected)
    return RetentionPlan(
        items=tuple(
            sorted(decisions.items, key=lambda item: (item.zone, item.target, item.action))
        ),
        protected=tuple(
            sorted(set(decisions.protected), key=lambda item: (item.zone, item.target, item.reason))
        ),
        now=now,
        policy_digest=policy.digest,
    )


def _aware(value: datetime | None) -> bool:
    return value is not None and value.utcoffset() is not None


def _selected(source_id: str | None, source_ids: frozenset[str] | None) -> bool:
    return source_ids is None or source_id in source_ids


def _plan_bronze(
    inventory: MaintenanceInventory,
    policy: MaintenancePolicy,
    now: datetime,
    source_ids: frozenset[str] | None,
) -> _Decisions:
    decisions = _Decisions()
    for table in inventory.bronze:
        if source_ids is not None and source_ids.isdisjoint(table.source_ids):
            continue
        if table.unavailable_reason is not None:
            decisions.items.append(
                PlannedItem(
                    "bronze",
                    table.table_identifier,
                    "expire_snapshots",
                    {},
                    skipped_reason=table.unavailable_reason,
                )
            )
            continue
        _bronze_table(table, policy, now, decisions)
    return decisions


def _bronze_table(
    table: BronzeTableInventory, policy: MaintenancePolicy, now: datetime, decisions: _Decisions
) -> None:
    effective = table.override if table.override is not None else policy.bronze
    cutoff = now - timedelta(days=effective.older_than_days)
    aged = sorted(
        (entry for entry in table.snapshots if _aware(entry.committed_at)),
        key=lambda entry: (entry.committed_at, entry.snapshot_id),
        reverse=True,
    )
    newest = {entry.snapshot_id for entry in aged[: effective.retain_last]}
    candidates = []
    for entry in table.snapshots:
        reason = None
        if entry.is_current:
            reason = "current_snapshot"
        elif not _aware(entry.committed_at):
            reason = "unaged"
        elif entry.snapshot_id in newest:
            reason = "retain_last"
        elif entry.committed_at >= cutoff:
            reason = "within_window"
        if reason is not None:
            decisions.protect("bronze", f"{table.table_identifier}#{entry.snapshot_id}", reason)
        else:
            candidates.append(entry.snapshot_id)
    if candidates:
        decisions.items.append(
            PlannedItem(
                "bronze",
                table.table_identifier,
                "expire_snapshots",
                {
                    "older_than": cutoff.isoformat(),
                    "retain_last": str(effective.retain_last),
                    "snapshot_ids": json.dumps(sorted(candidates)),
                },
            )
        )
    # Overrides replace the two retention fields, not profile-level optional procedures.
    if policy.bronze.remove_orphan_files:
        decisions.items.append(
            PlannedItem(
                "bronze",
                table.table_identifier,
                "remove_orphan_files",
                {
                    "orphan_older_than": (
                        now - timedelta(days=policy.bronze.orphan_older_than_days)
                    ).isoformat(),
                },
            )
        )
    if policy.bronze.compact_enabled:
        target = policy.bronze.compact_target_file_size_mb
        if target is None:
            raise ValueError("enabled bronze compaction requires target_file_size_mb")
        decisions.items.append(
            PlannedItem(
                "bronze",
                table.table_identifier,
                "rewrite_data_files",
                {
                    "target_file_size_bytes": str(target * 1024 * 1024),
                },
            )
        )


def _kept_runs(metadata: MetadataZoneInventory, keep_last: int) -> set[tuple[str, str]]:
    by_source: dict[str, dict[str, datetime]] = {}
    for entry in metadata.artifacts:
        if (
            entry.kind == "runs"
            and entry.source_id is not None
            and entry.run_id is not None
            and entry.read_error is None
            and entry.timestamp is not None
            and _aware(entry.timestamp)
        ):
            runs = by_source.setdefault(entry.source_id, {})
            runs[entry.run_id] = max(runs.get(entry.run_id, entry.timestamp), entry.timestamp)
    return {
        (source, run_id)
        for source, runs in by_source.items()
        for run_id in sorted(runs, key=lambda run_id: (runs[run_id], run_id), reverse=True)[
            :keep_last
        ]
    }


def _artifact_protection(
    entry: RunArtifactEntry, metadata: MetadataZoneInventory, kept: set[tuple[str, str]]
) -> str | None:
    if entry.path in metadata.protected_paths:
        return "state_file"
    if entry.read_error is not None:
        return "unreadable"
    if not _aware(entry.timestamp):
        return "unaged"
    if entry.source_id is not None and entry.run_id is not None:
        if (entry.source_id, entry.run_id) in kept:
            return "keep_last_runs"
        segment = metadata.live_raw_run_segments.get(entry.source_id)
        if segment is not None and raw_run_path_segment(entry.run_id) == segment:
            return "live_progress"
    return None


def _plan_metadata(
    inventory: MaintenanceInventory,
    policy: MaintenancePolicy,
    now: datetime,
    source_ids: frozenset[str] | None,
) -> _Decisions:
    decisions = _Decisions()
    metadata = inventory.metadata
    if metadata is None:
        return decisions
    kept = _kept_runs(metadata, policy.metadata.keep_last_runs)
    cutoff = now - timedelta(days=policy.metadata.older_than_days)
    for path in metadata.protected_paths:
        decisions.protect("metadata", str(path), "state_file")
    for entry in metadata.artifacts:
        if not _selected(entry.source_id, source_ids):
            # A shared summary cannot be attributed to the selected sources safely.
            if entry.source_id is None:
                decisions.protect("metadata", str(entry.path), "source_filter")
            continue
        reason = _artifact_protection(entry, metadata, kept)
        if reason is not None:
            decisions.protect("metadata", str(entry.path), reason)
        elif entry.timestamp is not None and entry.timestamp < cutoff:
            decisions.items.append(PlannedItem("metadata", str(entry.path), "delete_file", {}))
        else:
            decisions.protect("metadata", str(entry.path), "within_window")
    return decisions


def _plan_lineage(
    inventory: MaintenanceInventory,
    policy: MaintenancePolicy,
    now: datetime,
    _source_ids: frozenset[str] | None,
) -> _Decisions:
    # Shared day files are aged as a whole; EventFileEntry has no source attribution.
    decisions = _Decisions()
    if not inventory.lineage_events:
        return decisions
    latest = max(entry.day for entry in inventory.lineage_events)
    cutoff = now.date() - timedelta(days=policy.lineage_events.older_than_days)
    for entry in inventory.lineage_events:
        reason = None
        if entry.day == now.date():
            reason = "today"
        elif entry.day == latest:
            reason = "latest_file"
        elif entry.day >= cutoff:
            reason = "within_window"
        if reason is not None:
            decisions.protect("lineage", str(entry.path), reason)
        else:
            decisions.items.append(PlannedItem("lineage", str(entry.path), "delete_file", {}))
    return decisions


def _plan_runs_table(
    inventory: MaintenanceInventory,
    policy: MaintenancePolicy,
    now: datetime,
    _source_ids: frozenset[str] | None,
) -> _Decisions:
    decisions = _Decisions()
    cutoff_day = now.date() - timedelta(days=policy.runs_table.older_than_days)
    cutoff = datetime.combine(cutoff_day, datetime.min.time(), tzinfo=UTC).isoformat()
    for entry in inventory.runs_table:
        day = entry.emitted_at_day.isoformat()
        if entry.emitted_at_day >= cutoff_day:
            decisions.protect("runs-table", day, "within_window")
            continue
        detail = {"older_than": cutoff}
        if entry.row_count is not None:
            detail["row_count"] = str(entry.row_count)
        decisions.items.append(PlannedItem("runs-table", day, "delete_partition", detail))
    if decisions.items:
        # Logical target: the executor resolves the configured identifier/catalog (D-12).
        # Expire only after a partition delete; with no candidates a repeated plan is empty.
        decisions.items.append(
            PlannedItem(
                "runs-table",
                "metadata.runs",
                "expire_snapshots",
                {
                    "older_than": cutoff,
                    "retain_last": str(policy.bronze.retain_last),
                },
            )
        )
    return decisions


def _raw_source(
    entries: list[RawRunPrefixEntry],
    policy: MaintenancePolicy,
    now: datetime,
    live: Mapping[str, str | None],
    decisions: _Decisions,
) -> None:
    source = entries[0].source_id
    if source in live and live[source] is None:
        decisions.items.append(
            PlannedItem("raw", source, "delete_prefix", {}, skipped_reason="legacy_progress_prefix")
        )
        for entry in entries:
            decisions.protect("raw", str(entry.path), "legacy_progress_prefix")
        return
    successful = sorted(
        (entry for entry in entries if entry.run_succeeded is True),
        key=lambda entry: (entry.ingestion_date, entry.run_segment, str(entry.path)),
        reverse=True,
    )
    kept = {entry.path for entry in successful[: policy.raw.keep_last_runs]}
    cutoff = now.date() - timedelta(days=policy.raw.older_than_days)
    for entry in entries:
        reason = None
        if entry.run_succeeded is None:
            reason = "unknown_run_status"
        elif entry.run_segment == live.get(source):
            reason = "live_progress"
        elif entry.path in kept:
            reason = "keep_last_runs"
        elif entry.ingestion_date >= cutoff:
            reason = "within_window"
        if reason is not None:
            decisions.protect("raw", str(entry.path), reason)
        else:
            decisions.items.append(PlannedItem("raw", str(entry.path), "delete_prefix", {}))


def _plan_raw(
    inventory: MaintenanceInventory,
    policy: MaintenancePolicy,
    now: datetime,
    source_ids: frozenset[str] | None,
) -> _Decisions:
    decisions = _Decisions()
    if not policy.raw.enabled:
        return decisions
    by_source: dict[str, list[RawRunPrefixEntry]] = {}
    for entry in inventory.raw:
        if _selected(entry.source_id, source_ids):
            by_source.setdefault(entry.source_id, []).append(entry)
    live = inventory.metadata.live_raw_run_segments if inventory.metadata is not None else {}
    for entries in by_source.values():
        _raw_source(entries, policy, now, live, decisions)
    return decisions
