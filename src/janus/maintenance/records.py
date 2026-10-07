"""Versioned retention evidence and its atomic store under the shared metadata root."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from janus.lineage.persistence import write_json_atomic
from janus.maintenance.planning import PlannedItem, ProtectedItem, RetentionPlan
from janus.strategies.http.errors import RESPONSE_BODY_EXCERPT_LIMIT
from janus.utils.logging import (
    REDACTED_VALUE,
    SENSITIVE_FIELD_MARKERS,
    StructuredLogger,
    sanitize_log_payload,
)
from janus.utils.storage import StorageLayout

MAINTENANCE_RECORD_SCHEMA_VERSION = 1
MAINTENANCE_DIRECTORY = "maintenance"
MAINTENANCE_ITEM_PLANNED = "maintenance_item_planned"
MAINTENANCE_ITEM_APPLIED = "maintenance_item_applied"
MAINTENANCE_ITEM_SKIPPED = "maintenance_item_skipped"
MAINTENANCE_ITEM_FAILED = "maintenance_item_failed"
_ITEM_EVENTS = {
    "planned": MAINTENANCE_ITEM_PLANNED,
    "applied": MAINTENANCE_ITEM_APPLIED,
    "skipped": MAINTENANCE_ITEM_SKIPPED,
    "failed": MAINTENANCE_ITEM_FAILED,
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PLAN_DIGEST = re.compile(r"[0-9a-f]{64}")
_ID_DIGEST_LENGTH = 8
_MAX_ID_LENGTH = 96
_URI = re.compile(r"\b(?:jdbc:|[a-z][a-z0-9+.-]*://)[^\s\"'<>]+", re.IGNORECASE)
_SENSITIVE_KEY = "|".join(
    re.escape(marker).replace(r"\-", "[-_]") for marker in SENSITIVE_FIELD_MARKERS
)
_ASSIGNMENT = re.compile(
    rf"(?<![\w-])([\w-]*(?:{_SENSITIVE_KEY})[\w-]*)(\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|\{[^}]*\}|[^\s,;]+)",
    re.IGNORECASE,
)


def validate_maintenance_run_id(maintenance_run_id: str) -> str:
    """Refuse unsafe path segments before an id is interpolated into a path."""
    if (
        not _SAFE_ID.fullmatch(maintenance_run_id)
        or ".." in maintenance_run_id
        or len(maintenance_run_id) > _MAX_ID_LENGTH
    ):
        raise ValueError(
            f"maintenance_run_id {maintenance_run_id!r} is not usable as a path component: "
            f"use at most {_MAX_ID_LENGTH} letters, digits, '.', '-' or '_', "
            "start with a letter or digit, "
            "and do not include '..'"
        )
    return maintenance_run_id


def default_maintenance_run_id(plan_digest: str, started_at: datetime) -> str:
    """Derive a reproducible UTC-second identity with a plan-derived hex suffix."""
    _require_aware("started_at", started_at)
    if not _PLAN_DIGEST.fullmatch(plan_digest):
        raise ValueError("plan_digest must be a lowercase SHA-256 hex digest")
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return validate_maintenance_run_id(f"maintenance-{timestamp}-{plan_digest[:_ID_DIGEST_LENGTH]}")


def _redact_assignment(match: re.Match[str]) -> str:
    key, separator, value = match.groups()
    return f"{key}{separator}{sanitize_log_payload(value, field_name=key)}"


def _safe_text(value: str) -> str:
    # Scrub quoted/braced JDBC properties before removing the URI: whitespace in
    # a password must not leave its suffix behind when the URI match ends there.
    without_credentials = _ASSIGNMENT.sub(_redact_assignment, value)
    redacted = sanitize_log_payload(_URI.sub(REDACTED_VALUE, without_credentials))
    assert isinstance(redacted, str)
    return redacted


def _bounded_failure_message(value: str) -> str:
    # Redact before truncating: a boundary through a password must never expose a prefix.
    redacted = _safe_text(" ".join(value.split()))
    if len(redacted) <= RESPONSE_BODY_EXCERPT_LIMIT:
        return redacted
    return redacted[: RESPONSE_BODY_EXCERPT_LIMIT - 1] + "…"


def _require_aware(name: str, value: datetime) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_nonnegative(name: str, value: int | float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


def _planned_detail(item: PlannedItem) -> dict[str, str]:
    detail = dict(item.detail)
    if item.skipped_reason is not None:
        detail["skipped_reason"] = item.skipped_reason
    return detail


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    """What happened to one planned item, including failures and refusals."""

    zone: str
    target: str
    action: str
    status: str
    detail: Mapping[str, str]
    removed_count: int | None = None
    removed_bytes: int | None = None
    expired_snapshot_ids: tuple[int, ...] = ()
    failure_type: str | None = None
    failure_message: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.status not in _ITEM_EVENTS:
            raise ValueError(f"unsupported maintenance item status: {self.status!r}")
        has_failure = self.failure_type is not None and self.failure_message is not None
        if (self.status == "failed") != has_failure or (
            self.status != "failed"
            and (self.failure_type is not None or self.failure_message is not None)
        ):
            raise ValueError("only failed items must carry failure_type and failure_message")
        for name in ("removed_count", "removed_bytes", "duration_seconds"):
            _require_nonnegative(name, getattr(self, name))
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))
        if self.failure_message is not None:
            object.__setattr__(
                self, "failure_message", _bounded_failure_message(self.failure_message)
            )

    @classmethod
    def from_planned_item(cls, item: PlannedItem) -> Self:
        """Capture a dry-run estimate, or the planner's reason for skipping an item."""
        if item.skipped_reason is not None:
            return cls(item.zone, item.target, item.action, "skipped", _planned_detail(item))
        snapshot_ids = tuple(json.loads(item.detail.get("snapshot_ids", "[]")))
        count = None
        if item.action == "delete_file":
            count = 1
        elif item.action == "expire_snapshots" and "snapshot_ids" in item.detail:
            count = len(snapshot_ids)
        elif item.action == "delete_partition" and "row_count" in item.detail:
            count = int(item.detail["row_count"])
        return cls(
            zone=item.zone,
            target=item.target,
            action=item.action,
            status="planned",
            detail=_planned_detail(item),
            removed_count=count,
            removed_bytes=item.estimated_bytes,
            expired_snapshot_ids=snapshot_ids,
        )

    @classmethod
    def pending_apply(cls, item: PlannedItem) -> Self:
        """Pending execution has unknown measurements, rather than dry-run estimates."""
        if item.skipped_reason is not None:
            return cls.from_planned_item(item)
        return cls(item.zone, item.target, item.action, "planned", item.detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "target": self.target,
            "action": self.action,
            "status": self.status,
            "detail": dict(self.detail),
            "removed_count": self.removed_count,
            "removed_bytes": self.removed_bytes,
            "expired_snapshot_ids": list(self.expired_snapshot_ids),
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "duration_seconds": self.duration_seconds,
        }

    def log(self, logger: StructuredLogger, *, maintenance_run_id: str) -> None:
        """Emit one status event without arguments, payloads, credentials, or URLs."""
        fields = {
            "zone": _safe_text(self.zone),
            "target": _safe_text(self.target),
            "action": _safe_text(self.action),
            "maintenance_run_id": validate_maintenance_run_id(maintenance_run_id),
        }
        if self.status == "failed":
            assert self.failure_message is not None
            fields["failure_message"] = self.failure_message
            logger.error(_ITEM_EVENTS[self.status], **fields)
        else:
            logger.info(_ITEM_EVENTS[self.status], **fields)


@dataclass(frozen=True, slots=True)
class ZoneSummary:
    zone: str
    items_planned: int
    items_applied: int
    items_skipped: int
    items_failed: int
    removed_count: int
    removed_bytes: int

    @classmethod
    def from_items(cls, zone: str, items: tuple[ItemOutcome, ...]) -> Self:
        """Count every planned item; totals include any measured partial failure."""
        selected = tuple(item for item in items if item.zone == zone)
        return cls(
            zone=zone,
            items_planned=len(selected),
            items_applied=sum(item.status == "applied" for item in selected),
            items_skipped=sum(item.status == "skipped" for item in selected),
            items_failed=sum(item.status == "failed" for item in selected),
            removed_count=sum(item.removed_count or 0 for item in selected),
            removed_bytes=sum(item.removed_bytes or 0 for item in selected),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "items_planned": self.items_planned,
            "items_applied": self.items_applied,
            "items_skipped": self.items_skipped,
            "items_failed": self.items_failed,
            "removed_count": self.removed_count,
            "removed_bytes": self.removed_bytes,
        }


@dataclass(frozen=True, slots=True)
class RecordFailure:
    """A command or session failure independent of any retention item."""

    stage: str
    failure_type: str
    failure_message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_message", _bounded_failure_message(self.failure_message))

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True, slots=True)
class MaintenanceRecord:
    maintenance_run_id: str
    schema_version: int
    environment: str
    dry_run: bool
    zones: tuple[str, ...]
    source_ids: tuple[str, ...]
    policy_digest: str
    plan_digest: str
    lock: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    zone_summaries: tuple[ZoneSummary, ...]
    items: tuple[ItemOutcome, ...]
    failures: tuple[RecordFailure, ...] = ()
    protected: tuple[ProtectedItem, ...] = ()

    def __post_init__(self) -> None:
        validate_maintenance_run_id(self.maintenance_run_id)
        if self.schema_version != MAINTENANCE_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported maintenance record schema_version: {self.schema_version}"
            )
        _require_aware("started_at", self.started_at)
        _require_aware("ended_at", self.ended_at)
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        _require_nonnegative("duration_seconds", self.duration_seconds)

    @classmethod
    def from_plan(
        cls,
        plan: RetentionPlan,
        *,
        environment: str,
        dry_run: bool,
        zones: Iterable[str],
        started_at: datetime,
        ended_at: datetime,
        source_ids: Iterable[str] = (),
        items: tuple[ItemOutcome, ...] | None = None,
    ) -> Self:
        """Finalize complete evidence; applied items must be supplied in plan order."""
        _require_aware("started_at", started_at)
        _require_aware("ended_at", ended_at)
        started_at, ended_at = started_at.astimezone(UTC), ended_at.astimezone(UTC)
        if items is None:
            if not dry_run and plan.items:
                raise ValueError("an applied record requires an outcome for every planned item")
            items = tuple(ItemOutcome.from_planned_item(item) for item in plan.items)
        _validate_outcomes(plan, items, dry_run=dry_run)
        selected_zones = tuple(sorted(set(zones)))
        if any(item.zone not in selected_zones for item in items):
            raise ValueError("every item zone must be included in the record's zones")
        return cls(
            maintenance_run_id=default_maintenance_run_id(plan.digest, started_at),
            schema_version=MAINTENANCE_RECORD_SCHEMA_VERSION,
            environment=environment,
            dry_run=dry_run,
            zones=selected_zones,
            source_ids=tuple(sorted(set(source_ids))),
            policy_digest=plan.policy_digest,
            plan_digest=plan.digest,
            lock="none",
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=(ended_at - started_at).total_seconds(),
            zone_summaries=tuple(ZoneSummary.from_items(zone, items) for zone in selected_zones),
            items=items,
            protected=plan.protected,
        )

    def with_interrupted_outcomes(
        self, plan: RetentionPlan, items: tuple[ItemOutcome, ...]
    ) -> Self:
        """Keep partial apply evidence, including every still-planned operation."""
        if self.plan_digest != plan.digest or self.policy_digest != plan.policy_digest:
            raise ValueError("interrupted outcomes must belong to this record's plan")
        _validate_outcomes(plan, items, dry_run=False, interrupted=True)
        return replace(
            self,
            dry_run=False,
            items=items,
            zone_summaries=tuple(ZoneSummary.from_items(zone, items) for zone in self.zones),
        )

    @property
    def has_failures(self) -> bool:
        return bool(self.failures) or any(item.status == "failed" for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        protected: dict[tuple[str, str], set[str]] = {}
        for item in self.protected:
            protected.setdefault((item.zone, item.target), set()).add(item.reason)
        return {
            "maintenance_run_id": self.maintenance_run_id,
            "schema_version": self.schema_version,
            "environment": self.environment,
            "dry_run": self.dry_run,
            "zones": list(self.zones),
            "source_ids": list(self.source_ids),
            "policy_digest": self.policy_digest,
            "plan_digest": self.plan_digest,
            "lock": self.lock,
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "ended_at": self.ended_at.astimezone(UTC).isoformat(),
            "duration_seconds": self.duration_seconds,
            "zone_summaries": [summary.to_dict() for summary in self.zone_summaries],
            "items": [item.to_dict() for item in self.items],
            "failures": [failure.to_dict() for failure in self.failures],
            "protected": [
                {"zone": zone, "target": target, "reasons": sorted(reasons)}
                for (zone, target), reasons in sorted(protected.items())
            ],
        }


def _validate_outcomes(
    plan: RetentionPlan,
    items: tuple[ItemOutcome, ...],
    *,
    dry_run: bool,
    interrupted: bool = False,
) -> None:
    if len(items) != len(plan.items):
        raise ValueError("the record requires exactly one outcome per planned item")
    allowed = {"planned", "skipped"} if dry_run else {"applied", "skipped", "failed"}
    if interrupted:
        allowed.add("planned")
    for planned, outcome in zip(plan.items, items, strict=True):
        arguments = _planned_detail(planned)
        # Execution measurements may extend detail, but every declared argument
        # must remain identical so the plan digest continues to identify the action.
        detail_matches = (
            dict(outcome.detail) == arguments
            if dry_run
            else arguments.items() <= outcome.detail.items()
        )
        if (planned.zone, planned.target, planned.action) != (
            outcome.zone,
            outcome.target,
            outcome.action,
        ) or not detail_matches:
            raise ValueError("outcomes must match every planned item and its arguments in order")
        if outcome.status not in allowed:
            raise ValueError(f"status {outcome.status!r} is invalid for dry_run={dry_run}")


@dataclass(frozen=True, slots=True)
class MaintenanceRecordStore:
    storage_layout: StorageLayout

    def record_path(self, maintenance_run_id: str) -> Path:
        """Resolve <metadata>/maintenance/<id>.json without escaping the shared root."""
        safe_id = validate_maintenance_run_id(maintenance_run_id)
        metadata_root = self.storage_layout.metadata_dir.resolve()
        candidate = (metadata_root / MAINTENANCE_DIRECTORY / f"{safe_id}.json").resolve()
        if not candidate.is_relative_to(metadata_root):
            raise ValueError(
                f"Maintenance record path {candidate} escapes metadata root {metadata_root}"
            )
        return candidate

    def persist(self, record: MaintenanceRecord) -> Path:
        """Publish dry-run and applied evidence through the established atomic writer."""
        return write_json_atomic(self.record_path(record.maintenance_run_id), record.to_dict())
