"""The explicitly declared ``maintenance`` block of an environment profile.

Validation raises on the first problem, unlike ``SourceConfig.from_mapping``, which
collects issues. A profile is one operator's file with a handful of keys and the CLI
prints one error; the source-model issue collector does not belong in this package.
Only the maintenance command resolves this block, so a retention typo cannot fail
an ingestion run. No engine or I/O is needed to resolve or fingerprint a policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from janus.maintenance.errors import MaintenanceProfileError
from janus.utils.environment import non_empty_text

MAINTENANCE_BLOCK_KEY = "maintenance"
SUPPORTED_MAINTENANCE_KEYS = frozenset(
    {"bronze", "metadata", "lineage_events", "runs_table", "raw", "item_timeout_seconds"}
)
SUPPORTED_BRONZE_KEYS = frozenset(
    {"retain_last", "older_than_days", "remove_orphan_files", "orphan_older_than_days", "compact"}
)
SUPPORTED_METADATA_KEYS = frozenset({"keep_last_runs", "older_than_days"})
SUPPORTED_AGE_KEYS = frozenset({"older_than_days"})
SUPPORTED_RAW_KEYS = frozenset({"enabled", "keep_last_runs", "older_than_days"})
SUPPORTED_COMPACT_KEYS = frozenset({"enabled", "target_file_size_mb"})


@dataclass(frozen=True, slots=True)
class BronzeRetentionPolicy:
    """Snapshot retention and optional orphan removal and compaction."""

    retain_last: int
    older_than_days: int
    remove_orphan_files: bool = False
    orphan_older_than_days: int = 3
    compact_enabled: bool = False
    compact_target_file_size_mb: int | None = None


@dataclass(frozen=True, slots=True)
class MetadataRetentionPolicy:
    """Run history retention, with the newest runs protected."""

    keep_last_runs: int
    older_than_days: int


@dataclass(frozen=True, slots=True)
class LineageEventsRetentionPolicy:
    """Age threshold for OpenLineage day files."""

    older_than_days: int


@dataclass(frozen=True, slots=True)
class RunsTableRetentionPolicy:
    """Age threshold for runs-table partitions."""

    older_than_days: int


@dataclass(frozen=True, slots=True)
class RawRetentionPolicy:
    """Opt-in raw retention; disabled unless explicitly enabled."""

    enabled: bool = False
    keep_last_runs: int = 0
    older_than_days: int = 0


@dataclass(frozen=True, slots=True)
class MaintenancePolicy:
    """One profile's resolved policy, including the per-item procedure timeout."""

    bronze: BronzeRetentionPolicy
    metadata: MetadataRetentionPolicy
    lineage_events: LineageEventsRetentionPolicy
    runs_table: RunsTableRetentionPolicy
    raw: RawRetentionPolicy
    item_timeout_seconds: float = 1800.0

    @property
    def digest(self) -> str:
        """A stable SHA-256 over resolved fields, independent of YAML key order."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_maintenance_settings(config: Mapping[str, Any]) -> MaintenancePolicy:
    """Resolve the declared policy, or refuse naming the first unusable key.

    Unlike OpenLineage, an absent block is a refusal rather than a disabled default:
    retention nobody declared is a guess about what may be deleted.
    """
    declared = _declared(config, MAINTENANCE_BLOCK_KEY)
    if declared is None:
        raise MaintenanceProfileError(
            "Environment config has no 'maintenance' block; "
            "'janus maintain' applies only a declared policy"
        )
    block = _mapping(declared, MAINTENANCE_BLOCK_KEY, SUPPORTED_MAINTENANCE_KEYS)
    bronze = _bronze_policy(_subblock(block, "bronze", SUPPORTED_BRONZE_KEYS))
    metadata_block = _subblock(block, "metadata", SUPPORTED_METADATA_KEYS)
    metadata = MetadataRetentionPolicy(
        keep_last_runs=_integer(
            metadata_block, "keep_last_runs", "maintenance.metadata", minimum=1
        ),
        older_than_days=_integer(metadata_block, "older_than_days", "maintenance.metadata"),
    )
    lineage_block = _subblock(block, "lineage_events", SUPPORTED_AGE_KEYS)
    lineage = LineageEventsRetentionPolicy(
        older_than_days=_integer(lineage_block, "older_than_days", "maintenance.lineage_events")
    )
    runs_block = _subblock(block, "runs_table", SUPPORTED_AGE_KEYS)
    runs = RunsTableRetentionPolicy(
        older_than_days=_integer(runs_block, "older_than_days", "maintenance.runs_table")
    )
    raw = _raw_policy(_subblock(block, "raw", SUPPORTED_RAW_KEYS))
    # FR-8: retained bronze snapshots need their raw input to remain rebuildable.
    # The raw builder cannot see bronze: this cross-zone guard belongs after both resolve.
    if raw.enabled and raw.keep_last_runs < bronze.retain_last:
        raise MaintenanceProfileError(
            f"maintenance.raw.keep_last_runs ({raw.keep_last_runs}) must be at least "
            f"maintenance.bronze.retain_last ({bronze.retain_last}) so every retained "
            "bronze snapshot's raw input is retained too"
        )
    return MaintenancePolicy(bronze, metadata, lineage, runs, raw, _timeout_seconds(block))


def _declared(block: Mapping[str, Any], key: str) -> Any:
    """Keep the original type; the shared empty-guard does not coerce numbers or flags."""
    value = block.get(key)
    return None if non_empty_text(value) is None else value


def _mapping(value: Any, path: str, supported_keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaintenanceProfileError(f"{path} must be a mapping")
    _reject_unsupported_keys(value, supported_keys, path)
    return value


def _subblock(
    block: Mapping[str, Any], key: str, supported_keys: frozenset[str]
) -> Mapping[str, Any]:
    path = f"{MAINTENANCE_BLOCK_KEY}.{key}"
    value = _declared(block, key)
    if value is None:
        raise MaintenanceProfileError(f"Environment config must set {path}")
    return _mapping(value, path, supported_keys)


def _reject_unsupported_keys(
    block: Mapping[str, Any], supported_keys: frozenset[str], path: str
) -> None:
    """Name one unknown key, never its value; mixed YAML key types are also refused."""
    unsupported = sorted((key for key in block if key not in supported_keys), key=repr)
    if unsupported:
        supported = ", ".join(sorted(supported_keys))
        raise MaintenanceProfileError(
            f"Environment config has an unsupported {path}: {unsupported[0]!r}; "
            f"supported keys: {supported}"
        )


def _integer(
    block: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: int = 0,
    default: int | None = None,
) -> int:
    qualified = f"{path}.{key}"
    value = _declared(block, key)
    if value is None:
        if default is None:
            raise MaintenanceProfileError(f"Environment config must set {qualified}")
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaintenanceProfileError(
            f"Environment config has a non-integer {qualified}: {value!r}"
        )
    if value < minimum:
        rule = "must not be negative" if minimum == 0 else f"must be at least {minimum}"
        raise MaintenanceProfileError(f"Environment config {qualified} {rule}")
    return value


def _boolean(block: Mapping[str, Any], key: str, path: str) -> bool:
    value = _declared(block, key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise MaintenanceProfileError(
            f"Environment config has a non-boolean {path}.{key}: {value!r}"
        )
    return value


def _bronze_policy(block: Mapping[str, Any]) -> BronzeRetentionPolicy:
    path = "maintenance.bronze"
    compact_value = _declared(block, "compact")
    compact = (
        {}
        if compact_value is None
        else _mapping(compact_value, f"{path}.compact", SUPPORTED_COMPACT_KEYS)
    )
    enabled = _boolean(compact, "enabled", f"{path}.compact")
    target = _declared(compact, "target_file_size_mb")
    if enabled and target is None:
        raise MaintenanceProfileError(
            "Environment config must set maintenance.bronze.compact.target_file_size_mb "
            "when compaction is enabled"
        )
    return BronzeRetentionPolicy(
        retain_last=_integer(block, "retain_last", path, minimum=1),
        older_than_days=_integer(block, "older_than_days", path),
        remove_orphan_files=_boolean(block, "remove_orphan_files", path),
        orphan_older_than_days=_integer(
            block, "orphan_older_than_days", path, minimum=1, default=3
        ),
        compact_enabled=enabled,
        compact_target_file_size_mb=(
            None
            if target is None
            else _integer(compact, "target_file_size_mb", f"{path}.compact", minimum=1)
        ),
    )


def _raw_policy(block: Mapping[str, Any]) -> RawRetentionPolicy:
    path = "maintenance.raw"
    enabled = _boolean(block, "enabled", path)
    return RawRetentionPolicy(
        enabled=enabled,
        keep_last_runs=_integer(block, "keep_last_runs", path, default=None if enabled else 0),
        older_than_days=_integer(block, "older_than_days", path, default=None if enabled else 0),
    )


def _timeout_seconds(block: Mapping[str, Any]) -> float:
    qualified = "maintenance.item_timeout_seconds"
    value = _declared(block, "item_timeout_seconds")
    if value is None:
        return 1800.0
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MaintenanceProfileError(
            f"Environment config has a non-numeric {qualified}: {value!r}"
        )
    try:
        timeout = float(value)
    except OverflowError as exc:
        raise MaintenanceProfileError(f"Environment config {qualified} must be finite") from exc
    if not math.isfinite(timeout):
        raise MaintenanceProfileError(f"Environment config {qualified} must be finite")
    if timeout <= 0:
        raise MaintenanceProfileError(f"Environment config {qualified} must be positive")
    return timeout
