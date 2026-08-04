from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from janus.models import ExecutionPlan
from janus.models.source_config import OutputTarget
from janus.utils.environment import resolve_project_path

SUPPORTED_STORAGE_ZONES = frozenset({"bronze", "metadata", "raw"})
ICEBERG_BRONZE_NAMESPACE = "bronze"
_CANONICAL_ZONE_PREFIXES = {
    "raw": Path("data") / "raw",
    "bronze": Path("data") / "bronze",
    "metadata": Path("data") / "metadata",
}


@dataclass(frozen=True, slots=True)
class ResolvedOutputTarget:
    """Configured output target resolved against the active runtime storage layout."""

    zone: str
    configured_path: str
    resolved_path: Path
    format: str

    def __post_init__(self) -> None:
        _validate_zone(self.zone)
        if not self.configured_path.strip():
            raise ValueError("configured_path must not be empty")
        if not self.format.strip():
            raise ValueError("format must not be empty")
        if not self.resolved_path.is_absolute():
            raise ValueError("resolved_path must be absolute")

    def child(self, relative_path: str | Path) -> Path:
        return self.resolved_path / _normalize_relative_path(relative_path)


@dataclass(frozen=True, slots=True)
class StorageLayout:
    """Environment-aware raw, bronze, and metadata zone roots."""

    project_root: Path
    root_dir: Path
    raw_dir: Path
    bronze_dir: Path
    metadata_dir: Path

    def __post_init__(self) -> None:
        absolute_paths = (
            self.project_root,
            self.root_dir,
            self.raw_dir,
            self.bronze_dir,
            self.metadata_dir,
        )
        if any(not path.is_absolute() for path in absolute_paths):
            raise ValueError("storage paths must be absolute")

    @classmethod
    def from_environment_config(
        cls,
        config: Mapping[str, Any],
        project_root: Path,
    ) -> Self:
        storage_config = config.get("storage")
        if not isinstance(storage_config, Mapping):
            raise ValueError("storage must be a mapping")

        resolved_project_root = project_root.resolve()
        return cls(
            project_root=resolved_project_root,
            root_dir=resolve_project_path(
                resolved_project_root,
                _require_non_empty_string(storage_config, "root_dir"),
            ),
            raw_dir=resolve_project_path(
                resolved_project_root,
                _require_non_empty_string(storage_config, "raw_dir"),
            ),
            bronze_dir=resolve_project_path(
                resolved_project_root,
                _require_non_empty_string(storage_config, "bronze_dir"),
            ),
            metadata_dir=resolve_project_path(
                resolved_project_root,
                _require_non_empty_string(storage_config, "metadata_dir"),
            ),
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "project_root": self.project_root,
            "root_dir": self.root_dir,
            "raw_dir": self.raw_dir,
            "bronze_dir": self.bronze_dir,
            "metadata_dir": self.metadata_dir,
        }

    def zone_path(self, zone: str) -> Path:
        _validate_zone(zone)
        if zone == "raw":
            return self.raw_dir
        if zone == "bronze":
            return self.bronze_dir
        return self.metadata_dir

    def resolve_output_path(self, zone: str, configured_path: str) -> Path:
        _validate_zone(zone)
        normalized_path = Path(configured_path)
        if normalized_path.is_absolute():
            return normalized_path
        return self.zone_path(zone) / _relative_zone_suffix(zone, normalized_path)

    def resolve_target(self, zone: str, output_target: OutputTarget) -> ResolvedOutputTarget:
        return ResolvedOutputTarget(
            zone=zone,
            configured_path=output_target.path,
            resolved_path=self.resolve_output_path(zone, output_target.path),
            format=output_target.format,
        )

    def resolve_output(self, plan: ExecutionPlan, zone: str) -> ResolvedOutputTarget:
        return self.resolve_target(zone, _output_target_for_zone(plan, zone))


def bronze_table_identifier(
    configured_path: str,
    *,
    fallback_name: str,
    namespace: str | None = None,
    table_name: str | None = None,
) -> str:
    """Return the deterministic Iceberg table identifier for one bronze target."""
    resolved_namespace = sanitize_identifier_segment(namespace or ICEBERG_BRONZE_NAMESPACE)
    if not resolved_namespace:
        resolved_namespace = ICEBERG_BRONZE_NAMESPACE

    resolved_table = sanitize_identifier_segment(table_name) if table_name else ""
    if not resolved_table:
        raw_parts = [part for part in Path(configured_path).parts if part not in {"", ".", "/"}]
        normalized_parts = [sanitize_identifier_segment(part) for part in raw_parts]
        normalized_parts = [part for part in normalized_parts if part]

        bronze_indexes = [
            index for index, part in enumerate(normalized_parts) if part == ICEBERG_BRONZE_NAMESPACE
        ]
        if bronze_indexes:
            normalized_parts = normalized_parts[bronze_indexes[-1] + 1 :]

        if not normalized_parts:
            fallback = sanitize_identifier_segment(fallback_name)
            normalized_parts = [fallback or "dataset"]

        resolved_table = "__".join(normalized_parts)

    return f"{resolved_namespace}.{resolved_table}"


def sanitize_identifier_segment(value: str) -> str:
    """Normalize one identifier segment to ``[a-z0-9_]``, or ``""`` if nothing survives.

    Public because it is the single normalization every JANUS identifier goes through
    before it reaches a catalog or a SQL statement — the bronze table identifier above and
    the bronze staging-view name in :mod:`janus.writers.identifiers`. A second regex
    elsewhere would be a second answer to "what is a safe identifier".
    """
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        return ""
    if normalized[0].isdigit():
        return f"t_{normalized}"
    return normalized


def _output_target_for_zone(plan: ExecutionPlan, zone: str) -> OutputTarget:
    if zone == "raw":
        return plan.raw_output
    if zone == "bronze":
        return plan.bronze_output
    if zone == "metadata":
        return plan.metadata_output
    _validate_zone(zone)
    raise AssertionError("unreachable")


def _relative_zone_suffix(zone: str, configured_path: Path) -> Path:
    normalized = _normalize_relative_path(configured_path, allow_empty=True)
    canonical_prefix = _CANONICAL_ZONE_PREFIXES[zone]
    if normalized == canonical_prefix:
        return Path()
    if normalized.is_relative_to(canonical_prefix):
        return normalized.relative_to(canonical_prefix)

    zone_prefix = Path(zone)
    if normalized == zone_prefix:
        return Path()
    if normalized.is_relative_to(zone_prefix):
        return normalized.relative_to(zone_prefix)
    return normalized


def _normalize_relative_path(
    value: str | Path,
    *,
    allow_empty: bool = False,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("relative paths must not be absolute")
    if ".." in path.parts:
        raise ValueError("relative paths must not contain '..'")
    if not allow_empty and str(path).strip() in {"", "."}:
        raise ValueError("relative paths must not be empty")
    if allow_empty and str(path).strip() in {"", "."}:
        return Path()
    return path


def _require_non_empty_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_zone(zone: str) -> None:
    if zone not in SUPPORTED_STORAGE_ZONES:
        allowed = ", ".join(sorted(SUPPORTED_STORAGE_ZONES))
        raise ValueError(f"zone must be one of: {allowed}")
