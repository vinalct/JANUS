from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan
from janus.utils.environment import resolve_project_path


def resolve_schema_path_for_plan(plan: ExecutionPlan) -> Path | None:
    """Return the explicit schema path configured for one plan, if any."""
    if plan.source_config.schema.mode != "explicit" or not plan.source_config.schema.path:
        return None

    configured_path = Path(plan.source_config.schema.path)
    if configured_path.is_absolute():
        return configured_path

    runtime_path = resolve_project_path(plan.run_context.project_root, str(configured_path))
    if runtime_path.exists():
        return runtime_path

    config_path = plan.source_config.config_path.resolve()
    for parent in config_path.parents:
        candidate = parent / configured_path
        if candidate.exists():
            return candidate

    return runtime_path


def resolve_spark_schema_for_plan(plan: ExecutionPlan) -> Any | None:
    """Return a Spark schema object when the plan declares an explicit schema contract."""
    schema_path = resolve_schema_path_for_plan(plan)
    if schema_path is None:
        return None
    if not schema_path.exists():
        raise FileNotFoundError(f"Configured schema path does not exist: {schema_path}")
    return load_spark_schema_from_schema_path(schema_path)


def load_spark_schema_from_schema_path(path: Path) -> Any:
    """Load a Spark-readable schema from one JSON contract file."""
    try:
        from pyspark.sql.types import StringType, StructField, StructType
    except ImportError:
        return _FieldNameSchema(load_expected_fields_from_schema_path(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(raw, Mapping)
        and raw.get("type") == "struct"
        and isinstance(raw.get("fields"), Sequence)
    ):
        return StructType.fromJson(raw)

    field_names = load_expected_fields_from_schema_path(path)
    return StructType([StructField(field_name, StringType(), True) for field_name in field_names])


@dataclass(frozen=True, slots=True)
class _FieldNameSchema:
    """Minimal schema facade for unit tests that run without PySpark installed."""

    field_names: tuple[str, ...]

    def fieldNames(self) -> list[str]:
        return list(self.field_names)


def load_expected_fields_from_schema_path(path: Path) -> tuple[str, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return _field_names_from_payload(raw, path)
    if isinstance(raw, Mapping):
        for key in ("fields", "columns"):
            if key in raw:
                return _field_names_from_payload(raw[key], path)
        if isinstance(raw.get("schema"), Mapping):
            nested_schema = raw["schema"]
            for key in ("fields", "columns"):
                if key in nested_schema:
                    return _field_names_from_payload(nested_schema[key], path)
    raise ValueError(
        f"Schema file {path} must be a JSON array of field names or a mapping with fields/columns"
    )


def _field_names_from_payload(value: Any, path: Path) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"Schema file {path} fields/columns must be an array")

    field_names: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            field_names.append(entry)
            continue
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            field_names.append(entry["name"])
            continue
        raise ValueError(
            f"Schema file {path} entries must be strings or objects containing a 'name' field"
        )
    return _normalize_field_names(field_names)


def _normalize_field_names(field_names: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for field_name in field_names:
        candidate = str(field_name).strip()
        if not candidate:
            raise ValueError("Schema field names must not be empty")
        if candidate in seen:
            raise ValueError(f"Schema field names must be unique; duplicate {candidate!r}")
        normalized.append(candidate)
        seen.add(candidate)
    return tuple(normalized)


__all__ = [
    "load_expected_fields_from_schema_path",
    "load_spark_schema_from_schema_path",
    "resolve_schema_path_for_plan",
    "resolve_spark_schema_for_plan",
]
