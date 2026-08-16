"""Reshape an ``ExecutionPlan`` so a replay reads and writes where the caller meant.

Three adjustments, all made before rehydration starts: redirect the bronze target to the
``--bronze-table`` the caller named, point the plan's raw root at a *historical* run
instead of the one this run would have written, and resolve which historical run that is
when none was named.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan
from janus.models.source_config import OutputTarget
from janus.utils.storage import bronze_table_identifier


def _override_bronze_output(plan: ExecutionPlan, bronze_table: str) -> ExecutionPlan:
    """Redirect the bronze target to ``bronze_table`` for a replay."""
    normalized_target = bronze_table.strip()
    if not normalized_target:
        raise ValueError("bronze_table must not be empty")

    namespace = plan.bronze_output.namespace
    table_name = normalized_target

    if "." in normalized_target:
        if normalized_target.count(".") != 1:
            raise ValueError(
                "bronze_table must be a table name or one namespace.table identifier"
            )
        namespace, table_name = normalized_target.split(".", 1)
        namespace = namespace.strip() or None
        table_name = table_name.strip()
        if not table_name:
            raise ValueError("bronze_table table name must not be empty")

    return replace(
        plan,
        bronze_output=OutputTarget(
            path=plan.bronze_output.path,
            format=plan.bronze_output.format,
            namespace=namespace,
            table_name=table_name,
        ),
    )


def _plan_with_active_raw_root(plan: ExecutionPlan) -> ExecutionPlan:
    raw_root = Path(plan.raw_output.path)
    latest_run_root = _latest_raw_run_root(raw_root)
    if latest_run_root is None:
        return plan
    return replace(plan, raw_output=replace(plan.raw_output, path=str(latest_run_root)))


def _latest_raw_run_root(raw_root: Path) -> Path | None:
    """Newest ``runs/ingestion_date=…/run_id=…`` root under ``raw_root``, or ``None``."""
    runs_root = raw_root / "runs"
    if not runs_root.exists():
        return None

    candidates = sorted(
        path
        for path in runs_root.glob("ingestion_date=*/run_id=*")
        if path.is_dir()
    )
    if not candidates:
        return None
    return candidates[-1]


def _bronze_target_identifier(plan: ExecutionPlan) -> str:
    return bronze_table_identifier(
        plan.bronze_output.path,
        fallback_name=plan.source.source_id,
        namespace=plan.bronze_output.namespace,
        table_name=plan.bronze_output.table_name,
    )


def _freeze_string_mapping(values: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()

    frozen_items: list[tuple[str, str]] = []
    for key, value in values.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if not normalized_key or not normalized_value:
            continue
        frozen_items.append((normalized_key, normalized_value))
    return tuple(sorted(frozen_items))
