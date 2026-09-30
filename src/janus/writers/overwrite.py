"""Pure planning and SQL builders for a full refresh of an Iceberg bronze table.

An ``INSERT OVERWRITE`` preserves snapshot ancestry. On the pinned Iceberg pair,
``REPLACE TABLE`` creates a new snapshot root: older snapshots remain readable by ID,
but rollback to them is unavailable because they are no longer ancestors. The caller
records ``history_reset_reason`` whenever it chooses replacement.

The evolution planner decides schema compatibility before this planner runs. Additions
and promotions are applied to the live table first, so this planner sees their final
types. A declared breaking change on a full refresh forces replacement; partition
spec drift retains the replacement behavior. Calls without an evolution
plan preserve the earlier pure planner behavior for existing callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from janus.writers.evolution import EvolutionPlan
from janus.writers.identifiers import partition_clause, quote_identifier
from janus.writers.schema_ddl import render_projection

_MECHANISMS = frozenset({"insert_overwrite", "replace_table"})


@dataclass(frozen=True, slots=True)
class FullRefreshOverwritePlan:
    """How one full-refresh run replaces the contents of an existing bronze table."""

    mechanism: str  # one of _MECHANISMS
    projection: tuple[str, ...] = ()  # target column order for the SELECT
    add_columns: tuple[tuple[str, str], ...] = ()  # (name, spark type string)
    reason: str = ""  # human-readable derivation, for logs/metadata

    def __post_init__(self) -> None:
        if self.mechanism not in _MECHANISMS:
            allowed = ", ".join(sorted(_MECHANISMS))
            raise ValueError(f"mechanism must be one of: {allowed}")
        if self.mechanism == "insert_overwrite" and not self.projection:
            raise ValueError("insert_overwrite requires an explicit column projection")
        if self.mechanism == "replace_table" and (self.projection or self.add_columns):
            raise ValueError("replace_table recreates the table; it takes no projection")

    @property
    def preserves_history(self) -> bool:
        """Return whether this mechanism appends to the snapshot log instead of resetting it."""
        return self.mechanism == "insert_overwrite"


def plan_full_refresh_overwrite(
    *,
    source_columns: Sequence[tuple[str, str]],  # (name, type) in DataFrame order
    target_columns: Sequence[tuple[str, str]],  # (name, type) in table order
    configured_partitions: Sequence[str],
    target_partitions: Sequence[str] | None,  # None => unreadable, assume drift
    evolution: EvolutionPlan | None = None,
) -> FullRefreshOverwritePlan:
    """Decide how a full refresh overwrites an existing bronze table.

    With ``evolution``, the declared compatibility decision is already settled.
    Without it, this function retains the earlier total fallback behavior.

    Name comparison is case-sensitive and order-independent. Iceberg column names round-trip
    exactly, so lower-casing would merge columns the table keeps distinct; and the explicit
    projection makes the DataFrame's column order irrelevant, so a pure reordering is absorbed
    rather than treated as drift.
    """
    breaking = evolution is not None and evolution.outcome == "breaking_replace"
    source_types = dict(source_columns)
    target_names = tuple(name for name, _ in target_columns)

    if not source_columns or breaking:
        return FullRefreshOverwritePlan(
            mechanism="replace_table",
            reason=(
                evolution.reason
                if breaking and evolution is not None
                else "source schema is empty; there is nothing to project"
            ),
        )

    if target_partitions is None:
        return FullRefreshOverwritePlan(
            mechanism="replace_table",
            reason="target partition spec could not be read",
        )
    if tuple(target_partitions) != tuple(configured_partitions):
        return FullRefreshOverwritePlan(
            mechanism="replace_table",
            reason=(
                "partition spec changed: "
                f"{_render_spec(target_partitions)} -> {_render_spec(configured_partitions)}"
            ),
        )

    dropped = [name for name in target_names if name not in source_types]
    if dropped:
        return FullRefreshOverwritePlan(
            mechanism="replace_table",
            reason=f"columns removed from the source schema: {', '.join(dropped)}",
        )

    # Conservative on purpose: Iceberg permits some widening promotions, but encoding a
    # promotion matrix belongs to the schema-evolution work, and the fallback is always safe.
    retyped = [
        f"{name} {target_type} -> {source_types[name]}"
        for name, target_type in target_columns
        if source_types[name] != target_type
    ]
    if retyped:
        return FullRefreshOverwritePlan(
            mechanism="replace_table",
            reason=f"column type changed: {'; '.join(retyped)}",
        )

    added = tuple(
        (name, column_type) for name, column_type in source_columns if name not in set(target_names)
    )
    if added:
        reason = (
            f"columns added to the target: {', '.join(name for name, _ in added)}; "
            "table history retained"
        )
    else:
        reason = "schema and partition spec unchanged; table history retained"

    return FullRefreshOverwritePlan(
        mechanism="insert_overwrite",
        projection=target_names + tuple(name for name, _ in added),
        add_columns=added,
        reason=reason,
    )


def build_insert_overwrite_sql(
    *,
    table_identifier: str,
    source_view: str,
    projection: Sequence[str],
) -> str:
    """Render the history-preserving full-refresh overwrite.

    Pure — strings in, string out, no Spark. ``INSERT OVERWRITE`` with no ``PARTITION`` clause
    replaces every partition of the table under Spark's *static* partition-overwrite mode, in
    one atomic Iceberg commit that appends to the snapshot log instead of resetting it. The
    caller is responsible for pinning that mode — the SQL alone does not express it, and under
    ``dynamic`` the same statement would only replace the partitions the SELECT produces.

    Columns are listed explicitly, in target-table order, so the positional INSERT can never
    depend on the DataFrame's column order; each one is quoted with the same defence as the
    table identifier.
    """
    if not projection:
        raise ValueError("insert overwrite requires at least one projected column")

    quoted_table = quote_identifier(table_identifier)
    quoted_view = quote_identifier(source_view)
    columns = render_projection(projection)
    return f"INSERT OVERWRITE {quoted_table}\nSELECT {columns} FROM {quoted_view}"


def build_create_table_as_select_sql(
    *,
    table_identifier: str,
    source_view: str,
    partition_columns: Sequence[str],
) -> str:
    """Render the first bronze write, which creates the table from the staged view."""
    return (
        f"CREATE TABLE {quote_identifier(table_identifier)} USING iceberg "
        f"{partition_clause(partition_columns)} AS SELECT * FROM "
        f"{quote_identifier(source_view)}"
    )


def build_replace_table_as_select_sql(
    *,
    table_identifier: str,
    source_view: str,
    partition_columns: Sequence[str],
) -> str:
    """Render the history-resetting fallback overwrite.

    Only for drift :func:`plan_full_refresh_overwrite` refuses to reconcile. The table is
    dropped and recreated, so its snapshot log starts empty — the caller must say so in the
    write metadata rather than let an operator assume time travel still reaches the prior run.
    """
    return (
        f"REPLACE TABLE {quote_identifier(table_identifier)} USING iceberg "
        f"{partition_clause(partition_columns)} AS SELECT * FROM "
        f"{quote_identifier(source_view)}"
    )


def _render_spec(partition_columns: Sequence[str]) -> str:
    return ", ".join(partition_columns) if partition_columns else "(none)"
