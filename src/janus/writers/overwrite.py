"""How a full refresh replaces the contents of an existing bronze table.

``REPLACE TABLE ... AS SELECT`` drops and recreates the Iceberg table: new table UUID, empty
snapshot log, history gone — which defeats time travel and rollback, one of the main reasons
to use Iceberg at all. ``INSERT OVERWRITE`` writes *into* the existing table instead, so the
snapshot log grows rather than resetting. The price is that three kinds of schema drift the
recreate absorbed silently now have to be decided explicitly:

===========================  =========================================================
Drift                        Decision
===========================  =========================================================
source has a new column      ``ALTER TABLE ... ADD COLUMNS``, then insert
source dropped a column      fall back to ``REPLACE TABLE`` (it would survive as NULLs)
a column changed type        fall back to ``REPLACE TABLE``
``spark.partition_by`` moved fall back to ``REPLACE TABLE`` (physical layout differs)
===========================  =========================================================

The fallback is what keeps the change output-neutral: every full refresh that succeeds today
still succeeds, and the only observable difference is that history is retained in the common
case. When the fallback fires the caller records *why* in the write metadata — a silent
history reset would be worse than the current behaviour, because operators would believe in
time travel that is not there.

**Additive columns are not gated on ``quality.allow_schema_evolution``.** The merge path gates
them because a MERGE against an older table is a genuine incremental-semantics question. A full
refresh has *always* accepted new columns — ``REPLACE TABLE`` recreated the schema every run —
so gating them here would be a regression, not a hardening.

Everything in this module is pure: the planner takes the two schemas and the two partition
specs as plain tuples and the builders take strings, so the interesting logic is unit-testable
on a host without PySpark. This is the same split :func:`janus.models.resolve_bronze_write_intent`
already uses for the write decision itself; the difference in scope is that ``BronzeWriteIntent``
answers "what does the *contract* ask for" from config alone, while this answers "what can the
*target table* accept", which needs the live table's state — hence ``writers/``, not ``models/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from janus.writers.identifiers import partition_clause, quote_identifier

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
) -> FullRefreshOverwritePlan:
    """Decide how a full refresh overwrites an existing bronze table.

    Total: every input returns a plan, drift never raises — an unreadable partition spec or a
    schema this module will not reconcile degrades to ``REPLACE TABLE``, which is always safe
    because it is exactly what the writer does today.

    Name comparison is case-sensitive and order-independent. Iceberg column names round-trip
    exactly, so lower-casing would merge columns the table keeps distinct; and the explicit
    projection makes the DataFrame's column order irrelevant, so a pure reordering is absorbed
    rather than treated as drift.
    """
    source_types = dict(source_columns)
    target_names = tuple(name for name, _ in target_columns)

    if not source_columns:
        return FullRefreshOverwritePlan(
            mechanism="replace_table",
            reason="source schema is empty; there is nothing to project",
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
        (name, column_type)
        for name, column_type in source_columns
        if name not in set(target_names)
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
    columns = ", ".join(quote_identifier(column) for column in projection)
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
