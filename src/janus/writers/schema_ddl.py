"""Pure SQL builders and column-order planning for Iceberg writes.

Spark's ``INSERT INTO ... SELECT`` resolves columns by position. Until the evolution plan
can supply missing nullable columns, an append requires the source and target to have the
same names and selects those names in target order. The pinned Spark/Iceberg merge check
found that ``UPDATE SET *`` and ``INSERT *`` resolve by name.
"""

from __future__ import annotations

from collections.abc import Sequence

from janus.writers.identifiers import quote_identifier


class UnreconciledAppendError(ValueError):
    """An append needs a schema change or a missing-column projection before it can write."""


def render_projection(columns: Sequence[str]) -> str:
    """Render a nonempty, quoted SELECT column list."""
    if not columns:
        raise ValueError("insert requires at least one projected column")
    return ", ".join(quote_identifier(column) for column in columns)


def build_insert_into_sql(
    *, table_identifier: str, source_view: str, projection: Sequence[str]
) -> str:
    """Render an append whose SELECT follows target-table column order."""
    return (
        f"INSERT INTO {quote_identifier(table_identifier)}\n"
        f"SELECT {render_projection(projection)} FROM {quote_identifier(source_view)}"
    )


def plan_append_projection(
    *,
    source_columns: Sequence[tuple[str, str]],
    target_columns: Sequence[tuple[str, str]],
    table_identifier: str | None = None,
) -> tuple[str, ...]:
    """Return the target order when source and target have the same column names."""
    source_names = {name for name, _ in source_columns}
    target_names = {name for name, _ in target_columns}
    missing_in_target = [name for name, _ in source_columns if name not in target_names]
    missing_in_source = [name for name, _ in target_columns if name not in source_names]
    if missing_in_target or missing_in_source:
        details = []
        if missing_in_target:
            details.append(f"lacks columns present in this batch: {', '.join(missing_in_target)}")
        if missing_in_source:
            details.append(f"has columns missing from this batch: {', '.join(missing_in_source)}")
        target = f"bronze target {table_identifier!r}" if table_identifier else "bronze target"
        raise UnreconciledAppendError(
            target + " " + "; ".join(details)
            + " — the contract's compatibility decides whether they may be added or projected."
        )
    return tuple(name for name, _ in target_columns)


def build_add_columns_sql(
    *,
    table_identifier: str,
    columns: Sequence[tuple[str, str]],
) -> str | None:
    """Render ``ALTER TABLE ... ADD COLUMNS`` for schema evolution, or ``None`` if empty."""
    if not columns:
        return None

    quoted_table = quote_identifier(table_identifier)
    rendered = ", ".join(
        f"{quote_identifier(name)} {column_type}" for name, column_type in columns
    )
    return f"ALTER TABLE {quoted_table} ADD COLUMNS ({rendered})"


def build_alter_column_type_sql(
    *, table_identifier: str, column: str, spark_type: str
) -> str:
    raise NotImplementedError("implements build_alter_column_type_sql")


def build_set_contract_properties_sql(
    *, table_identifier: str, contract_id: str, contract_version: str, schema_version: str
) -> str:
    raise NotImplementedError("implements build_set_contract_properties_sql")
