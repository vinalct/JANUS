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


def render_projection(columns: Sequence[str | tuple[str, str | None]]) -> str:
    """Render a nonempty, quoted SELECT column list with optional NULL aliases."""
    if not columns:
        raise ValueError("insert requires at least one projected column")
    return ", ".join(
        quote_identifier(column)
        if isinstance(column, str)
        else (
            f"NULL AS {quote_identifier(column[0])}"
            if column[1] is None
            else f"{quote_identifier(column[1])} AS {quote_identifier(column[0])}"
        )
        for column in columns
    )


def build_insert_into_sql(
    *, table_identifier: str, source_view: str, projection: Sequence[str | tuple[str, str | None]]
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
            target
            + " "
            + "; ".join(details)
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
    rendered = ", ".join(f"{quote_identifier(name)} {column_type}" for name, column_type in columns)
    return f"ALTER TABLE {quoted_table} ADD COLUMNS ({rendered})"


def build_alter_column_type_sql(*, table_identifier: str, column: str, spark_type: str) -> str:
    return (
        f"ALTER TABLE {quote_identifier(table_identifier)} "
        f"ALTER COLUMN {quote_identifier(column)} TYPE {spark_type}"
    )


CONTRACT_PROPERTY_KEYS = (
    "janus.contract_id",
    "janus.contract_version",
    "janus.schema_version",
)


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_set_contract_properties_sql(
    *, table_identifier: str, contract_id: str, contract_version: str, schema_version: str
) -> str:
    values = (contract_id, contract_version, schema_version)
    properties = ", ".join(
        f"{_quote_literal(key)} = {_quote_literal(value)}"
        for key, value in zip(CONTRACT_PROPERTY_KEYS, values, strict=True)
    )
    return f"ALTER TABLE {quote_identifier(table_identifier)} SET TBLPROPERTIES ({properties})"


def build_show_contract_properties_sql(*, table_identifier: str) -> str:
    return f"SHOW TBLPROPERTIES {quote_identifier(table_identifier)}"


def project_append_columns(
    *,
    source_names: set[str],
    target_names: tuple[str, ...],
    required_names: set[str],
    table_identifier: str,
) -> tuple[str | tuple[str, None], ...]:
    """Build target-order projection, filling only absent nullable columns with NULL."""
    target_set = set(target_names)
    extra = sorted(source_names - target_set)
    if extra:
        raise UnreconciledAppendError(
            f"bronze target {table_identifier!r} lacks columns present in this batch: "
            + ", ".join(extra)
        )
    missing_required = sorted(required_names & (target_set - source_names))
    if missing_required:
        raise UnreconciledAppendError(
            f"bronze target {table_identifier!r} has required columns missing from this batch: "
            + ", ".join(missing_required)
        )
    return tuple(name if name in source_names else (name, None) for name in target_names)
