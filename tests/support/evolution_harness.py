"""Real Iceberg tables for the contract evolution matrix.

Each cell owns one table in the shared profile-backed test catalog. Both data
writes use the production writer. Only the unstamped-table fixture changes table
properties directly after its first write.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from janus.models.data_contracts import load_data_contract, spark_sql_type
from janus.writers import quote_identifier
from tests.support.contract_enforcement import (
    EnforcementCase,
    plan_case,
    snapshot_count,
    table_properties,
    table_uuid,
    write_case_project,
    write_frame,
)

_STARTED_AT = datetime(2026, 9, 29, 12, tzinfo=UTC)
_WHEN = datetime(2026, 9, 29, 12)
_INGESTED_AT = datetime(2026, 9, 29, 13)


@dataclass(frozen=True)
class EvolutionCell:
    id: str
    live_contract: Path
    next_contract: Path
    compatibility: str
    strategy: str
    live_extra_columns: tuple[str, ...] = ()
    expected_outcome: str = "noop"
    expected_ddl: tuple[str, ...] = ()
    next_version: str | None = None
    stamp_live_table: bool = True


@dataclass
class CellResult:
    error: Exception | None
    statements: list[str]
    write_metadata: dict[str, str]
    snapshots_before: int
    snapshots_after: int
    run_1_snapshot_id: int
    new_snapshot_id: int
    uuid_before: str
    uuid_after: str
    schema_before: tuple[tuple[str, str, bool], ...]
    schema_after: tuple[tuple[str, str, bool], ...]
    stamp_before: dict[str, str]
    stamp_after: dict[str, str]
    next_contract_version: str
    next_schema_version: str
    run_1_rows_expected: list[dict[str, Any]]
    run_1_rows_after: list[dict[str, Any]]
    run_1_is_current_ancestor: bool | None
    new_snapshot_parent_id: int | None
    rollback_to_run_1_error: str
    _write_again: Callable[[Path], CellResult] = field(repr=False)

    def write_again(self, *, next_contract: Path) -> CellResult:
        """Try a further write against this table without recreating it."""
        return self._write_again(next_contract)


def run_cell(spark: Any, tmp_path: Path, cell: EvolutionCell) -> CellResult:
    """Create a stamped table, then attempt the cell's contract change."""
    source_id = "evolution_" + re.sub(r"[^a-z0-9]+", "_", cell.id.lower()).strip("_")
    live = _case(source_id, cell.live_contract, cell.strategy)
    root = write_case_project(tmp_path, live)
    first_plan = plan_case(root, live, run_id=f"{source_id}_first", started_at=_STARTED_AT).plan
    first_rows, first_schema = _frame_data(live, "old", extra=cell.live_extra_columns)
    write_frame(spark, first_plan, first_rows, first_schema)
    table = live.bronze_table

    if not cell.stamp_live_table:
        spark.sql(
            f"ALTER TABLE {quote_identifier(table)} UNSET TBLPROPERTIES "
            "('janus.contract_id', 'janus.contract_version', 'janus.schema_version')"
        )

    first_snapshot = _latest_snapshot(spark, table)

    def attempt(path: Path, version: str | None = None) -> CellResult:
        following = _case(source_id, path, cell.strategy, compatibility=cell.compatibility)
        if version is not None:
            following = replace(following, version=version)
        write_case_project(root, following)
        plan = plan_case(
            root, following, run_id=f"{source_id}_next", started_at=_STARTED_AT
        ).plan
        declared = load_data_contract(root / following.contract_path)
        before_schema = _schema(spark, table)
        before_snapshots = snapshot_count(spark, table)
        before_uuid = table_uuid(spark, table)
        before_stamp = _stamp(spark, table)
        before_rows = _rows(spark, table, tuple(name for name, _, _ in before_schema))

        rows, ddl = _frame_data(
            following, "new", include_old=cell.strategy == "replace_table"
        )
        statements: list[str] = []
        original_sql = spark.sql

        def record_sql(statement: str, *args: Any, **kwargs: Any) -> Any:
            statements.append(statement)
            return original_sql(statement, *args, **kwargs)

        error: Exception | None = None
        write_metadata: dict[str, str] = {}
        with patch.object(spark, "sql", side_effect=record_sql):
            try:
                written = write_frame(spark, plan, rows, ddl)
                write_metadata = written.metadata_as_dict()
            except Exception as exc:
                error = exc

        after_schema = _schema(spark, table)
        after_snapshots = snapshot_count(spark, table)
        after_uuid = table_uuid(spark, table)
        after_stamp = _stamp(spark, table)
        after_rows = _rows(spark, table, tuple(name for name, _, _ in after_schema))
        expected_old = _expected_old_row(before_rows, after_schema)
        actual_old = [row for row in after_rows if row["id"] == "old"]
        history = _history(spark, table)
        newest = _latest_snapshot(spark, table)
        rollback_error = ""
        if write_metadata.get("schema_evolution") == "breaking_replace":
            try:
                spark.sql(
                    "CALL janus.system.rollback_to_snapshot("
                    f"table => '{table}', snapshot_id => {first_snapshot})"
                ).collect()
            except Exception as exc:
                rollback_error = str(exc)

        return CellResult(
            error=error,
            statements=statements,
            write_metadata=write_metadata,
            snapshots_before=before_snapshots,
            snapshots_after=after_snapshots,
            run_1_snapshot_id=first_snapshot,
            new_snapshot_id=newest,
            uuid_before=before_uuid,
            uuid_after=after_uuid,
            schema_before=before_schema,
            schema_after=after_schema,
            stamp_before=before_stamp,
            stamp_after=after_stamp,
            next_contract_version=declared.version,
            next_schema_version=declared.schema_version,
            run_1_rows_expected=expected_old,
            run_1_rows_after=actual_old,
            run_1_is_current_ancestor=history.get(first_snapshot),
            new_snapshot_parent_id=_snapshot_parent(spark, table, newest),
            rollback_to_run_1_error=rollback_error,
            _write_again=lambda next_contract: attempt(next_contract),
        )

    return attempt(cell.next_contract, cell.next_version)


def _case(
    source_id: str, contract_path: Path, strategy: str, *, compatibility: str | None = None
) -> EnforcementCase:
    if strategy not in {"insert", "merge_on_keys", "replace_table"}:
        raise ValueError(f"unknown evolution strategy: {strategy}")
    options: dict[str, Any] = {
        "write_mode": "overwrite" if strategy == "replace_table" else "append",
        "extraction_mode": "incremental" if strategy == "merge_on_keys" else "full_refresh",
    }
    if strategy == "merge_on_keys":
        options["checkpoint_field"] = "when"
    if compatibility is not None:
        options["compatibility"] = compatibility
    return EnforcementCase.from_contract_file(source_id, contract_path, **options)


def _frame_data(
    case: EnforcementCase,
    row_id: str,
    *,
    include_old: bool = False,
    extra: tuple[str, ...] = (),
) -> tuple[list[tuple[Any, ...]], str]:
    columns = [(prop["name"], spark_sql_type(prop["physicalType"])) for prop in case.properties]
    columns.extend((item.split()[0], item.split(maxsplit=1)[1]) for item in extra)
    columns.append(("ingestion_timestamp", "timestamp"))
    ddl = ", ".join(f"{quote_identifier(name)} {type_}" for name, type_ in columns)
    ids = ("old", "new") if include_old else (row_id,)
    rows = [tuple(_value(name, type_, id_) for name, type_ in columns) for id_ in ids]
    return rows, ddl


def _value(name: str, type_: str, row_id: str) -> Any:
    old = row_id == "old"
    if name == "id":
        value: Any = row_id
    elif name in {"when", "ingestion_timestamp"}:
        value = _WHEN if name == "when" else _INGESTED_AT
    elif name == "label":
        value = "original" if old else "following"
    elif name == "note" and old:
        value = None
    elif name in {"note", "code", "title", "legacy_col"}:
        value = f"{name}-{row_id}"
    elif type_.startswith("decimal"):
        value = Decimal("1.25") if old else Decimal("2.50")
    elif type_ in {"float", "double"}:
        value = 1.0 if old else 2.0
    elif type_ in {"int", "bigint"}:
        value = 1 if old else 2
    else:
        value = "1" if old else "2"
    return value


def _schema(spark: Any, table: str) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (field.name, field.dataType.simpleString(), field.nullable)
        for field in spark.table(table).schema.fields
    )


def _stamp(spark: Any, table: str) -> dict[str, str]:
    return {
        key: value
        for key, value in table_properties(spark, table).items()
        if key.startswith("janus.")
    }


def _rows(spark: Any, table: str, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    return [row.asDict() for row in spark.table(table).select(*columns).collect()]


def _expected_old_row(
    before_rows: list[dict[str, Any]], schema: tuple[tuple[str, str, bool], ...]
) -> list[dict[str, Any]]:
    names = [name for name, _, _ in schema]
    return [
        {name: row.get(name) for name in names}
        for row in before_rows
        if row["id"] == "old"
    ]


def _history(spark: Any, table: str) -> dict[int, bool]:
    return {
        row["snapshot_id"]: row["is_current_ancestor"]
        for row in spark.table(f"{table}.history")
        .select("snapshot_id", "is_current_ancestor")
        .collect()
    }


def _latest_snapshot(spark: Any, table: str) -> int:
    return int(
        spark.table(f"{table}.history")
        .orderBy("made_current_at", ascending=False)
        .select("snapshot_id")
        .first()["snapshot_id"]
    )


def _snapshot_parent(spark: Any, table: str, snapshot_id: int) -> int | None:
    return (
        spark.table(f"{table}.snapshots")
        .where(f"snapshot_id = {snapshot_id}")
        .select("parent_id")
        .first()["parent_id"]
    )
