"""The append path resolves by name: ``INSERT INTO … SELECT <target order>``, never ``SELECT *``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from test_bronze_overwrite_writer import (
    ENVIRONMENT_CONFIG,
    FakeDataFrame,
    FakeField,
    FakeSchema,
    FakeSparkSession,
    FakeTable,
    FakeType,
    _plan,
)

from janus.models import BronzeWriteIntent, ExecutionPlan, WriteResult
from janus.models.data_contracts import DataContract, load_data_contract
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

pytestmark = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"

BASE_TABLE = (
    ("id", "string"),
    ("label", "string"),
    ("amount", "bigint"),
    ("when", "timestamp"),
    ("ingestion_date", "date"),
)
INSERT_INTENT = BronzeWriteIntent(
    strategy="insert", configured_mode="append", partition_columns=("ingestion_date",)
)
REPLACE_INTENT = BronzeWriteIntent(
    strategy="replace_table", configured_mode="overwrite", partition_columns=("ingestion_date",)
)


_SPARK_JSON = {"bigint": "long", "int": "integer", "smallint": "short", "tinyint": "byte"}


# ── the contract-aware recorder ──────────────────────────────────────────────


class TypedFakeType(FakeType):
    def jsonValue(self) -> Any:
        return _SPARK_JSON.get(self._simple, self._simple)


def typed_schema(columns: Sequence[tuple[str, str]]) -> FakeSchema:
    return FakeSchema(tuple(FakeField(name, TypedFakeType(spelled)) for name, spelled in columns))


class Row(dict):
    """A result row readable as ``row["key"]`` or ``row.key``, like a Spark ``Row``."""

    def __getattr__(self, name: str) -> Any:
        return self[name]


class Result:
    def __init__(self, rows: Sequence[Row] = ()) -> None:
        self._rows = list(rows)

    def collect(self) -> list[Row]:
        return list(self._rows)


class ContractAwareSession(FakeSparkSession):
    """The overwrite recorder, answering the stamp read and applying the DDL it records."""

    def __init__(self, *, stamp: Mapping[str, str] | None = None, **options: Any) -> None:
        options.setdefault("target_columns", BASE_TABLE)
        super().__init__(**options)
        self.stamp = dict(stamp or {})

    def sql(self, statement: str) -> Result:
        super().sql(statement)
        if statement.startswith("SHOW TBLPROPERTIES"):
            return Result([Row(key=key, value=value) for key, value in self.stamp.items()])
        self._apply(statement)
        return Result()

    def table(self, identifier: str) -> FakeTable:
        table = super().table(identifier)
        if identifier.endswith(".partitions"):
            return table
        return FakeTable(typed_schema(self.target_columns))

    def _apply(self, statement: str) -> None:
        added = re.search(r"ADD COLUMNS \((.*)\)$", statement)
        if added:
            for column in added.group(1).split(", "):
                name, spelled = column.split(" ", 1)
                self.target_columns = (*self.target_columns, (name.strip("`"), spelled))
        altered = re.search(r"ALTER COLUMN `([^`]+)` TYPE (\S+)$", statement)
        if altered:
            name, spelled = altered.groups()
            self.target_columns = tuple(
                (column, spelled if column == name else current)
                for column, current in self.target_columns
            )
        for key, value in re.findall(r"'(janus\.[a-z_]+)' = '((?:[^']|'')*)'", statement):
            self.stamp[key] = value.replace("''", "'")


class TypedFakeDataFrame(FakeDataFrame):
    def __init__(self, session: FakeSparkSession, columns: Sequence[tuple[str, str]]) -> None:
        super().__init__(session, tuple(columns))
        self.schema = typed_schema(columns)


def contract(name: str) -> DataContract:
    return load_data_contract(HOSTILE / f"{name}.yaml")


def stamp_of(declared: DataContract) -> dict[str, str]:
    """The table properties a write under ``declared`` leaves behind (D-8)."""
    return {
        "janus.contract_id": declared.id,
        "janus.contract_version": declared.version,
        "janus.schema_version": declared.schema_version,
    }


def contract_plan(tmp_path: Path, declared: DataContract) -> ExecutionPlan:
    return _plan(tmp_path).with_data_contract(declared)


def write(
    tmp_path: Path,
    session: FakeSparkSession,
    declared: DataContract,
    frame_columns: Sequence[tuple[str, str]],
    *,
    intent: BronzeWriteIntent = INSERT_INTENT,
) -> WriteResult:
    writer = SparkDatasetWriter(StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path))
    return writer.write(
        TypedFakeDataFrame(session, frame_columns),
        contract_plan(tmp_path, declared),
        "bronze",
        intent=intent,
        count_records=True,
    )


def inserts(session: FakeSparkSession) -> list[str]:
    return [statement for statement in session.statements if statement.startswith("INSERT")]


def projection_of(statement: str) -> str:
    return statement.split("\nSELECT ", 1)[1].split(" FROM ", 1)[0]


# ── the append path ──────────────────────────────────────────────────────────


def test_a_reordered_batch_is_projected_in_target_table_order(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))
    reordered = (BASE_TABLE[2], BASE_TABLE[0], BASE_TABLE[4], BASE_TABLE[3], BASE_TABLE[1])

    write(tmp_path, session, declared, reordered)

    [statement] = inserts(session)
    assert statement.startswith("INSERT INTO `bronze_test`.`bronze_overwrite_fixture`\n")
    assert projection_of(statement) == "`id`, `label`, `amount`, `when`, `ingestion_date`"


def test_no_append_ever_selects_star(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))

    write(tmp_path, session, declared, BASE_TABLE)

    assert len(inserts(session)) == 1
    assert not any("SELECT *" in statement for statement in session.statements)


def test_a_batch_column_the_table_lacks_is_refused_before_any_insert(tmp_path):
    from janus.writers.schema_ddl import UnreconciledAppendError

    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))

    with pytest.raises(UnreconciledAppendError) as raised:
        write(tmp_path, session, declared, (*BASE_TABLE, ("note", "string")))

    assert issubclass(UnreconciledAppendError, ValueError)
    assert "note" in str(raised.value)
    assert inserts(session) == []
    assert len(session.dropped_temp_views) == 1


def test_a_table_column_the_batch_lacks_is_refused_before_any_insert(tmp_path):
    from janus.writers.schema_ddl import UnreconciledAppendError

    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))
    without_label = tuple(column for column in BASE_TABLE if column[0] != "label")

    with pytest.raises(UnreconciledAppendError) as raised:
        write(tmp_path, session, declared, without_label)

    assert "label" in str(raised.value)
    assert inserts(session) == []


def test_plan_append_projection_is_pure_and_returns_target_order():
    from janus.writers.schema_ddl import plan_append_projection

    projection = plan_append_projection(
        source_columns=(("label", "string"), ("id", "string")),
        target_columns=(("id", "string"), ("label", "string")),
    )

    assert projection == ("id", "label")


def test_the_full_refresh_projection_is_unchanged(tmp_path):
    from janus.writers.schema_ddl import render_projection

    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))

    write(tmp_path, session, declared, BASE_TABLE, intent=REPLACE_INTENT)

    [statement] = inserts(session)
    assert statement.startswith("INSERT OVERWRITE ")
    assert projection_of(statement) == render_projection(tuple(name for name, _ in BASE_TABLE))
