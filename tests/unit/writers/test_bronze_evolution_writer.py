"""The writer executes the evolution plan with Iceberg-native statements, then stamps the table."""

from __future__ import annotations

from typing import Any

import pytest

from test_bronze_append_by_name import (
    BASE_TABLE,
    INSERT_INTENT,
    REPLACE_INTENT,
    ContractAwareSession,
    TypedFakeDataFrame,
    contract,
    inserts,
    projection_of,
    stamp_of,
    write,
)
from test_bronze_overwrite_writer import ENVIRONMENT_CONFIG, PARTITION_OVERWRITE_MODE, _plan

from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

RED_TASK = pytest.mark.xfail(strict=True, reason="red until implementation finishes")
RED_TASK_2 = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

INT_TABLE = tuple(
    (name, "int" if name == "amount" else spelled) for name, spelled in BASE_TABLE
)
PLUS_NOTE = (*BASE_TABLE, ("note", "string"))
V2_TABLE = tuple(column for column in BASE_TABLE if column[0] != "label")


def _kind(statement: str) -> str | None:
    """What one recorded statement does to the table; ``None`` for reads and bookkeeping."""
    for marker, kind in (
        ("ADD COLUMNS", "add_columns"),
        ("ALTER COLUMN", "alter_type"),
        ("SET TBLPROPERTIES", "stamp"),
    ):
        if marker in statement:
            return kind
    for prefix, kind in (
        ("INSERT OVERWRITE", "insert_overwrite"),
        ("INSERT INTO", "insert_into"),
        ("REPLACE TABLE", "replace_table"),
        ("CREATE TABLE", "create_table"),
        ("MERGE INTO", "merge"),
    ):
        if statement.startswith(prefix):
            return kind
    return None


def _effects(session: ContractAwareSession) -> list[str]:
    return [kind for statement in session.statements if (kind := _kind(statement))]


def _metadata(result: Any) -> dict[str, str]:
    return result.metadata_as_dict()


# ── the matching append is by name and changes nothing else ─────────


@RED_TASK
def test_a_matching_append_is_by_name_and_issues_no_ddl(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))
    reordered = tuple(reversed(BASE_TABLE))

    write(tmp_path, session, declared, reordered)

    assert _effects(session) == ["insert_into"]
    assert projection_of(inserts(session)[0]) == (
        "`id`, `label`, `amount`, `when`, `ingestion_date`"
    )


# ── additions, promotions, the stamp ────────────────────────────────


@RED_TASK_2
def test_a_nullable_addition_alters_then_appends_then_stamps(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))
    declared = contract("base_plus_nullable")

    result = write(tmp_path, session, declared, PLUS_NOTE)

    assert _effects(session) == ["add_columns", "insert_into", "stamp"]
    assert "ADD COLUMNS (`note` string)" in session.statements[
        _effects_index(session, "add_columns")
    ]
    assert projection_of(inserts(session)[0]).endswith("`ingestion_date`, `note`")
    assert _metadata(result)["schema_evolution"] == "added:note"
    assert session.stamp == stamp_of(declared)


@RED_TASK_2
def test_a_backward_promotion_alters_the_type_and_keeps_history_on_a_full_refresh(tmp_path):
    session = ContractAwareSession(
        stamp=stamp_of(contract("base_int")),
        target_columns=INT_TABLE,
        conf_values={PARTITION_OVERWRITE_MODE: "dynamic"},
    )
    declared = contract("base_backward")

    result = write(tmp_path, session, declared, BASE_TABLE, intent=REPLACE_INTENT)

    assert _effects(session) == ["alter_type", "insert_overwrite", "stamp"]
    assert "ALTER COLUMN `amount` TYPE bigint" in session.statements[
        _effects_index(session, "alter_type")
    ]
    metadata = _metadata(result)
    assert metadata["schema_evolution"] == "promoted:amount(integer->long)"
    assert metadata["overwrite_mechanism"] == "insert_overwrite"
    assert "history_reset_reason" not in metadata
    assert session.conf.as_dict()[PARTITION_OVERWRITE_MODE] == "dynamic"


@RED_TASK_2
@pytest.mark.parametrize(
    "intent", [INSERT_INTENT, REPLACE_INTENT], ids=["append", "full_refresh"]
)
def test_a_promotion_an_additive_contract_refuses_raises_before_anything_writes(tmp_path, intent):
    session = ContractAwareSession(stamp=stamp_of(contract("base_int")), target_columns=INT_TABLE)

    with pytest.raises(Exception) as raised:
        write(tmp_path, session, contract("base"), BASE_TABLE, intent=intent)

    assert type(raised.value).__name__ == "SchemaEvolutionRefusedError"
    assert raised.value.failure_stage == "schema_evolution"
    assert "amount" in str(raised.value)
    assert _effects(session) == []


@RED_TASK_2
def test_a_major_bump_on_a_full_refresh_replaces_the_table_and_says_why(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))
    declared = contract("base_v2")

    result = write(tmp_path, session, declared, V2_TABLE, intent=REPLACE_INTENT)

    assert _effects(session) == ["replace_table", "stamp"]
    metadata = _metadata(result)
    assert metadata["overwrite_mechanism"] == "replace_table"
    assert metadata["history_reset_reason"].startswith("contract major version 1 -> 2")
    assert metadata["schema_evolution"] == "breaking_replace"
    assert session.stamp == stamp_of(declared)


@RED_TASK_2
def test_a_major_bump_on_an_append_is_refused(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))

    with pytest.raises(Exception) as raised:
        write(tmp_path, session, contract("base_v2"), V2_TABLE, intent=INSERT_INTENT)

    assert type(raised.value).__name__ == "SchemaEvolutionRefusedError"
    assert not any(kind in _effects(session) for kind in ("replace_table", "insert_into"))


@RED_TASK_2
def test_a_matching_table_with_an_equal_stamp_is_not_restamped(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))

    result = write(tmp_path, session, declared, BASE_TABLE)

    assert "stamp" not in _effects(session)
    assert any(statement.startswith("SHOW TBLPROPERTIES") for statement in session.statements)
    assert _metadata(result)["schema_evolution"] == "none"


@RED_TASK_2
def test_a_first_write_creates_the_table_then_stamps_it(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(table_exists=False)

    result = write(tmp_path, session, declared, BASE_TABLE)

    assert _effects(session) == ["create_table", "stamp"]
    assert _metadata(result)["schema_evolution"] == "none"
    assert session.stamp == stamp_of(declared)


@RED_TASK_2
def test_a_plan_without_a_contract_is_refused_by_the_writer(tmp_path):
    session = ContractAwareSession()
    writer = SparkDatasetWriter(StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path))

    with pytest.raises(Exception) as raised:
        writer.write(
            TypedFakeDataFrame(session, BASE_TABLE),
            _plan(tmp_path),
            "bronze",
            intent=INSERT_INTENT,
        )

    assert type(raised.value).__name__ == "MissingContractError"
    assert _effects(session) == []


# ── helpers ──────────────────────────────────────────────────────────────────


def _effects_index(session: ContractAwareSession, kind: str) -> int:
    return next(
        index for index, statement in enumerate(session.statements) if _kind(statement) == kind
    )


def test_the_recorder_applies_the_ddl_it_records(tmp_path):
    """Not red: the fake itself must model an ALTER, or the cases above prove nothing."""
    session = ContractAwareSession()

    session.sql("ALTER TABLE `t` ADD COLUMNS (`note` string, `price` decimal(18,2))")
    session.sql("ALTER TABLE `t` ALTER COLUMN `amount` TYPE bigint")
    session.sql(
        "ALTER TABLE `t` SET TBLPROPERTIES ('janus.contract_id' = 'o''brien', "
        "'janus.contract_version' = '2.0.0')"
    )
    stamp = session.sql("SHOW TBLPROPERTIES `t`").collect()

    assert session.target_columns[-2:] == (("note", "string"), ("price", "decimal(18,2)"))
    assert dict(session.target_columns)["amount"] == "bigint"
    assert {row["key"]: row.value for row in stamp} == {
        "janus.contract_id": "o'brien",
        "janus.contract_version": "2.0.0",
    }
    assert [field.dataType.jsonValue() for field in session.table("t").schema.fields][:3] == [
        "string",
        "string",
        "long",
    ]
