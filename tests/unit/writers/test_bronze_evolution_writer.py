"""The writer executes the evolution plan with Iceberg-native statements, then stamps the table."""

from __future__ import annotations

from dataclasses import replace
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

INT_TABLE = tuple((name, "int" if name == "amount" else spelled) for name, spelled in BASE_TABLE)
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


def test_a_nullable_addition_alters_then_appends_then_stamps(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))
    declared = contract("base_plus_nullable")

    result = write(tmp_path, session, declared, PLUS_NOTE)

    assert _effects(session) == ["add_columns", "insert_into", "stamp"]
    assert (
        "ADD COLUMNS (`note` string)" in session.statements[_effects_index(session, "add_columns")]
    )
    assert projection_of(inserts(session)[0]).endswith("`ingestion_date`, `note`")
    assert _metadata(result)["schema_evolution"] == "added:note"
    assert session.stamp == stamp_of(declared)


def test_a_backward_promotion_alters_the_type_and_keeps_history_on_a_full_refresh(tmp_path):
    session = ContractAwareSession(
        stamp=stamp_of(contract("base_int")),
        target_columns=INT_TABLE,
        conf_values={PARTITION_OVERWRITE_MODE: "dynamic"},
    )
    declared = contract("base_backward")

    result = write(tmp_path, session, declared, BASE_TABLE, intent=REPLACE_INTENT)

    assert _effects(session) == ["alter_type", "insert_overwrite", "stamp"]
    assert (
        "ALTER COLUMN `amount` TYPE bigint"
        in session.statements[_effects_index(session, "alter_type")]
    )
    metadata = _metadata(result)
    assert metadata["schema_evolution"] == "promoted:amount(integer->long)"
    assert metadata["overwrite_mechanism"] == "insert_overwrite"
    assert "history_reset_reason" not in metadata
    assert session.conf.as_dict()[PARTITION_OVERWRITE_MODE] == "dynamic"


@pytest.mark.parametrize("intent", [INSERT_INTENT, REPLACE_INTENT], ids=["append", "full_refresh"])
def test_a_promotion_an_additive_contract_refuses_raises_before_anything_writes(tmp_path, intent):
    session = ContractAwareSession(stamp=stamp_of(contract("base_int")), target_columns=INT_TABLE)

    with pytest.raises(Exception) as raised:
        write(tmp_path, session, contract("base"), BASE_TABLE, intent=intent)

    assert type(raised.value).__name__ == "SchemaEvolutionRefusedError"
    assert raised.value.failure_stage == "schema_evolution"
    assert "amount" in str(raised.value)
    assert _effects(session) == []


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


def test_partition_drift_restamps_after_replace_even_if_the_old_stamp_matched(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared), target_partitions=("id",))

    result = write(tmp_path, session, declared, BASE_TABLE, intent=REPLACE_INTENT)

    assert _effects(session) == ["replace_table", "stamp"]
    assert result.metadata_as_dict()["history_reset_reason"].startswith("partition spec changed")


def test_a_major_bump_on_an_append_is_refused(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))

    with pytest.raises(Exception) as raised:
        write(tmp_path, session, contract("base_v2"), V2_TABLE, intent=INSERT_INTENT)

    assert type(raised.value).__name__ == "SchemaEvolutionRefusedError"
    assert not any(kind in _effects(session) for kind in ("replace_table", "insert_into"))


def test_a_matching_table_with_an_equal_stamp_is_not_restamped(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(stamp=stamp_of(declared))

    result = write(tmp_path, session, declared, BASE_TABLE)

    assert "stamp" not in _effects(session)
    assert any(statement.startswith("SHOW TBLPROPERTIES") for statement in session.statements)
    assert _metadata(result)["schema_evolution"] == "none"


def test_a_first_write_creates_the_table_then_stamps_it(tmp_path):
    declared = contract("base")
    session = ContractAwareSession(table_exists=False)

    result = write(tmp_path, session, declared, BASE_TABLE)

    assert _effects(session) == ["create_table", "stamp"]
    assert _metadata(result)["schema_evolution"] == "none"
    assert session.stamp == stamp_of(declared)


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


def test_addition_and_promotion_apply_in_order_before_the_append(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base_int")), target_columns=INT_TABLE)
    declared = contract("base_plus_nullable")
    declared = replace(declared, janus=replace(declared.janus, compatibility="backward"))

    result = write(tmp_path, session, declared, PLUS_NOTE)

    assert _effects(session) == ["add_columns", "alter_type", "insert_into", "stamp"]
    assert _metadata(result)["schema_evolution"] == ("added:note;promoted:amount(integer->long)")


def test_new_nullable_column_can_be_projected_when_the_batch_omits_it(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))

    result = write(tmp_path, session, contract("base_plus_nullable"), BASE_TABLE)

    assert _effects(session) == ["add_columns", "insert_into", "stamp"]
    assert projection_of(inserts(session)[0]).endswith("NULL AS `note`")
    assert _metadata(result)["schema_evolution"] == "added:note"


def test_a_changed_contract_stamp_is_set_after_a_matching_write(tmp_path):
    session = ContractAwareSession(stamp=stamp_of(contract("base")))
    declared = contract("base_backward")

    result = write(tmp_path, session, declared, BASE_TABLE)

    assert _effects(session) == ["insert_into", "stamp"]
    assert _metadata(result)["schema_evolution"] == "none"
    assert session.stamp == stamp_of(declared)


def test_failed_bronze_statement_does_not_stamp_the_contract(tmp_path):
    original = stamp_of(contract("base"))
    session = ContractAwareSession(stamp=original, failing_statement="INSERT INTO")

    with pytest.raises(RuntimeError, match="commit failed"):
        write(tmp_path, session, contract("base_backward"), BASE_TABLE)

    assert _effects(session) == ["insert_into"]
    assert session.stamp == original


def test_a_path_based_bronze_write_without_a_contract_is_refused(tmp_path):
    session = ContractAwareSession()
    writer = SparkDatasetWriter(StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path))

    with pytest.raises(Exception) as raised:
        writer.write(
            TypedFakeDataFrame(session, BASE_TABLE),
            _plan(tmp_path),
            "bronze",
            format_name="parquet",
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
