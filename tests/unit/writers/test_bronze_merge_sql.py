"""Spark-free coverage of the bronze upsert SQL builders.

The SQL that makes an incremental bronze write idempotent is built by pure functions
(strings in, string out) so it is covered by the fast unit gate rather than only inside a
live Spark session. These tests pin the exact rendered SQL, the null-safe equality, the
identifier quoting on the table, the temp view and *every* key column, and the injection
escaping of a backtick in a key name.
"""

from __future__ import annotations

import pytest

from janus.writers.spark import build_add_columns_sql, build_merge_sql


def test_build_merge_sql_single_key_is_null_safe_and_fully_quoted():
    sql = build_merge_sql(
        table_identifier="bronze_test.incremental_upsert_fixture",
        source_view="janus_bronze_fixture_abc",
        merge_keys=["event_id"],
    )

    assert sql == (
        "MERGE INTO `bronze_test`.`incremental_upsert_fixture` AS janus_target\n"
        "USING `janus_bronze_fixture_abc` AS janus_source\n"
        "ON janus_target.`event_id` <=> janus_source.`event_id`\n"
        "WHEN MATCHED THEN UPDATE SET *\n"
        "WHEN NOT MATCHED THEN INSERT *"
    )


def test_build_merge_sql_composite_key_joins_conditions_with_null_safe_equality():
    sql = build_merge_sql(
        table_identifier="ns.tbl",
        source_view="janus_bronze_v",
        merge_keys=["k1", "k2"],
    )

    assert sql == (
        "MERGE INTO `ns`.`tbl` AS janus_target\n"
        "USING `janus_bronze_v` AS janus_source\n"
        "ON janus_target.`k1` <=> janus_source.`k1`\n"
        "   AND janus_target.`k2` <=> janus_source.`k2`\n"
        "WHEN MATCHED THEN UPDATE SET *\n"
        "WHEN NOT MATCHED THEN INSERT *"
    )


def test_build_merge_sql_escapes_backtick_in_key_name():
    sql = build_merge_sql(
        table_identifier="ns.tbl",
        source_view="v",
        merge_keys=["weird`col"],
    )

    assert "janus_target.`weird``col` <=> janus_source.`weird``col`" in sql
    # The escaped backtick must not break out of the quoted identifier.
    assert "`weird`col`" not in sql


def test_build_merge_sql_rejects_empty_keys():
    with pytest.raises(ValueError, match="at least one merge key"):
        build_merge_sql(table_identifier="ns.tbl", source_view="v", merge_keys=[])


def test_build_add_columns_sql_quotes_names_and_emits_types():
    sql = build_add_columns_sql(
        table_identifier="ns.tbl",
        columns=[("new_col", "string"), ("amount", "bigint")],
    )

    assert sql == "ALTER TABLE `ns`.`tbl` ADD COLUMNS (`new_col` string, `amount` bigint)"


def test_build_add_columns_sql_escapes_backtick_in_column_name():
    sql = build_add_columns_sql(
        table_identifier="ns.tbl",
        columns=[("weird`col", "string")],
    )

    assert sql == "ALTER TABLE `ns`.`tbl` ADD COLUMNS (`weird``col` string)"


def test_build_add_columns_sql_returns_none_for_empty_columns():
    assert build_add_columns_sql(table_identifier="ns.tbl", columns=[]) is None
