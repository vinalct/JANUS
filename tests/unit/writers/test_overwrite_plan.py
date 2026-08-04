"""Spark-free coverage of the full-refresh overwrite planner and its SQL builders.

The decision of *how* a full refresh replaces bronze — ``INSERT OVERWRITE`` when the target
can accept the batch, ``REPLACE TABLE`` when the table's shape genuinely changed — is a pure
function of two schemas and two partition specs, so it is covered by the fast unit gate rather
than only inside a live Spark session. Nothing here imports ``pyspark``.

The last test is the cheapest possible proof that extracting the two pre-existing statements
into builders changed nothing: it compares them against the literal f-strings they replaced,
double space and all.
"""

from __future__ import annotations

import inspect

import pytest

from janus.writers.overwrite import (
    FullRefreshOverwritePlan,
    build_create_table_as_select_sql,
    build_insert_overwrite_sql,
    build_replace_table_as_select_sql,
    plan_full_refresh_overwrite,
)

# A normalized bronze batch, trimmed to what the planner actually compares.
BASE_COLUMNS = (
    ("janus_run_id", "string"),
    ("event_id", "string"),
    ("amount", "bigint"),
    ("ingestion_date", "date"),
)
PARTITIONS = ("ingestion_date",)


def test_identical_schema_and_partitions_uses_insert_overwrite():
    plan = plan_full_refresh_overwrite(
        source_columns=BASE_COLUMNS,
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "insert_overwrite"
    assert plan.preserves_history is True
    assert plan.add_columns == ()
    assert plan.projection == tuple(name for name, _ in BASE_COLUMNS)
    assert plan.reason == "schema and partition spec unchanged; table history retained"


def test_added_column_is_reconciled_and_appended_to_the_projection():
    plan = plan_full_refresh_overwrite(
        source_columns=(*BASE_COLUMNS, ("c", "string")),
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "insert_overwrite"
    assert plan.add_columns == (("c", "string"),)
    # ALTER TABLE ... ADD COLUMNS appends, so the new column lands last in the table too.
    assert plan.projection == (*(name for name, _ in BASE_COLUMNS), "c")
    assert plan.reason == "columns added to the target: c; table history retained"


def test_several_added_columns_keep_source_order():
    plan = plan_full_refresh_overwrite(
        source_columns=(*BASE_COLUMNS, ("c", "string"), ("d", "int")),
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=PARTITIONS,
    )

    assert plan.add_columns == (("c", "string"), ("d", "int"))
    assert plan.projection[-2:] == ("c", "d")
    assert plan.reason == "columns added to the target: c, d; table history retained"


def test_added_columns_are_not_gated_on_schema_evolution():
    """A full refresh has always accepted new columns; gating them would be a regression.

    ``REPLACE TABLE`` recreated the bronze schema on every run, so an added column has never
    needed ``quality.allow_schema_evolution``. The merge path gates them because a MERGE
    against an older table is a genuine incremental-semantics question — this is not that.
    The planner therefore takes no evolution flag at all, and cannot grow one silently.
    """
    parameters = set(inspect.signature(plan_full_refresh_overwrite).parameters)

    assert parameters == {
        "source_columns",
        "target_columns",
        "configured_partitions",
        "target_partitions",
    }
    assert not any("evolution" in name or "quality" in name for name in parameters)


def test_dropped_column_falls_back_to_replace_table():
    plan = plan_full_refresh_overwrite(
        source_columns=tuple(c for c in BASE_COLUMNS if c[0] != "amount"),
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "replace_table"
    assert plan.preserves_history is False
    assert plan.projection == ()
    assert plan.reason == "columns removed from the source schema: amount"


def test_type_change_falls_back_to_replace_table():
    plan = plan_full_refresh_overwrite(
        source_columns=tuple(
            (name, "string" if name == "amount" else column_type)
            for name, column_type in BASE_COLUMNS
        ),
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "replace_table"
    assert plan.reason == "column type changed: amount bigint -> string"


def test_partition_spec_change_falls_back_to_replace_table():
    plan = plan_full_refresh_overwrite(
        source_columns=BASE_COLUMNS,
        target_columns=BASE_COLUMNS,
        configured_partitions=("ingestion_date", "janus_run_id"),
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "replace_table"
    assert plan.reason == (
        "partition spec changed: ingestion_date -> ingestion_date, janus_run_id"
    )


def test_partition_spec_change_to_unpartitioned_names_both_specs():
    plan = plan_full_refresh_overwrite(
        source_columns=BASE_COLUMNS,
        target_columns=BASE_COLUMNS,
        configured_partitions=(),
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "replace_table"
    assert plan.reason == "partition spec changed: ingestion_date -> (none)"


def test_unpartitioned_on_both_sides_is_not_drift():
    plan = plan_full_refresh_overwrite(
        source_columns=BASE_COLUMNS,
        target_columns=BASE_COLUMNS,
        configured_partitions=(),
        target_partitions=(),
    )

    assert plan.mechanism == "insert_overwrite"


def test_unreadable_partition_spec_falls_back_to_replace_table():
    plan = plan_full_refresh_overwrite(
        source_columns=BASE_COLUMNS,
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=None,
    )

    assert plan.mechanism == "replace_table"
    assert plan.reason == "target partition spec could not be read"


def test_column_order_difference_alone_still_uses_insert_overwrite():
    plan = plan_full_refresh_overwrite(
        source_columns=tuple(reversed(BASE_COLUMNS)),
        target_columns=BASE_COLUMNS,
        configured_partitions=PARTITIONS,
        target_partitions=PARTITIONS,
    )

    assert plan.mechanism == "insert_overwrite"
    assert plan.add_columns == ()
    # The explicit projection is in *target* order, which is what absorbs the reordering.
    assert plan.projection == tuple(name for name, _ in BASE_COLUMNS)


def test_case_difference_in_a_column_name_is_treated_as_drift():
    """Iceberg column names round-trip exactly, so lower-casing would merge distinct columns."""
    plan = plan_full_refresh_overwrite(
        source_columns=(("Event_Id", "string"),),
        target_columns=(("event_id", "string"),),
        configured_partitions=(),
        target_partitions=(),
    )

    assert plan.mechanism == "replace_table"
    assert plan.reason == "columns removed from the source schema: event_id"


def test_planner_is_total_for_an_empty_source_schema():
    plan = plan_full_refresh_overwrite(
        source_columns=(),
        target_columns=(),
        configured_partitions=(),
        target_partitions=(),
    )

    assert plan.mechanism == "replace_table"
    assert plan.reason == "source schema is empty; there is nothing to project"


def test_plan_rejects_inconsistent_construction():
    with pytest.raises(ValueError, match="mechanism must be one of"):
        FullRefreshOverwritePlan(mechanism="truncate")

    with pytest.raises(ValueError, match="explicit column projection"):
        FullRefreshOverwritePlan(mechanism="insert_overwrite")

    with pytest.raises(ValueError, match="takes no projection"):
        FullRefreshOverwritePlan(mechanism="replace_table", projection=("a",))

    with pytest.raises(ValueError, match="takes no projection"):
        FullRefreshOverwritePlan(mechanism="replace_table", add_columns=(("a", "string"),))


def test_insert_overwrite_sql_quotes_every_identifier():
    sql = build_insert_overwrite_sql(
        table_identifier="bronze_test.events",
        source_view="janus_bronze_fixture_abc",
        projection=("col_a", "col_b", "weird`col"),
    )

    assert sql == (
        "INSERT OVERWRITE `bronze_test`.`events`\n"
        "SELECT `col_a`, `col_b`, `weird``col` FROM `janus_bronze_fixture_abc`"
    )
    # No PARTITION clause: under static mode this replaces every partition of the table.
    assert "PARTITION (" not in sql
    assert "REPLACE TABLE" not in sql


def test_insert_overwrite_sql_rejects_an_empty_projection():
    with pytest.raises(ValueError, match="at least one projected column"):
        build_insert_overwrite_sql(
            table_identifier="ns.tbl", source_view="v", projection=()
        )


def test_create_and_replace_sql_match_the_previous_literals():
    """Byte-identical to the f-strings these builders replaced, including the double space."""
    partitioned = dict(
        table_identifier="bronze_test.events",
        source_view="janus_bronze_fixture_abc",
        partition_columns=("ingestion_date",),
    )
    unpartitioned = dict(partitioned, partition_columns=())

    assert build_create_table_as_select_sql(**partitioned) == (
        "CREATE TABLE `bronze_test`.`events` USING iceberg "
        "PARTITIONED BY (`ingestion_date`) AS SELECT * FROM `janus_bronze_fixture_abc`"
    )
    assert build_create_table_as_select_sql(**unpartitioned) == (
        "CREATE TABLE `bronze_test`.`events` USING iceberg "
        " AS SELECT * FROM `janus_bronze_fixture_abc`"
    )
    assert build_replace_table_as_select_sql(**partitioned) == (
        "REPLACE TABLE `bronze_test`.`events` USING iceberg "
        "PARTITIONED BY (`ingestion_date`) AS SELECT * FROM `janus_bronze_fixture_abc`"
    )
    assert build_replace_table_as_select_sql(**unpartitioned) == (
        "REPLACE TABLE `bronze_test`.`events` USING iceberg "
        " AS SELECT * FROM `janus_bronze_fixture_abc`"
    )


def test_no_pyspark_import_in_the_pure_modules():
    """The planner and the builders must stay runnable on a host without PySpark."""
    from janus.writers import identifiers, overwrite

    for module in (identifiers, overwrite):
        source = inspect.getsource(module)
        assert "pyspark" not in source, f"{module.__name__} must not import pyspark"
