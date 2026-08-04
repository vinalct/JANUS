"""Real-Iceberg proof that history-preserving full refresh still replaces rows."""

from __future__ import annotations

from janus.models import resolve_bronze_write_intent

PARTITIONED_SOURCE_ID = "full_refresh_history_partitioned"
UNPARTITIONED_SOURCE_ID = "full_refresh_history_unpartitioned"
PARTITION_OVERWRITE_MODE = "spark.sql.sources.partitionOverwriteMode"
PARTITIONED_SCHEMA = "id string, value string, partition_key string"
UNPARTITIONED_SCHEMA = "id string, value string"


def test_full_refresh_replaces_all_rows(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        PARTITIONED_SOURCE_ID,
        run_id="run-replace-all-001",
    )
    full_refresh_harness.write_rows(
        [("old-1", "alpha", "A"), ("old-2", "beta", "B"), ("old-3", "gamma", "C")],
        PARTITIONED_SCHEMA,
        plan_one,
    )

    run_two_rows = [("new-1", "delta", "A"), ("new-2", "epsilon", "B")]
    plan_two = full_refresh_harness.plan(
        PARTITIONED_SOURCE_ID,
        run_id="run-replace-all-002",
    )
    result = full_refresh_harness.write_rows(run_two_rows, PARTITIONED_SCHEMA, plan_two)

    assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"
    assert _table_rows(
        full_refresh_harness.spark,
        result.path,
        ("id", "value", "partition_key"),
    ) == sorted(run_two_rows)


def test_partitioned_full_refresh_drops_stale_partitions(full_refresh_harness):
    spark = full_refresh_harness.spark
    previous_mode = spark.conf.get(PARTITION_OVERWRITE_MODE, None)
    spark.conf.set(PARTITION_OVERWRITE_MODE, "dynamic")
    try:
        plan_one = full_refresh_harness.plan(
            PARTITIONED_SOURCE_ID,
            run_id="run-dynamic-partitions-001",
        )
        full_refresh_harness.write_rows(
            [
                ("old-a", "alpha", "A"),
                ("old-b", "beta", "B"),
                ("old-c", "gamma", "C"),
            ],
            PARTITIONED_SCHEMA,
            plan_one,
        )

        plan_two = full_refresh_harness.plan(
            PARTITIONED_SOURCE_ID,
            run_id="run-dynamic-partitions-002",
        )
        result = full_refresh_harness.write_rows(
            [("new-a", "delta", "A")],
            PARTITIONED_SCHEMA,
            plan_two,
        )

        assert spark.conf.get(PARTITION_OVERWRITE_MODE) == "dynamic"
        assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"
        assert _table_rows(spark, result.path, ("id", "value", "partition_key")) == [
            ("new-a", "delta", "A")
        ]
        assert {
            row["partition_key"]
            for row in spark.table(result.path).select("partition_key").distinct().collect()
        } == {"A"}
    finally:
        if previous_mode is None:
            spark.conf.unset(PARTITION_OVERWRITE_MODE)
        else:
            spark.conf.set(PARTITION_OVERWRITE_MODE, previous_mode)


def test_unpartitioned_full_refresh_replaces_all_rows(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-unpartitioned-001",
    )
    full_refresh_harness.write_rows(
        [("old-1", "alpha"), ("old-2", "beta"), ("old-3", "gamma")],
        UNPARTITIONED_SCHEMA,
        plan_one,
    )

    run_two_rows = [("new-1", "delta"), ("new-2", "epsilon")]
    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-unpartitioned-002",
    )
    result = full_refresh_harness.write_rows(run_two_rows, UNPARTITIONED_SCHEMA, plan_two)

    assert result.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"
    assert _table_rows(full_refresh_harness.spark, result.path, ("id", "value")) == sorted(
        run_two_rows
    )


def test_schema_and_partition_spec_unchanged_after_overwrite(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        PARTITIONED_SOURCE_ID,
        run_id="run-shape-stability-001",
    )
    first = full_refresh_harness.write_rows(
        [("old-1", "alpha", "A"), ("old-2", "beta", "B")],
        PARTITIONED_SCHEMA,
        plan_one,
    )
    first_schema = _table_schema(full_refresh_harness.spark, first.path)
    first_partition_spec = _partition_spec(full_refresh_harness.spark, first.path)

    assert first_schema == (
        ("id", "string"),
        ("value", "string"),
        ("partition_key", "string"),
    )
    assert first_partition_spec == (("partition_key", "string"),)
    plan_two = full_refresh_harness.plan(
        PARTITIONED_SOURCE_ID,
        run_id="run-shape-stability-002",
    )
    full_refresh_harness.write_rows(
        [("new-1", "gamma", "A")],
        PARTITIONED_SCHEMA,
        plan_two,
    )

    assert _table_schema(full_refresh_harness.spark, first.path) == first_schema
    assert _partition_spec(full_refresh_harness.spark, first.path) == first_partition_spec


def test_multi_batch_handoff_overwrites_once_then_appends(full_refresh_harness):
    seed_plan = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-multi-batch-seed",
    )
    full_refresh_harness.write_rows(
        [("stale", "old")],
        UNPARTITIONED_SCHEMA,
        seed_plan,
    )

    batch_plan = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-multi-batch-001",
    )
    run_intent = resolve_bronze_write_intent(batch_plan)
    first_batch = full_refresh_harness.write_rows(
        [("batch-1", "alpha")],
        UNPARTITIONED_SCHEMA,
        batch_plan,
        intent=run_intent.for_batch(1),
    )
    second_batch = full_refresh_harness.write_rows(
        [("batch-2", "beta")],
        UNPARTITIONED_SCHEMA,
        batch_plan,
        intent=run_intent.for_batch(2),
    )

    assert first_batch.metadata_as_dict()["overwrite_mechanism"] == "insert_overwrite"
    assert second_batch.mode == "append"
    assert _table_rows(
        full_refresh_harness.spark,
        second_batch.path,
        ("id", "value"),
    ) == [("batch-1", "alpha"), ("batch-2", "beta")]


def test_dropped_column_falls_back_and_records_the_reason(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-dropped-column-001",
    )
    full_refresh_harness.write_rows(
        [("old-1", "alpha", "legacy")],
        "id string, value string, legacy string",
        plan_one,
    )

    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-dropped-column-002",
    )
    result = full_refresh_harness.write_rows(
        [("new-1", "beta")],
        UNPARTITIONED_SCHEMA,
        plan_two,
    )

    metadata = result.metadata_as_dict()
    assert metadata["overwrite_mechanism"] == "replace_table"
    assert "legacy" in metadata["history_reset_reason"]
    assert _table_schema(full_refresh_harness.spark, result.path) == (
        ("id", "string"),
        ("value", "string"),
    )
    assert _table_rows(full_refresh_harness.spark, result.path, ("id", "value")) == [
        ("new-1", "beta")
    ]


def test_added_column_preserves_history(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-added-column-001",
    )
    first = full_refresh_harness.write_rows(
        [("old-1", "alpha")],
        UNPARTITIONED_SCHEMA,
        plan_one,
    )
    first_snapshot_id = _only_snapshot_id(full_refresh_harness.spark, first.path)

    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-added-column-002",
    )
    run_two_rows = [
        ("old-1", "alpha", None),
        ("new-1", "beta", "extra-value"),
    ]
    result = full_refresh_harness.write_rows(
        run_two_rows,
        "id string, value string, extra string",
        plan_two,
    )

    metadata = result.metadata_as_dict()
    assert metadata["overwrite_mechanism"] == "insert_overwrite"
    assert "history_reset_reason" not in metadata
    snapshot_ids = _snapshot_ids(full_refresh_harness.spark, result.path)
    assert len(snapshot_ids) >= 2
    assert first_snapshot_id in snapshot_ids
    assert len(_history_snapshot_ids(full_refresh_harness.spark, result.path)) >= 2
    assert _table_schema(full_refresh_harness.spark, result.path) == (
        ("id", "string"),
        ("value", "string"),
        ("extra", "string"),
    )
    assert _table_rows(
        full_refresh_harness.spark,
        result.path,
        ("id", "value", "extra"),
    ) == sorted(run_two_rows)

    prior_rows = _read_snapshot(
        full_refresh_harness.spark,
        result.path,
        first_snapshot_id,
    ).select("id", "value")
    assert _collected_rows(prior_rows, ("id", "value")) == [("old-1", "alpha")]


def test_partition_spec_change_falls_back(full_refresh_harness):
    schema = "id string, value string, partition_a string, partition_b string"
    plan_one = full_refresh_harness.plan(
        PARTITIONED_SOURCE_ID,
        run_id="run-partition-change-001",
        partition_by=("partition_a",),
    )
    first = full_refresh_harness.write_rows(
        [("old-1", "alpha", "A", "X")],
        schema,
        plan_one,
    )
    assert _active_partition_fields(full_refresh_harness.spark, first.path) == {
        "partition_a"
    }

    plan_two = full_refresh_harness.plan(
        PARTITIONED_SOURCE_ID,
        run_id="run-partition-change-002",
        partition_by=("partition_b",),
    )
    result = full_refresh_harness.write_rows(
        [("new-1", "beta", "B", "Y")],
        schema,
        plan_two,
    )

    metadata = result.metadata_as_dict()
    assert metadata["overwrite_mechanism"] == "replace_table"
    assert "partition spec changed" in metadata["history_reset_reason"]
    assert _active_partition_fields(full_refresh_harness.spark, result.path) == {
        "partition_b"
    }
    assert _table_rows(
        full_refresh_harness.spark,
        result.path,
        ("id", "value", "partition_a", "partition_b"),
    ) == [("new-1", "beta", "B", "Y")]


def _table_rows(
    spark,
    table_identifier: str,
    columns: tuple[str, ...],
) -> list[tuple]:
    dataframe = spark.table(table_identifier).select(*columns)
    return _collected_rows(dataframe, columns)


def _collected_rows(dataframe, columns: tuple[str, ...]) -> list[tuple]:
    return sorted(tuple(row[column] for column in columns) for row in dataframe.collect())


def _table_schema(spark, table_identifier: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field.name, field.dataType.simpleString())
        for field in spark.table(table_identifier).schema.fields
    )


def _partition_spec(spark, table_identifier: str) -> tuple[tuple[str, str], ...]:
    partition_fields = spark.table(f"{table_identifier}.partitions").schema.fields
    partition_struct = next(
        (field for field in partition_fields if field.name == "partition"),
        None,
    )
    if partition_struct is None:
        return ()
    return tuple(
        (field.name, field.dataType.simpleString())
        for field in partition_struct.dataType.fields
    )


def _active_partition_fields(spark, table_identifier: str) -> set[str]:
    active_fields: set[str] = set()
    partition_rows = (
        spark.table(f"{table_identifier}.partitions")
        .select("partition")
        .collect()
    )
    for row in partition_rows:
        partition = row["partition"]
        if partition is None:
            continue
        active_fields.update(
            name for name, value in partition.asDict().items() if value is not None
        )
    return active_fields


def _only_snapshot_id(spark, table_identifier: str) -> int:
    snapshot_ids = _snapshot_ids(spark, table_identifier)
    assert len(snapshot_ids) == 1
    return next(iter(snapshot_ids))


def _snapshot_ids(spark, table_identifier: str) -> set[int]:
    return {
        row["snapshot_id"]
        for row in spark.table(f"{table_identifier}.snapshots")
        .select("snapshot_id")
        .collect()
    }


def _history_snapshot_ids(spark, table_identifier: str) -> set[int]:
    return {
        row["snapshot_id"]
        for row in spark.table(f"{table_identifier}.history")
        .select("snapshot_id")
        .collect()
    }


def _read_snapshot(spark, table_identifier: str, snapshot_id: int):
    return (
        spark.read.format("iceberg")
        .option("snapshot-id", str(snapshot_id))
        .load(table_identifier)
    )
