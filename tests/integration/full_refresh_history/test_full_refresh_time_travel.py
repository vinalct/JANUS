"""Real-Iceberg proof that full refresh preserves history and time travel."""

from __future__ import annotations

UNPARTITIONED_SOURCE_ID = "full_refresh_history_unpartitioned"


def test_full_refresh_retains_previous_snapshot(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-history-retention-001",
    )
    first = full_refresh_harness.write_rows(
        [("old-1", "alpha"), ("old-2", "beta")],
        "id string, value string",
        plan_one,
    )
    first_snapshot_id = _only_snapshot_id(full_refresh_harness.spark, first.path)

    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-history-retention-002",
    )
    second = full_refresh_harness.write_rows(
        [("new-1", "gamma"), ("new-2", "delta")],
        "id string, value string",
        plan_two,
    )

    snapshot_rows = _snapshot_rows(full_refresh_harness.spark, second.path)
    snapshot_ids = set(snapshot_rows)
    history_rows = _history_rows(full_refresh_harness.spark, second.path)
    history_ids = set(history_rows)
    assert len(snapshot_ids) >= 2
    assert len(history_ids) >= 2
    assert first_snapshot_id in snapshot_ids
    assert first_snapshot_id in history_ids
    assert snapshot_ids.issubset(history_ids)
    second_snapshot_ids = snapshot_ids - {first_snapshot_id}
    assert len(second_snapshot_ids) == 1
    second_snapshot_id = next(iter(second_snapshot_ids))
    assert snapshot_rows[second_snapshot_id] == first_snapshot_id
    assert history_rows[first_snapshot_id]
    assert history_rows[second_snapshot_id]


def test_time_travel_reads_previous_run_rows(full_refresh_harness):
    run_one_rows = [("old-1", "alpha"), ("old-2", "beta")]
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-time-travel-001",
    )
    first = full_refresh_harness.write_rows(
        run_one_rows,
        "id string, value string",
        plan_one,
    )
    first_snapshot_id = _only_snapshot_id(full_refresh_harness.spark, first.path)

    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-time-travel-002",
    )
    full_refresh_harness.write_rows(
        [("new-1", "gamma"), ("new-2", "delta")],
        "id string, value string",
        plan_two,
    )

    previous_run = _read_snapshot(
        full_refresh_harness.spark,
        first.path,
        first_snapshot_id,
    ).select("id", "value")
    assert _collected_rows(previous_run, ("id", "value")) == sorted(run_one_rows)


def test_table_identity_is_stable_across_runs(full_refresh_harness):
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-table-identity-001",
    )
    first = full_refresh_harness.write_rows(
        [("old-1", "alpha")],
        "id string, value string",
        plan_one,
    )
    first_snapshot_id = _only_snapshot_id(full_refresh_harness.spark, first.path)

    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-table-identity-002",
    )
    second = full_refresh_harness.write_rows(
        [("new-1", "beta")],
        "id string, value string",
        plan_two,
    )

    snapshot_rows = _snapshot_rows(full_refresh_harness.spark, second.path)
    second_snapshot_ids = set(snapshot_rows) - {first_snapshot_id}
    assert len(second_snapshot_ids) == 1
    second_snapshot_id = next(iter(second_snapshot_ids))
    assert snapshot_rows[second_snapshot_id] == first_snapshot_id

    prior_metadata_snapshot_ids = {
        row["latest_snapshot_id"]
        for row in full_refresh_harness.spark.table(
            f"{first.path}.metadata_log_entries"
        )
        .select("latest_snapshot_id")
        .collect()
    }
    assert first_snapshot_id in prior_metadata_snapshot_ids


def test_rollback_to_previous_run_restores_prior_bronze(full_refresh_harness):
    run_one_rows = [("old-1", "alpha"), ("old-2", "beta")]
    plan_one = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-rollback-001",
    )
    first = full_refresh_harness.write_rows(
        run_one_rows,
        "id string, value string",
        plan_one,
    )
    first_snapshot_id = _only_snapshot_id(full_refresh_harness.spark, first.path)

    plan_two = full_refresh_harness.plan(
        UNPARTITIONED_SOURCE_ID,
        run_id="run-rollback-002",
    )
    second = full_refresh_harness.write_rows(
        [("new-1", "gamma")],
        "id string, value string",
        plan_two,
    )
    second_snapshot_ids = _snapshot_ids(full_refresh_harness.spark, second.path) - {
        first_snapshot_id
    }
    assert len(second_snapshot_ids) == 1
    second_snapshot_id = next(iter(second_snapshot_ids))

    rollback_result = full_refresh_harness.spark.sql(
        "CALL janus.system.rollback_to_snapshot("
        f"table => '{first.path}', snapshot_id => {first_snapshot_id})"
    ).first()

    assert rollback_result["previous_snapshot_id"] == second_snapshot_id
    assert rollback_result["current_snapshot_id"] == first_snapshot_id
    restored = full_refresh_harness.spark.table(first.path).select("id", "value")
    assert _collected_rows(restored, ("id", "value")) == sorted(run_one_rows)


def _only_snapshot_id(spark, table_identifier: str) -> int:
    snapshot_ids = _snapshot_ids(spark, table_identifier)
    assert len(snapshot_ids) == 1
    return next(iter(snapshot_ids))


def _snapshot_ids(spark, table_identifier: str) -> set[int]:
    return set(_snapshot_rows(spark, table_identifier))


def _snapshot_rows(spark, table_identifier: str) -> dict[int, int | None]:
    return {
        row["snapshot_id"]: row["parent_id"]
        for row in spark.table(f"{table_identifier}.snapshots")
        .select("snapshot_id", "parent_id")
        .collect()
    }


def _history_rows(spark, table_identifier: str) -> dict[int, bool]:
    return {
        row["snapshot_id"]: row["is_current_ancestor"]
        for row in spark.table(f"{table_identifier}.history")
        .select("snapshot_id", "is_current_ancestor")
        .collect()
    }


def _read_snapshot(spark, table_identifier: str, snapshot_id: int):
    return (
        spark.read.format("iceberg")
        .option("snapshot-id", str(snapshot_id))
        .load(table_identifier)
    )


def _collected_rows(dataframe, columns: tuple[str, ...]) -> list[tuple]:
    return sorted(tuple(row[column] for column in columns) for row in dataframe.collect())
