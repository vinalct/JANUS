"""Real command proof that expiration bounds history and reclaims data files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from janus.writers.identifiers import quote_identifier

pytestmark = pytest.mark.parametrize(
    "maintenance_harness",
    ("full_refresh_history_unpartitioned", "full_refresh_history_partitioned"),
    indirect=True,
    ids=("unpartitioned", "partitioned"),
)


@dataclass(frozen=True)
class ThreeRuns:
    snapshot_ids: tuple[int, ...]
    files: tuple[frozenset[Path], ...]
    rows: tuple[list[tuple], ...]
    columns: tuple[str, ...]
    now: datetime


def _snapshot_ids(harness) -> set[int]:
    return {row.snapshot_id for row in harness.spark.table(f"{harness.table}.snapshots").collect()}


def _current_snapshot(harness) -> int:
    (main,) = harness.spark.table(f"{harness.table}.refs").filter("name = 'main'").collect()
    return main.snapshot_id


def _three_runs(harness) -> ThreeRuns:
    partitioned = harness.source_id == "full_refresh_history_partitioned"
    columns = ("id", "value", "partition_key") if partitioned else ("id", "value")
    schema = ", ".join(f"{column} string" for column in columns)
    ids, files, rows = [], [], []
    for index in range(3):
        batch = (
            [
                (f"run-{index}-a", f"value-{index}", "first"),
                (f"run-{index}-b", f"value-{index}", "second"),
            ]
            if partitioned
            else [(f"run-{index}", f"value-{index}")]
        )
        plan = harness.history.plan(harness.source_id, run_id=f"retention-{index}")
        harness.history.write_rows(batch, schema, plan)
        rows.append(batch)
        ids.append(_current_snapshot(harness))
        files.append(harness.data_files(harness.table))
    assert len(set(ids)) == 3
    assert _snapshot_ids(harness) == set(ids)
    assert all(len(paths) == (2 if partitioned else 1) for paths in files)
    assert all(path.stat().st_size > 0 for paths in files for path in paths)
    assert files[0].isdisjoint(files[1] | files[2])
    committed = harness.spark.sql(
        f"SELECT unix_millis(committed_at) AS committed_ms "
        f"FROM {quote_identifier(harness.table + '.snapshots')}"
    ).collect()

    last_commit = datetime.fromtimestamp(max(row.committed_ms for row in committed) / 1000, UTC)
    now = last_commit + timedelta(seconds=1)
    assert all(datetime.fromtimestamp(row.committed_ms / 1000, UTC) < now for row in committed)
    return ThreeRuns(tuple(ids), tuple(files), tuple(rows), columns, now)


def _read_rows(harness, columns, *, snapshot_id=None) -> list[tuple]:
    version = f" VERSION AS OF {snapshot_id}" if snapshot_id is not None else ""
    frame = harness.spark.sql(f"SELECT * FROM {quote_identifier(harness.table)}{version}")
    return sorted(tuple(row) for row in frame.select(*columns).collect())


def _assert_retained(harness, runs) -> None:
    assert _snapshot_ids(harness) == set(runs.snapshot_ids[1:])
    assert _current_snapshot(harness) == runs.snapshot_ids[-1]
    assert _read_rows(harness, runs.columns, snapshot_id=runs.snapshot_ids[1]) == sorted(
        runs.rows[1]
    )
    assert _read_rows(harness, runs.columns) == sorted(runs.rows[-1])
    for path in runs.files[0]:
        with pytest.raises(FileNotFoundError):
            path.stat()
    assert all(path.stat().st_size > 0 for path in runs.files[1] | runs.files[2])


def test_three_runs_retain_two_and_preserve_time_travel(maintenance_harness, record_property):
    harness = maintenance_harness
    runs = _three_runs(harness)
    current = _current_snapshot(harness)
    sizes = {str(path): path.stat().st_size for paths in runs.files for path in paths}
    before = harness.warehouse_digest()
    policy = harness.policy(retain_last=2, older_than_days=0)

    dry = harness.run_maintain(policy, apply=False, now=runs.now)
    assert dry["dry_run"] is True
    assert dry["policy_digest"] == policy.digest
    assert dry["zones"] == ["bronze"]
    assert dry["source_ids"] == [harness.source_id]
    (planned,) = dry["items"]
    assert planned["action"] == "expire_snapshots" and planned["status"] == "planned"
    assert json.loads(planned["detail"]["snapshot_ids"]) == [runs.snapshot_ids[0]]
    assert planned["detail"]["older_than"] == runs.now.isoformat()
    assert planned["detail"]["retain_last"] == "2"
    assert _snapshot_ids(harness) == set(runs.snapshot_ids)
    assert _current_snapshot(harness) == current
    after_dry = harness.warehouse_digest()
    assert after_dry == before

    applied = harness.run_maintain(policy, apply=True, now=runs.now)
    assert applied["dry_run"] is False and applied["failures"] == []
    (outcome,) = applied["items"]
    assert outcome["status"] == "applied"
    assert outcome["expired_snapshot_ids"] == [runs.snapshot_ids[0]]
    assert json.loads(outcome["detail"]["predicted_but_retained_snapshot_ids"]) == []
    assert json.loads(outcome["detail"]["unexpected_expired_snapshot_ids"]) == []
    assert applied["plan_digest"] == dry["plan_digest"]
    _assert_retained(harness, runs)

    repeat = harness.run_maintain(policy, apply=True, now=runs.now)
    assert repeat["items"] == [] and repeat["failures"] == []
    assert all(summary["items_planned"] == 0 for summary in repeat["zone_summaries"])
    _assert_retained(harness, runs)
    after_apply = harness.warehouse_digest()
    reclaimed = sum(sizes[str(path)] for path in runs.files[0])
    assert reclaimed > 0
    assert after_apply["bytes"] < before["bytes"]
    record_property(
        "snapshot_expiration",
        json.dumps(
            {
                "source_id": harness.source_id,
                "snapshots_before": runs.snapshot_ids,
                "snapshots_after": sorted(_snapshot_ids(harness)),
                "current_snapshot": current,
                "cutoff": runs.now.isoformat(),
                "dry_run": dry,
                "applied": applied,
                "repeat": repeat,
                "warehouse_before": before,
                "warehouse_after_dry_run": after_dry,
                "warehouse_after_apply": after_apply,
                "data_files": [
                    {"path": path, "bytes_before": size, "exists_after": Path(path).exists()}
                    for path, size in sorted(sizes.items())
                ],
                "data_bytes_reclaimed": reclaimed,
                "warehouse_bytes_reclaimed": before["bytes"] - after_apply["bytes"],
            },
            sort_keys=True,
        ),
    )


def test_second_apply_has_no_items(maintenance_harness):
    harness = maintenance_harness
    runs = _three_runs(harness)
    policy = harness.policy(retain_last=2, older_than_days=0)
    first = harness.run_maintain(policy, apply=True, now=runs.now)
    assert len(first["items"]) == 1
    _assert_retained(harness, runs)
    before = harness.warehouse_digest()

    second = harness.run_maintain(policy, apply=True, now=runs.now)
    assert second["dry_run"] is False and second["failures"] == []
    assert second["items"] == []
    assert all(summary["items_planned"] == 0 for summary in second["zone_summaries"])
    assert harness.warehouse_digest() == before
    _assert_retained(harness, runs)


def test_retain_three_preserves_history_time_travel_and_rollback(maintenance_harness):
    harness = maintenance_harness
    runs = _three_runs(harness)
    before = harness.warehouse_digest()
    record = harness.run_maintain(harness.policy(retain_last=3), apply=True, now=runs.now)
    assert record["items"] == [] and record["failures"] == []
    assert harness.warehouse_digest() == before

    assert len(_snapshot_ids(harness)) >= 2
    assert _snapshot_ids(harness) == set(runs.snapshot_ids)
    assert _current_snapshot(harness) == runs.snapshot_ids[-1]
    history = harness.spark.table(f"{harness.table}.history").collect()
    assert {row.snapshot_id for row in history} == set(runs.snapshot_ids)
    assert all(row.is_current_ancestor for row in history)
    for snapshot_id, rows in zip(runs.snapshot_ids, runs.rows, strict=True):
        assert _read_rows(harness, runs.columns, snapshot_id=snapshot_id) == sorted(rows)
    assert all(path.stat().st_size > 0 for paths in runs.files for path in paths)

    catalog = harness.spark.conf.get("spark.sql.defaultCatalog")
    rollback = harness.spark.sql(
        f"CALL {quote_identifier(catalog)}.system.rollback_to_snapshot("
        f"table => '{harness.table}', snapshot_id => {runs.snapshot_ids[0]})"
    ).first()
    assert rollback.previous_snapshot_id == runs.snapshot_ids[-1]
    assert rollback.current_snapshot_id == runs.snapshot_ids[0]
    assert _current_snapshot(harness) == runs.snapshot_ids[0]
    assert _read_rows(harness, runs.columns) == sorted(runs.rows[0])
    assert _snapshot_ids(harness) == set(runs.snapshot_ids)
