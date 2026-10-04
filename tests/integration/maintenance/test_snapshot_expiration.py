import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from tests.integration.full_refresh_history.conftest import ENVIRONMENT_CONFIG
from tests.support.retention_baseline import filesystem_state

SOURCE = "full_refresh_history_unpartitioned"
NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


@pytest.mark.xfail(
    strict=True, reason="snapshot maintenance command/executor absent"
)
def test_three_runs_retain_two_and_preserve_time_travel(full_refresh_harness, run_maintenance):
    harness = full_refresh_harness
    session = harness.spark
    ids, files, rows_by_run = [], [], []
    for index in range(3):
        plan = harness.plan(SOURCE, run_id=f"retention-{index}")
        rows = [(f"run-{index}", f"value-{index}")]
        result = harness.write_rows(rows, "id string, value string", plan)
        rows_by_run.append(rows)
        ids.append(
            session.table(f"{result.path}.history")
            .orderBy("made_current_at")
            .collect()[-1]
            .snapshot_id
        )
        files.append(
            {
                Path(unquote(urlsplit(row.file_path).path))
                for row in session.table(f"{result.path}.files").collect()
            }
        )
    assert len(set(ids)) == 3
    assert all(path.exists() for paths in files for path in paths)
    location = next(
        row.data_type
        for row in session.sql(f"DESCRIBE TABLE EXTENDED {result.path}").collect()
        if row.col_name == "Location"
    )
    warehouse = Path(unquote(urlsplit(location).path))
    config = ENVIRONMENT_CONFIG
    paths = {"metadata_dir": harness.project_root / "metadata"}
    arguments = dict(now=NOW, zone="bronze", source_id=SOURCE, table_name=harness.table_name)
    before = filesystem_state(warehouse)
    dry = run_maintenance(session, config, paths, **arguments)
    assert dry["dry_run"] is True
    planned_ids = {
        int(snapshot)
        for item in dry["items"]
        if item["action"] == "expire_snapshots"
        for snapshot in json.loads(item["detail"]["snapshot_ids"])
    }
    assert planned_ids == {ids[0]}
    assert {row.snapshot_id for row in session.table(f"{result.path}.snapshots").collect()} == set(
        ids
    )
    assert filesystem_state(warehouse) == before
    record = run_maintenance(session, config, paths, apply=True, **arguments)
    assert record["dry_run"] is False
    assert {row.snapshot_id for row in session.table(f"{result.path}.snapshots").collect()} == set(
        ids[1:]
    )
    current = session.table(f"{result.path}.history").orderBy("made_current_at").collect()[-1]
    assert current.snapshot_id == ids[-1]
    previous = session.read.option("snapshot-id", str(ids[1])).table(result.path)
    assert sorted(tuple(row) for row in previous.select("id", "value").collect()) == rows_by_run[1]
    assert files[0].isdisjoint(files[1] | files[2])
    assert all(not path.exists() for path in files[0])
    assert all(path.exists() for path in files[1] | files[2])
    assert {snapshot for item in record["items"] for snapshot in item["expired_snapshot_ids"]} == {
        ids[0]
    }
    assert dry["plan_digest"] == record["plan_digest"]
    repeat = run_maintenance(session, config, paths, apply=True, **arguments)
    assert repeat["items"] == []
    assert {row.snapshot_id for row in session.table(f"{result.path}.snapshots").collect()} == set(
        ids[1:]
    )
