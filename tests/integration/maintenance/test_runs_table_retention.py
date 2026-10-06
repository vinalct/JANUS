import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from janus.maintenance.inventory import collect_runs_table_inventory
from janus.maintenance.settings import resolve_maintenance_settings
from janus.observability import IcebergAppendOutcome, append_run_record
from tests.integration.catalog_commits.test_queryable_observability import _seed_record
from tests.support.maintenance_zone import PROJECT_ROOT
from tests.support.retention_baseline import filesystem_state
from tests.support.spark_sessions import catalog_acceptance_prerequisites_available

pytestmark = pytest.mark.skipif(
    not catalog_acceptance_prerequisites_available(),
    reason="runs-table retention needs Spark, PyIceberg and seeded jars",
)
NOW = datetime(2030, 9, 30, 12, tzinfo=UTC)
CUTOFF = datetime(2030, 9, 27, tzinfo=UTC)


def _policy():
    profile = json.loads((PROJECT_ROOT / "tests/fixtures/maintenance/policy.json").read_text())
    policy = resolve_maintenance_settings(profile)
    return replace(
        policy,
        bronze=replace(policy.bronze, retain_last=2, older_than_days=0),
        runs_table=replace(policy.runs_table, older_than_days=3),
    )


def _seed_rows(token):
    records = []
    for index in range(7):
        day = NOW - timedelta(days=6 - index)
        if day.date() == CUTOFF.date():
            day = CUTOFF  # The partition boundary itself must survive.
        if index == 2:
            day = CUTOFF - timedelta(seconds=1)
        record = _seed_record(
            run_id=f"{token}_{index}",
            source_id=token,
            status="failed",
            started_at=day,
            emitted_at=day,
            quality_outcome="failed",
            failed_checks=("test.quality",),
            error_type="ContractViolationError",
        )
        records.append(
            replace(
                record,
                config_version=str(index) * 64,
                pipeline_run_id=token,
                pipeline_attempt=1,
                checkpoint_decision="advanced",
                checkpoint_advanced=True,
                checkpoint_field="updated_at",
                checkpoint_strategy="max_value",
                checkpoint_value=day.isoformat(),
            )
        )
    
    records.append(replace(records[-1], emitted_at=NOW + timedelta(hours=1)))
    return records


def _query_results(session, qualified, token):
    queries = sorted((PROJECT_ROOT / "docs/queries/observability").glob("*.sql"))
    assert len(queries) == 7  
    results = {}
    for query in queries:
        sql = (
            query.read_text()
            .replace("janus.metadata.runs", qualified)
            .replace("2026-", "2030-")
            .replace("replace-with-pipeline-run-id", token)
        )
        results[query.name] = session.sql(sql).collect()
    return results


def test_old_partitions_removed_snapshots_expired_and_queries_retained(
    catalog_target, shared_catalog_session, run_maintenance, tmp_path, monkeypatch
):
    token = f"retention_{uuid4().hex}"
    identifier = f"metadata.{token}"
    config = catalog_target.environment_config()
    config["observability"] = {"runs_table": identifier}
    records = _seed_rows(token)
    for record in records:
        outcome = append_run_record(record, config, catalog_target.resolved_paths)
        assert outcome.outcome is IcebergAppendOutcome.EMITTED, outcome
    session = shared_catalog_session
    catalog_name = session.conf.get("spark.sql.defaultCatalog")
    qualified = f"{catalog_name}.{identifier}"
    statements = []
    sql = session.sql

    def capture(statement, *args, **kwargs):
        if statement.startswith(("DELETE", "CALL")):
            statements.append(statement)
        return sql(statement, *args, **kwargs)

    monkeypatch.setattr(session, "sql", capture)
    before = collect_runs_table_inventory(session, catalog_name=catalog_name, identifier=identifier)
    assert len(before) == 7
    assert [entry.row_count for entry in before] == [1, 1, 1, 1, 1, 1, 2]
    before_snapshots = {
        row.snapshot_id for row in session.table(f"{qualified}.snapshots").collect()
    }
    assert len(before_snapshots) == 8
    before_files = session.sql(
        f"SELECT file_path, partition.emitted_at_day AS day FROM {qualified}.files"
    ).collect()
    before_queries = _query_results(session, qualified, token)
    digest = filesystem_state(catalog_target.warehouse_dir)
    dry = run_maintenance(
        session, config, catalog_target.resolved_paths, now=NOW, zone="runs-table", policy=_policy()
    )
    assert not statements
    assert filesystem_state(catalog_target.warehouse_dir) == digest
    assert {
        row.snapshot_id for row in session.table(f"{qualified}.snapshots").collect()
    } == before_snapshots
    deletes = [item for item in dry["items"] if item["action"] == "delete_partition"]
    assert [item["target"] for item in deletes] == ["2030-09-24", "2030-09-25", "2030-09-26"]
    assert all(item["detail"]["row_count"] == "1" for item in deletes)
    assert all(item["detail"]["older_than"] == CUTOFF.isoformat() for item in dry["items"])
    applied = run_maintenance(
        session,
        config,
        catalog_target.resolved_paths,
        now=NOW,
        zone="runs-table",
        apply=True,
        policy=_policy(),
    )
    assert all(item["status"] == "applied" for item in applied["items"])
    assert [
        {key: item[key] for key in ("zone", "target", "action")} for item in applied["items"]
    ] == [{key: item[key] for key in ("zone", "target", "action")} for item in dry["items"]]
    expected_ids = {item.run_id for item in records[3:]}
    rows = session.table(qualified).collect()
    assert len(rows) == 5 and {row.run_id for row in rows} == expected_ids
    assert sum(row.run_id == records[-1].run_id for row in rows) == 2
    after = collect_runs_table_inventory(session, catalog_name=catalog_name, identifier=identifier)
    assert after == before[3:]
    # A partition metadata delete leaves surviving data files intact, without a rewrite.
    assert {row.file_path for row in session.table(f"{qualified}.files").collect()} == {
        row.file_path for row in before_files if row.day >= CUTOFF.date()
    }
    snapshots = {row.snapshot_id for row in session.table(f"{qualified}.snapshots").collect()}
    assert len(snapshots) == _policy().bronze.retain_last
    assert set(applied["items"][-1]["expired_snapshot_ids"]) == before_snapshots - snapshots
    after_queries = _query_results(session, qualified, token)
    for name, result in after_queries.items():
        assert result, name
        if "run_id" in result[0].asDict():
            ids = [row.run_id for row in result]
            assert len(ids) == len(set(ids)) and set(ids) == expected_ids, name
        if "source_id" in result[0].asDict():
            assert {row.source_id for row in result} == {token}, name
    assert sum(row.run_count for row in after_queries["runs-by-source-over-time.sql"]) == 4
    assert after_queries["checkpoint-decisions-by-source.sql"][0].decision_count == 4
    assert after_queries["quality-breaches-by-source.sql"][0].breached_runs == 4
    # The published September window reaches into deleted days and silently shrinks to four runs.
    assert len(before_queries["failed-runs-in-window.sql"]) == 7
    assert len(after_queries["failed-runs-in-window.sql"]) == 4
    writes = list(statements)
    second = run_maintenance(
        session,
        config,
        catalog_target.resolved_paths,
        now=NOW,
        zone="runs-table",
        apply=True,
        policy=_policy(),
    )
    assert second["items"] == [] and statements == writes
    assert len(writes) == 2 and writes[0].startswith("DELETE") and writes[1].startswith("CALL")
    (tmp_path / "runs-table-evidence.json").write_text(
        json.dumps(
            {
                "before": [
                    {"day": entry.emitted_at_day.isoformat(), "rows": entry.row_count}
                    for entry in before
                ],
                "after": [
                    {"day": entry.emitted_at_day.isoformat(), "rows": entry.row_count}
                    for entry in after
                ],
                "statements": writes,
                "snapshot_count": len(snapshots),
                "query_counts": {
                    name: {"before": len(before_queries[name]), "after": len(result)}
                    for name, result in after_queries.items()
                },
                "deduplicated_runs": len(expected_ids),
                "retained_rows_including_retry": len(rows),
                "second_apply_items": second["items"],
            },
            indent=2,
        )
        + "\n"
    )


def test_absent_runs_table_is_skipped(catalog_target, shared_catalog_session, run_maintenance):
    config = catalog_target.environment_config()
    config["observability"] = {"runs_table": f"missing_{uuid4().hex}.runs"}
    record = run_maintenance(
        shared_catalog_session,
        config,
        catalog_target.resolved_paths,
        now=NOW,
        zone="runs-table",
        apply=True,
        policy=_policy(),
    )
    assert len(record["items"]) == 1
    assert record["items"][0]["status"] == "skipped"
    assert record["items"][0]["detail"]["skipped_reason"] == "absent_table"
