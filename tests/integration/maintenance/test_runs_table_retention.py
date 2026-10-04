from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from janus.observability import IcebergAppendOutcome, append_run_record
from tests.integration.catalog_commits.test_queryable_observability import _seed_record
from tests.support.maintenance_zone import PROJECT_ROOT
from tests.support.spark_sessions import catalog_acceptance_prerequisites_available

pytestmark = pytest.mark.skipif(
    not catalog_acceptance_prerequisites_available(),
    reason="runs-table retention needs Spark, PyIceberg and seeded jars",
)
NOW = datetime(2030, 10, 5, 12, tzinfo=UTC)


@pytest.mark.xfail(strict=True, reason="runs-table maintenance executor absent")
def test_old_partitions_removed_snapshots_expired_and_queries_retained(
    catalog_target,
    shared_catalog_session,
    run_maintenance,
):
    token = f"retention_{uuid4().hex}"
    identifier = f"metadata.{token}"
    config = catalog_target.environment_config()
    config["observability"] = {"runs_table": identifier}
    records = []
    for index, day in enumerate(
        (
            datetime(2030, 8, 1, tzinfo=UTC),
            datetime(2030, 9, 15, tzinfo=UTC),
            datetime(2030, 9, 16, tzinfo=UTC),
        )
    ):
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
        outcome = append_run_record(records[-1], config, catalog_target.resolved_paths)
        assert outcome.outcome is IcebergAppendOutcome.EMITTED, outcome
    session = shared_catalog_session
    catalog_name = session.conf.get("spark.sql.defaultCatalog")
    qualified = f"{catalog_name}.{identifier}"
    before_snapshots = {
        row.snapshot_id for row in session.table(f"{qualified}.snapshots").collect()
    }
    assert len(before_snapshots) == 3
    record = run_maintenance(
        session, config, catalog_target.resolved_paths, now=NOW, zone="runs-table", apply=True
    )
    assert all(item["status"] == "applied" for item in record["items"])
    assert {row.run_id for row in session.table(qualified).collect()} == {
        item.run_id for item in records[1:]
    }
    days = {
        str(row.partition.emitted_at_day)
        for row in session.table(f"{qualified}.partitions").collect()
    }
    assert days == {"2030-09-15", "2030-09-16"}
    after = {row.snapshot_id for row in session.table(f"{qualified}.snapshots").collect()}
    assert len(after) == 2
    assert before_snapshots - after
    queries = sorted((PROJECT_ROOT / "docs/queries/observability").glob("*.sql"))
    assert len(queries) == 7  # TASK-01 revision includes order-19's schema-drift query.
    for query in queries:
        sql = query.read_text().replace("janus.metadata.runs", qualified).replace("2026-", "2030-")
        rows = session.sql(sql).collect()
        assert rows, query.name
        if "run_id" in session.sql(sql).columns:
            assert records[0].run_id not in {row.run_id for row in rows}
        if "source_id" in session.sql(sql).columns:
            assert {row.source_id for row in rows} == {token}
