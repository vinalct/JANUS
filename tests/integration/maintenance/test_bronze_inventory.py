"""Collector parity with the real history harness, including rollback and dry runs."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

from janus.maintenance.inventory import collect_bronze_inventory
from janus.writers.identifiers import quote_identifier
from tests.integration.full_refresh_history.conftest import ENVIRONMENT_CONFIG
from tests.support.retention_baseline import filesystem_state
from tests.support.spark_sessions import DEFAULT_CATALOG_NAME

SOURCE = "full_refresh_history_unpartitioned"


def _write_history(harness):
    plans = []
    for index in range(3):
        plan = harness.plan(SOURCE, run_id=f"inventory-{index}")
        harness.write_rows([(f"row-{index}", f"value-{index}")], "id string, value string", plan)
        plans.append(plan)
    return plans[-1]


def test_collector_matches_snapshots_and_main_ref_after_rollback(full_refresh_harness):
    harness = full_refresh_harness
    plan = _write_history(harness)
    registry = replace(harness.registry, sources=(plan.source_config,))
    identifier = registry.graph.nodes[0].bronze_table
    qualified = f"{DEFAULT_CATALOG_NAME}.{identifier}"
    rows = harness.spark.sql(
        f"SELECT * FROM {quote_identifier(qualified + '.snapshots')} ORDER BY committed_at"
    ).collect()
    assert len(rows) == 3
    previous = rows[0].snapshot_id
    harness.spark.sql(
        f"CALL {quote_identifier(DEFAULT_CATALOG_NAME + '.system.rollback_to_snapshot')}("
        f"table => '{identifier}', snapshot_id => {previous})"
    ).collect()
    (table,) = collect_bronze_inventory(
        registry,
        catalog_name=DEFAULT_CATALOG_NAME,
        source_ids=frozenset({SOURCE}),
        session=harness.spark,
    )
    assert table.table_identifier == identifier
    assert table.unavailable_reason is None
    assert [
        (entry.snapshot_id, entry.parent_id, entry.committed_at) for entry in table.snapshots
    ] == [(row.snapshot_id, row.parent_id, row.committed_at.astimezone(UTC)) for row in rows]
    assert [entry.snapshot_id for entry in table.snapshots if entry.is_current] == [previous]
    assert previous != rows[-1].snapshot_id
    assert all(entry.committed_at.tzinfo is UTC for entry in table.snapshots)


def test_bronze_command_dry_run_preserves_history_and_warehouse(
    full_refresh_harness, run_maintenance
):
    harness = full_refresh_harness
    _write_history(harness)
    qualified = f"{DEFAULT_CATALOG_NAME}.bronze_full_refresh_history.{harness.table_name}"
    snapshots_query = (
        f"SELECT * FROM {quote_identifier(qualified + '.snapshots')} ORDER BY committed_at"
    )
    before = harness.spark.sql(snapshots_query).collect()
    warehouse = harness.spark.conf.get(f"spark.sql.catalog.{DEFAULT_CATALOG_NAME}.warehouse")
    directory = Path(unquote(urlsplit(warehouse).path))
    digest = filesystem_state(directory)
    record = run_maintenance(
        harness.spark,
        ENVIRONMENT_CONFIG,
        {"metadata_dir": harness.project_root / "metadata"},
        now=datetime(2030, 1, 1, tzinfo=UTC),
        zone="bronze",
        source_id=SOURCE,
        table_name=harness.table_name,
    )
    assert record["dry_run"] is True and record["failures"] == []
    assert len(record["items"]) == 1
    assert record["items"][0]["expired_snapshot_ids"] == [before[0].snapshot_id]
    assert harness.spark.sql(snapshots_query).collect() == before
    assert filesystem_state(directory) == digest


def test_never_written_namespace_is_absent_in_the_real_catalog(full_refresh_harness):
    harness = full_refresh_harness
    plan = harness.plan(SOURCE, run_id="inventory-absent")
    source = replace(
        plan.source_config,
        outputs=replace(
            plan.source_config.outputs,
            bronze=replace(plan.bronze_output, namespace="bronze_inventory_never_written"),
        ),
    )
    registry = replace(harness.registry, sources=(source,))
    (table,) = collect_bronze_inventory(
        registry, catalog_name=DEFAULT_CATALOG_NAME, source_ids=None, session=harness.spark
    )
    assert table.unavailable_reason == "absent_table"
    assert table.snapshots == ()
