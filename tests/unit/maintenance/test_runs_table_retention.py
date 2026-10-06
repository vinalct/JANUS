from __future__ import annotations

import ast
import json
from collections import deque
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import janus.maintenance
from janus.cli import maintain
from janus.maintenance.execute import execute_retention
from janus.maintenance.inventory import (
    MaintenanceInventory,
    RunsTablePartitionEntry,
    collect_inventory,
    collect_runs_table_inventory,
)
from janus.maintenance.planning import plan_retention
from janus.maintenance.records import ItemOutcome
from janus.maintenance.settings import resolve_maintenance_settings
from tests.support.operator_cli import run_janus
from tests.unit.maintenance.test_session_free_zones import _project


class Session:
    def __init__(self, *, partitions=(), tables=("runs",), failure=None, snapshots=None):
        self.sparkContext = MagicMock()
        self.partitions = partitions
        self.tables = tables
        self.failure = failure
        self.snapshots = deque(snapshots or ((1, 2, 3, 4), (3, 4)))
        self.statements = []

    def sql(self, statement):
        self.statements.append(statement)

        def collect():
            if self.failure:
                result = self.failure(statement)
                if result is not None:
                    return result
            if statement.startswith("SHOW"):
                return [SimpleNamespace(tableName=name, isTemporary=False) for name in self.tables]
            if ".`partitions`" in statement:
                return [SimpleNamespace(day=day, rows=count) for day, count in self.partitions]
            if ".`snapshots`" in statement:
                return [(value,) for value in self.snapshots.popleft()]
            return []

        return SimpleNamespace(collect=collect)


def _plan(policy_config, now, *, identifier="metadata.runs", counts=(10, 20, 30)):
    policy = resolve_maintenance_settings(policy_config)
    entries = tuple(
        RunsTablePartitionEntry(now.date() - timedelta(days=100 + index), count, identifier)
        for index, count in enumerate(counts)
    )
    plan = plan_retention(
        MaintenanceInventory(runs_table=entries), policy, now, zones=frozenset({"runs-table"})
    )
    return policy, plan


def test_inventory_reads_only_partition_metadata():
    session = Session(partitions=((date(2026, 7, 1), 12), (date(2026, 10, 5), 3)))
    entries = collect_runs_table_inventory(
        session, catalog_name="janus", identifier="metadata.runs"
    )
    assert entries == (
        RunsTablePartitionEntry(date(2026, 7, 1), 12),
        RunsTablePartitionEntry(date(2026, 10, 5), 3),
    )
    assert session.statements == [
        "SHOW TABLES IN `janus`.`metadata`",
        "SELECT partition.emitted_at_day AS day, SUM(record_count) AS rows\n"
        "FROM `janus`.`metadata`.`runs`.`partitions`\nGROUP BY 1 ORDER BY 1",
    ]


@pytest.mark.parametrize("namespace_absent", [False, True])
def test_absent_table_is_one_skipped_item(policy_config, now, namespace_absent):
    class MissingNamespace(Exception):
        def getCondition(self):
            return "SCHEMA_NOT_FOUND"

    def fail(_statement):
        if namespace_absent:
            raise MissingNamespace()

    session = Session(tables=(), failure=fail)
    entries = collect_runs_table_inventory(
        session, catalog_name="janus", identifier="metadata.runs"
    )
    assert len(entries) == 1 and entries[0].unavailable_reason == "absent_table"
    policy = resolve_maintenance_settings(policy_config)
    plan = plan_retention(
        MaintenanceInventory(runs_table=entries), policy, now, zones=frozenset({"runs-table"})
    )
    assert plan.is_empty and len(plan.items) == 1
    outcomes = execute_retention(plan, policy=policy, session=None, catalog_name="janus")
    assert outcomes[0].status == "skipped"
    assert outcomes[0].detail["skipped_reason"] == "absent_table"
    assert len(session.statements) == 1


def test_partition_read_failure_is_evidence_without_a_delete():
    def fail(statement):
        if statement.startswith("SELECT"):
            raise RuntimeError("catalog unavailable")

    entries = collect_runs_table_inventory(
        Session(failure=fail), catalog_name="janus", identifier="metadata.runs"
    )
    assert entries[0].unavailable_reason == "partition_read_failed: RuntimeError"


def test_collection_uses_shared_catalog_derivation_and_override(policy_config, now):
    config = {
        **policy_config,
        "spark": {"iceberg": {"catalog_name": "custom"}},
        "observability": {"runs_table": "audit.events"},
    }
    session = Session(tables=("events",), partitions=((date(2026, 1, 1), 2),))
    inventory = collect_inventory(
        None,
        config,
        {},
        resolve_maintenance_settings(config),
        now,
        zones=frozenset({"runs-table"}),
        source_ids=None,
        session=session,
    )
    assert inventory.runs_table == (RunsTablePartitionEntry(date(2026, 1, 1), 2, "audit.events"),)
    assert "`custom`.`audit`.`events`.`partitions`" in session.statements[-1]
    with pytest.raises(ValueError, match="requires a Spark session"):
        collect_inventory(
            None,
            config,
            {},
            resolve_maintenance_settings(config),
            now,
            zones=frozenset({"runs-table"}),
            source_ids=None,
            session=None,
        )


def test_one_delete_for_three_days_then_expiration_with_exact_sql(policy_config, now):
    policy, plan = _plan(policy_config, now, identifier="audit.events")
    session = Session()
    outcomes = execute_retention(plan, policy=policy, session=session, catalog_name="custom")
    writes = [sql for sql in session.statements if sql.startswith(("DELETE", "CALL"))]
    assert writes == [
        "DELETE FROM `custom`.`audit`.`events` WHERE emitted_at < "
        "TIMESTAMP '2026-07-07T00:00:00+00:00'",
        "CALL `custom`.system.expire_snapshots(\n  table => '`audit`.`events`',\n"
        "  older_than => TIMESTAMP '2026-07-07T00:00:00+00:00',\n  retain_last => 3\n)",
    ]
    assert all(outcome.status == "applied" for outcome in outcomes)
    assert [item.target for item in outcomes] == [item.target for item in plan.items]
    assert [item.removed_count for item in outcomes[:3]] == [30, 20, 10]
    assert [item.detail["row_count"] for item in outcomes[:3]] == ["30", "20", "10"]
    assert outcomes[-1].expired_snapshot_ids == (1, 2)
    assert json.loads(outcomes[-1].detail["surviving_snapshot_ids"]) == [3, 4]


def test_unknown_row_count_stays_unknown_after_grouping(policy_config, now):
    policy, plan = _plan(policy_config, now, counts=(None, 5))
    outcomes = execute_retention(plan, policy=policy, session=Session(), catalog_name="janus")
    assert [item.removed_count for item in outcomes[:2]] == [5, None]


@pytest.mark.parametrize("failed_action", ["DELETE", "CALL"])
def test_failures_preserve_completed_deletions_and_skip_expiration_on_delete_failure(
    policy_config, now, failed_action
):
    def fail(statement):
        if statement.startswith(failed_action):
            raise RuntimeError("scripted failure")

    policy, plan = _plan(policy_config, now)
    ledger = [ItemOutcome.pending_apply(item) for item in plan.items]
    session = Session(failure=fail)
    outcomes = execute_retention(
        plan, policy=policy, session=session, catalog_name="janus", outcomes=ledger
    )
    assert outcomes == tuple(ledger)
    if failed_action == "DELETE":
        assert [item.status for item in ledger] == ["failed", "failed", "failed", "skipped"]
        assert ledger[-1].detail["skipped_reason"] == "partition_delete_failed"
        assert not any(sql.startswith("CALL") for sql in session.statements)
    else:
        assert [item.status for item in ledger] == ["applied", "applied", "applied", "failed"]
        assert [item.removed_count for item in ledger[:3]] == [30, 20, 10]


def test_delete_timeout_cancels_group_and_does_not_expire(policy_config, now):
    entered, release, finished = Event(), Event(), Event()

    def block(statement):
        if statement.startswith("DELETE"):
            entered.set()
            release.wait(5)
            finished.set()
            return []

    policy, plan = _plan(policy_config, now)
    policy = replace(policy, item_timeout_seconds=0.05)
    session = Session(failure=block)
    try:
        outcomes = execute_retention(plan, policy=policy, session=session, catalog_name="janus")
        assert entered.is_set() and not finished.is_set()
        assert all(item.failure_type == "MaintenanceItemTimeout" for item in outcomes[:3])
        assert outcomes[-1].status == "skipped"
        session.sparkContext.cancelJobGroup.assert_called_once()
    finally:
        release.set()
        assert finished.wait(1)


def test_cutoff_is_midnight_utc_after_timezone_conversion(policy_config, now):
    policy_config["maintenance"]["runs_table"]["older_than_days"] = 3
    local = datetime(2026, 10, 4, 22, 30, tzinfo=timezone(timedelta(hours=-3)))
    _, plan = _plan(policy_config, local)
    assert {item.detail["older_than"] for item in plan.items} == {"2026-10-02T00:00:00+00:00"}


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_real_runs_table_cli_acquires_once_and_stops_after_success_or_failure(
    tmp_path, policy_config, now, monkeypatch, apply, fail
):
    root = _project(tmp_path, policy_config)

    def failure(statement):
        if fail and statement.startswith("DELETE"):
            raise RuntimeError("scripted delete failure")

    session = Session(partitions=((date(2020, 1, 1), 10),), failure=failure)
    provider = MagicMock()
    provider.get.return_value = session
    provider.take_cleanup_failures.return_value = ()
    monkeypatch.setattr(maintain, "SparkSessionProvider", lambda *a: provider)
    argv = ["maintain", "--project-root", str(root), "--zone", "runs-table", "--format", "json"]
    if apply:
        argv.append("--apply")
    result = run_janus(argv)
    assert result.exit_code == (1 if fail and apply else 0), result.output
    provider.get.assert_called_once()
    provider.stop.assert_called_once()
    record = json.loads(result.stdout)
    assert record["items"][0]["target"] == "2020-01-01"
    assert record["items"][0]["detail"]["row_count"] == "10"
    if not apply:
        assert all(not sql.startswith(("DELETE", "CALL")) for sql in session.statements)


def _pyiceberg_imports(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if (
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "pyiceberg" for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pyiceberg")
    ]


def test_maintenance_package_has_no_pyiceberg_import():
    modules = sorted(Path(janus.maintenance.__file__).parent.rglob("*.py"))
    assert modules
    assert not [path for path in modules if _pyiceberg_imports(path.read_text())]


@pytest.mark.parametrize(
    "source", ["import pyiceberg", "from pyiceberg.catalog import load_catalog"]
)
def test_pyiceberg_detector_flags_imports(source):
    assert _pyiceberg_imports(source)


def test_pyiceberg_detector_allows_shared_catalog_derivation():
    assert not _pyiceberg_imports(
        "from janus.utils.catalog_properties import derive_pyiceberg_catalog_name"
    )
