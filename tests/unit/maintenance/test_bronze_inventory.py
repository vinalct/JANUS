"""FR-4: registry ownership, shared policies, snapshot refs and bounded reads."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from janus.cli.maintain import _render_text
from janus.maintenance.inventory import MaintenanceInventory, collect_bronze_inventory
from janus.maintenance.planning import plan_retention
from janus.maintenance.records import MaintenanceRecord
from janus.maintenance.settings import resolve_maintenance_settings
from janus.models import BronzeRetentionConfig
from janus.registry import load_registry
from janus.writers.identifiers import quote_identifier
from tests.integration.full_refresh_history.conftest import FIXTURE_PROJECT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_IDENTIFIERS = json.loads(
    (PROJECT_ROOT / "tests/fixtures/maintenance/bronze_identifiers.json").read_text()
)


class FakeSession:
    """Answer namespace, snapshots and refs reads; reject every other statement."""

    def __init__(self, tables=(), *, snapshots=None, current=None, failures=None, catalog="janus"):
        self.tables = set(tables)
        self.snapshots = snapshots or {}
        self.current = current or {}
        self.failures = failures or {}
        self.catalog_name = catalog
        self.queries = []

    def sql(self, query):
        self.queries.append(query)
        if query in self.failures:
            raise self.failures[query]
        if query.startswith("SHOW TABLES IN "):
            rows = [
                SimpleNamespace(tableName=table.rsplit(".", 1)[1], isTemporary=False)
                for table in sorted(self.tables)
                if query
                == "SHOW TABLES IN "
                + quote_identifier(self.catalog_name + "." + table.rsplit(".", 1)[0])
            ]
        else:
            rows = None
            for table in sorted(self.tables):
                qualified = self.catalog_name + "." + table
                if f"FROM {quote_identifier(qualified + '.snapshots')} " in query:
                    rows = self.snapshots.get(table, [])
                elif f"FROM {quote_identifier(qualified + '.refs')} " in query:
                    snapshot_id = self.current.get(table)
                    rows = [] if snapshot_id is None else [SimpleNamespace(snapshot_id=snapshot_id)]
            assert rows is not None, f"unexpected query: {query}"
        return SimpleNamespace(collect=lambda: rows)


@pytest.fixture
def registry():
    return load_registry(FIXTURE_PROJECT_ROOT)


def _source(registry, source_id, *, table="shared", override=None, shared_with=(), enabled=True):
    source = registry.get_source("full_refresh_history_unpartitioned")
    return replace(
        source,
        source_id=source_id,
        enabled=enabled,
        outputs=replace(
            source.outputs,
            bronze=replace(
                source.outputs.bronze,
                namespace="bronze",
                table_name=table,
                shared_with=shared_with,
                retention=override,
            ),
        ),
    )


def _registry(registry, *sources):
    contract = registry.contract_for("full_refresh_history_unpartitioned")
    return replace(
        registry,
        sources=tuple(sources),
        contracts={source.source_id: contract for source in sources},
    )


def _shared_registry(registry, first=None, second=None):
    return _registry(
        registry,
        _source(registry, "alpha", override=first, shared_with=("beta",)),
        _source(registry, "beta", override=second, shared_with=("alpha",), enabled=False),
        _source(registry, "other", table="other"),
    )


def _snapshots(now):
    return [
        SimpleNamespace(
            snapshot_id=value,
            parent_id=value - 1 if value > 1 else None,
            committed_at=now - timedelta(days=100 - value),
        )
        for value in range(1, 5)
    ]


def _collect(registry, session, source_ids=None, *, catalog_name="janus"):
    return collect_bronze_inventory(
        registry, catalog_name=catalog_name, source_ids=source_ids, session=session
    )


def test_all_31_checked_in_sources_include_30_disabled_and_30_unique_tables(now):
    registry = load_registry(PROJECT_ROOT)
    tables = set(EXPECTED_IDENTIFIERS.values())
    session = FakeSession(tables | {"bronze.foreign_table"})
    result = _collect(registry, session)

    assert len(registry.sources) == 31
    assert sum(not source.enabled for source in registry.sources) == 30
    assert len(result) == 30
    assert {source: table.table_identifier for table in result for source in table.source_ids} == (
        EXPECTED_IDENTIFIERS
    )
    assert [table.table_identifier for table in result] == sorted(tables)
    probes = [query for query in session.queries if query.startswith("SHOW TABLES")]
    assert len(probes) == len({table.split(".")[0] for table in tables}) == 6
    assert len(session.queries) == len(probes) + 2 * len(tables)
    assert all("foreign_table" not in query for query in session.queries)


@pytest.mark.parametrize("override", [None, BronzeRetentionConfig(2, 0)])
def test_shared_table_is_grouped_with_all_writers_and_agreed_override(registry, override):
    result = _collect(_shared_registry(registry, override, override), FakeSession())
    shared = next(table for table in result if table.table_identifier == "bronze.shared")
    assert len(result) == 2
    assert shared.source_ids == ("alpha", "beta")
    assert shared.override == override
    assert shared.unavailable_reason == "absent_table"


@pytest.mark.parametrize("first", [None, BronzeRetentionConfig(3, 30)])
def test_retention_conflict_names_both_writers_and_other_tables_proceed(
    registry, first, policy_config, now
):
    second = BronzeRetentionConfig(2, 0)
    session = FakeSession({"bronze.shared", "bronze.other"})
    result = _collect(
        _shared_registry(registry, first, second), session, frozenset({"alpha", "other"})
    )
    shared = next(table for table in result if table.table_identifier == "bronze.shared")
    assert shared.unavailable_reason.startswith("retention_conflict: alpha declares ")
    assert "beta declares retain_last=2, older_than_days=0" in shared.unavailable_reason
    assert shared.override is None and shared.snapshots == ()
    assert (
        next(
            table for table in result if table.table_identifier == "bronze.other"
        ).unavailable_reason
        is None
    )
    assert all("`shared`" not in query for query in session.queries)
    plan = plan_retention(
        MaintenanceInventory(bronze=result),
        resolve_maintenance_settings(policy_config),
        now,
        zones=frozenset({"bronze"}),
        source_ids=frozenset({"alpha", "other"}),
    )
    assert plan.items[0].skipped_reason == shared.unavailable_reason


def test_selection_of_one_shared_writer_keeps_table_and_records_inclusion(
    registry, policy_config, now
):
    session = FakeSession(
        {"bronze.shared", "bronze.other"},
        snapshots={"bronze.shared": _snapshots(now)},
        current={"bronze.shared": 4},
    )
    result = _collect(_shared_registry(registry), session, frozenset({"beta"}))
    assert len(result) == 1
    assert result[0].source_ids == ("alpha", "beta")
    assert result[0].selected_source_ids == ("beta",)
    assert all("`other`" not in query for query in session.queries)
    plan = plan_retention(
        MaintenanceInventory(bronze=result),
        resolve_maintenance_settings(policy_config),
        now,
        zones=frozenset({"bronze"}),
        source_ids=frozenset({"beta"}),
    )
    record = MaintenanceRecord.from_plan(
        plan, environment="local", dry_run=True, zones=("bronze",), started_at=now, ended_at=now
    )
    assert json.loads(record.items[0].detail["selected_source_ids"]) == ["beta"]
    assert "included via shared_with: selected beta; writers alpha, beta" in _render_text(
        plan, record
    )


def test_absent_table_is_skipped_without_snapshot_read(registry):
    registry = _registry(registry, _source(registry, "alpha", table="absent"))
    session = FakeSession({"bronze.foreign"})
    (entry,) = _collect(registry, session)
    assert entry.unavailable_reason == "absent_table" and entry.snapshots == ()
    assert session.queries == ["SHOW TABLES IN `janus`.`bronze`"]


@pytest.mark.parametrize("condition", ["SCHEMA_NOT_FOUND", "NAMESPACE_NOT_FOUND"])
def test_absent_namespace_skips_all_its_tables_with_one_probe(registry, condition):
    class MissingNamespace(Exception):
        def getCondition(self):
            return condition

    session = FakeSession(failures={"SHOW TABLES IN `janus`.`bronze`": MissingNamespace()})
    result = _collect(_shared_registry(registry), session)
    assert len(result) == 2
    assert all(table.unavailable_reason == "absent_table" for table in result)
    assert len(session.queries) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_iceberg_java_missing_namespace_skips_all_its_tables(registry, wrapped):
    def java_error(name, cause=None):
        return SimpleNamespace(
            getClass=lambda: SimpleNamespace(getName=lambda: name), getCause=lambda: cause
        )

    missing = java_error("org.apache.iceberg.exceptions.NoSuchNamespaceException")
    error = RuntimeError("java error")
    error.java_exception = (
        java_error("org.apache.spark.SparkException", missing) if wrapped else missing
    )
    session = FakeSession(failures={"SHOW TABLES IN `janus`.`bronze`": error})
    result = _collect(_shared_registry(registry), session)
    assert all(table.unavailable_reason == "absent_table" for table in result)
    assert len(session.queries) == 1


@pytest.mark.parametrize("metadata_table", ["snapshots", "refs"])
def test_snapshot_read_failure_isolated_to_one_table(registry, now, metadata_table):
    query = (
        "SELECT `snapshot_id`, `parent_id`, `committed_at` FROM "
        "`janus`.`bronze`.`shared`.`snapshots` ORDER BY `committed_at`"
        if metadata_table == "snapshots"
        else "SELECT `snapshot_id` FROM `janus`.`bronze`.`shared`.`refs` WHERE `name` = 'main'"
    )
    session = FakeSession(
        {"bronze.shared", "bronze.other"},
        snapshots={"bronze.other": _snapshots(now)},
        current={"bronze.other": 4},
        failures={query: RuntimeError("read refused")},
    )
    result = _collect(_shared_registry(registry), session)
    assert (
        next(
            table for table in result if table.table_identifier == "bronze.shared"
        ).unavailable_reason
        == "snapshot_read_failed: RuntimeError"
    )
    other = next(table for table in result if table.table_identifier == "bronze.other")
    assert len(other.snapshots) == 4 and other.unavailable_reason is None


def test_namespace_failure_isolated_and_recorded(registry):
    alpha = _source(registry, "alpha", table="alpha")
    beta = _source(registry, "beta", table="beta")
    beta = replace(
        beta, outputs=replace(beta.outputs, bronze=replace(beta.outputs.bronze, namespace="other"))
    )
    session = FakeSession(
        {"other.beta"}, failures={"SHOW TABLES IN `janus`.`bronze`": RuntimeError("catalog down")}
    )
    result = _collect(_registry(registry, alpha, beta), session)
    assert result[0].unavailable_reason == "snapshot_read_failed: RuntimeError"
    assert result[1].unavailable_reason is None


def test_non_iceberg_outputs_and_empty_selection_issue_no_queries(registry):
    alpha = _source(registry, "alpha", table="alpha")
    parquet = replace(
        alpha,
        outputs=replace(alpha.outputs, bronze=replace(alpha.outputs.bronze, format="parquet")),
    )
    session = FakeSession()
    assert _collect(_registry(registry, parquet), session) == ()
    assert _collect(_registry(registry, alpha), session, frozenset()) == ()
    assert session.queries == []


def test_current_snapshot_comes_from_main_ref_after_rollback(registry, now, policy_config):
    session = FakeSession(
        {"bronze.shared"},
        snapshots={"bronze.shared": _snapshots(now)},
        current={"bronze.shared": 1},
    )
    source = _source(registry, "alpha", override=BronzeRetentionConfig(2, 0))
    (table,) = _collect(_registry(registry, source), session)
    assert [snapshot.snapshot_id for snapshot in table.snapshots if snapshot.is_current] == [1]
    assert table.snapshots[-1].snapshot_id == 4
    assert [snapshot.parent_id for snapshot in table.snapshots] == [None, 1, 2, 3]
    plan = plan_retention(
        MaintenanceInventory(bronze=(table,)),
        resolve_maintenance_settings(policy_config),
        now,
        zones=frozenset({"bronze"}),
    )
    assert json.loads(plan.items[0].detail["snapshot_ids"]) == [2]
    assert any(
        item.target == "bronze.shared#1" and item.reason == "current_snapshot"
        for item in plan.protected
    )


@pytest.mark.parametrize("current", [None, 99])
def test_missing_or_inconsistent_current_ref_refuses_nonempty_table(registry, now, current):
    session = FakeSession(
        {"bronze.shared"},
        snapshots={"bronze.shared": _snapshots(now)},
        current={"bronze.shared": current},
    )
    (table,) = _collect(_registry(registry, _source(registry, "alpha")), session)
    assert table.unavailable_reason == "snapshot_read_failed: ValueError"
    assert table.snapshots == ()


@pytest.mark.parametrize("aware", [False, True])
def test_timestamps_are_normalized_to_aware_utc(registry, now, aware):
    timestamp = (
        now.astimezone(timezone(timedelta(hours=-3)))
        if aware
        else datetime.fromtimestamp(now.timestamp())
    )
    row = SimpleNamespace(snapshot_id=1, parent_id=None, committed_at=timestamp)
    session = FakeSession(
        {"bronze.shared"}, snapshots={"bronze.shared": [row]}, current={"bronze.shared": 1}
    )
    (table,) = _collect(_registry(registry, _source(registry, "alpha")), session)
    assert table.snapshots[0].committed_at == now
    assert table.snapshots[0].committed_at.tzinfo is UTC


def test_naive_spark_timestamp_keeps_the_instant_in_a_non_utc_python_timezone(
    registry, now, monkeypatch
):
    with monkeypatch.context() as patch:
        patch.setenv("TZ", "America/Sao_Paulo")
        time.tzset()
        try:
            row = SimpleNamespace(
                snapshot_id=1, parent_id=None, committed_at=datetime.fromtimestamp(now.timestamp())
            )
            session = FakeSession(
                {"bronze.shared"}, snapshots={"bronze.shared": [row]}, current={"bronze.shared": 1}
            )
            (table,) = _collect(_registry(registry, _source(registry, "alpha")), session)
            assert row.committed_at.hour == 9
            assert table.snapshots[0].committed_at == now
            assert table.snapshots[0].committed_at.tzinfo is UTC
        finally:
            patch.undo()
            time.tzset()


def test_hostile_catalog_identifier_is_quoted_in_every_query(registry):
    catalog = "catalog` WHERE 1=1 --"
    session = FakeSession({"bronze.shared"}, catalog=catalog)
    (table,) = _collect(
        _registry(registry, _source(registry, "alpha")), session, catalog_name=catalog
    )
    assert table.unavailable_reason is None
    assert all("`catalog`` WHERE 1=1 --`.`bronze`" in query for query in session.queries)
