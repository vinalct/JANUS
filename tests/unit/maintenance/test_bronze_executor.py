from __future__ import annotations

import ast
import inspect
import json
import re
from collections import deque
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

import janus
from janus.cli import maintain
from janus.maintenance import execute
from janus.maintenance.inventory import BronzeTableInventory, MaintenanceInventory, SnapshotEntry
from janus.maintenance.planning import PlannedItem, RetentionPlan, plan_retention
from janus.maintenance.records import ItemOutcome, MaintenanceRecord
from janus.maintenance.settings import resolve_maintenance_settings
from tests.support.operator_cli import run_janus
from tests.support.semantics_fixtures import CLEAN, CLEAN_PRODUCER, install_profile, materialize

CUTOFF = "2026-10-05T12:00:00+00:00"
COUNTS = {
    "deleted_data_files_count": 5,
    "deleted_position_delete_files_count": 1,
    "deleted_equality_delete_files_count": 2,
    "deleted_manifest_files_count": 3,
    "deleted_manifest_lists_count": 4,
    "deleted_statistics_files_count": 0,
}


class Row(tuple):
    def __new__(cls, **fields):
        result = super().__new__(cls, fields.values())
        result.fields = fields
        return result

    def asDict(self):
        return dict(self.fields)


class Session:
    def __init__(self, *, rows=None, before=(1, 2, 3), after=(2, 3), call=None):
        self.sparkContext = MagicMock()
        self.rows = [Row(**COUNTS)] if rows is None else rows
        self.snapshots = deque((before, after))
        self.call = call
        self.statements = []
        self.threads = []

    def sql(self, statement):
        self.statements.append(statement)
        self.threads.append(get_ident())

        def collect():
            if statement.startswith("SELECT"):
                return [(value,) for value in self.snapshots.popleft()]
            return self.call(statement) if self.call else self.rows

        return SimpleNamespace(collect=collect)


def _item(action="expire_snapshots", *, target="bronze.example", **detail):
    arguments = {
        "expire_snapshots": {"older_than": CUTOFF, "retain_last": "2", "snapshot_ids": "[1]"},
        "remove_orphan_files": {"orphan_older_than": "2026-10-02T12:00:00+00:00"},
        "rewrite_data_files": {"target_file_size_bytes": "134217728"},
    }
    return PlannedItem("bronze", target, action, {**arguments[action], **detail})


def _execute(session, item=None, **kwargs):
    return execute.execute_item(
        session, item or _item(), catalog_name="janus", timeout_seconds=1, **kwargs
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            "expire_snapshots",
            "CALL `janus`.system.expire_snapshots(\n"
            "  table => '`bronze`.`example`',\n"
            "  older_than => TIMESTAMP '2026-10-05T12:00:00+00:00',\n"
            "  retain_last => 2\n)",
        ),
        (
            "remove_orphan_files",
            "CALL `janus`.system.remove_orphan_files(\n"
            "  table => '`bronze`.`example`',\n"
            "  older_than => TIMESTAMP '2026-10-02T12:00:00+00:00'\n)",
        ),
        (
            "rewrite_data_files",
            "CALL `janus`.system.rewrite_data_files(\n"
            "  table => '`bronze`.`example`',\n"
            "  options => map('target-file-size-bytes', '134217728')\n)",
        ),
    ],
)
def test_procedure_sql_is_exactly_the_declared_plan(action, expected):
    session = Session()
    outcome = _execute(session, _item(action))
    assert outcome.status == "applied"
    assert [sql for sql in session.statements if sql.startswith("CALL")] == [expected]
    if action == "expire_snapshots":
        assert (
            session.statements[0]
            == session.statements[2]
            == ("SELECT `snapshot_id` FROM `janus`.`bronze`.`example`.`snapshots`")
        )


def test_identifiers_and_sql_string_are_escaped():
    session = Session()
    item = _item("remove_orphan_files", target="bron`ze.tab'le\\name")
    outcome = execute.execute_item(session, item, catalog_name="cat`alog", timeout_seconds=1)
    assert outcome.status == "applied"
    assert session.statements == [
        "CALL `cat``alog`.system.remove_orphan_files(\n"
        "  table => '`bron``ze`.`tab\\'le\\\\name`',\n"
        "  older_than => TIMESTAMP '2026-10-02T12:00:00+00:00'\n)"
    ]


def test_expiration_counts_use_names_and_record_actual_snapshot_changes():
    session = Session(rows=[Row(**dict(reversed(list(COUNTS.items()))))])
    outcome = _execute(session)
    assert outcome.removed_count == 15
    assert {key: outcome.detail[key] for key in COUNTS} == {
        key: str(value) for key, value in COUNTS.items()
    }
    assert outcome.expired_snapshot_ids == (1,)
    assert json.loads(outcome.detail["surviving_snapshot_ids"]) == [2, 3]
    assert outcome.removed_bytes is None
    assert outcome.detail["snapshot_ids"] == "[1]"


@pytest.mark.parametrize("row", [(5, 1, 2, 3, 4, 0), Row(a=5, b=1, c=2, d=3, e=4, f=0)])
def test_renamed_count_columns_fall_back_to_positions_and_keep_raw_row(row):
    outcome = _execute(Session(rows=[row]))
    assert outcome.status == "applied"
    assert outcome.removed_count == 15
    assert json.loads(outcome.detail["raw_result"]) == (
        row.asDict() if hasattr(row, "asDict") else list(row)
    )


def test_empty_procedure_counts_remain_unknown():
    outcome = _execute(Session(rows=[]))
    assert outcome.status == "applied"
    assert outcome.removed_count is None
    assert outcome.detail["raw_result"] == "[]"


def test_plan_outcome_divergence_is_evidence_without_failure():
    outcome = _execute(Session(before=(1, 2, 3, 4), after=(2, 4)), _item(snapshot_ids="[1, 2]"))
    assert outcome.status == "applied"
    assert outcome.expired_snapshot_ids == (1, 3)
    assert json.loads(outcome.detail["predicted_but_retained_snapshot_ids"]) == [2]
    assert json.loads(outcome.detail["unexpected_expired_snapshot_ids"]) == [3]


def test_verification_failure_keeps_measured_file_deletions():
    session = Session()
    session.snapshots = deque(((1, 2, 3),))
    outcome = _execute(session)
    assert outcome.status == "failed"
    assert outcome.removed_count == 15
    assert outcome.expired_snapshot_ids == ()
    assert outcome.failure_type == "IndexError"


@pytest.mark.parametrize("key", ["snapshot_ids", "older_than", "retain_last"])
def test_incomplete_expiration_plan_fails_before_any_sql(key):
    item = _item()
    detail = dict(item.detail)
    del detail[key]
    session = Session()
    outcome = _execute(session, replace(item, detail=detail))
    assert outcome.status == "failed"
    assert outcome.failure_type == "KeyError"
    assert session.statements == []


def test_missing_optional_count_names_do_not_duplicate_other_counts():
    counts = dict(COUNTS)
    del counts["deleted_position_delete_files_count"]
    outcome = _execute(Session(rows=[Row(**counts)]))
    assert outcome.status == "applied"
    assert outcome.removed_count == 14
    assert "deleted_position_delete_files_count" not in outcome.detail
    assert json.loads(outcome.detail["raw_result"]) == counts


def test_orphan_count_is_the_number_of_returned_locations():
    outcome = _execute(Session(rows=[("one",), ("two",)]), _item("remove_orphan_files"))
    assert outcome.removed_count == 2
    assert outcome.detail["orphan_files_count"] == "2"


def test_compaction_records_rewrite_counts_without_claiming_reclaimed_bytes():
    outcome = _execute(
        Session(
            rows=[
                Row(
                    rewritten_data_files_count=7,
                    added_data_files_count=2,
                    rewritten_bytes_count=1024,
                    failed_data_files_count=0,
                    removed_delete_files_count=1,
                )
            ]
        ),
        _item("rewrite_data_files"),
    )
    assert outcome.status == "applied"
    assert outcome.removed_count == 8
    assert outcome.removed_bytes is None
    assert outcome.detail["rewritten_bytes_count"] == "1024"


def test_second_of_four_tables_fails_and_the_rest_execute(now, policy_config):
    calls = []

    def call(statement):
        calls.append(statement)
        if "`table2`" in statement:
            raise RuntimeError("password=secret " + "x" * 1000)
        return [("removed",)]

    session = Session(call=call)
    plan = RetentionPlan(
        tuple(_item("remove_orphan_files", target=f"bronze.table{i}") for i in range(1, 5)),
        (),
        now,
        "a" * 64,
    )
    outcomes = execute.execute_retention(
        plan,
        policy=resolve_maintenance_settings(policy_config),
        session=session,
        catalog_name="janus",
    )
    assert [item.status for item in outcomes] == ["applied", "failed", "applied", "applied"]
    assert len(calls) == 4
    assert outcomes[1].failure_type == "RuntimeError"
    assert "secret" not in outcomes[1].failure_message
    assert len(outcomes[1].failure_message) <= 512


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_timeout_requests_cancellation_without_claiming_it_stopped(cancel_fails):
    entered, release, finished = Event(), Event(), Event()

    def blocked(_statement):
        entered.set()
        release.wait(5)
        finished.set()
        return [("late result",)]

    session = Session(call=blocked)
    if cancel_fails:
        session.sparkContext.cancelJobGroup.side_effect = RuntimeError("unavailable")
    try:
        outcome = execute.execute_item(
            session, _item("rewrite_data_files"), catalog_name="janus", timeout_seconds=0.05
        )
        assert entered.is_set() and not finished.is_set()
        assert outcome.status == "failed"
        assert outcome.failure_type == "MaintenanceItemTimeout"
        suffix = "cancellation request failed" if cancel_fails else "cancellation requested"
        assert (
            outcome.failure_message
            == f"timed out after 0.05s; {suffix}; operation may still be running"
        )
        assert "cancelled" not in outcome.failure_message
        assert outcome.detail["cancellation_requested"] == json.dumps(not cancel_fails)
        assert outcome.removed_count is None
        group_id = session.sparkContext.setJobGroup.call_args.args[0]
        session.sparkContext.cancelJobGroup.assert_called_once_with(group_id)
        captured = outcome.to_dict()
    finally:
        release.set()
        assert finished.wait(1)
    assert outcome.to_dict() == captured


def test_job_group_is_set_and_cleared_on_the_worker_thread():
    session = Session()
    group_threads = []
    session.sparkContext.setJobGroup.side_effect = lambda *a, **kw: group_threads.append(
        get_ident()
    )
    session.sparkContext.setLocalProperty.side_effect = lambda *a: group_threads.append(get_ident())
    assert _execute(session).status == "applied"
    assert len(group_threads) == 4
    assert set(group_threads) == set(session.threads)
    assert get_ident() not in group_threads
    assert session.sparkContext.setJobGroup.call_args.kwargs == {"interruptOnCancel": True}


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_worker_interruptions_propagate_unchanged(interruption):
    def interrupted(_statement):
        raise interruption

    session = Session(call=interrupted)
    with pytest.raises(type(interruption)) as caught:
        _execute(session)
    assert caught.value is interruption
    session.sparkContext.cancelJobGroup.assert_called_once()


def test_job_group_cleanup_interruption_propagates():
    session = Session()
    session.sparkContext.setLocalProperty.side_effect = KeyboardInterrupt("cleanup interrupted")
    with pytest.raises(KeyboardInterrupt, match="cleanup interrupted"):
        _execute(session)


def test_duration_uses_the_injected_clock():
    ticks = iter((10, 10, 10.25))
    outcome = _execute(Session(), clock=lambda: next(ticks))
    assert outcome.duration_seconds == 0.25


def test_skips_never_touch_spark():
    item = replace(_item(), skipped_reason="absent_table")
    outcome = _execute(None, item)
    assert outcome == ItemOutcome.from_planned_item(item)


def test_unsupported_zone_is_a_recorded_failure():
    item = PlannedItem("metadata", "history.json", "delete_file", {})
    outcome = _execute(None, item)
    assert outcome.status == "failed"
    assert outcome.failure_type == "MaintenanceExecutionUnavailable"


def test_expiration_then_orphans_then_compaction_follow_the_planner(now, policy_config):
    policy_config["maintenance"]["bronze"].update(
        remove_orphan_files=True, compact={"enabled": True, "target_file_size_mb": 128}
    )
    policy = resolve_maintenance_settings(policy_config)
    inventory = _inventory(now)
    plan = plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    session = Session()
    outcomes = execute.execute_retention(plan, policy=policy, session=session, catalog_name="janus")
    assert [item.action for item in outcomes] == [
        "expire_snapshots",
        "remove_orphan_files",
        "rewrite_data_files",
    ]
    assert all(item.status == "applied" for item in outcomes)


def _inventory(now, count=1):
    return MaintenanceInventory(
        bronze=tuple(
            BronzeTableInventory(
                f"bronze.table{index}",
                (CLEAN_PRODUCER,),
                tuple(
                    SnapshotEntry(value, now - timedelta(days=100 - value), value == 4)
                    for value in range(1, 5)
                ),
            )
            for index in range(count)
        )
    )


@pytest.fixture
def cli_project(tmp_path, policy_config):
    root = materialize(CLEAN, tmp_path / "project")
    path = install_profile(root, "local")
    profile = yaml.safe_load(path.read_text())
    profile.update(policy_config)
    path.write_text(yaml.safe_dump(profile))
    return root


def _cli_session(monkeypatch, session):
    provider = MagicMock()
    provider.get.return_value = session
    provider.take_cleanup_failures.return_value = ()
    monkeypatch.setattr(maintain, "SparkSessionProvider", lambda *a: provider)
    return provider


def test_dry_run_never_reaches_execute_item(cli_project, monkeypatch):
    session = Session()
    _cli_session(monkeypatch, session)
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: _inventory(a[4]))
    executor = MagicMock(side_effect=AssertionError("dry run executed a procedure"))
    monkeypatch.setattr(execute, "execute_item", executor)
    result = run_janus(["maintain", "--project-root", str(cli_project), "--zone", "bronze"])
    assert result.exit_code == 0, result.output
    executor.assert_not_called()
    assert session.statements == []


def test_apply_persists_extended_execution_details(cli_project, monkeypatch):
    session = Session(before=(1, 2, 3, 4), after=(2, 3, 4))
    provider = _cli_session(monkeypatch, session)
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: _inventory(a[4]))
    result = run_janus(
        [
            "maintain",
            "--project-root",
            str(cli_project),
            "--zone",
            "bronze",
            "--apply",
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    (path,) = (cli_project / "data/metadata/maintenance").glob("*.json")
    assert result.stdout.encode() == path.read_bytes()
    record = json.loads(path.read_text())
    assert record["items"][0]["expired_snapshot_ids"] == [1]
    assert record["items"][0]["detail"]["surviving_snapshot_ids"] == "[2, 3, 4]"
    provider.stop.assert_called_once()


def test_keyboard_interrupt_persists_completed_and_pending_items(cli_project, monkeypatch):
    calls = []

    def call(statement):
        calls.append(statement)
        if len(calls) == 2:
            raise KeyboardInterrupt("operator interruption")
        return [Row(**COUNTS)]

    session = Session(call=call)
    session.snapshots = deque(((1, 2, 3, 4), (2, 3, 4), (1, 2, 3, 4)))
    provider = _cli_session(monkeypatch, session)
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: _inventory(a[4], 4))
    with pytest.raises(KeyboardInterrupt, match="operator interruption"):
        run_janus(["maintain", "--project-root", str(cli_project), "--zone", "bronze", "--apply"])
    (path,) = (cli_project / "data/metadata/maintenance").glob("*.json")
    record = json.loads(path.read_text())
    assert record["dry_run"] is False
    assert [item["status"] for item in record["items"]] == [
        "applied",
        "planned",
        "planned",
        "planned",
    ]
    assert record["items"][0]["expired_snapshot_ids"] == [1]
    assert record["failures"][0]["stage"] == "interrupted"
    assert all(
        item["removed_count"] is None and item["expired_snapshot_ids"] == []
        for item in record["items"][1:]
    )
    assert record["zone_summaries"][0]["removed_count"] == 15
    assert len(calls) == 2
    provider.stop.assert_called_once()


def test_orphan_dry_run_renders_caution(cli_project, monkeypatch, policy_config):
    policy = resolve_maintenance_settings(policy_config)
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: _inventory(a[4]))
    monkeypatch.setattr(
        maintain,
        "resolve_maintenance_settings",
        lambda config: replace(policy, bronze=replace(policy.bronze, remove_orphan_files=True)),
    )
    _cli_session(monkeypatch, Session())
    result = run_janus(["maintain", "--project-root", str(cli_project), "--zone", "bronze"])
    assert result.exit_code == 0, result.output
    assert f"⚠ {maintain.ORPHAN_WARNING}" in result.stdout
    assert "⚠ remove_orphan_files" in result.stdout


def test_only_execute_module_can_call_maintenance_procedures():
    root = Path(inspect.getfile(janus)).parent
    modules = {str(path.relative_to(root)): path.read_text() for path in root.rglob("*.py")}
    assert len(modules) >= 180
    # Policy attributes and planner action labels share these names. The boundary
    # covers executable SQL; declarations are not procedure calls.
    pattern = re.compile(
        r"system\.(expire_snapshots|remove_orphan_files|rewrite_data_files|rewrite_manifests)\("
    )
    callers = {
        path
        for path, source in modules.items()
        if any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and pattern.search(node.value)
            for node in ast.walk(ast.parse(source))
        )
    }
    assert callers == {"maintenance/execute.py"}
    executor = modules["maintenance/execute.py"]
    assert all(
        name in executor
        for name in (
            "expire_snapshots",
            "remove_orphan_files",
            "rewrite_data_files",
            "rewrite_manifests",
        )
    )
    assert "timedelta" not in executor and "committed_at" not in executor
    assert all(word not in executor for word in ("for attempt", "max_attempts", "backoff"))
    assert len(executor.splitlines()) < 600


def test_execution_details_extend_but_cannot_replace_plan_arguments(now):
    plan = RetentionPlan((_item(),), (), now, "a" * 64)
    outcome = _execute(Session())
    kwargs = dict(
        environment="local", dry_run=False, zones=("bronze",), started_at=now, ended_at=now
    )
    assert MaintenanceRecord.from_plan(plan, items=(outcome,), **kwargs).items == (outcome,)
    wrong = replace(outcome, detail={**outcome.detail, "retain_last": "99"})
    with pytest.raises(ValueError, match="must match every planned item"):
        MaintenanceRecord.from_plan(plan, items=(wrong,), **kwargs)
