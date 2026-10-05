"""command selection, dry-run evidence, exit codes and deferred compute."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from janus.cli import maintain
from janus.maintenance.inventory import (
    BronzeTableInventory,
    MaintenanceInventory,
    MetadataZoneInventory,
    RunArtifactEntry,
    SnapshotEntry,
)
from janus.maintenance.planning import PlannedItem, ProtectedItem, RetentionPlan
from janus.maintenance.records import ItemOutcome, MaintenanceRecordStore
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.support.semantics_fixtures import (
    CLEAN,
    CLEAN_CONSUMER,
    CLEAN_PRODUCER,
    install_profile,
    materialize,
)


@pytest.fixture(autouse=True)
def fake_compute(monkeypatch):
    """Keep command tests independent of Spark; lifecycle cases install their own spy."""
    session = MagicMock()
    session.sql.return_value.collect.return_value = []

    class Provider:
        def __init__(self, *args):
            pass

        def get(self):
            return session

        def stop(self):
            pass

        def take_cleanup_failures(self):
            return ()

    monkeypatch.setattr(maintain, "SparkSessionProvider", Provider)
    return session


@pytest.fixture
def project(tmp_path, policy_config):
    root = materialize(CLEAN, tmp_path / "project")
    profile_path = install_profile(root, "local")
    profile = yaml.safe_load(profile_path.read_text())
    profile.update(policy_config)
    profile_path.write_text(yaml.safe_dump(profile))
    return root


def _invoke(project: Path, *args: str):
    return run_janus(["maintain", "--project-root", str(project), *args])


def _record(project: Path):
    (path,) = (project / "data/metadata/maintenance").glob("*.json")
    return path, json.loads(path.read_text())


def _change_profile(project: Path, update):
    path = project / "conf/environments/local.yaml"
    profile = yaml.safe_load(path.read_text())
    update(profile)
    path.write_text(yaml.safe_dump(profile))


def test_no_flags_default_to_dry_run_and_persist_evidence(project, monkeypatch):
    monkeypatch.chdir(project)

    result = run_janus(["maintain"])

    assert result.exit_code == 0, result.output
    _, record = _record(project)
    assert record["dry_run"] is True
    assert len(record["items"]) == 2
    assert all(item["detail"]["skipped_reason"] == "absent_table" for item in record["items"])
    assert record["zones"] == ["bronze", "lineage", "metadata", "runs-table"]
    assert record["lock"] == "none"
    assert "DRY RUN (nothing will be deleted)" in result.stdout
    assert maintain.LOCK_WARNING in result.stdout
    assert "skipped: absent_table" in result.stdout
    assert record["zone_summaries"][0]["items_skipped"] == 2


@pytest.mark.parametrize("mode", [(), ("--dry-run",), ("--apply",)])
def test_json_is_byte_identical_to_persisted_record(project, mode):
    result = _invoke(project, *mode, "--format", "json")

    assert result.exit_code == 0, result.output
    path, record = _record(project)
    assert result.stdout.encode() == path.read_bytes()
    assert json.loads(result.stdout) == record
    assert record["dry_run"] is (mode != ("--apply",))
    assert all(item["status"] == "skipped" for item in record["items"])
    assert maintain.LOCK_WARNING in result.stderr


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda p: p.pop("maintenance"), "maintenance"),
        (lambda p: p.update(maintenance="wrong"), "maintenance must be a mapping"),
        (lambda p: p["maintenance"]["bronze"].update(retain_last=0), "retain_last"),
        (lambda p: p["maintenance"].update(dry_run=False), "dry_run"),
    ],
    ids=["absent", "not-a-mapping", "invalid-floor", "apply-default-forbidden"],
)
def test_unusable_policy_refuses_before_inventory(project, monkeypatch, update, message):
    _change_profile(project, update)

    def forbidden(*args, **kwargs):
        pytest.fail("refused profile reached inventory")

    monkeypatch.setattr(maintain, "collect_inventory", forbidden)
    result = _invoke(project)

    assert result.exit_code == 2
    assert message in result.stderr
    assert result.stdout == ""
    assert not (project / "data/metadata/maintenance").exists()
    assert "\n" not in result.stderr.rstrip("\n")


@pytest.mark.parametrize("failure", ["profile-missing", "yaml-invalid", "registry-missing"])
def test_load_failures_exit_two(project, failure):
    profile = project / "conf/environments/local.yaml"
    if failure == "profile-missing":
        profile.rename(profile.with_suffix(".unused"))
    elif failure == "yaml-invalid":
        profile.write_text("maintenance: [\n")
    else:
        app = project / "conf/app.yaml"
        app.rename(app.with_suffix(".unused"))
    result = _invoke(project)

    assert result.exit_code == 2
    assert result.stderr.startswith("janus maintain: ")
    assert result.stdout == ""
    assert not (project / "data/metadata/maintenance").exists()


def test_runtime_permission_failure_uses_existing_message(project, monkeypatch):
    def denied(*args):
        raise PermissionError(13, "Permission denied", "/workspace/data/raw")

    monkeypatch.setattr(maintain, "prepare_runtime", denied)
    result = _invoke(project)

    assert result.exit_code == 2
    assert "JANUS could not prepare the runtime path" in result.stderr
    assert "make down && make up" in result.stderr


@pytest.mark.parametrize("source_id", ["does-not-exist", " "])
def test_unknown_source_refuses_with_exit_two(project, source_id):
    result = _invoke(project, "--zone", "bronze", "--source-id", source_id)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert not (project / "data/metadata/maintenance").exists()


def test_explicit_runs_table_cannot_be_filtered_by_source(project):
    result = _invoke(project, "--zone", "runs-table", "--source-id", CLEAN_PRODUCER)

    assert result.exit_code == 2
    assert "--source-id cannot restrict --zone runs-table" in result.stderr
    assert "shared" in result.stderr and "whole partitions" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "options",
    [
        ("--execute",),
        ("--run-id", "run-1"),
        ("--ingest-raw-to-bronze",),
        ("--with-spark",),
        ("--bronze-table", "bronze.example"),
        ("--include-disabled",),
        ("--tag", "daily"),
        ("--domain", "example"),
        ("--max-parallel", "2"),
        ("--started-at", "2026-10-05T12:00:00Z"),
        ("--run-id=run-1",),
        ("--run-id",),
    ],
)
def test_other_verbs_options_are_refused_by_name(project, options):
    result = _invoke(project, *options)

    assert result.exit_code == 2
    option = options[0].partition("=")[0]
    assert f"janus maintain: {option} is not a maintain option" in result.stderr
    assert "unrecognized arguments" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "options",
    [("--apply", "--dry-run"), ("--dry-run", "--apply"), ("--zone", "unknown")],
)
def test_parser_refuses_contradictory_modes_and_unknown_zones(project, options):
    result = _invoke(project, *options)

    assert result.exit_code == 2
    assert "janus maintain: error:" in result.stderr
    assert not (project / "data/metadata/maintenance").exists()


def test_raw_requires_explicit_enabled_policy(project):
    result = _invoke(project, "--zone", "raw")

    assert result.exit_code == 2
    assert "maintenance.raw.enabled: true" in result.stderr


def test_enabled_raw_is_in_default_zone_set(project):
    _change_profile(project, lambda p: p["maintenance"]["raw"].update(enabled=True))
    result = _invoke(project, "--format", "json")

    assert result.exit_code == 0
    assert "raw" in json.loads(result.stdout)["zones"]


def test_repeatable_selection_is_deduplicated_and_threaded(project, monkeypatch):
    received = {}

    def collect(registry, config, paths, policy, now, **kwargs):
        received.update(kwargs)
        assert {source.source_id for source in registry.sources} == {
            CLEAN_PRODUCER,
            CLEAN_CONSUMER,
        }
        return MaintenanceInventory()

    monkeypatch.setattr(maintain, "collect_inventory", collect)
    result = _invoke(
        project,
        "--zone",
        "bronze",
        "--zone",
        "metadata",
        "--zone",
        "bronze",
        "--source-id",
        CLEAN_CONSUMER,
        "--source-id",
        CLEAN_PRODUCER,
        "--source-id",
        CLEAN_CONSUMER,
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert record["zones"] == ["bronze", "metadata"]
    assert record["source_ids"] == sorted([CLEAN_PRODUCER, CLEAN_CONSUMER])
    assert received["zones"] == frozenset(record["zones"])
    assert received["source_ids"] == frozenset(record["source_ids"])


def test_default_shared_zones_remain_selected_with_source_filter(project):
    result = _invoke(project, "--source-id", CLEAN_PRODUCER, "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["zones"] == ["bronze", "lineage", "metadata", "runs-table"]


@pytest.mark.parametrize("zone", ["metadata", "lineage"])
@pytest.mark.parametrize("mode", ["--dry-run", "--apply"])
def test_file_zones_never_construct_provider(project, monkeypatch, zone, mode):
    arm_spark_tripwire(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("file maintenance constructed a provider")

    monkeypatch.setattr(maintain, "SparkSessionProvider", forbidden)
    result = _invoke(project, "--zone", zone, mode)

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("stage", ["success", "collect-failure", "apply-failure", "apply-success"])
def test_one_deferred_provider_stops_on_every_path(project, monkeypatch, stage):
    calls = []
    providers = []

    class Provider:
        def __init__(self, *args):
            calls.append("build")
            providers.append(self)

        def get(self):
            calls.append("get")
            return live_session

        def stop(self):
            calls.append("stop")

        def take_cleanup_failures(self):
            calls.append("cleanup")
            return ()

    live_session = object()

    def collect(*args, session, **kwargs):
        calls.append("collect")
        assert session is live_session
        if stage == "collect-failure":
            raise RuntimeError("inventory unavailable")
        if stage in {"apply-failure", "apply-success"}:
            return _bronze_inventory(args[4])
        return MaintenanceInventory()

    def execute(*args, session, **kwargs):
        calls.append("execute")
        assert session is live_session
        if stage == "apply-failure":
            raise RuntimeError("scripted apply failure")
        return tuple(
            ItemOutcome.from_planned_item(item)
            if item.skipped_reason is not None
            else replace(ItemOutcome.from_planned_item(item), status="applied")
            for item in args[0].items
        )

    monkeypatch.setattr(maintain, "SparkSessionProvider", Provider)
    monkeypatch.setattr(maintain, "collect_inventory", collect)
    monkeypatch.setattr(maintain, "execute_retention", execute)
    result = _invoke(project, "--zone", "bronze", "--apply")

    assert result.exit_code == (0 if stage in {"success", "apply-success"} else 1), result.output
    assert len(providers) == 1
    expected = ["build", "get", "collect"]
    if stage in {"apply-failure", "apply-success"}:
        expected.append("execute")
    assert calls == [*expected, "stop", "cleanup"]
    if stage == "apply-failure":
        _, record = _record(project)
        failed = next(item for item in record["items"] if item["status"] == "failed")
        assert failed["failure_message"] == "scripted apply failure"
        assert failed["action"] == "expire_snapshots"


def _bronze_inventory(now):
    return MaintenanceInventory(
        bronze=(
            BronzeTableInventory(
                "bronze.example",
                (CLEAN_PRODUCER,),
                tuple(
                    SnapshotEntry(value, now - timedelta(days=100 - value), value == 4)
                    for value in range(1, 5)
                ),
            ),
            BronzeTableInventory(
                "bronze.absent", (CLEAN_PRODUCER,), (), unavailable_reason="absent_table"
            ),
        )
    )


def test_text_renders_arguments_snapshot_ids_skips_and_protected_items(project, monkeypatch):
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: _bronze_inventory(a[4]))
    result = _invoke(project, "--zone", "bronze")

    assert result.exit_code == 0, result.output
    assert "older_than=" in result.stdout and "retain_last=3" in result.stdout
    assert "would expire 1 snapshots: 1" in result.stdout
    assert "skipped: absent_table" in result.stdout
    assert "protected (not candidates)" in result.stdout
    assert "bronze.example#4  current_snapshot" in result.stdout
    assert "bronze.example#3  retain_last" in result.stdout


def test_apply_records_unavailable_metadata_executor_and_exits_one(project, monkeypatch):
    candidate = project / "data/metadata/example/lineage/orphan.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text('{"retained": true}\n')

    def collect(*args, **kwargs):
        now = args[4]
        return MaintenanceInventory(
            metadata=MetadataZoneInventory(
                (
                    RunArtifactEntry(
                        "lineage", candidate, CLEAN_PRODUCER, None, now - timedelta(days=200)
                    ),
                ),
                frozenset(),
                {},
            )
        )

    monkeypatch.setattr(maintain, "collect_inventory", collect)
    before = candidate.read_bytes()
    result = _invoke(project, "--zone", "metadata", "--apply", "--format", "json")

    assert result.exit_code == 1, result.output
    path, record = _record(project)
    assert result.stdout.encode() == path.read_bytes()
    assert record["dry_run"] is False
    failures = [item for item in record["items"] if item["status"] == "failed"]
    assert len(failures) == 1
    assert failures[0]["failure_type"] == "MaintenanceExecutionUnavailable"
    assert failures[0]["removed_count"] == failures[0]["removed_bytes"] == 0
    assert record["zone_summaries"][0]["items_failed"] == 1
    assert candidate.read_bytes() == before
    events = [
        json.loads(line)["event"] for line in result.stderr.splitlines() if line.startswith("{")
    ]
    assert "maintenance_item_failed" in events


def test_persistence_failure_still_prints_record_and_exits_one(project, monkeypatch):
    def denied(*args):
        raise PermissionError("evidence destination is read-only")

    monkeypatch.setattr(MaintenanceRecordStore, "persist", denied)
    result = _invoke(project, "--format", "json")

    assert result.exit_code == 1
    assert all(item["status"] == "skipped" for item in json.loads(result.stdout)["items"])
    assert "could not persist maintenance record" in result.stderr


def test_unusable_maintenance_id_refuses_before_apply(project, monkeypatch):
    import janus.maintenance.records as records

    monkeypatch.setattr(records, "default_maintenance_run_id", lambda *args: "../escape")

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe identity reached apply")

    monkeypatch.setattr(maintain, "execute_retention", forbidden)
    result = _invoke(project, "--apply")

    assert result.exit_code == 2
    assert "maintenance_run_id" in result.stderr and "not usable" in result.stderr
    assert not (project / "data/metadata/maintenance").exists()


def test_planner_and_record_share_one_wall_clock_read(project, monkeypatch):
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    clocks = []
    observed = []

    class Clock:
        @staticmethod
        def now(*, tz):
            clocks.append(tz)
            return now

    def collect(*args, **kwargs):
        observed.append(args[4])
        return _bronze_inventory(now)

    monkeypatch.setattr(maintain, "datetime", Clock)
    monkeypatch.setattr(maintain, "collect_inventory", collect)
    result = _invoke(project, "--zone", "bronze", "--format", "json")

    assert result.exit_code == 0, result.output
    _, record = _record(project)
    assert clocks == [UTC] and observed == [now]
    assert record["started_at"] == now.isoformat()
    action = next(item for item in record["items"] if item["status"] == "planned")
    assert action["detail"]["older_than"] == (now - timedelta(days=30)).isoformat()
    assert datetime.fromisoformat(record["ended_at"]) - now == timedelta(
        seconds=record["duration_seconds"]
    )


def test_default_inventory_changes_no_history_or_warehouse_file(project):
    for relative in (
        "data/metadata/example/checkpoints/current.json",
        "data/metadata/example/runs/old.json",
        "data/metadata/example/extraction_progress.json",
        "data/metadata/lineage/openlineage/events-2020-01-01.ndjson",
        "data/bronze/example/metadata/snapshot.json",
        "data/raw/example/runs/old/data.csv",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserved\n")

    def digest():
        return {
            str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (project / "data").rglob("*")
            if path.is_file() and "maintenance" not in path.parts
        }

    before = digest()
    result = _invoke(project)

    assert result.exit_code == 0
    assert digest() == before


def test_run_execute_ignores_malformed_maintenance_block(project, monkeypatch):
    from janus.cli import run

    _change_profile(project, lambda p: p.update(maintenance="invalid on purpose"))
    seen = []

    class Executor:
        def __init__(self, **kwargs):
            pass

        def execute(self, planned_run, provider, config):
            seen.append(planned_run.plan.source_config.source_id)
            assert config["maintenance"] == "invalid on purpose"
            assert provider.was_started is False
            return SimpleNamespace(is_successful=True, to_summary=lambda: {"status": "succeeded"})

    def forbidden(*args):
        pytest.fail("ingestion consulted maintenance settings")

    monkeypatch.setattr(maintain, "resolve_maintenance_settings", forbidden)
    monkeypatch.setattr(run, "SourceExecutor", Executor)
    arm_spark_tripwire(monkeypatch)
    result = run_janus(
        [
            "run",
            "--project-root",
            str(project),
            "--source-id",
            CLEAN_PRODUCER,
            "--execute",
        ]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["executed_run"]["status"] == "succeeded"
    assert seen == [CLEAN_PRODUCER]


def test_source_filter_does_not_filter_shared_lineage_events(project, monkeypatch):
    from janus.maintenance.inventory import EventFileEntry

    def collect(*args, **kwargs):
        now = args[4]
        return MaintenanceInventory(
            lineage_events=(
                EventFileEntry(
                    Path("/metadata/events-old.ndjson"), now.date() - timedelta(days=200)
                ),
                EventFileEntry(Path("/metadata/events-today.ndjson"), now.date()),
            )
        )

    monkeypatch.setattr(maintain, "collect_inventory", collect)
    result = _invoke(
        project, "--zone", "lineage", "--source-id", CLEAN_PRODUCER, "--format", "json"
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["items"][0]["target"] == "/metadata/events-old.ndjson"


def test_text_apply_renders_failed_items_and_zero_empty_zones(project, monkeypatch, fake_compute):
    fake_compute.sql.side_effect = RuntimeError("catalog unavailable")
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: _bronze_inventory(a[4]))
    result = _invoke(project, "--apply")

    assert result.exit_code == 1
    assert "janus maintain — APPLY" in result.stdout
    assert "failed: RuntimeError: catalog unavailable" in result.stdout
    assert "nothing was deleted in: metadata" in result.stdout
    assert maintain.LOCK_WARNING in result.stdout


def test_empty_apply_retains_skipped_items(project, monkeypatch):
    inventory = MaintenanceInventory(
        bronze=(
            BronzeTableInventory(
                "bronze.absent", (CLEAN_PRODUCER,), (), unavailable_reason="absent_table"
            ),
        )
    )
    monkeypatch.setattr(maintain, "collect_inventory", lambda *a, **kw: inventory)
    result = _invoke(project, "--zone", "bronze", "--apply", "--format", "json")

    assert result.exit_code == 0, result.output
    _, record = _record(project)
    assert [item["status"] for item in record["items"]] == ["skipped"]


def test_text_renderer_explains_file_counts_and_protected_state(now):
    plan = RetentionPlan(
        (PlannedItem("metadata", "/metadata/runs/old.json", "delete_file", {}, 128),),
        (ProtectedItem("metadata", "/metadata/checkpoints/current.json", "state_file"),),
        now,
        "a" * 64,
    )
    from janus.maintenance.records import MaintenanceRecord

    record = MaintenanceRecord.from_plan(
        plan, environment="local", dry_run=True, zones=("metadata",), started_at=now, ended_at=now
    )
    output = maintain._render_text(plan, record)

    assert "planned: count=1 bytes=128" in output
    assert "/metadata/checkpoints/current.json  state_file" in output
