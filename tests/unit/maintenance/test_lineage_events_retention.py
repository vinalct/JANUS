import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from threading import Event

import pytest
import yaml

from janus.cli import maintain
from janus.maintenance import execute, inventory
from janus.maintenance.errors import MaintenanceInvariantError
from janus.maintenance.planning import PlannedItem
from janus.maintenance.records import MaintenanceRecord
from janus.observability.openlineage.settings import resolve_openlineage_settings
from janus.observability.openlineage.transport import (
    FileOpenLineageTransport,
    resolve_openlineage_transport,
)
from janus.runtime import SparkSessionProvider
from tests.support.maintenance_zone import SOURCES, zone_plans
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.support.semantics_fixtures import (
    CLEAN,
    CLEAN_CONSUMER,
    install_profile,
    materialize,
)


@pytest.fixture
def lineage_project(tmp_path, planted, policy_config, monkeypatch):
    root = materialize(CLEAN, tmp_path / "project")
    profile_path = install_profile(root, "local")
    profile = yaml.safe_load(profile_path.read_text())
    profile.update(policy_config)
    profile["storage"]["metadata_dir"] = str(planted.root)
    profile["observability"] = {
        "openlineage": {"transport": "file", "path": f"{SOURCES[0]}/lineage/openlineage"}
    }
    profile_path.write_text(yaml.safe_dump(profile))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return planted.now

    monkeypatch.setattr(maintain, "datetime", Clock)
    arm_spark_tripwire(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("lineage maintenance constructed a Spark provider")

    monkeypatch.setattr(SparkSessionProvider, "__init__", forbidden)
    return root


def _invoke(root, *args):
    return run_janus(["maintain", "--project-root", str(root), "--zone", "lineage", *args])


def _plan(planted, config):
    inventory = import_module("janus.maintenance.inventory")
    settings = resolve_openlineage_settings(
        {"observability": {"openlineage": {"transport": "file", "path": "lineage/openlineage"}}}
    )
    events = inventory.collect_lineage_event_files(zone_plans(planted), settings, now=planted.now)
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(config)
    plan = import_module("janus.maintenance.planning").plan_retention(
        inventory.MaintenanceInventory(lineage_events=events),
        policy,
        planted.now,
        zones=frozenset({"lineage"}),
    )
    return events, plan


def test_filename_day_is_used_instead_of_contents_or_mtime(planted, policy_config, monkeypatch):
    # Every payload carries today's eventTime, even in the ancient day files.
    plans = zone_plans(planted)
    settings = resolve_openlineage_settings(
        {"observability": {"openlineage": {"transport": "file"}}}
    )
    for path in planted.root.glob("*/lineage/openlineage/*.ndjson"):
        os.utime(path, (planted.now.timestamp(), planted.now.timestamp()))

    def guarded_open(*args, **kwargs):
        pytest.fail("collector opened a file")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        events = inventory.collect_lineage_event_files(plans, settings, now=planted.now)
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(policy_config)
    plan = import_module("janus.maintenance.planning").plan_retention(
        inventory.MaintenanceInventory(lineage_events=events),
        policy,
        planted.now,
        zones=frozenset({"lineage"}),
    )
    assert {Path(item.target) for item in plan.items} == {
        path for path in planted.candidate_paths if path.suffix == ".ndjson"
    }


def test_today_and_exact_boundary_are_protected(planted, policy_config):
    _, plan = _plan(planted, policy_config)
    targets = {Path(item.target) for item in plan.items}
    for age in (0, 90):
        day = (planted.now - timedelta(days=age)).date()
        for source in SOURCES:
            path = planted.root / source / "lineage/openlineage" / f"events-{day}.ndjson"
            assert path in planted.protected_paths
            assert path not in targets


def _leave_only_ancient_days(planted):
    cutoff = planted.now.date() - timedelta(days=90)
    for path in planted.root.glob("*/lineage/openlineage/events-*.ndjson"):
        if path.stem.removeprefix("events-") >= cutoff.isoformat():
            path.unlink()


def test_most_recent_file_is_protected_even_when_old(planted, policy_config):
    import_module("janus.maintenance.inventory")
    _leave_only_ancient_days(planted)
    events, plan = _plan(planted, policy_config)
    targets = {item.target for item in plan.items}
    latest = max(entry.day for entry in events)
    assert latest < planted.now.date() - timedelta(days=90)
    assert all(str(entry.path) not in targets for entry in events if entry.day == latest)
    assert {item.reason for item in plan.protected} == {"most_recent_file"}
    assert targets  # older files are actually condemned


def test_today_and_latest_reasons_share_one_record_entry(planted, policy_config):
    _, plan = _plan(planted, policy_config)
    today = planted.now.date()
    paths = {
        str(planted.root / source / "lineage/openlineage" / f"events-{today}.ndjson")
        for source in SOURCES
    }
    for path in paths:
        assert {item.reason for item in plan.protected if item.target == path} == {
            "todays_file",
            "most_recent_file",
        }
    record = MaintenanceRecord.from_plan(
        plan,
        environment="local",
        dry_run=True,
        zones=("lineage",),
        started_at=planted.now,
        ended_at=planted.now,
    ).to_dict()
    protected = [item for item in record["protected"] if item["target"] in paths]
    assert len(protected) == len(paths)
    assert all(item["reasons"] == ["most_recent_file", "todays_file"] for item in protected)


@pytest.mark.parametrize(
    "name",
    [
        "events-.ndjson",
        "events-2026-13-45.ndjson",
        "events-renamed.ndjson",
        "events-undated.ndjson",
        "events-20260607.ndjson",
        "events-2026-W01-1.ndjson",
    ],
)
def test_invalid_filename_is_skipped_and_recorded(planted, policy_config, name):
    path = planted.root / SOURCES[0] / "lineage/openlineage" / name
    path.write_bytes(b"this payload is never read")
    events, plan = _plan(planted, policy_config)
    (entry,) = (entry for entry in events if entry.path == path)
    assert entry.day is None and entry.skipped_reason == "invalid_event_filename"
    (item,) = (item for item in plan.items if item.target == str(path))
    assert item.skipped_reason == "invalid_event_filename"
    outcome = execute.execute_metadata_item(item)
    assert outcome.status == "skipped"
    assert outcome.detail["skipped_reason"] == "invalid_event_filename"
    assert path.read_bytes() == b"this payload is never read"


@pytest.mark.parametrize("transport", ["http", "disabled", "absent"])
def test_nonfile_transport_never_walks_or_consumes_plans(planted, monkeypatch, transport):
    block = {"transport": transport, "url": "https://lineage.example.invalid"}
    settings = resolve_openlineage_settings(
        {} if transport == "absent" else {"observability": {"openlineage": block}}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("nonfile transport walked a directory")

    def plans():
        pytest.fail("nonfile transport consumed plans")
        yield

    monkeypatch.setattr(Path, "glob", forbidden)
    assert inventory.collect_lineage_event_files(plans(), settings, now=planted.now) == ()


def test_configured_directory_matches_the_transport_and_is_walked_once(planted, monkeypatch):
    directory = planted.root / "custom/events"
    directory.mkdir(parents=True)
    path = directory / "events-2020-01-01.ndjson"
    path.write_bytes(b"")
    settings = resolve_openlineage_settings(
        {"observability": {"openlineage": {"transport": "file", "path": "custom/events"}}}
    )
    paths = {"metadata_dir": planted.root}
    transport = resolve_openlineage_transport(settings, paths)
    assert transport.directory == directory
    walked = []
    original = Path.glob

    def glob(path, pattern):
        walked.append(path)
        return original(path, pattern)

    monkeypatch.setattr(Path, "glob", glob)
    events = inventory.collect_lineage_event_files(
        (),
        settings,
        now=planted.now,
        resolved_paths=paths,
    )
    assert [entry.path for entry in events] == [path]
    assert walked == [transport.directory]


@pytest.mark.parametrize("directory", ["../escaped", "/tmp/escaped"])
def test_events_directory_uses_the_transport_containment_rule(planted, directory):
    settings = resolve_openlineage_settings(
        {"observability": {"openlineage": {"transport": "file", "path": directory}}}
    )
    with pytest.raises(ValueError, match="inside the metadata zone"):
        inventory.collect_lineage_event_files(zone_plans(planted), settings, now=planted.now)


def test_missing_directory_is_empty_and_not_created(planted):
    plans = zone_plans(planted)
    settings = resolve_openlineage_settings(
        {"observability": {"openlineage": {"transport": "file", "path": "missing/events"}}}
    )
    assert inventory.collect_lineage_event_files(plans, settings, now=planted.now) == ()
    assert all(not (planted.root / source / "missing").exists() for source in SOURCES)


def test_duplicate_plan_roots_and_matching_directories_are_not_entries(planted):
    plans = zone_plans(planted)
    path = planted.root / SOURCES[0] / "lineage/openlineage/events-1990-01-01.ndjson"
    path.mkdir()
    settings = resolve_openlineage_settings(
        {"observability": {"openlineage": {"transport": "file"}}}
    )
    events = inventory.collect_lineage_event_files((*plans, *plans), settings, now=planted.now)
    assert len(events) == 10
    assert len({entry.path for entry in events}) == len(events)
    assert path not in {entry.path for entry in events}


def test_lineage_executor_refuses_protected_paths_before_io(planted, monkeypatch):
    path = planted.root / SOURCES[0] / "lineage/openlineage" / f"events-{planted.now.date()}.ndjson"

    def forbidden(*args, **kwargs):
        pytest.fail("protected event file was touched")

    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(MaintenanceInvariantError, match="protected metadata path"):
        execute.execute_metadata_item(
            PlannedItem("lineage", str(path), "delete_file", {}),
            protected_paths=frozenset({path}),
        )


def test_cli_dry_run_apply_evidence_and_idempotence(lineage_project, planted):
    directory = planted.root / SOURCES[0] / "lineage/openlineage"
    invalid = directory / "events-undated.ndjson"
    invalid.write_bytes(b"not read")
    before = {path: path.read_bytes() for path in planted.root.rglob("*.ndjson")}
    dry = _invoke(lineage_project, "--format", "json", "--source-id", CLEAN_CONSUMER)
    assert dry.exit_code == 0, dry.output
    dry_record = json.loads(dry.stdout)
    assert dry_record["dry_run"] is True
    assert {path: path.read_bytes() for path in before} == before
    candidates = [item for item in dry_record["items"] if item["status"] == "planned"]
    assert {Path(item["target"]) for item in candidates} == {
        path for path in planted.candidate_paths if path.parent == directory
    }
    text = _invoke(lineage_project)
    today = directory / f"events-{planted.now.date()}.ndjson"
    assert text.exit_code == 0, text.output
    assert text.stdout.count(str(today)) == 1
    assert f"{today}  most_recent_file, todays_file" in text.stdout
    applied = _invoke(lineage_project, "--apply", "--format", "json")
    assert applied.exit_code == 0, applied.output
    record = json.loads(applied.stdout)
    assert record["plan_digest"] == dry_record["plan_digest"]
    assert record["protected"] == dry_record["protected"]
    assert record["zone_summaries"][0]["removed_bytes"] == sum(
        len(before[Path(item["target"])]) for item in candidates
    )
    for item in record["items"]:
        path = Path(item["target"])
        if path == invalid:
            assert item["status"] == "skipped"
            assert item["detail"]["skipped_reason"] == "invalid_event_filename"
            assert path.exists()
        else:
            assert item["status"] == "applied" and not path.exists()
    for path, payload in before.items():
        if path.exists():
            assert path.read_bytes() == payload
    persisted = planted.root / "maintenance" / f"{record['maintenance_run_id']}.json"
    assert json.loads(persisted.read_text()) == record
    repeated = _invoke(lineage_project, "--apply", "--format", "json")
    assert repeated.exit_code == 0, repeated.output
    assert all(item["status"] == "skipped" for item in json.loads(repeated.stdout)["items"])


@pytest.mark.parametrize("day_kind", ["today", "latest_old"])
def test_concurrent_append_loses_no_events(
    lineage_project,
    planted,
    day_kind,
    monkeypatch,
):
    if day_kind == "latest_old":
        _leave_only_ancient_days(planted)
    instant = planted.now - timedelta(days=100 if day_kind == "latest_old" else 0)
    directory = planted.root / SOURCES[0] / "lineage/openlineage"
    transport = FileOpenLineageTransport(directory)
    started, during_walk, appended_during_walk = Event(), Event(), Event()
    expected = {f"append-{index}" for index in range(50)}

    def append():
        for index in range(50):
            if index == 1:
                started.set()
                assert during_walk.wait(timeout=10)
            event = {"eventTime": instant.isoformat(), "run": {"runId": f"append-{index}"}}
            assert transport.send(event, budget_seconds=1).emitted
            if index == 1:
                appended_during_walk.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(append)
        try:
            assert started.wait(timeout=10)
            original = Path.unlink

            def unlink(path, *args, **kwargs):
                during_walk.set()
                assert appended_during_walk.wait(timeout=10)
                return original(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(Path, "unlink", unlink)
                result = _invoke(lineage_project, "--apply", "--format", "json")
                assert result.exit_code == 0, result.output
                record = json.loads(result.stdout)
                assert record["items"]
                assert all(item["status"] == "applied" for item in record["items"])
        finally:
            during_walk.set()
        writer.result(timeout=10)
    path = directory / f"events-{instant.date()}.ndjson"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    actual = [
        event["run"]["runId"] for event in events if event["run"]["runId"].startswith("append-")
    ]
    assert set(actual) == expected
    assert len(actual) == len(expected)
    assert all(not Path(item["target"]).exists() for item in record["items"])
