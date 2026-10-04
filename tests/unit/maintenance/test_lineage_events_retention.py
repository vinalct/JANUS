import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from threading import Event

import pytest

from janus.observability.openlineage.settings import resolve_openlineage_settings
from janus.observability.openlineage.transport import FileOpenLineageTransport
from tests.support.maintenance_zone import SOURCES, zone_plans

pytestmark = pytest.mark.xfail(
    strict=True, reason="lineage event collector absent"
)


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
    original = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".ndjson":
            pytest.fail("collector read event payloads")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        _, plan = _plan(planted, policy_config)
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
    assert targets  # older files are actually condemned


@pytest.mark.parametrize("day_kind", ["today", "latest_old"])
def test_concurrent_append_loses_no_events(planted, policy_config, day_kind, monkeypatch):
    import_module("janus.maintenance.inventory")
    execute = import_module("janus.maintenance.execute")
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
            _, plan = _plan(planted, policy_config)
            original = Path.unlink

            def unlink(path, *args, **kwargs):
                during_walk.set()
                assert appended_during_walk.wait(timeout=10)
                return original(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(Path, "unlink", unlink)
                for item in plan.items:
                    assert execute.execute_metadata_item(item).status == "applied"
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
    assert all(not Path(item.target).exists() for item in plan.items)
