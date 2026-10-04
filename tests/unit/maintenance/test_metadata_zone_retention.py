from importlib import import_module
from pathlib import Path

import pytest

from tests.support.maintenance_zone import SOURCES, zone_layout, zone_plans

pytestmark = pytest.mark.xfail(
    strict=True, reason="metadata collector/executor absent"
)


def _collect(planted):
    api = import_module("janus.maintenance.inventory")
    return api, api.collect_metadata_inventory(
        zone_plans(planted), zone_layout(planted), source_ids=None
    )


def _plan(planted, config):
    api, metadata = _collect(planted)
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(config)
    return import_module("janus.maintenance.planning").plan_retention(
        api.MaintenanceInventory(metadata=metadata),
        policy,
        planted.now,
        zones=frozenset({"metadata"}),
    )


@pytest.mark.parametrize(
    "shape",
    ["checkpoint", "dead_letters", "progress", "latest_run", "latest_validation", "progress_run"],
)
def test_each_protected_shape_survives(planted, policy_config, shape):
    plan = _plan(planted, policy_config)
    execute = import_module("janus.maintenance.execute")
    newest = planted.run_ids_by_source[SOURCES[0]][-1]
    live = planted.run_ids_by_source[SOURCES[0]][0]
    relative = {
        "checkpoint": "checkpoints/current.json",
        "dead_letters": "dead_letters/current.json",
        "progress": "extraction_progress.json",
        "latest_run": f"runs/{newest}.json",
        "latest_validation": f"validations/{newest}.json",
        "progress_run": f"runs/{live}.json",
    }
    path = planted.root / SOURCES[0] / relative[shape]
    assert path in planted.protected_paths
    before = path.read_bytes()
    for item in plan.items:
        assert execute.execute_metadata_item(item).status == "applied"
    assert path.read_bytes() == before
    assert all(protected.exists() for protected in planted.protected_paths)


def test_collector_candidates_equal_fixture_declaration(planted, policy_config):
    plan = _plan(planted, policy_config)
    expected = {path for path in planted.candidate_paths if path.parent.name != "openlineage"}
    assert {Path(item.target) for item in plan.items} == expected
    assert {planted.root / source / "lineage/orphan.json" for source in SOURCES} <= expected


def test_unparseable_timestamp_is_recorded_as_unaged(planted, policy_config):
    _, inventory = _collect(planted)
    path = planted.root / SOURCES[0] / "runs/unaged.json"
    entry = next(entry for entry in inventory.artifacts if entry.path == path)
    assert entry.timestamp is None
    plan = _plan(planted, policy_config)
    assert any(item.target == str(path) and item.reason == "unaged" for item in plan.protected)


def test_invalid_json_is_recorded_and_skipped(planted, policy_config):
    _, inventory = _collect(planted)
    path = planted.root / SOURCES[0] / "runs/unreadable.json"
    entry = next(entry for entry in inventory.artifacts if entry.path == path)
    assert entry.read_error
    plan = _plan(planted, policy_config)
    assert str(path) not in {item.target for item in plan.items}
    assert any(item.target == str(path) and item.reason == "unreadable" for item in plan.protected)


def test_progress_prefix_is_compared_in_sanitized_space(planted):
    _, inventory = _collect(planted)
    assert inventory.live_raw_run_segments[SOURCES[0]] == f"{SOURCES[0]}-run-00"
    assert inventory.live_raw_run_segments[SOURCES[1]] is None


def test_removal_failure_is_recorded_and_other_items_continue(planted, policy_config, monkeypatch):
    plan = _plan(planted, policy_config)
    execute = import_module("janus.maintenance.execute")
    blocked = Path(plan.items[0].target)
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("scripted removal failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    sizes = {item.target: Path(item.target).stat().st_size for item in plan.items}
    outcomes = [execute.execute_metadata_item(item) for item in plan.items]
    assert outcomes[0].status == "failed"
    assert outcomes[0].failure_type == "PermissionError"
    assert blocked.exists()
    assert all(outcome.status == "applied" for outcome in outcomes[1:])
    assert all(
        outcome.removed_count == 1 and outcome.removed_bytes == sizes[outcome.target]
        for outcome in outcomes[1:]
    )
    assert all(path.exists() for path in planted.protected_paths)


def test_executor_refuses_handbuilt_state_deletion(planted):
    execute = import_module("janus.maintenance.execute")
    planning = import_module("janus.maintenance.planning")
    errors = import_module("janus.maintenance.errors")
    path = planted.root / SOURCES[0] / "checkpoints/current.json"
    assert path in planted.protected_paths
    item = planning.PlannedItem("metadata", str(path), "delete_file", {})
    with pytest.raises(errors.MaintenanceInvariantError):
        execute.execute_metadata_item(item)
    assert path.exists()
