import json
from dataclasses import replace
from datetime import timedelta
from importlib import import_module
from pathlib import Path

import pytest

from tests.support.maintenance_zone import FAMILIES, SOURCES, zone_layout, zone_plans


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
    [
        "checkpoint",
        "dead_letters",
        "progress",
        "latest_run",
        "latest_validation",
        "progress_run",
        "null_pipeline",
    ],
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
    path = (
        planted.root / "pipelines/null/summary.json"
        if shape == "null_pipeline"
        else planted.root / SOURCES[0] / relative[shape]
    )
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
    plan = replace(plan, items=plan.items[:5])
    blocked = Path(plan.items[2].target)
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("scripted removal failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    sizes = {item.target: Path(item.target).stat().st_size for item in plan.items}
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(policy_config)
    outcomes = execute.execute_retention(plan, policy=policy, session=None, catalog_name="janus")
    assert outcomes[2].status == "failed"
    assert outcomes[2].failure_type == "PermissionError"
    assert blocked.exists()
    removed = [outcome for outcome in outcomes if outcome.target != str(blocked)]
    assert len(removed) == 4
    assert all(
        outcome.status == "applied" and not Path(outcome.target).exists() for outcome in removed
    )
    assert all(
        outcome.removed_count == 1 and outcome.removed_bytes == sizes[outcome.target]
        for outcome in removed
    )
    assert all(path.exists() for path in planted.protected_paths)


@pytest.mark.parametrize(
    "relative",
    ["checkpoints/current.json", "dead_letters/current.json", "extraction_progress.json"],
)
def test_executor_refuses_handbuilt_state_deletion(planted, relative):
    execute = import_module("janus.maintenance.execute")
    planning = import_module("janus.maintenance.planning")
    errors = import_module("janus.maintenance.errors")
    path = planted.root / SOURCES[0] / relative
    assert path in planted.protected_paths
    item = planning.PlannedItem("metadata", str(path), "delete_file", {})
    with pytest.raises(errors.MaintenanceInvariantError):
        execute.execute_metadata_item(item)
    assert path.exists()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_state_paths_are_protected_even_when_absent(planted):
    expected = {
        planted.root / source / relative
        for source in SOURCES
        for relative in (
            "checkpoints/current.json",
            "dead_letters/current.json",
            "extraction_progress.json",
        )
    }
    for path in expected:
        path.unlink()
    _, inventory = _collect(planted)
    assert inventory.protected_paths == expected
    assert inventory.live_raw_run_segments == {}


@pytest.mark.parametrize("family", FAMILIES)
def test_kept_and_live_runs_protect_all_four_artifact_kinds(planted, policy_config, family):
    plan = _plan(planted, policy_config)
    reasons = {item.target: item.reason for item in plan.protected}
    for source in SOURCES:
        for run_id in planted.run_ids_by_source[source][-20:]:
            path = planted.root / source / family / f"{run_id}.json"
            assert reasons[str(path)] == "keep_last_runs"
    live = planted.run_ids_by_source[SOURCES[0]][0]
    assert reasons[str(planted.root / SOURCES[0] / family / f"{live}.json")] == "live_progress"


@pytest.mark.parametrize("family,field", FAMILIES.items())
def test_timestamp_comes_only_from_the_artifact_field(planted, family, field):
    path = planted.root / SOURCES[0] / family / "timestamp.json"
    timestamp = planted.now - timedelta(days=120)
    _write(
        path,
        {
            "run_id": "timestamp",
            **dict.fromkeys(FAMILIES.values(), "wrong"),
            field: timestamp.isoformat(),
        },
    )
    _, inventory = _collect(planted)
    entry = next(entry for entry in inventory.artifacts if entry.path == path)
    assert entry.timestamp == timestamp
    assert entry.read_error is None


@pytest.mark.parametrize("timestamp", [None, "bad", "2026-01-01T12:00:00", 123, "missing"])
def test_missing_null_invalid_or_naive_timestamp_is_protected(planted, policy_config, timestamp):
    path = planted.root / SOURCES[0] / "runs/timestamp.json"
    payload = {"run_id": "timestamp"}
    if timestamp != "missing":
        payload["started_at"] = timestamp
    _write(path, payload)
    plan = _plan(planted, policy_config)
    assert any(item.target == str(path) and item.reason == "unaged" for item in plan.protected)
    assert str(path) not in {item.target for item in plan.items}


@pytest.mark.parametrize("family", FAMILIES)
def test_mismatched_run_id_is_unreadable_and_preserved(planted, policy_config, family):
    path = planted.root / SOURCES[0] / family / "mismatch.json"
    _write(path, {"run_id": "different", FAMILIES[family]: "2020-01-01T00:00:00Z"})
    _, inventory = _collect(planted)
    entry = next(entry for entry in inventory.artifacts if entry.path == path)
    assert entry.read_error == "run_id_mismatch"
    plan = _plan(planted, policy_config)
    assert any(item.target == str(path) and item.reason == "unreadable" for item in plan.protected)


@pytest.mark.parametrize("payload", [[], 1, "text", None])
def test_nonmapping_record_is_protected(planted, policy_config, payload):
    path = planted.root / SOURCES[0] / "runs/nonmapping.json"
    _write(path, payload)
    plan = _plan(planted, policy_config)
    assert any(item.target == str(path) and item.reason == "unreadable" for item in plan.protected)


def test_unreadable_record_is_protected_and_collection_continues(
    planted, policy_config, monkeypatch
):
    api = import_module("janus.maintenance.inventory")
    blocked = planted.root / SOURCES[0] / "runs/unaged.json"
    original = api.read_json_mapping

    def read(path):
        if path == blocked:
            raise PermissionError("scripted read failure")
        return original(path)

    monkeypatch.setattr(api, "read_json_mapping", read)
    _, inventory = _collect(planted)
    assert next(entry for entry in inventory.artifacts if entry.path == blocked).read_error == (
        "PermissionError"
    )
    plan = _plan(planted, policy_config)
    assert any(
        item.target == str(blocked) and item.reason == "unreadable" for item in plan.protected
    )
    assert plan.items


def test_shared_pipeline_root_is_walked_once(planted):
    _, inventory = _collect(planted)
    pipelines = [entry for entry in inventory.artifacts if entry.kind == "pipelines"]
    assert len(pipelines) == 3
    assert all(entry.source_id is None for entry in pipelines)
    assert {entry.run_id for entry in pipelines} == {"old", "recent", "null"}
    null = next(entry for entry in pipelines if entry.run_id == "null")
    assert null.timestamp is None


def test_temporary_and_unowned_history_files_are_never_listed(planted):
    extras = (
        planted.root / SOURCES[0] / "runs/ignored.txt",
        planted.root / SOURCES[0] / "lineage/openlineage/event.json",
        planted.root / SOURCES[0] / "dead_letters/history/old.json",
        planted.root / "pipelines/ignored/not-summary.json",
    )
    for path in extras:
        _write(path, {"run_id": path.stem, "started_at": "2020-01-01T00:00:00Z"})
    _, inventory = _collect(planted)
    listed = {entry.path for entry in inventory.artifacts}
    assert listed.isdisjoint(extras)
    assert not any(path.suffix == ".tmp" or "openlineage" in path.parts for path in listed)


def test_sources_compute_keep_last_runs_independently(planted, policy_config):
    policy_config["maintenance"]["metadata"].update(keep_last_runs=2, older_than_days=0)
    for source_index, source in enumerate(SOURCES):
        for index, run_id in enumerate(planted.run_ids_by_source[source]):
            path = planted.root / source / "runs" / f"{run_id}.json"
            _write(
                path,
                {
                    "run_id": run_id,
                    "started_at": (
                        planted.now - timedelta(days=200 + 100 * source_index - index)
                    ).isoformat(),
                },
            )
    plan = _plan(planted, policy_config)
    reasons = {item.target: item.reason for item in plan.protected}
    candidates = {item.target for item in plan.items}
    for source in SOURCES:
        for run_id in planted.run_ids_by_source[source][-2:]:
            for family in FAMILIES:
                assert reasons[str(planted.root / source / family / f"{run_id}.json")] == (
                    "keep_last_runs"
                )
        old = planted.run_ids_by_source[source][2]
        assert str(planted.root / source / "runs" / f"{old}.json") in candidates


def test_source_filter_keeps_other_sources_and_shared_summaries(planted, policy_config):
    api = import_module("janus.maintenance.inventory")
    selected = frozenset({SOURCES[0]})
    inventory = api.collect_metadata_inventory(
        zone_plans(planted), zone_layout(planted), source_ids=selected
    )
    assert {entry.source_id for entry in inventory.artifacts} == {SOURCES[0], None}
    assert len(inventory.protected_paths) == 3
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(policy_config)
    plan = import_module("janus.maintenance.planning").plan_retention(
        api.MaintenanceInventory(metadata=inventory),
        policy,
        planted.now,
        zones=frozenset({"metadata"}),
        source_ids=selected,
    )
    assert all(str(planted.root / SOURCES[0]) in item.target for item in plan.items)
    assert any(item.reason == "source_filter" for item in plan.protected)


def test_executor_checks_the_full_protected_set(planted, policy_config):
    execute = import_module("janus.maintenance.execute")
    planning = import_module("janus.maintenance.planning")
    errors = import_module("janus.maintenance.errors")
    plan = _plan(planted, policy_config)
    path = planted.root / "pipelines/null/summary.json"
    item = planning.PlannedItem("metadata", str(path), "delete_file", {})
    with pytest.raises(errors.MaintenanceInvariantError):
        execute.execute_metadata_item(item, protected_paths=frozenset({path}))
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(policy_config)
    (outcome,) = execute.execute_retention(
        replace(plan, items=(item,)), policy=policy, session=None, catalog_name="janus"
    )
    assert outcome.status == "failed"
    assert outcome.failure_type == "MaintenanceInvariantError"
    assert path.exists()


def test_stat_precedes_unlink_and_empty_directory_survives(tmp_path, monkeypatch):
    execute = import_module("janus.maintenance.execute")
    planning = import_module("janus.maintenance.planning")
    path = tmp_path / "runs/old.json"
    _write(path, {"run_id": "old"})
    size = path.stat().st_size
    calls = []
    original_stat, original_unlink = Path.stat, Path.unlink

    def stat(target, *args, **kwargs):
        if target == path:
            calls.append("stat")
        return original_stat(target, *args, **kwargs)

    def unlink(target, *args, **kwargs):
        if target == path:
            calls.append("unlink")
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "unlink", unlink)
    item = planning.PlannedItem("metadata", str(path), "delete_file", {})
    outcome = execute.execute_metadata_item(item)
    assert calls == ["stat", "unlink"]
    assert outcome.removed_count == 1 and outcome.removed_bytes == size
    assert path.parent.is_dir()
    again = execute.execute_metadata_item(item)
    assert again.status == "skipped" and again.detail["skipped_reason"] == "already_absent"
    assert again.removed_count == again.removed_bytes == 0


def test_applied_inventory_has_an_empty_second_plan(planted, policy_config):
    execute = import_module("janus.maintenance.execute")
    plan = _plan(planted, policy_config)
    for item in plan.items:
        assert execute.execute_metadata_item(item).status == "applied"
    assert _plan(planted, policy_config).is_empty
    assert all(path.exists() for path in planted.protected_paths)


@pytest.mark.parametrize("failure", [PermissionError, OSError])
@pytest.mark.parametrize("operation", ["stat", "unlink"])
def test_file_operation_failures_have_zero_removed_measurements(
    tmp_path, monkeypatch, failure, operation
):
    execute = import_module("janus.maintenance.execute")
    planning = import_module("janus.maintenance.planning")
    path = tmp_path / "old.json"
    _write(path, {"run_id": "old"})
    original = getattr(Path, operation)

    def denied(target, *args, **kwargs):
        if target == path:
            raise failure("scripted failure")
        return original(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, denied)
        outcome = execute.execute_metadata_item(
            planning.PlannedItem(
                "metadata",
                str(path),
                "delete_file",
                {},
            )
        )
    assert outcome.status == "failed" and outcome.failure_type == failure.__name__
    assert outcome.removed_count == outcome.removed_bytes == 0
    assert path.exists()


def test_disappearance_between_stat_and_unlink_is_skipped(tmp_path, monkeypatch):
    execute = import_module("janus.maintenance.execute")
    planning = import_module("janus.maintenance.planning")
    path = tmp_path / "old.json"
    _write(path, {"run_id": "old"})
    original = Path.unlink

    def removed_elsewhere(target, *args, **kwargs):
        if target == path:
            original(target)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", removed_elsewhere)
    outcome = execute.execute_metadata_item(
        planning.PlannedItem(
            "metadata",
            str(path),
            "delete_file",
            {},
        )
    )
    assert outcome.status == "skipped" and outcome.detail["skipped_reason"] == "already_absent"
    assert outcome.removed_count == outcome.removed_bytes == 0


def test_sanitized_run_id_collisions_preserve_all_matching_evidence(planted, policy_config):
    run_id = f"{SOURCES[0]}-run-00"
    for family, field in FAMILIES.items():
        path = planted.root / SOURCES[0] / family / f"{run_id}.json"
        _write(path, {"run_id": run_id, field: "2020-01-01T00:00:00Z"})
    plan = _plan(planted, policy_config)
    reasons = {item.target: item.reason for item in plan.protected}
    for family in FAMILIES:
        assert (
            reasons[str(planted.root / SOURCES[0] / family / f"{run_id}.json")] == "live_progress"
        )


def test_unreadable_progress_aborts_before_any_retention_can_act(planted):
    path = planted.root / SOURCES[0] / "extraction_progress.json"
    path.write_text("{invalid json")
    before = {path: path.read_bytes() for path in planted.root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        _collect(planted)
    assert {path: path.read_bytes() for path in before} == before
