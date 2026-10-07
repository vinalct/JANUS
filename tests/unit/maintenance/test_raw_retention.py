"""Raw retention preserves rebuild inputs and resume positions, with measured deletions."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from janus.cli import maintain
from janus.lineage.persistence import MetadataZonePaths
from janus.maintenance.errors import MaintenanceInvariantError, MaintenanceProfileError
from janus.maintenance.execute import delete_prefix, execute_retention
from janus.maintenance.inventory import (
    MaintenanceInventory,
    collect_inventory,
    collect_metadata_inventory,
    collect_raw_inventory,
)
from janus.maintenance.planning import PlannedItem, ProtectedItem, plan_retention
from janus.maintenance.raw_inventory import collect_raw_run_statuses
from janus.maintenance.settings import resolve_maintenance_settings
from janus.planner import Planner, PlanningRequest
from janus.registry import load_registry
from janus.strategies.common import raw_run_path_segment
from janus.utils.environment import load_environment_config
from janus.utils.storage import StorageLayout
from tests.support.maintenance_zone import PROJECT_ROOT, SOURCES, zone_layout, zone_plans
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.unit.maintenance.test_session_free_zones import _project

RAW_ZONE = frozenset({"raw"})


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def raw_zone(planted):
    plans = tuple(
        replace(plan, raw_output=replace(plan.raw_output, path=f"data/raw/{plan.source.source_id}"))
        for plan in zone_plans(planted)
    )
    for plan in plans:
        (MetadataZonePaths.from_plan(plan).base_dir / "extraction_progress.json").unlink()
    return SimpleNamespace(plans=plans, layout=zone_layout(planted), now=planted.now)


def _plant(zone, run_id, *, age=120, status="failed", source_index=0):
    plan = zone.plans[source_index]
    root = zone.layout.resolve_output(plan, "raw").resolved_path
    day = (zone.now - timedelta(days=age)).date()
    prefix = root / "runs" / f"ingestion_date={day}" / f"run_id={raw_run_path_segment(run_id)}"
    _write(prefix / "pages/data.json", {"run_id": run_id})
    (prefix / "pages/data.json.sha256").write_text("a" * 64 + "\n")
    if status is not None:
        _write(
            MetadataZonePaths.from_plan(plan).runs_dir / f"{run_id}.json",
            {
                "source_id": plan.source.source_id,
                "run_id": run_id,
                "status": status,
                "started_at": (zone.now - timedelta(days=age)).isoformat(),
            },
        )
    return prefix


def _inventory(zone):
    return MaintenanceInventory(
        raw=collect_raw_inventory(
            zone.plans,
            zone.layout,
            source_ids=None,
            run_status_by_segment=collect_raw_run_statuses(zone.plans),
        ),
        metadata=collect_metadata_inventory(zone.plans, zone.layout, source_ids=None),
    )


def _policy(config, *, enabled=True, keep_last=3):
    policy = resolve_maintenance_settings(config)
    return replace(policy, raw=replace(policy.raw, enabled=enabled, keep_last_runs=keep_last))


def _plan(zone, config, *, enabled=True, zones=RAW_ZONE, source_ids=None):
    return plan_retention(
        _inventory(zone),
        _policy(config, enabled=enabled),
        zone.now,
        zones=zones,
        source_ids=source_ids,
    )


def _targets(plan):
    return {Path(item.target) for item in plan.items if item.skipped_reason is None}


@pytest.mark.parametrize("enabled,zones", [(False, RAW_ZONE), (True, frozenset())])
def test_disabled_or_unselected_raw_has_no_candidates(raw_zone, policy_config, enabled, zones):
    prefix = _plant(raw_zone, "very-old", age=1000)
    plan = _plan(raw_zone, policy_config, enabled=enabled, zones=zones)
    assert plan.items == () and plan.protected == ()
    assert (
        execute_retention(
            plan, policy=_policy(policy_config, enabled=enabled), session=None, catalog_name="janus"
        )
        == ()
    )
    assert prefix.exists()


def test_newest_successful_runs_per_source_survive_and_failed_and_unknown_differ(
    raw_zone, policy_config
):
    expected = set()
    for source_index in range(2):
        expected.add(
            _plant(raw_zone, "old-success", age=200, status="succeeded", source_index=source_index)
        )
        for index in range(3):
            _plant(
                raw_zone,
                f"kept-{index}",
                age=180 - index,
                status="succeeded",
                source_index=source_index,
            )
        expected.add(_plant(raw_zone, "failed", age=100, source_index=source_index))
        unknown = _plant(raw_zone, "unknown", age=99, status=None, source_index=source_index)
        running = _plant(raw_zone, "running", age=98, status="running", source_index=source_index)
        plan = _plan(raw_zone, policy_config)
        reasons = {Path(item.target): item.reason for item in plan.protected}
        assert reasons[unknown] == reasons[running] == "unknown_run_status"
    assert _targets(plan) == expected
    assert sum(item.reason == "keep_last_runs" for item in plan.protected) == 6


def test_live_progress_matches_a_run_id_that_normalizes_and_collisions(raw_zone, policy_config):
    prefix = _plant(raw_zone, "run:with:spaces", age=200)
    assert prefix.name == "run_id=run-with-spaces"
    root = raw_zone.layout.resolve_output(raw_zone.plans[0], "raw").resolved_path
    progress = MetadataZonePaths.from_plan(raw_zone.plans[0]).base_dir / "extraction_progress.json"
    collision = _plant(raw_zone, "run-with-spaces", age=199)
    _write(progress, {"raw_path_prefix": str(prefix.relative_to(root))})
    assert _plan(raw_zone, policy_config).is_empty
    assert {
        Path(item.target)
        for item in _plan(raw_zone, policy_config).protected
        if item.reason == "live_progress"
    } == {prefix, collision}
    progress.unlink()
    assert _targets(_plan(raw_zone, policy_config)) == {prefix, collision}


def test_legacy_progress_records_one_skip_and_other_sources_continue(raw_zone, policy_config):
    protected = _plant(raw_zone, "legacy")
    removed = _plant(raw_zone, "other", source_index=1)
    progress = MetadataZonePaths.from_plan(raw_zone.plans[0]).base_dir / "extraction_progress.json"
    _write(progress, {"request_index": 8})
    plan = _plan(raw_zone, policy_config)
    assert _targets(plan) == {removed}
    skipped = [item for item in plan.items if item.skipped_reason == "legacy_progress_prefix"]
    assert len(skipped) == 1 and skipped[0].target == SOURCES[0]
    outcomes = execute_retention(
        plan, policy=_policy(policy_config), session=None, catalog_name="janus"
    )
    assert {item.status for item in outcomes} == {"applied", "skipped"}
    assert protected.exists() and not removed.exists()


def test_flat_layout_is_recorded_once_and_never_aged_or_deleted(raw_zone, policy_config):
    root = raw_zone.layout.resolve_output(raw_zone.plans[0], "raw").resolved_path
    flat = _write(root / "pages/old.json", {"old": True})
    _write(root / "another-flat-directory/old.json", {"old": True})
    scoped = _plant(raw_zone, "scoped-old")
    plan = _plan(raw_zone, policy_config)
    skips = [item for item in plan.items if item.skipped_reason == "flat_layout_present"]
    assert len(skips) == 1 and skips[0].detail["source_id"] == SOURCES[0]
    assert _targets(plan) == {scoped}
    outcomes = execute_retention(
        plan, policy=_policy(policy_config), session=None, catalog_name="janus"
    )
    assert {item.status for item in outcomes} == {"applied", "skipped"}
    assert flat.exists() and not scoped.exists()
    assert _plan(raw_zone, policy_config).is_empty


def test_settings_and_planner_enforce_the_bronze_rebuild_floor(raw_zone, policy_config):
    policy_config["maintenance"]["raw"].update(enabled=True, keep_last_runs=2)
    with pytest.raises(MaintenanceProfileError, match="maintenance.raw.keep_last_runs"):
        resolve_maintenance_settings(policy_config)
    policy_config["maintenance"]["raw"]["keep_last_runs"] = 3
    policy = _policy(policy_config, keep_last=2)
    with pytest.raises(MaintenanceInvariantError, match="bronze.retain_last"):
        plan_retention(_inventory(raw_zone), policy, raw_zone.now, zones=RAW_ZONE)


def test_collector_uses_writer_resolver_and_directories_only(raw_zone, policy_config, tmp_path):
    raw_zone.layout = replace(raw_zone.layout, raw_dir=tmp_path / "relocated-raw")
    prefix = _plant(raw_zone, "record:that:normalizes")
    fake = prefix.parent / "run_id=not-a-directory"
    fake.write_text("{}")
    os.utime(prefix, (0, 0))
    inventory = _inventory(raw_zone)
    assert len(inventory.raw) == 1
    entry = inventory.raw[0]
    assert entry.path == prefix
    assert entry.raw_root == tmp_path / "relocated-raw" / SOURCES[0]
    assert entry.ingestion_date == (raw_zone.now - timedelta(days=120)).date()
    assert entry.run_segment == "record-that-normalizes" and entry.run_succeeded is False
    assert _targets(_plan(raw_zone, policy_config)) == {prefix}


def test_absolute_targets_and_source_filter(raw_zone, policy_config, tmp_path):
    first, second = raw_zone.plans
    raw_zone.plans = (
        replace(first, raw_output=replace(first.raw_output, path=str(tmp_path / "absolute"))),
        second,
    )
    prefix = _plant(raw_zone, "selected")
    _plant(raw_zone, "other", source_index=1)
    selected = frozenset({SOURCES[0]})
    entries = collect_raw_inventory(
        raw_zone.plans,
        raw_zone.layout,
        source_ids=selected,
        run_status_by_segment=collect_raw_run_statuses(raw_zone.plans),
    )
    assert len(entries) == 1 and entries[0].path == prefix
    assert entries[0].raw_root == tmp_path / "absolute"
    assert _targets(_plan(raw_zone, policy_config, source_ids=selected)) == {prefix}


@pytest.mark.parametrize("status", [None, "unexpected", {}, "invalid-json", "wrong-run-id"])
def test_unusable_run_records_are_unknown_and_protected(raw_zone, policy_config, status):
    prefix = _plant(raw_zone, "unreadable", status="failed")
    record = MetadataZonePaths.from_plan(raw_zone.plans[0]).runs_dir / "unreadable.json"
    if status == "invalid-json":
        record.write_text("{bad")
    elif status == "wrong-run-id":
        _write(record, {"run_id": "some-other-run", "status": "failed"})
    else:
        _write(record, {"run_id": "unreadable", "status": status})
    plan = _plan(raw_zone, policy_config)
    assert plan.is_empty
    assert any(
        item.target == str(prefix) and item.reason == "unknown_run_status"
        for item in plan.protected
    )


def test_conflicting_sanitized_statuses_protect_both_runs(raw_zone, policy_config):
    first = _plant(raw_zone, "collision:run", status="succeeded")
    second = _plant(raw_zone, "collision-run", age=121)
    plan = _plan(raw_zone, policy_config)
    assert plan.is_empty
    assert {
        Path(item.target) for item in plan.protected if item.reason == "unknown_run_status"
    } == {first, second}


@pytest.mark.parametrize(
    "directory",
    [
        "ingestion_date=bad/run_id=old",
        "ingestion_date=20260101/run_id=old",
        "ingestion_date=2020-01-01/run_id=",
    ],
)
def test_invalid_raw_prefix_is_skipped(raw_zone, policy_config, directory):
    root = raw_zone.layout.resolve_output(raw_zone.plans[0], "raw").resolved_path
    path = root / "runs" / directory
    path.mkdir(parents=True)
    plan = _plan(raw_zone, policy_config)
    assert plan.is_empty and len(plan.items) == 1
    assert plan.items[0].skipped_reason == "invalid_raw_prefix" and path.exists()


def test_raw_day_boundary_is_strict(raw_zone, policy_config):
    prefixes = {_plant(raw_zone, f"age-{age}", age=age): age for age in (91, 90, 89)}
    assert _targets(_plan(raw_zone, policy_config)) == {
        path for path, age in prefixes.items() if age > 90
    }


@pytest.mark.parametrize("escape", ["outside", "symlink", "parent-traversal", "root", "flat"])
def test_delete_prefix_refuses_escape_root_and_flat_layout(raw_zone, tmp_path, escape):
    prefix = _plant(raw_zone, "old")
    root = raw_zone.layout.resolve_output(raw_zone.plans[0], "raw").resolved_path
    outside = tmp_path / "outside"
    _write(outside / "data.json", {"keep": True})
    if escape == "symlink":
        prefix = prefix.parent / "run_id=escaped"
        prefix.symlink_to(outside, target_is_directory=True)
    else:
        prefix = {
            "outside": outside,
            "parent-traversal": root / ".." / ".." / "outside",
            "root": root,
            "flat": root / "pages",
        }[escape]
    item = PlannedItem("raw", str(prefix), "delete_prefix", {})
    with pytest.raises(MaintenanceInvariantError):
        delete_prefix(item, raw_root=root)
    assert (outside / "data.json").exists()


def test_executor_checks_resolved_protected_paths_and_dispatch_forwards_them(
    raw_zone, policy_config
):
    prefix = _plant(raw_zone, "old")
    root = raw_zone.layout.resolve_output(raw_zone.plans[0], "raw").resolved_path
    item = PlannedItem("raw", str(prefix), "delete_prefix", {"raw_root": str(root)})
    with pytest.raises(MaintenanceInvariantError, match="protected"):
        delete_prefix(item, raw_root=root, protected_paths=frozenset({prefix / "pages" / ".."}))
    plan = _plan(raw_zone, policy_config)
    protected_plan = replace(plan, protected=(ProtectedItem("raw", str(prefix), "live_progress"),))
    outcomes = execute_retention(
        protected_plan, policy=_policy(policy_config), session=None, catalog_name="janus"
    )
    assert (
        outcomes[0].status == "failed" and outcomes[0].failure_type == "MaintenanceInvariantError"
    )
    assert prefix.exists()


def test_success_removes_data_and_sidecars_with_real_counts_and_bytes(raw_zone, policy_config):
    prefix = _plant(raw_zone, "old")
    plan = _plan(raw_zone, policy_config)
    sizes = {path: path.stat().st_size for path in prefix.rglob("*") if path.is_file()}
    outcomes = execute_retention(
        plan, policy=_policy(policy_config), session=None, catalog_name="janus"
    )
    assert len(outcomes) == 1 and outcomes[0].status == "applied"
    assert outcomes[0].removed_count == len(sizes) == 2
    assert outcomes[0].removed_bytes == sum(sizes.values())
    assert not prefix.exists()
    assert _plan(raw_zone, policy_config).is_empty


def test_partial_rmtree_failure_records_survivors_and_continues(
    raw_zone, policy_config, monkeypatch
):
    blocked = _plant(raw_zone, "blocked")
    other = _plant(raw_zone, "other")
    file = _write(blocked / "pages/blocked.json", {"blocked": True})
    file_size = file.stat().st_size
    total = sum(path.stat().st_size for path in blocked.rglob("*") if path.is_file())
    original = os.unlink

    def unlink(path, *args, **kwargs):
        if Path(path).name == file.name:
            raise PermissionError("scripted file refusal")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    outcomes = execute_retention(
        _plan(raw_zone, policy_config),
        policy=_policy(policy_config),
        session=None,
        catalog_name="janus",
    )
    by_target = {Path(item.target): item for item in outcomes}
    failed = by_target[blocked]
    assert failed.status == "failed" and failed.failure_type == "PermissionError"
    assert failed.detail["surviving_count"] == "1"
    assert any(
        entry["failure_type"] == "PermissionError"
        for entry in json.loads(failed.detail["file_failures"])
    )
    assert failed.removed_count == 2 and failed.removed_bytes == total - file_size
    assert file.exists() and not (file.parent / "data.json.sha256").exists()
    assert by_target[other].status == "applied" and not other.exists()


def test_missing_prefix_is_skipped(raw_zone, policy_config):
    prefix = _plant(raw_zone, "missing")
    plan = _plan(raw_zone, policy_config)
    for file in prefix.rglob("*"):
        if file.is_file():
            file.unlink()
    (prefix / "pages").rmdir()
    prefix.rmdir()
    (outcome,) = execute_retention(
        plan, policy=_policy(policy_config), session=None, catalog_name="janus"
    )
    assert outcome.status == "skipped" and outcome.detail["skipped_reason"] == "already_absent"
    assert outcome.removed_count == outcome.removed_bytes == 0


@pytest.fixture
def raw_project(tmp_path, policy_config, now, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(maintain, "datetime", Clock)
    policy_config["maintenance"]["raw"]["enabled"] = True
    root = _project(tmp_path, policy_config)
    registry = load_registry(root)
    config = load_environment_config("local", root)
    plans = tuple(
        Planner()
        .plan(
            PlanningRequest.create(
                source_id=source.source_id,
                environment="local",
                project_root=root,
                run_id="raw-retention-test",
                started_at=now,
                include_disabled=True,
            ),
            registry=registry,
        )
        .plan
        for source in registry.list_sources(enabled_only=False)
    )
    return SimpleNamespace(
        root=root,
        plans=plans,
        config=config,
        registry=registry,
        layout=StorageLayout.from_environment_config(config, root),
        now=now,
    )


def _invoke(zone, *args):
    return run_janus(
        ["maintain", "--project-root", str(zone.root), "--zone", "raw", "--format", "json", *args]
    )


def test_real_raw_cli_dry_run_apply_and_repeat_are_session_free(raw_project, monkeypatch):
    zone = raw_project
    prefix = _plant(zone, "run:old")
    unknown = _plant(zone, "no-record", status=None)
    arm_spark_tripwire(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("raw maintenance constructed a Spark provider")

    monkeypatch.setattr(maintain.SparkSessionProvider, "__init__", forbidden)
    snapshot = {path: path.read_bytes() for path in prefix.rglob("*") if path.is_file()}
    dry_run = _invoke(zone)
    assert dry_run.exit_code == 0, dry_run.output
    planned = json.loads(dry_run.stdout)
    assert planned["zones"] == ["raw"] and len(planned["items"]) == 1
    assert all(path.read_bytes() == data for path, data in snapshot.items())
    applied = _invoke(zone, "--apply")
    assert applied.exit_code == 0, applied.output
    record = json.loads(applied.stdout)
    assert record["plan_digest"] == planned["plan_digest"]
    assert record["protected"] == planned["protected"]
    assert record["items"][0]["status"] == "applied" and not prefix.exists()
    assert record["items"][0]["removed_count"] == 2
    assert record["items"][0]["removed_bytes"] == sum(map(len, snapshot.values()))
    persisted = zone.root / "data/metadata/maintenance" / f"{record['maintenance_run_id']}.json"
    assert persisted.read_text() == applied.stdout
    assert unknown.exists()
    repeated = _invoke(zone, "--apply")
    assert repeated.exit_code == 0 and json.loads(repeated.stdout)["items"] == []


@pytest.mark.parametrize("flat_only", [False, True])
def test_legacy_raw_cli_skip_exits_zero_without_deletion(raw_project, flat_only):
    zone = raw_project
    prefix = _plant(zone, "live-legacy")
    progress = MetadataZonePaths.from_plan(zone.plans[0]).base_dir / "extraction_progress.json"
    _write(progress, {"request_index": 8})
    if flat_only:
        prefix.rename(prefix.parents[2] / "pages")
    result = _invoke(zone, "--apply")
    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert all(item["status"] == "skipped" for item in record["items"])
    assert any(
        item["detail"]["skipped_reason"] == "legacy_progress_prefix" for item in record["items"]
    )
    assert record["failures"] == [] and progress.exists()


def test_disabled_raw_inventory_never_reads_sources(raw_project, policy_config, monkeypatch):
    policy = _policy(policy_config, enabled=False)

    def forbidden(*args, **kwargs):
        pytest.fail("disabled raw read source plans or raw paths")

    monkeypatch.setattr(raw_project.registry.__class__, "list_sources", forbidden)
    inventory = collect_inventory(
        raw_project.registry,
        raw_project.config,
        {},
        policy,
        raw_project.now,
        zones=RAW_ZONE,
        source_ids=None,
        session=None,
    )
    assert inventory.raw == () and inventory.metadata is None


def test_every_shipped_profile_keeps_raw_disabled():
    paths = tuple((PROJECT_ROOT / "conf/environments").glob("*.yaml"))
    assert paths
    for path in paths:
        profile = yaml.safe_load(path.read_text())
        assert profile.get("maintenance", {}).get("raw", {}).get("enabled", False) is False
