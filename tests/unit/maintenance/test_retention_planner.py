from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

import pytest

from janus.maintenance.inventory import (
    BronzeTableInventory,
    EventFileEntry,
    MaintenanceInventory,
    MetadataZoneInventory,
    RawRunPrefixEntry,
    RunArtifactEntry,
    RunsTablePartitionEntry,
    SnapshotEntry,
)
from janus.maintenance.planning import PlannedItem, ProtectedItem, RetentionPlan, plan_retention
from janus.maintenance.settings import resolve_maintenance_settings
from janus.strategies.common import _safe_raw_path_segment, raw_run_path_segment
from tests.support.maintenance_zone import FAMILIES, SOURCES


def _api(config):
    inventory = import_module("janus.maintenance.inventory")
    planning = import_module("janus.maintenance.planning")
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(config)
    return inventory, planning, policy


def _snapshot_ids(plan):
    return {
        int(snapshot)
        for item in plan.items
        if item.zone == "bronze"
        and item.action == "expire_snapshots"
        and item.skipped_reason is None
        for snapshot in json.loads(item.detail["snapshot_ids"])
    }


def _bronze(inv, now, ages, *, current=None, override=None):
    snapshots = tuple(
        inv.SnapshotEntry(
            index + 1,
            now - timedelta(days=age),
            index == (len(ages) - 1 if current is None else current),
        )
        for index, age in enumerate(ages)
    )
    return inv.MaintenanceInventory(
        bronze=(
            inv.BronzeTableInventory("bronze.fixture", (SOURCES[0],), snapshots, override=override),
        )
    )


@pytest.mark.parametrize(
    ("retain", "ages", "expected"),
    [
        (2, (10, 9, 8), {1}),
        (3, (10, 9, 8), set()),
        (5, (10, 9, 8), set()),
        (1, (10, 9, 8), {1, 2}),
    ],
)
def test_retain_last_composes_with_age(policy_config, now, retain, ages, expected):
    inv, planning, policy = _api(policy_config)
    policy = replace(policy, bronze=replace(policy.bronze, retain_last=retain, older_than_days=0))
    plan = planning.plan_retention(
        _bronze(inv, now, ages), policy, now, zones=frozenset({"bronze"})
    )
    assert _snapshot_ids(plan) == expected


def test_exact_age_boundary_survives(policy_config, now):
    inv, planning, policy = _api(policy_config)
    policy = replace(policy, bronze=replace(policy.bronze, retain_last=1, older_than_days=30))
    inventory = _bronze(inv, now, (31, 30, 29, 0))
    plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    assert _snapshot_ids(plan) == {1}


def test_current_snapshot_survives_even_when_not_newest(policy_config, now):
    inv, planning, policy = _api(policy_config)
    policy = replace(policy, bronze=replace(policy.bronze, retain_last=1, older_than_days=0))
    plan = planning.plan_retention(
        _bronze(inv, now, (10, 9, 8), current=0), policy, now, zones=frozenset({"bronze"})
    )
    assert _snapshot_ids(plan) == {2}


def test_source_override_precedes_profile(policy_config, now):
    inv, planning, policy = _api(policy_config)
    config = import_module("janus.models.config.types").BronzeRetentionConfig
    inventory = _bronze(inv, now, (60, 50, 40), override=config(2, 0))
    plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    assert _snapshot_ids(plan) == {1}
    without = replace(inventory, bronze=(replace(inventory.bronze[0], override=None),))
    assert planning.plan_retention(without, policy, now, zones=frozenset({"bronze"})).is_empty


def _metadata(inv, planted):
    """Inventory constructed by the test, not by the missing collector."""
    artifacts = []
    for source, run_ids in planted.run_ids_by_source.items():
        for run_id in run_ids:
            for family, field in FAMILIES.items():
                path = planted.root / source / family / f"{run_id}.json"
                timestamp = datetime.fromisoformat(json.loads(path.read_text())[field])
                kind = "checkpoint_history" if family == "checkpoints/history" else family
                artifacts.append(inv.RunArtifactEntry(kind, path, source, run_id, timestamp))
        path = planted.root / source / "lineage/orphan.json"
        artifacts.append(
            inv.RunArtifactEntry(
                "lineage", path, source, "orphan", planted.now - timedelta(days=120)
            )
        )
    state = frozenset(
        path
        for path in planted.protected_paths
        if path.name in {"current.json", "extraction_progress.json"}
    )
    return inv.MaintenanceInventory(
        metadata=inv.MetadataZoneInventory(
            tuple(artifacts), state, {SOURCES[0]: f"{SOURCES[0]}-run-00", SOURCES[1]: None}
        )
    )


def test_protected_runs_apply_to_every_family_per_source(policy_config, planted, now):
    inv, planning, policy = _api(policy_config)
    inventory = _metadata(inv, planted)
    plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    actual = {Path(item.target) for item in plan.items if item.action == "delete_file"}
    expected = {
        path
        for path in planted.candidate_paths
        if path.parts[-2] != "openlineage" and "pipelines" not in path.parts
    }
    assert actual == expected
    assert actual.isdisjoint(planted.protected_paths)


def test_state_and_unaged_artifacts_are_protected(policy_config, planted, now):
    inv, planning, policy = _api(policy_config)
    paths = sorted(
        path
        for path in planted.protected_paths
        if path.name
        in {"current.json", "extraction_progress.json", "unaged.json", "unreadable.json"}
    )
    state = frozenset(
        path for path in paths if path.name in {"current.json", "extraction_progress.json"}
    )
    artifacts = tuple(
        inv.RunArtifactEntry(
            "lineage" if path in state else "runs",
            path,
            SOURCES[0],
            None,
            now - timedelta(days=120) if path in state else None,
            read_error="invalid_json" if path.stem == "unreadable" else None,
        )
        for path in paths
    )
    metadata = inv.MetadataZoneInventory(artifacts, state, {})
    plan = planning.plan_retention(
        inv.MaintenanceInventory(metadata=metadata), policy, now, zones=frozenset({"metadata"})
    )
    assert plan.is_empty
    assert {Path(item.target) for item in plan.protected} >= set(paths)


def test_metadata_boundary_is_strict(policy_config, now, tmp_path):
    inv, planning, policy = _api(policy_config)
    entries = tuple(
        inv.RunArtifactEntry(
            "lineage", tmp_path / f"{days}.json", SOURCES[0], None, now - timedelta(days=days)
        )
        for days in (91, 90, 89)
    )
    inventory = inv.MaintenanceInventory(
        metadata=inv.MetadataZoneInventory(entries, frozenset(), {})
    )
    plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert [item.target for item in plan.items] == [str(tmp_path / "91.json")]


def test_post_apply_inventory_is_empty(policy_config, planted, now):
    inv, planning, policy = _api(policy_config)
    inventory = _metadata(inv, planted)
    plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert not plan.is_empty
    removed = {item.target for item in plan.items}
    after = replace(
        inventory,
        metadata=replace(
            inventory.metadata,
            artifacts=tuple(
                entry for entry in inventory.metadata.artifacts if str(entry.path) not in removed
            ),
        ),
    )
    assert planning.plan_retention(after, policy, now, zones=frozenset({"metadata"})).is_empty


def test_permuted_inventory_produces_identical_plan_and_digest(policy_config, planted, now):
    inv, planning, policy = _api(policy_config)
    inventory = _metadata(inv, planted)
    baseline = planning.plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    for entries in (
        inventory.metadata.artifacts[::-1],
        inventory.metadata.artifacts[1:] + inventory.metadata.artifacts[:1],
    ):
        permuted = replace(inventory, metadata=replace(inventory.metadata, artifacts=entries))
        actual = planning.plan_retention(permuted, policy, now, zones=frozenset({"metadata"}))
        assert actual == baseline
        assert actual.digest == baseline.digest


def test_planner_performs_no_io(policy_config, now, monkeypatch):
    inv, planning, policy = _api(policy_config)
    inventory = _bronze(inv, now, (60, 50, 40, 0))

    def forbidden(*args, **kwargs):
        pytest.fail("pure planner performed I/O")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr(Path, "exists", forbidden)
        patch.setattr(Path, "unlink", forbidden)
        plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    assert _snapshot_ids(plan) == {1}


def test_bronze_post_apply_inventory_has_no_more_expiration(policy_config, now):
    inv, planning, policy = _api(policy_config)
    policy = replace(policy, bronze=replace(policy.bronze, retain_last=2, older_than_days=0))
    inventory = _bronze(inv, now, (60, 50, 40))
    plan = planning.plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    expired = _snapshot_ids(plan)
    assert expired == {1}
    table = inventory.bronze[0]
    after = replace(
        inventory,
        bronze=(
            replace(
                table,
                snapshots=tuple(
                    snapshot for snapshot in table.snapshots if snapshot.snapshot_id not in expired
                ),
            ),
        ),
    )
    assert planning.plan_retention(after, policy, now, zones=frozenset({"bronze"})).is_empty


def _artifact(path, now, *, kind="lineage", source="alpha", run=None, age=120, error=None):
    return RunArtifactEntry(kind, Path(path), source, run, now - timedelta(days=age), error)


def _metadata_inventory(entries=(), *, state=frozenset(), live=None):
    return MaintenanceInventory(metadata=MetadataZoneInventory(tuple(entries), state, live or {}))


def _raw(path, now, *, source="alpha", segment=None, age=120, succeeded=True):
    return RawRunPrefixEntry(
        Path(path), source, segment or path, now.date() - timedelta(days=age), succeeded
    )


def _targets(plan, action=None):
    return {
        item.target
        for item in plan.items
        if item.skipped_reason is None and (action is None or item.action == action)
    }


def _reasons(plan):
    return {item.target: item.reason for item in plan.protected}


@pytest.mark.parametrize("reason", ["absent_table", "retention_conflict"])
def test_unavailable_table_is_visible_and_does_not_stop_other_tables(policy_config, now, reason):
    policy = resolve_maintenance_settings(policy_config)
    snapshots = tuple(SnapshotEntry(i, now - timedelta(days=60), False) for i in range(5))
    inventory = MaintenanceInventory(
        bronze=(
            BronzeTableInventory("bronze.absent", ("alpha",), (), unavailable_reason=reason),
            BronzeTableInventory("bronze.ready", ("beta",), snapshots),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    skipped = [item for item in plan.items if item.skipped_reason is not None]
    assert len(skipped) == 1 and skipped[0].skipped_reason == reason
    assert skipped[0].target == "bronze.absent"
    assert _snapshot_ids(plan) == {0, 1}
    only_skipped = replace(plan, items=tuple(skipped))
    assert only_skipped.is_empty


def test_snapshot_timestamp_ties_use_descending_snapshot_id(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    inventory = MaintenanceInventory(
        bronze=(
            BronzeTableInventory(
                "bronze.ties",
                ("alpha",),
                tuple(SnapshotEntry(i, now - timedelta(days=60), False) for i in (3, 1, 5, 2, 4)),
            ),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    assert _snapshot_ids(plan) == {1, 2}
    assert _reasons(plan) == {f"bronze.ties#{i}": "retain_last" for i in (3, 4, 5)}


def test_snapshots_outside_count_floor_survive_when_within_age_window(policy_config, now):
    inv, _, policy = _api(policy_config)
    policy = replace(policy, bronze=replace(policy.bronze, retain_last=1))
    plan = plan_retention(
        _bronze(inv, now, (31, 30, 29, 0)), policy, now, zones=frozenset({"bronze"})
    )
    assert _snapshot_ids(plan) == {1}
    assert _reasons(plan)["bronze.fixture#2"] == "within_window"
    assert _reasons(plan)["bronze.fixture#3"] == "within_window"
    assert _reasons(plan)["bronze.fixture#4"] == "current_snapshot"


def test_zero_day_expiration_carries_injected_now_and_ids(policy_config, now):
    inv, _, policy = _api(policy_config)
    policy = replace(policy, bronze=replace(policy.bronze, retain_last=1, older_than_days=0))
    plan = plan_retention(_bronze(inv, now, (10, 9, 0)), policy, now, zones=frozenset({"bronze"}))
    assert plan.items[0].detail == {
        "older_than": now.isoformat(),
        "retain_last": "1",
        "snapshot_ids": "[1, 2]",
    }


def test_override_keeps_profile_optional_procedures_in_order(policy_config, now):
    inv, _, policy = _api(policy_config)
    config = import_module("janus.models").BronzeRetentionConfig
    policy = replace(
        policy,
        bronze=replace(
            policy.bronze,
            remove_orphan_files=True,
            orphan_older_than_days=7,
            compact_enabled=True,
            compact_target_file_size_mb=128,
        ),
    )
    plan = plan_retention(
        _bronze(inv, now, (60, 50, 40), override=config(2, 0)),
        policy,
        now,
        zones=frozenset({"bronze"}),
    )
    assert [item.action for item in plan.items] == [
        "expire_snapshots",
        "remove_orphan_files",
        "rewrite_data_files",
    ]
    assert plan.items[0].detail["retain_last"] == "2"
    assert plan.items[1].detail["orphan_older_than"] == (now - timedelta(days=7)).isoformat()
    assert plan.items[2].detail["target_file_size_bytes"] == str(128 * 1024 * 1024)


def test_optional_procedures_are_planned_without_snapshot_candidates(policy_config, now):
    inv, _, policy = _api(policy_config)
    policy = replace(
        policy,
        bronze=replace(
            policy.bronze,
            remove_orphan_files=True,
            compact_enabled=True,
            compact_target_file_size_mb=64,
        ),
    )
    plan = plan_retention(_bronze(inv, now, (0,)), policy, now, zones=frozenset({"bronze"}))
    assert [item.action for item in plan.items] == ["remove_orphan_files", "rewrite_data_files"]


def test_shared_table_matches_any_selected_source(policy_config, now):
    inv, _, policy = _api(policy_config)
    inventory = _bronze(inv, now, (60, 50, 40, 0))
    table = replace(inventory.bronze[0], source_ids=("alpha", "beta"))
    inventory = replace(inventory, bronze=(table,))
    selected = plan_retention(
        inventory, policy, now, zones=frozenset({"bronze"}), source_ids=frozenset({"beta"})
    )
    assert _snapshot_ids(selected) == {1}
    for sources in (frozenset(), frozenset({"other"})):
        assert plan_retention(
            inventory, policy, now, zones=frozenset({"bronze"}), source_ids=sources
        ).is_empty


@pytest.mark.parametrize(
    "kind", ["runs", "lineage", "checkpoint_history", "validations", "pipelines"]
)
def test_latest_run_protects_each_artifact_family_even_when_artifact_is_older(
    policy_config, now, kind
):
    policy = resolve_maintenance_settings(policy_config)
    policy = replace(policy, metadata=replace(policy.metadata, keep_last_runs=1))
    inventory = _metadata_inventory(
        (
            _artifact("new-run", now, kind="runs", run="new", age=100),
            _artifact("old-run", now, kind="runs", run="old", age=110),
            _artifact("linked", now, kind=kind, run="new", age=300),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert _targets(plan) == {"old-run"}
    assert _reasons(plan)["linked"] == "keep_last_runs"


def test_latest_run_ties_and_count_floor_are_per_source(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    policy = replace(policy, metadata=replace(policy.metadata, keep_last_runs=1))
    inventory = _metadata_inventory(
        tuple(
            _artifact(f"{source}/{run}", now, kind="runs", source=source, run=run)
            for source in ("alpha", "beta")
            for run in ("a", "b")
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert _targets(plan) == {"alpha/a", "beta/a"}
    assert _reasons(plan) == {"alpha/b": "keep_last_runs", "beta/b": "keep_last_runs"}


def test_latest_runs_count_distinct_ids_only_from_readable_runs_entries(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    policy = replace(policy, metadata=replace(policy.metadata, keep_last_runs=2))
    inventory = _metadata_inventory(
        (
            _artifact("a", now, kind="runs", run="a", age=100),
            _artifact("b", now, kind="runs", run="b", age=110),
            _artifact("a-copy", now, kind="runs", run="a", age=101),
            _artifact("orphan", now, run="orphan", age=99),
            _artifact("bad-run", now, kind="runs", run="bad", age=1, error="invalid_json"),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert _targets(plan) == {"orphan"}
    assert _reasons(plan)["b"] == "keep_last_runs"


@pytest.mark.parametrize("run", ["run:a", "run/a", "run a"])
def test_live_progress_uses_sanitized_space_and_protects_colliding_ids(policy_config, now, run):
    policy = resolve_maintenance_settings(policy_config)
    inventory = _metadata_inventory(
        (
            _artifact("live", now, run=run),
            _artifact("other-source", now, source="beta", run=run),
        ),
        live={"alpha": "run-a"},
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert _targets(plan) == {"other-source"}
    assert _reasons(plan)["live"] == "live_progress"


@pytest.mark.parametrize(
    "path",
    [
        "alpha/checkpoints/current.json",
        "alpha/dead_letters/current.json",
        "alpha/extraction_progress.json",
    ],
)
def test_each_state_path_is_protected_even_when_planted_as_history(policy_config, now, path):
    policy = resolve_maintenance_settings(policy_config)
    inventory = _metadata_inventory((_artifact(path, now),), state=frozenset({Path(path)}))
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert plan.is_empty
    assert _reasons(plan) == {path: "state_file"}


def test_protected_state_paths_are_recorded_without_artifact_entries(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    inventory = _metadata_inventory(state=frozenset({Path("current.json")}))
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert _reasons(plan) == {"current.json": "state_file"}


@pytest.mark.parametrize(
    ("timestamp", "error", "reason"),
    [
        (None, None, "unaged"),
        (datetime(2020, 1, 1), None, "unaged"),
        (datetime(2020, 1, 1, tzinfo=UTC), "invalid_json", "unreadable"),
        (None, "", "unreadable"),
    ],
)
def test_unaged_or_unreadable_history_cannot_be_deleted(
    policy_config, now, timestamp, error, reason
):
    policy = resolve_maintenance_settings(policy_config)
    entry = RunArtifactEntry("lineage", Path("history"), "alpha", None, timestamp, error)
    plan = plan_retention(_metadata_inventory((entry,)), policy, now, zones=frozenset({"metadata"}))
    assert plan.is_empty
    assert _reasons(plan) == {"history": reason}


@pytest.mark.parametrize("kind", ["lineage", "pipelines"])
@pytest.mark.parametrize("run_id", [None, "orphan"])
def test_orphan_and_shared_pipeline_artifacts_are_aged_by_their_own_timestamp(
    policy_config, now, kind, run_id
):
    policy = resolve_maintenance_settings(policy_config)
    inventory = _metadata_inventory(
        (
            _artifact("old", now, kind=kind, source=None, run=run_id, age=91),
            _artifact("boundary", now, kind=kind, source=None, run=run_id, age=90),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"metadata"}))
    assert _targets(plan) == {"old"}
    assert _reasons(plan)["boundary"] == "within_window"


def test_metadata_source_filter_preserves_unattributed_summaries(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    inventory = _metadata_inventory(
        tuple(_artifact(str(source), now, source=source) for source in ("alpha", "beta", None))
    )
    plan = plan_retention(
        inventory, policy, now, zones=frozenset({"metadata"}), source_ids=frozenset({"alpha"})
    )
    assert _targets(plan) == {"alpha"}
    assert _reasons(plan)["None"] == "source_filter"


@pytest.mark.parametrize("older_than_days", [0, 90])
def test_lineage_event_boundary_and_today_are_protected(policy_config, now, older_than_days):
    policy = resolve_maintenance_settings(policy_config)
    policy = replace(
        policy, lineage_events=replace(policy.lineage_events, older_than_days=older_than_days)
    )
    inventory = MaintenanceInventory(
        lineage_events=tuple(
            EventFileEntry(Path(str(age)), now.date() - timedelta(days=age))
            for age in sorted({0, older_than_days, older_than_days + 1})
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"lineage"}))
    assert _targets(plan) == {str(older_than_days + 1)}
    assert _reasons(plan)["0"] == "todays_file"


def test_all_files_on_most_recent_event_day_are_protected_even_if_ancient(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    inventory = MaintenanceInventory(
        lineage_events=(
            EventFileEntry(Path("alpha/older"), now.date() - timedelta(days=101)),
            EventFileEntry(Path("alpha/latest"), now.date() - timedelta(days=100)),
            EventFileEntry(Path("beta/latest"), now.date() - timedelta(days=100)),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"lineage"}))
    assert _targets(plan) == {"alpha/older"}
    assert _reasons(plan) == {"alpha/latest": "most_recent_file", "beta/latest": "most_recent_file"}


def test_event_day_and_partition_cutoffs_use_utc_day(policy_config):
    policy = resolve_maintenance_settings(policy_config)
    policy = replace(
        policy,
        lineage_events=replace(policy.lineage_events, older_than_days=0),
        runs_table=replace(policy.runs_table, older_than_days=0),
    )
    local_now = datetime(2026, 10, 4, 23, 30, tzinfo=timezone(timedelta(hours=-3)))
    utc_now = local_now.astimezone(UTC)
    inventory = MaintenanceInventory(
        lineage_events=(
            EventFileEntry(Path("yesterday"), local_now.date()),
            EventFileEntry(Path("today"), utc_now.date()),
        ),
        runs_table=(
            RunsTablePartitionEntry(local_now.date()),
            RunsTablePartitionEntry(utc_now.date()),
        ),
    )
    plan = plan_retention(inventory, policy, local_now, zones=frozenset({"lineage", "runs-table"}))
    assert _targets(plan, "delete_file") == {"yesterday"}
    assert _targets(plan, "delete_partition") == {"2026-10-04"}
    assert _reasons(plan)["today"] == "todays_file"
    assert plan.now == utc_now


def test_runs_table_partitions_and_expiration_carry_partition_aligned_arguments(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    inventory = MaintenanceInventory(
        runs_table=tuple(
            RunsTablePartitionEntry(now.date() - timedelta(days=age), count)
            for age, count in ((100, 12), (91, None), (90, 30), (0, 4))
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"runs-table"}))
    deletes = [item for item in plan.items if item.action == "delete_partition"]
    assert len(deletes) == 2
    cutoff = datetime.combine(now.date() - timedelta(days=90), datetime.min.time(), tzinfo=UTC)
    assert deletes[0].detail == {
        "older_than": cutoff.isoformat(),
        "row_count": "12",
        "table_identifier": "metadata.runs",
    }
    assert deletes[1].detail == {
        "older_than": cutoff.isoformat(),
        "table_identifier": "metadata.runs",
    }
    assert plan.items[-1].action == "expire_snapshots"
    assert plan.items[-1].target == "metadata.runs"
    assert plan.items[-1].detail["retain_last"] == str(policy.bronze.retain_last)
    assert plan.items[-1].detail["older_than"] == cutoff.isoformat()
    assert len(plan.protected) == 2


def test_runs_table_without_old_partitions_has_no_expiration_item(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    for partitions in ((), (RunsTablePartitionEntry(now.date()),)):
        plan = plan_retention(
            MaintenanceInventory(runs_table=partitions),
            policy,
            now,
            zones=frozenset({"runs-table"}),
        )
        assert plan.items == () and plan.is_empty


def _raw_policy(config, *, keep_last=1, enabled=True):
    policy = resolve_maintenance_settings(config)
    # Keep the explicitly validated raw/bronze floor consistent in these focused fixtures.
    return replace(
        policy,
        bronze=replace(policy.bronze, retain_last=keep_last),
        raw=replace(policy.raw, enabled=enabled, keep_last_runs=keep_last),
    )


def test_raw_is_opt_in_and_requires_zone_selection(policy_config, now):
    inventory = MaintenanceInventory(raw=(_raw("old", now, succeeded=False),))
    assert plan_retention(
        inventory, _raw_policy(policy_config, enabled=False), now, zones=frozenset({"raw"})
    ).is_empty
    assert plan_retention(inventory, _raw_policy(policy_config), now, zones=frozenset()).is_empty
    assert _targets(
        plan_retention(inventory, _raw_policy(policy_config), now, zones=frozenset({"raw"}))
    ) == {"old"}


def test_raw_keeps_latest_successful_prefixes_per_source_only(policy_config, now):
    policy = _raw_policy(policy_config, keep_last=2)
    inventory = MaintenanceInventory(
        raw=tuple(
            _raw(f"{source}/{name}", now, source=source, age=age, succeeded=status)
            for source in ("alpha", "beta")
            for name, age, status in (
                ("old", 150, True),
                ("kept-1", 140, True),
                ("kept-2", 130, True),
                ("failed", 120, False),
                ("unknown", 110, None),
            )
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"raw"}))
    assert _targets(plan) == {
        f"{source}/{name}" for source in ("alpha", "beta") for name in ("old", "failed")
    }
    for source in ("alpha", "beta"):
        assert _reasons(plan)[f"{source}/kept-1"] == "keep_last_runs"
        assert _reasons(plan)[f"{source}/kept-2"] == "keep_last_runs"
        assert _reasons(plan)[f"{source}/unknown"] == "unknown_run_status"


def test_raw_same_day_success_ties_use_descending_segment(policy_config, now):
    inventory = MaintenanceInventory(raw=tuple(_raw(name, now) for name in ("a", "c", "b")))
    plan = plan_retention(inventory, _raw_policy(policy_config), now, zones=frozenset({"raw"}))
    assert _targets(plan) == {"a", "b"}
    assert _reasons(plan) == {"c": "keep_last_runs"}


def test_raw_live_progress_protects_all_matching_segments_even_for_failed_runs(policy_config, now):
    inventory = replace(
        _metadata_inventory(live={"alpha": "run-a"}),
        raw=(
            _raw("first", now, segment="run-a", succeeded=False),
            _raw("collision", now, segment="run-a", succeeded=False),
            _raw("other-source", now, source="beta", segment="run-a", succeeded=False),
        ),
    )
    plan = plan_retention(inventory, _raw_policy(policy_config), now, zones=frozenset({"raw"}))
    assert _targets(plan) == {"other-source"}
    assert _reasons(plan) == {"first": "live_progress", "collision": "live_progress"}


def test_raw_legacy_progress_skips_whole_source_without_aborting_other_sources(policy_config, now):
    inventory = replace(
        _metadata_inventory(live={"alpha": None}),
        raw=(
            _raw("alpha/1", now, succeeded=False),
            _raw("alpha/2", now, succeeded=False),
            _raw("beta/1", now, source="beta", succeeded=False),
        ),
    )
    plan = plan_retention(inventory, _raw_policy(policy_config), now, zones=frozenset({"raw"}))
    assert _targets(plan) == {"beta/1"}
    skipped = [item for item in plan.items if item.skipped_reason is not None]
    assert len(skipped) == 1
    assert skipped[0].target == "alpha"
    assert skipped[0].skipped_reason == "legacy_progress_prefix"
    assert _reasons(plan) == {
        "alpha/1": "legacy_progress_prefix",
        "alpha/2": "legacy_progress_prefix",
    }


def test_raw_strict_day_boundary_and_source_filter(policy_config, now):
    inventory = MaintenanceInventory(
        raw=tuple(
            _raw(f"{source}/{age}", now, source=source, age=age, succeeded=False)
            for source in ("alpha", "beta")
            for age in (91, 90, 89)
        )
    )
    plan = plan_retention(
        inventory,
        _raw_policy(policy_config),
        now,
        zones=frozenset({"raw"}),
        source_ids=frozenset({"alpha"}),
    )
    assert _targets(plan) == {"alpha/91"}
    assert _reasons(plan) == {"alpha/90": "within_window", "alpha/89": "within_window"}


def _all_zones_inventory(now):
    return MaintenanceInventory(
        bronze=(
            BronzeTableInventory(
                "bronze.alpha",
                ("alpha",),
                tuple(
                    SnapshotEntry(i, now - timedelta(days=age), i == 3)
                    for i, age in enumerate((120, 110, 100, 0))
                ),
            ),
        ),
        metadata=MetadataZoneInventory(
            (
                _artifact("old-history", now),
                _artifact("young-history", now, age=1),
            ),
            frozenset({Path("current.json")}),
            {},
        ),
        lineage_events=tuple(
            EventFileEntry(Path(f"event/{age}"), now.date() - timedelta(days=age))
            for age in (100, 90, 0)
        ),
        runs_table=tuple(
            RunsTablePartitionEntry(now.date() - timedelta(days=age), age) for age in (100, 90, 0)
        ),
        raw=tuple(_raw(f"raw/{age}", now, age=age) for age in (120, 100, 0)),
    )


ALL_ZONES = frozenset({"bronze", "metadata", "lineage", "runs-table", "raw"})


@pytest.mark.parametrize("zone", sorted(ALL_ZONES))
def test_each_zone_is_idempotent_after_applying_its_own_candidates(policy_config, now, zone):
    inventory = _all_zones_inventory(now)
    policy = _raw_policy(policy_config)
    zones = frozenset({zone})
    plan = plan_retention(inventory, policy, now, zones=zones)
    assert not plan.is_empty
    removed = _targets(plan)
    expired = _snapshot_ids(plan)
    assert inventory.metadata is not None
    after = replace(
        inventory,
        bronze=tuple(
            replace(
                table,
                snapshots=tuple(
                    entry for entry in table.snapshots if entry.snapshot_id not in expired
                ),
            )
            for table in inventory.bronze
        ),
        metadata=replace(
            inventory.metadata,
            artifacts=tuple(
                entry for entry in inventory.metadata.artifacts if str(entry.path) not in removed
            ),
        ),
        lineage_events=tuple(
            entry for entry in inventory.lineage_events if str(entry.path) not in removed
        ),
        runs_table=tuple(
            entry
            for entry in inventory.runs_table
            if entry.emitted_at_day.isoformat() not in removed
        ),
        raw=tuple(entry for entry in inventory.raw if str(entry.path) not in removed),
    )
    assert plan_retention(after, policy, now, zones=zones).is_empty


def test_all_zone_inventory_permutations_have_identical_plans_and_both_digests(policy_config, now):
    inventory = _all_zones_inventory(now)
    policy = _raw_policy(policy_config)
    table = inventory.bronze[0]
    assert inventory.metadata is not None
    baseline = plan_retention(inventory, policy, now, zones=ALL_ZONES)
    assert [item.zone for item in baseline.items] == sorted(item.zone for item in baseline.items)
    for offset in range(3):

        def permute(entries, offset=offset):
            rotated = entries[offset:] + entries[:offset]
            return rotated[::-1]

        permuted = replace(
            inventory,
            bronze=(replace(table, snapshots=permute(table.snapshots)),),
            metadata=replace(inventory.metadata, artifacts=permute(inventory.metadata.artifacts)),
            lineage_events=permute(inventory.lineage_events),
            runs_table=permute(inventory.runs_table),
            raw=permute(inventory.raw),
        )
        actual = plan_retention(permuted, policy, now, zones=ALL_ZONES)
        assert actual == baseline
        assert actual.digest == baseline.digest
        assert actual.policy_digest == baseline.policy_digest == policy.digest


def test_plan_digest_canonicalizes_item_and_detail_order_and_ignores_execution_fields(now):
    a = PlannedItem("bronze", "table", "expire_snapshots", {"b": "2", "a": "1"})
    b = PlannedItem("metadata", "path", "delete_file", {})
    first = RetentionPlan((a, b), (), now, "policy")
    second = replace(
        first,
        items=(b, replace(a, detail={"a": "1", "b": "2"})),
        protected=(ProtectedItem("metadata", "state", "state_file"),),
        now=now + timedelta(seconds=1),
        policy_digest="other",
    )
    assert first.digest == second.digest
    for changed in (
        replace(a, zone="raw"),
        replace(a, target="other"),
        replace(a, action="remove_orphan_files"),
        replace(a, detail={"a": "3"}),
    ):
        assert replace(first, items=(changed, b)).digest != first.digest


def test_all_zone_planning_performs_no_io(policy_config, now, monkeypatch):
    inventory = _all_zones_inventory(now)
    policy = _raw_policy(policy_config)

    def forbidden(*args, **kwargs):
        pytest.fail("pure planner performed I/O")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "exists", forbidden)
        patch.setattr(Path, "unlink", forbidden)
        plan = plan_retention(inventory, policy, now, zones=ALL_ZONES)
        digest = plan.digest
    assert not plan.is_empty and len(digest) == 64


def test_planner_transitive_imports_are_engine_free_in_a_fresh_interpreter():
    script = """
import importlib.abc
import json
import sys

sys.path.insert(0, sys.argv[1])

class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('pyspark', 'pyiceberg', 'janus.runtime', 'janus.writers')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('planner imported ' + fullname)
        if fullname.startswith('janus.strategies.') and fullname != 'janus.strategies.common':
            raise AssertionError('planner imported strategy ' + fullname)

sys.meta_path.insert(0, Tripwire())
from janus.maintenance.planning import plan_retention
print(json.dumps(sorted(name for name in sys.modules if name.startswith('janus.strategies'))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[3] / "src")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == ["janus.strategies", "janus.strategies.common"]


@pytest.mark.parametrize(
    ("run_id", "expected"),
    [
        ("run:a/b c", "run-a-b-c"),
        ("  __id=.x--  ", "__id=.x"),
        (" / : ", "run"),
        ("", "run"),
        ("ação", "a-o"),
    ],
)
def test_raw_path_helper_preserves_normalization_and_private_alias(run_id, expected):
    assert raw_run_path_segment(run_id) == expected
    assert _safe_raw_path_segment is raw_run_path_segment


@pytest.mark.parametrize("zone", sorted(ALL_ZONES))
def test_empty_inventory_is_empty_in_every_zone(policy_config, now, zone):
    plan = plan_retention(
        MaintenanceInventory(), _raw_policy(policy_config), now, zones=frozenset({zone})
    )
    assert plan.is_empty and plan.items == () and plan.protected == ()


def test_now_must_be_aware_and_unknown_zones_are_rejected(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    with pytest.raises(ValueError, match="timezone-aware"):
        plan_retention(MaintenanceInventory(), policy, now.replace(tzinfo=None), zones=ALL_ZONES)
    with pytest.raises(ValueError, match="unsupported maintenance zones: typo"):
        plan_retention(MaintenanceInventory(), policy, now, zones=frozenset({"typo"}))


@pytest.mark.parametrize(("days", "expected"), [(0, {1, 2}), (100, set())])
def test_override_age_window_independently_replaces_profile_window(
    policy_config, now, days, expected
):
    inv, _, policy = _api(policy_config)
    config = import_module("janus.models").BronzeRetentionConfig
    inventory = _bronze(inv, now, (60, 50, 0), override=config(1, days))
    plan = plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    assert _snapshot_ids(plan) == expected


def test_naive_snapshot_timestamp_is_protected(policy_config, now):
    policy = resolve_maintenance_settings(policy_config)
    inventory = MaintenanceInventory(
        bronze=(
            BronzeTableInventory(
                "bronze.unknown",
                ("alpha",),
                (SnapshotEntry(1, datetime(2020, 1, 1), False),),
            ),
        )
    )
    plan = plan_retention(inventory, policy, now, zones=frozenset({"bronze"}))
    assert plan.is_empty
    assert _reasons(plan) == {"bronze.unknown#1": "unaged"}


def test_shared_lineage_and_runs_table_are_not_filtered_by_source(policy_config, now):
    inventory = _all_zones_inventory(now)
    policy = _raw_policy(policy_config)
    zones = frozenset({"lineage", "runs-table"})
    baseline = plan_retention(inventory, policy, now, zones=zones)
    actual = plan_retention(
        inventory, policy, now, zones=zones, source_ids=frozenset({"unrelated"})
    )
    assert actual == baseline


def test_multiple_bronze_tables_have_stable_plan_order(policy_config, now):
    inventory = _all_zones_inventory(now)
    alpha = inventory.bronze[0]
    beta = replace(alpha, table_identifier="bronze.beta", source_ids=("beta",))
    policy = _raw_policy(policy_config)
    baseline = plan_retention(
        replace(inventory, bronze=(alpha, beta)), policy, now, zones=frozenset({"bronze"})
    )
    permuted = plan_retention(
        replace(inventory, bronze=(beta, alpha)), policy, now, zones=frozenset({"bronze"})
    )
    assert baseline == permuted and baseline.digest == permuted.digest
    assert [item.target for item in baseline.items] == ["bronze.alpha", "bronze.beta"]


def test_same_instant_in_different_timezones_yields_identical_plan(policy_config, now):
    policy = _raw_policy(policy_config)
    inventory = _all_zones_inventory(now)
    local_now = now.astimezone(timezone(timedelta(hours=-3)))
    baseline = plan_retention(inventory, policy, now, zones=ALL_ZONES)
    assert plan_retention(inventory, policy, local_now, zones=ALL_ZONES) == baseline


def test_plan_preserves_input_inventory_and_policy(policy_config, now):
    inventory = _all_zones_inventory(now)
    policy = _raw_policy(policy_config)
    before = repr(inventory), repr(policy), policy.digest
    plan_retention(inventory, policy, now, zones=ALL_ZONES)
    assert (repr(inventory), repr(policy), policy.digest) == before
