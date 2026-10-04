from dataclasses import replace
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path

import pytest

from tests.support.maintenance_zone import FAMILIES, SOURCES

pytestmark = pytest.mark.xfail(
    strict=True, reason="pure retention planner absent"
)


def _api(config):
    inventory = import_module("janus.maintenance.inventory")
    planning = import_module("janus.maintenance.planning")
    policy = import_module("janus.maintenance.settings").resolve_maintenance_settings(config)
    return inventory, planning, policy


def _snapshot_ids(plan):
    import json

    return {
        int(snapshot)
        for item in plan.items
        if item.action == "expire_snapshots"
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
    import json

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
