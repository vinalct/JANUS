from importlib import import_module
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.xfail(strict=True, reason="maintenance settings absent")


def test_absent_block_refuses():
    settings = import_module("janus.maintenance.settings")
    errors = import_module("janus.maintenance.errors")
    with pytest.raises(errors.MaintenanceProfileError, match="maintenance"):
        settings.resolve_maintenance_settings({})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("maintenance", []),
        ("maintenance.bronze.retain_last", 0),
        ("maintenance.bronze.retain_last", True),
        ("maintenance.bronze.retain_last", "3"),
        ("maintenance.bronze.older_than_days", -1),
        ("maintenance.metadata.keep_last_runs", 0),
        ("maintenance.lineage_events.older_than_days", "90"),
        ("maintenance.raw.enabled", "false"),
        ("maintenance.bronze.orphan_older_than_days", 0),
        ("maintenance.bronze.compact.enabled", "yes"),
    ],
)
def test_wrong_value_names_key(policy_config, path, value):
    settings = import_module("janus.maintenance.settings")
    errors = import_module("janus.maintenance.errors")
    target = policy_config
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(errors.MaintenanceProfileError, match=path.replace(".", r"\.")):
        settings.resolve_maintenance_settings(policy_config)


@pytest.mark.parametrize(
    "block", ["maintenance", "maintenance.bronze", "maintenance.bronze.compact", "maintenance.raw"]
)
def test_unknown_key_names_key(policy_config, block):
    settings = import_module("janus.maintenance.settings")
    errors = import_module("janus.maintenance.errors")
    target = policy_config
    for key in block.split("."):
        target = target[key]
    target["typo"] = 42
    with pytest.raises(errors.MaintenanceProfileError, match="typo"):
        settings.resolve_maintenance_settings(policy_config)


def test_raw_floor_cannot_undercut_bronze(policy_config):
    settings = import_module("janus.maintenance.settings")
    errors = import_module("janus.maintenance.errors")
    policy_config["maintenance"]["raw"].update(enabled=True, keep_last_runs=2)
    with pytest.raises(errors.MaintenanceProfileError, match=r"raw\.keep_last_runs"):
        settings.resolve_maintenance_settings(policy_config)


@pytest.mark.parametrize("block", ["bronze", "metadata", "lineage_events", "runs_table", "raw"])
def test_required_subblock_is_not_defaulted(policy_config, block):
    settings = import_module("janus.maintenance.settings")
    errors = import_module("janus.maintenance.errors")
    del policy_config["maintenance"][block]
    with pytest.raises(errors.MaintenanceProfileError, match=block):
        settings.resolve_maintenance_settings(policy_config)


def test_quarantined_local_profile_has_no_policy_and_is_refused():
    settings = import_module("janus.maintenance.settings")
    errors = import_module("janus.maintenance.errors")
    root = Path(__file__).resolve().parents[3]
    # The supplemental local profile is quarantined by the catalog-containment rule.
    profiles = sorted((root / "conf/environments").glob("local-*.yaml"))
    assert len(profiles) == 1
    config = yaml.safe_load(profiles[0].read_text())
    assert "maintenance" not in config
    with pytest.raises(errors.MaintenanceProfileError, match="maintenance"):
        settings.resolve_maintenance_settings(config)
