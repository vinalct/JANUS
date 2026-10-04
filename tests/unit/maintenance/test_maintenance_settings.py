"""Fail-closed retention profiles and the fingerprint recorded with maintenance evidence."""

from __future__ import annotations

import ast
import os
import re
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest

from janus.maintenance import (
    MaintenancePolicy,
    MaintenanceProfileError,
    resolve_maintenance_settings,
)
from janus.maintenance.errors import MaintenanceError
from janus.maintenance.settings import (
    BronzeRetentionPolicy,
    LineageEventsRetentionPolicy,
    MetadataRetentionPolicy,
    RawRetentionPolicy,
    RunsTableRetentionPolicy,
)
from janus.utils.environment import load_environment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBBLOCKS = ("bronze", "metadata", "lineage_events", "runs_table", "raw")
INTEGER_PATHS = (
    "maintenance.bronze.retain_last",
    "maintenance.bronze.older_than_days",
    "maintenance.bronze.orphan_older_than_days",
    "maintenance.bronze.compact.target_file_size_mb",
    "maintenance.metadata.keep_last_runs",
    "maintenance.metadata.older_than_days",
    "maintenance.lineage_events.older_than_days",
    "maintenance.runs_table.older_than_days",
    "maintenance.raw.keep_last_runs",
    "maintenance.raw.older_than_days",
)
REQUIRED_PATHS = (
    "maintenance.bronze.retain_last",
    "maintenance.bronze.older_than_days",
    "maintenance.metadata.keep_last_runs",
    "maintenance.metadata.older_than_days",
    "maintenance.lineage_events.older_than_days",
    "maintenance.runs_table.older_than_days",
)


def _target(config, path):
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        target = target[key]
    return target, keys[-1]


def _set(config, path, value):
    target, key = _target(config, path)
    target[key] = value


def _assert_error(config, path, detail):
    with pytest.raises(MaintenanceProfileError) as raised:
        resolve_maintenance_settings(config)
    message = str(raised.value)
    assert path in message
    assert detail in message
    assert len(message.splitlines()) == 1
    assert str(PROJECT_ROOT) not in message
    assert "Traceback" not in message
    return message


@pytest.mark.parametrize("config", [{}, {"maintenance": None}, {"maintenance": " \n "}])
def test_absent_block_refuses(config):
    assert _assert_error(config, "maintenance", "no 'maintenance' block") == (
        "Environment config has no 'maintenance' block; "
        "'janus maintain' applies only a declared policy"
    )


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
    _set(policy_config, path, value)
    _assert_error(policy_config, path, "maintenance")


@pytest.mark.parametrize(
    "block", ["maintenance", "maintenance.bronze.compact", *(f"maintenance.{b}" for b in SUBBLOCKS)]
)
def test_unknown_key_names_key(policy_config, block):
    target = policy_config
    for key in block.split("."):
        target = target[key]
    target["typo\nkey"] = "/outside/profile/sensitive-value"
    message = _assert_error(policy_config, block, "unsupported")
    assert repr("typo\nkey") in message
    assert "supported keys:" in message
    assert "/outside/profile" not in message


def test_raw_floor_cannot_undercut_bronze(policy_config):
    policy_config["maintenance"]["raw"].update(enabled=True, keep_last_runs=2)
    assert _assert_error(policy_config, "maintenance.raw.keep_last_runs", "must be at least") == (
        "maintenance.raw.keep_last_runs (2) must be at least maintenance.bronze.retain_last (3) "
        "so every retained bronze snapshot's raw input is retained too"
    )


@pytest.mark.parametrize("block", SUBBLOCKS)
def test_required_subblock_is_not_defaulted(policy_config, block):
    del policy_config["maintenance"][block]
    _assert_error(policy_config, f"maintenance.{block}", "must set")


@pytest.mark.parametrize(
    ("profile", "overlay"),
    [
        ("local", None),
        ("cluster", None),
        ("cluster", "cluster-rest.env.example"),
        ("local-hadoop", None),
    ],
    ids=["local", "cluster", "cluster-rest", "local-hadoop"],
)
def test_shipped_profile_has_no_policy_and_is_refused(monkeypatch, profile, overlay):
    for key in list(os.environ):
        if key.startswith("JANUS_"):
            monkeypatch.delenv(key)
    if overlay is not None:
        # cluster-rest is an environment overlay on cluster.yaml, not a second YAML profile.
        for line in (PROJECT_ROOT / "conf/environments" / overlay).read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and not key.startswith("#"):
                monkeypatch.setenv(key, value)
    config = load_environment_config(profile, PROJECT_ROOT)
    if overlay is not None:
        assert config["spark"]["iceberg"]["catalog_type"] == "rest"
    assert "maintenance" not in config
    _assert_error(config, "maintenance", "no 'maintenance' block")


def test_profile_error_uses_the_existing_configuration_exception():
    assert issubclass(MaintenanceProfileError, ValueError)
    assert issubclass(MaintenanceProfileError, MaintenanceError)


def test_valid_policy_resolves_every_field_without_mutating_input(policy_config):
    original = deepcopy(policy_config)
    policy = resolve_maintenance_settings(policy_config)
    assert policy == MaintenancePolicy(
        bronze=BronzeRetentionPolicy(3, 30),
        metadata=MetadataRetentionPolicy(20, 90),
        lineage_events=LineageEventsRetentionPolicy(90),
        runs_table=RunsTableRetentionPolicy(90),
        raw=RawRetentionPolicy(False, 3, 90),
    )
    assert policy_config == original


@pytest.mark.parametrize("path", INTEGER_PATHS)
@pytest.mark.parametrize("value", [True, False, 1.0, "3", [], {}, "bad\nvalue"])
def test_integer_values_are_strict(policy_config, path, value):
    _set(policy_config, path, value)
    _assert_error(policy_config, path, "non-integer")


@pytest.mark.parametrize(
    "path",
    [
        "maintenance.bronze.remove_orphan_files",
        "maintenance.bronze.compact.enabled",
        "maintenance.raw.enabled",
    ],
)
@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}, "bad\nvalue"])
def test_boolean_values_are_strict(policy_config, path, value):
    _set(policy_config, path, value)
    _assert_error(policy_config, path, "non-boolean")


@pytest.mark.parametrize("path", INTEGER_PATHS)
def test_integer_floors_name_the_key(policy_config, path):
    _set(policy_config, path, -1)
    rule = (
        "must be at least 1"
        if path.endswith(("retain_last", "orphan_older_than_days", "target_file_size_mb"))
        or path == "maintenance.metadata.keep_last_runs"
        else "must not be negative"
    )
    _assert_error(policy_config, path, rule)


@pytest.mark.parametrize(
    "path",
    [
        "maintenance.bronze.retain_last",
        "maintenance.bronze.orphan_older_than_days",
        "maintenance.bronze.compact.target_file_size_mb",
        "maintenance.metadata.keep_last_runs",
    ],
)
def test_positive_integer_floors_reject_zero(policy_config, path):
    _set(policy_config, path, 0)
    _assert_error(policy_config, path, "must be at least 1")


@pytest.mark.parametrize("path", REQUIRED_PATHS)
@pytest.mark.parametrize("value", [None, "", " \n "])
def test_empty_required_value_is_absent(policy_config, path, value):
    _set(policy_config, path, value)
    _assert_error(policy_config, path, "must set")


@pytest.mark.parametrize("path", REQUIRED_PATHS)
def test_missing_required_value_is_absent(policy_config, path):
    target, key = _target(policy_config, path)
    del target[key]
    _assert_error(policy_config, path, "must set")


@pytest.mark.parametrize("block", SUBBLOCKS)
@pytest.mark.parametrize("value", [None, "", " \n "])
def test_empty_required_subblock_is_absent(policy_config, block, value):
    _set(policy_config, f"maintenance.{block}", value)
    _assert_error(policy_config, f"maintenance.{block}", "must set")


@pytest.mark.parametrize(
    "path", ["maintenance", "maintenance.bronze.compact", *(f"maintenance.{b}" for b in SUBBLOCKS)]
)
@pytest.mark.parametrize("value", [[], True, 3, "wrong\nshape"])
def test_blocks_must_be_mappings(policy_config, path, value):
    _set(policy_config, path, value)
    _assert_error(policy_config, path, "must be a mapping")


@pytest.mark.parametrize("value", [None, "", " \n "])
def test_empty_optional_values_use_defaults(policy_config, value):
    block = policy_config["maintenance"]
    block["item_timeout_seconds"] = value
    block["bronze"].update(remove_orphan_files=value, orphan_older_than_days=value, compact=value)
    block["raw"] = {"enabled": value, "keep_last_runs": value, "older_than_days": value}
    policy = resolve_maintenance_settings(policy_config)
    assert policy.bronze == BronzeRetentionPolicy(3, 30)
    assert policy.raw == RawRetentionPolicy()
    assert policy.item_timeout_seconds == 1800.0


def test_omitted_optional_values_use_defaults(policy_config):
    block = policy_config["maintenance"]
    block["bronze"] = {"retain_last": 3, "older_than_days": 30}
    block["raw"] = {}
    policy = resolve_maintenance_settings(policy_config)
    assert policy.bronze == BronzeRetentionPolicy(3, 30)
    assert policy.raw == RawRetentionPolicy()


@pytest.mark.parametrize("value", [None, "", " \n "])
def test_compaction_requires_nonempty_target_size(policy_config, value):
    policy_config["maintenance"]["bronze"]["compact"].update(
        enabled=True, target_file_size_mb=value
    )
    _assert_error(
        policy_config,
        "maintenance.bronze.compact.target_file_size_mb",
        "when compaction is enabled",
    )


def test_compaction_requires_target_size_when_omitted(policy_config):
    policy_config["maintenance"]["bronze"]["compact"]["enabled"] = True
    _assert_error(
        policy_config,
        "maintenance.bronze.compact.target_file_size_mb",
        "when compaction is enabled",
    )


@pytest.mark.parametrize("key", ["keep_last_runs", "older_than_days"])
@pytest.mark.parametrize("value", [None, "", " \n "])
def test_enabled_raw_requires_nonempty_policy(policy_config, key, value):
    policy_config["maintenance"]["raw"].update(enabled=True, **{key: value})
    _assert_error(policy_config, f"maintenance.raw.{key}", "must set")


@pytest.mark.parametrize("key", ["keep_last_runs", "older_than_days"])
def test_enabled_raw_requires_explicit_policy(policy_config, key):
    raw = policy_config["maintenance"]["raw"]
    raw["enabled"] = True
    del raw[key]
    _assert_error(policy_config, f"maintenance.raw.{key}", "must set")


@pytest.mark.parametrize("keep_last_runs", [3, 4])
def test_enabled_raw_respects_bronze_floor(policy_config, keep_last_runs):
    policy_config["maintenance"]["raw"].update(enabled=True, keep_last_runs=keep_last_runs)
    assert resolve_maintenance_settings(policy_config).raw == RawRetentionPolicy(
        True, keep_last_runs, 90
    )


def test_zero_age_is_valid_for_every_zone(policy_config):
    for block in SUBBLOCKS:
        policy_config["maintenance"][block]["older_than_days"] = 0
    policy_config["maintenance"]["bronze"]["retain_last"] = 1
    policy_config["maintenance"]["metadata"]["keep_last_runs"] = 1
    policy_config["maintenance"]["raw"].update(enabled=True, keep_last_runs=1)
    policy = resolve_maintenance_settings(policy_config)
    assert all(getattr(policy, block).older_than_days == 0 for block in SUBBLOCKS)


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        (0, "positive"),
        (-1.0, "positive"),
        (True, "non-numeric"),
        ("30", "non-numeric"),
        ([], "non-numeric"),
        ({}, "non-numeric"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
        (10**400, "finite"),
    ],
)
def test_unusable_timeout_is_refused(policy_config, value, detail):
    policy_config["maintenance"]["item_timeout_seconds"] = value
    _assert_error(policy_config, "maintenance.item_timeout_seconds", detail)


@pytest.mark.parametrize("value", [1, 0.5, 1800.0])
def test_positive_numeric_timeout_resolves(policy_config, value):
    policy_config["maintenance"]["item_timeout_seconds"] = value
    assert resolve_maintenance_settings(policy_config).item_timeout_seconds == float(value)


def test_unknown_nonstring_keys_raise_profile_errors(policy_config):
    policy_config["maintenance"].update({1: "hidden", None: "hidden"})
    _assert_error(policy_config, "maintenance", "unsupported")


def test_first_error_does_not_collect_source_model_issues(policy_config):
    policy_config["maintenance"]["bronze"]["retain_last"] = 0
    policy_config["maintenance"]["metadata"]["keep_last_runs"] = 0
    message = _assert_error(policy_config, "maintenance.bronze.retain_last", "at least 1")
    assert "maintenance.metadata" not in message


def _reverse_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_keys(item) for key, item in reversed(value.items())}
    return value


def test_digest_is_stable_across_resolutions_and_key_order(policy_config):
    digest = resolve_maintenance_settings(policy_config).digest
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert resolve_maintenance_settings(deepcopy(policy_config)).digest == digest
    assert resolve_maintenance_settings(_reverse_keys(policy_config)).digest == digest


def test_digest_hashes_resolved_defaults_and_numeric_timeout(policy_config):
    first = resolve_maintenance_settings(policy_config)
    block = policy_config["maintenance"]
    block["item_timeout_seconds"] = 1800
    del block["bronze"]["remove_orphan_files"]
    del block["bronze"]["orphan_older_than_days"]
    del block["bronze"]["compact"]
    assert resolve_maintenance_settings(policy_config).digest == first.digest


@pytest.mark.parametrize(
    ("path", "value"),
    [
        *((path, 100) for path in INTEGER_PATHS),
        ("maintenance.bronze.remove_orphan_files", True),
        ("maintenance.bronze.compact.enabled", True),
        ("maintenance.raw.enabled", True),
        ("maintenance.item_timeout_seconds", 900.5),
    ],
)
def test_digest_changes_with_every_resolved_field(policy_config, path, value):
    policy_config["maintenance"]["bronze"]["compact"]["target_file_size_mb"] = 128
    before = resolve_maintenance_settings(policy_config).digest
    _set(policy_config, path, value)
    assert resolve_maintenance_settings(policy_config).digest != before


@pytest.mark.parametrize("block", [*SUBBLOCKS, None])
def test_policy_dataclasses_are_frozen_and_slotted(policy_config, block):
    policy = resolve_maintenance_settings(policy_config)
    instance = policy if block is None else getattr(policy, block)
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, fields(instance)[0].name, None)


def test_maintenance_package_has_no_engine_imports():
    modules = sorted((PROJECT_ROOT / "src/janus/maintenance").rglob("*.py"))
    assert modules
    for module in modules:
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in {"pyspark", "pyiceberg"} for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"pyspark", "pyiceberg"}
