"""Command and lifecycle reds; metadata and lineage must not construct a provider."""

import json
from importlib import import_module
from unittest.mock import MagicMock

import pytest
import yaml

from janus.runtime import SparkSessionProvider
from janus.utils.logging import REDACTED_VALUE
from tests.support.operator_cli import arm_spark_tripwire, run_janus


def _project(tmp_path, config):
    from tests.support.semantics_fixtures import CLEAN, install_profile, materialize

    root = materialize(CLEAN, tmp_path / "project")
    path = install_profile(root, "local")
    profile = yaml.safe_load(path.read_text())
    profile.update(config)
    path.write_text(yaml.safe_dump(profile))
    return root


@pytest.mark.parametrize("zone", ["metadata", "lineage"])
def test_file_zones_do_not_construct_or_acquire_provider(
    tmp_path, policy_config, monkeypatch, zone
):
    import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    arm_spark_tripwire(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("file zone constructed a Spark provider")

    monkeypatch.setattr(SparkSessionProvider, "__init__", forbidden)
    result = run_janus(
        [
            "maintain",
            "--environment",
            "local",
            "--project-root",
            str(root),
            "--apply",
            "--zone",
            zone,
            "--format",
            "json",
        ]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dry_run"] is False


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_bronze_acquires_once_and_stops_in_finally(tmp_path, policy_config, monkeypatch, fail):
    import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    calls = []
    session = MagicMock()
    session.sql.return_value.collect.return_value = []

    def get(provider):
        calls.append("get")
        if fail:
            raise RuntimeError("scripted acquisition failure")
        return session

    monkeypatch.setattr(SparkSessionProvider, "get", get)
    monkeypatch.setattr(SparkSessionProvider, "stop", lambda provider: calls.append("stop"))
    try:
        result = run_janus(
            ["maintain", "--environment", "local", "--project-root", str(root), "--zone", "bronze"]
        )
        assert result.exit_code == (1 if fail else 0), result.output
    finally:
        assert calls == ["get", "stop"]


@pytest.mark.parametrize(
    "zones", [("bronze",), ("runs-table",), ("bronze", "runs-table", "metadata")]
)
@pytest.mark.parametrize("apply", [False, True])
def test_compute_zones_share_one_provider_and_one_session(
    tmp_path, policy_config, monkeypatch, zones, apply
):
    command = import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    session = object()
    calls = []
    received = []

    class Provider:
        def __init__(self, config, paths, logger):
            assert config["maintenance"] == policy_config["maintenance"]
            assert paths["metadata_dir"] == root / "data/metadata"
            assert logger is not None
            calls.append("build")

        def get(self):
            calls.append("get")
            return session

        def stop(self):
            calls.append("stop")

        def take_cleanup_failures(self):
            calls.append("cleanup")
            return ()

    def collect(*args, session, **kwargs):
        from janus.maintenance.inventory import MaintenanceInventory

        received.append(session)
        return MaintenanceInventory()

    monkeypatch.setattr(command, "SparkSessionProvider", Provider)
    monkeypatch.setattr(command, "collect_inventory", collect)
    argv = ["maintain", "--project-root", str(root), "--format", "json"]
    for zone in zones:
        argv.extend(("--zone", zone))
    if apply:
        argv.append("--apply")
    result = run_janus(argv)
    assert result.exit_code == 0, result.output
    assert received == [session]
    assert calls == ["build", "get", "stop", "cleanup"]


@pytest.mark.parametrize("apply", [False, True])
def test_cleanup_failure_is_recorded_and_drives_exit_one(
    tmp_path, policy_config, monkeypatch, apply
):
    root = _project(tmp_path, policy_config)
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    session.stop.side_effect = RuntimeError("scripted teardown failure token=planted-secret")
    monkeypatch.setattr(SparkSessionProvider, "_build_session", lambda provider: session)
    argv = ["maintain", "--project-root", str(root), "--zone", "bronze", "--format", "json"]
    if apply:
        argv.append("--apply")
    result = run_janus(argv)
    assert result.exit_code == 1, result.output
    record = json.loads(result.stdout)
    assert record["failures"] == [
        {
            "stage": "spark_cleanup",
            "failure_type": "RuntimeError",
            "failure_message": f"scripted teardown failure token={REDACTED_VALUE}",
        }
    ]
    assert "planted-secret" not in result.stdout
    (path,) = (root / "data/metadata/maintenance").glob("*.json")
    assert json.loads(path.read_text()) == record
    session.stop.assert_called_once()


def test_acquisition_failure_and_cleanup_are_both_preserved(tmp_path, policy_config, monkeypatch):
    root = _project(tmp_path, policy_config)
    calls = []

    def get(provider):
        calls.append("get")
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(SparkSessionProvider, "get", get)
    monkeypatch.setattr(SparkSessionProvider, "stop", lambda provider: calls.append("stop"))
    monkeypatch.setattr(
        SparkSessionProvider,
        "take_cleanup_failures",
        lambda provider: (RuntimeError("cleanup failed"),),
    )
    result = run_janus(
        ["maintain", "--project-root", str(root), "--zone", "bronze", "--format", "json"]
    )
    assert result.exit_code == 1, result.output
    record = json.loads(result.stdout)
    assert [failure["failure_message"] for failure in record["failures"]] == [
        "acquisition failed",
        "cleanup failed",
    ]
    assert calls == ["get", "stop"]


def test_missing_policy_refuses_with_exit_two(tmp_path, monkeypatch):
    import_module("janus.cli.maintain")
    from tests.support.semantics_fixtures import CLEAN, install_profile, materialize

    root = materialize(CLEAN, tmp_path / "project")
    profile_path = install_profile(root, "local")
    profile = yaml.safe_load(profile_path.read_text())
    profile.pop("maintenance", None)
    profile_path.write_text(yaml.safe_dump(profile))
    arm_spark_tripwire(monkeypatch)
    result = run_janus(["maintain", "--environment", "local", "--project-root", str(root)])
    assert result.exit_code == 2
    assert "maintenance" in result.output
