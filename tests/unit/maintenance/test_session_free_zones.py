"""Command and lifecycle reds; metadata and lineage must not construct a provider."""

import json
from importlib import import_module
from unittest.mock import MagicMock

import pytest
import yaml

from janus.runtime import SparkSessionProvider
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


@pytest.mark.xfail(
    strict=True, reason="bronze maintain provider lifecycle absent"
)
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_bronze_acquires_once_and_stops_in_finally(tmp_path, policy_config, monkeypatch, fail):
    import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    calls = []
    session = MagicMock()
    session.catalog.tableExists.return_value = False

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
