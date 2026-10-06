"""AC-9: file zones need no provider; compute is acquired once and always released."""

import json
from dataclasses import replace
from datetime import timedelta
from importlib import import_module
from unittest.mock import MagicMock

import pytest
import yaml

from janus.maintenance.inventory import (
    BronzeTableInventory,
    EventFileEntry,
    MaintenanceInventory,
    MetadataZoneInventory,
    RunArtifactEntry,
    RunsTablePartitionEntry,
    SnapshotEntry,
)
from janus.maintenance.records import ItemOutcome
from janus.runtime import SparkSessionProvider
from janus.utils.logging import REDACTED_VALUE
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.unit.runtime.test_spark_lifecycle_evidence import ArmedSpyProvider


def _project(tmp_path, config):
    from tests.support.semantics_fixtures import CLEAN, install_profile, materialize

    root = materialize(CLEAN, tmp_path / "project")
    path = install_profile(root, "local")
    profile = yaml.safe_load(path.read_text())
    profile.update(config)
    path.write_text(yaml.safe_dump(profile))
    return root


def _invoke(root, zones, *, apply=False):
    argv = ["maintain", "--project-root", str(root), "--format", "json"]
    for zone in zones:
        argv.extend(("--zone", zone))
    if apply:
        argv.append("--apply")
    return run_janus(argv)


@pytest.fixture
def lifecycle_spy(monkeypatch):
    """Reuse the extraction-boundary spy with counted calls and an injected session."""
    command = import_module("janus.cli.maintain")
    provider = ArmedSpyProvider()
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    provider._session_factory = lambda: session
    for method in ("get", "stop", "take_cleanup_failures"):
        monkeypatch.setattr(provider, method, MagicMock(wraps=getattr(provider, method)))
    factory = MagicMock(return_value=provider)
    monkeypatch.setattr(command, "SparkSessionProvider", factory)
    return provider, factory, session


def _actionable_inventory(root, now, zones):
    """Give the real planner candidates, so apply must actually reach the executor."""
    old = now - timedelta(days=120)
    metadata = None
    if "metadata" in zones:
        path = root / "data/metadata/pipelines/old/summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        metadata = MetadataZoneInventory(
            (RunArtifactEntry("pipeline_summary", path, None, "old", old),),
            frozenset(),
            {},
        )
    events = ()
    if "lineage" in zones:
        entries = []
        for day in (old.date(), now.date()):
            path = root / "data/metadata/lineage/openlineage" / f"events-{day}.ndjson"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
            entries.append(EventFileEntry(path, day))
        events = tuple(entries)
    return MaintenanceInventory(
        bronze=(
            BronzeTableInventory(
                "bronze.example",
                ("semantics_clean_producer",),
                tuple(
                    SnapshotEntry(value, old + timedelta(days=value), value == 4)
                    for value in range(1, 5)
                ),
            ),
        )
        if "bronze" in zones
        else (),
        metadata=metadata,
        lineage_events=events,
        runs_table=(RunsTablePartitionEntry(old.date(), row_count=2),)
        if "runs-table" in zones
        else (),
    )


@pytest.mark.parametrize("zones", [("metadata",), ("lineage",), ("metadata", "lineage")])
@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "apply"])
def test_file_zones_do_not_construct_or_acquire_provider(
    tmp_path, policy_config, monkeypatch, lifecycle_spy, zones, apply
):
    command = import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    provider, factory, session = lifecycle_spy
    arm_spark_tripwire(monkeypatch)
    factory.side_effect = AssertionError("file zone constructed a Spark provider")

    def collect(*args, session, zones, **kwargs):
        assert session is None
        return _actionable_inventory(root, args[4], zones)

    monkeypatch.setattr(command, "collect_inventory", collect)
    result = _invoke(root, zones, apply=apply)
    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert record["dry_run"] is not apply
    assert {item["zone"] for item in record["items"]} == set(zones)
    assert all(item["status"] == ("applied" if apply else "planned") for item in record["items"])
    for item in record["items"]:
        assert (root / item["target"]).exists() is not apply
    factory.assert_not_called()
    provider.get.assert_not_called()
    provider.stop.assert_not_called()
    provider.take_cleanup_failures.assert_not_called()
    session.stop.assert_not_called()


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
    "zones",
    [("bronze",), ("runs-table",), ("bronze", "metadata"), ("bronze", "runs-table", "metadata")],
)
@pytest.mark.parametrize("apply", [False, True])
def test_compute_zones_share_one_provider_and_one_session(
    tmp_path, policy_config, monkeypatch, lifecycle_spy, zones, apply
):
    command = import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    provider, factory, session = lifecycle_spy
    provider.arm()

    def collect(*args, session, zones, **kwargs):
        assert session is lifecycle_spy[2]
        return _actionable_inventory(root, args[4], zones)

    original_execute = command.execute_retention

    def execute(plan, *, session, **kwargs):
        assert session is lifecycle_spy[2]
        # File-only execution must work without a session even in a mixed selection.
        files = replace(plan, items=tuple(item for item in plan.items if item.zone == "metadata"))
        file_outcomes = original_execute(
            files,
            session=None,
            **{key: value for key, value in kwargs.items() if key != "outcomes"},
        )
        file_results = iter(file_outcomes)
        return tuple(
            next(file_results)
            if item.zone == "metadata"
            else (replace(ItemOutcome.from_planned_item(item), status="applied"))
            for item in plan.items
        )

    collector = MagicMock(side_effect=collect)
    executor = MagicMock(side_effect=execute)
    monkeypatch.setattr(command, "collect_inventory", collector)
    monkeypatch.setattr(command, "execute_retention", executor)
    result = _invoke(root, zones, apply=apply)
    assert result.exit_code == 0, result.output
    factory.assert_called_once()
    config, paths, logger = factory.call_args.args
    assert config["maintenance"] == policy_config["maintenance"]
    assert paths["metadata_dir"] == root / "data/metadata"
    assert logger is not None
    collector.assert_called_once()
    assert executor.call_count == int(apply)
    provider.get.assert_called_once_with()
    provider.stop.assert_called_once_with()
    provider.take_cleanup_failures.assert_called_once_with()
    session.stop.assert_called_once_with()
    assert (provider.start_count, provider.stop_count) == (1, 1)
    record = json.loads(result.stdout)
    assert {item["zone"] for item in record["items"]} == set(zones)
    assert all(item["status"] == ("applied" if apply else "planned") for item in record["items"])


@pytest.mark.parametrize("zones", [("bronze",), ("runs-table",), ("bronze", "metadata")])
@pytest.mark.parametrize("stage", ["collect", "execute"])
def test_compute_stops_in_finally_when_work_raises(
    tmp_path, policy_config, monkeypatch, lifecycle_spy, zones, stage
):
    command = import_module("janus.cli.maintain")
    root = _project(tmp_path, policy_config)
    provider, factory, session = lifecycle_spy
    provider.arm()
    order = []

    def collect(*args, session, zones, **kwargs):
        assert session is lifecycle_spy[2]
        order.append("collect")
        if stage == "collect":
            raise RuntimeError("scripted collect failure")
        return _actionable_inventory(root, args[4], zones)

    def execute(plan, *, session, **kwargs):
        assert session is lifecycle_spy[2]
        assert not plan.is_empty
        assert {item.zone for item in plan.items} == set(zones)
        order.append("execute")
        raise RuntimeError("scripted execute failure")

    session.stop.side_effect = lambda: order.append("stop")
    monkeypatch.setattr(command, "collect_inventory", collect)
    executor = MagicMock(side_effect=execute)
    monkeypatch.setattr(command, "execute_retention", executor)
    result = _invoke(root, zones, apply=True)

    assert result.exit_code == 1, result.output
    factory.assert_called_once()
    provider.get.assert_called_once_with()
    provider.stop.assert_called_once_with()
    provider.take_cleanup_failures.assert_called_once_with()
    session.stop.assert_called_once_with()
    assert (provider.start_count, provider.stop_count) == (1, 1)
    assert order == (["collect", "stop"] if stage == "collect" else ["collect", "execute", "stop"])
    assert executor.call_count == int(stage == "execute")
    record = json.loads(result.stdout)
    if stage == "collect":
        failures = record["failures"]
    else:
        failures = record["items"]
        assert all(item["status"] == "failed" for item in failures)
    assert {failure["failure_message"] for failure in failures} == {f"scripted {stage} failure"}
    (path,) = (root / "data/metadata/maintenance").glob("*.json")
    assert json.loads(path.read_text()) == record


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("zone", ["bronze", "runs-table"])
def test_cleanup_failure_is_recorded_and_drives_exit_one(
    tmp_path, policy_config, monkeypatch, apply, zone
):
    root = _project(tmp_path, policy_config)
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    session.stop.side_effect = RuntimeError("scripted teardown failure token=planted-secret")
    monkeypatch.setattr(SparkSessionProvider, "_build_session", lambda provider: session)
    argv = ["maintain", "--project-root", str(root), "--zone", zone, "--format", "json"]
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
