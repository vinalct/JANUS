from __future__ import annotations

import argparse
import contextlib
import io
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from janus.cli import maintain
from janus.maintenance.inventory import BronzeTableInventory, MaintenanceInventory, SnapshotEntry
from janus.maintenance.locking import (
    LOCKED_SOURCE_REASON,
    MaintenanceLock,
    NullMaintenanceLock,
)
from tests.support.operator_cli import CliResult, arm_spark_tripwire
from tests.support.semantics_fixtures import CLEAN, install_profile, materialize

SOURCE_IDS = ("source_a", "source_b", "source_c")
WARNING = (
    "warning: no source lock is held — do not run maintain while a run of the same source "
    "is in flight"
)


class FakeLock:
    name = "fake-source-lock"

    def __init__(self, refused=()):
        self.refused = frozenset(refused)
        self.acquired = []
        self.released = []
        self.held = set()

    def acquire(self, source_id):
        self.acquired.append(source_id)
        if source_id in self.refused:
            return False
        assert source_id not in self.held
        self.held.add(source_id)
        return True

    def release(self, source_id):
        assert source_id in self.held
        self.held.remove(source_id)
        self.released.append(source_id)


@pytest.fixture
def project(tmp_path, policy_config, now, monkeypatch):
    root = materialize(CLEAN, tmp_path / "project")
    source_dir = root / "conf/sources"
    source = yaml.safe_load((source_dir / "01_producer.yaml").read_text())
    for path in source_dir.glob("*.yaml"):
        path.unlink()
    for source_id in SOURCE_IDS:
        config = deepcopy(source)
        config.update(source_id=source_id, enabled=source_id != "source_c")
        for zone in ("raw", "bronze", "metadata"):
            config["outputs"][zone]["path"] = f"data/{zone}/{source_id}"
        config["outputs"]["bronze"]["table_name"] = source_id
        (source_dir / f"{source_id}.yaml").write_text(yaml.safe_dump(config))
        for run_id, days in (("old", 200), ("newest", 1)):
            timestamp = now - timedelta(days=days)
            _write(
                root / f"data/metadata/{source_id}/runs/{run_id}.json",
                json.dumps(
                    {
                        "source_id": source_id,
                        "run_id": run_id,
                        "started_at": timestamp.isoformat(),
                        "status": "succeeded",
                    }
                ),
            )
            _write(
                root
                / f"data/raw/{source_id}/runs/ingestion_date={timestamp.date()}"
                / f"run_id={run_id}/pages/data.json",
                '{"data": true}\n',
            )
    profile_path = install_profile(root, "local")
    profile = yaml.safe_load(profile_path.read_text())
    policy_config["maintenance"]["bronze"]["retain_last"] = 1
    policy_config["maintenance"]["metadata"]["keep_last_runs"] = 1
    policy_config["maintenance"]["raw"].update(enabled=True, keep_last_runs=1)
    profile.update(policy_config)
    profile_path.write_text(yaml.safe_dump(profile))

    class Clock:
        @staticmethod
        def now(*, tz):
            return now

    monkeypatch.setattr(maintain, "datetime", Clock)
    return root


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _invoke(project, *options, lock=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--environment", default="local")
    maintain.configure(parser)
    args = parser.parse_args(options)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = (
            maintain.maintain_command(args)
            if lock is None
            else maintain.maintain_command(args, lock=lock)
        )
    return CliResult(code, stdout.getvalue(), stderr.getvalue())


def _record(project):
    (path,) = (project / "data/metadata/maintenance").glob("*.json")
    return json.loads(path.read_text())


def test_null_lock_acquires_every_source_and_conforms_to_protocol():
    lock = NullMaintenanceLock()
    assert isinstance(lock, MaintenanceLock)
    assert lock.name == "none"
    for source_id in SOURCE_IDS:
        assert lock.acquire(source_id) is True
        assert lock.release(source_id) is None


def test_duck_typed_lock_needs_no_adapter():
    class FutureSourceStateLock:
        name = "source-state"

        def acquire(self, source_id):
            return True

        def release(self, source_id):
            pass

    assert isinstance(FutureSourceStateLock(), MaintenanceLock)


@pytest.mark.parametrize("zones", [("metadata",), ("raw",), ("metadata", "raw")])
@pytest.mark.parametrize("apply", [False, True])
def test_locked_source_is_unread_untouched_and_skipped_while_peers_proceed(
    project, monkeypatch, zones, apply
):
    arm_spark_tripwire(monkeypatch)
    lock = FakeLock(refused=("source_b",))
    collect = maintain.collect_inventory
    execute = maintain.execute_retention
    reads = []

    def locked_collect(*args, **kwargs):
        assert kwargs["source_ids"] == frozenset({"source_a", "source_c"})
        assert lock.held == {"source_a", "source_c"}
        reads.append(kwargs["source_ids"])
        return collect(*args, **kwargs)

    def locked_execute(*args, **kwargs):
        assert lock.held == {"source_a", "source_c"}
        return execute(*args, **kwargs)

    monkeypatch.setattr(maintain, "collect_inventory", locked_collect)
    monkeypatch.setattr(maintain, "execute_retention", locked_execute)
    before = {
        path: path.read_bytes()
        for zone in ("metadata", "raw")
        for path in (project / f"data/{zone}/source_b").rglob("*")
        if path.is_file()
    }
    options = [option for zone in zones for option in ("--zone", zone)]
    result = _invoke(
        project, *options, "--format", "json", *(("--apply",) if apply else ()), lock=lock
    )

    assert result.exit_code == 0, result.output
    record = _record(project)
    assert json.loads(result.stdout) == record
    assert record["lock"] == lock.name
    assert WARNING not in result.output
    assert reads and lock.acquired == list(SOURCE_IDS)
    assert set(lock.released) == {"source_a", "source_c"} and not lock.held
    assert all(path.read_bytes() == data for path, data in before.items())
    for zone in zones:
        items = [item for item in record["items"] if item["zone"] == zone]
        skipped = [item for item in items if item["status"] == "skipped"]
        assert len(skipped) == 1
        assert skipped[0]["target"] == "source_b"
        assert skipped[0]["detail"] == {
            "source_id": "source_b",
            "skipped_reason": LOCKED_SOURCE_REASON,
        }
        peers = [item for item in items if item["status"] != "skipped"]
        assert len(peers) == 2
        assert all(item["status"] == ("applied" if apply else "planned") for item in peers)
        assert all(Path(item["target"]).exists() is not apply for item in peers)


def test_all_sources_locked_never_collect_or_execute(project, monkeypatch):
    lock = FakeLock(refused=SOURCE_IDS)

    def forbidden(*args, **kwargs):
        pytest.fail("a refused source was inspected or executed")

    monkeypatch.setattr(maintain, "collect_inventory", forbidden)
    monkeypatch.setattr(maintain, "execute_retention", forbidden)
    result = _invoke(project, "--zone", "metadata", "--zone", "raw", "--apply", lock=lock)
    assert result.exit_code == 0, result.output
    record = _record(project)
    assert len(record["items"]) == 6
    assert all(item["status"] == "skipped" for item in record["items"])
    assert record["failures"] == [] and lock.released == []


@pytest.mark.parametrize("failure_stage", ["collect", "execute", "plan"])
def test_locks_release_on_failures(project, monkeypatch, failure_stage):
    lock = FakeLock()

    def fail(*args, **kwargs):
        assert lock.held == set(SOURCE_IDS)
        if failure_stage == "plan":
            raise ValueError("scripted refusal")
        raise RuntimeError("scripted failure")

    target = {
        "collect": "collect_inventory",
        "execute": "execute_retention",
        "plan": "plan_retention",
    }
    monkeypatch.setattr(maintain, target[failure_stage], fail)
    result = _invoke(project, "--zone", "metadata", "--apply", "--format", "json", lock=lock)
    assert result.exit_code == (2 if failure_stage == "plan" else 1), result.output
    assert set(lock.released) == set(SOURCE_IDS) and not lock.held
    if failure_stage != "plan":
        assert _record(project)["lock"] == lock.name


@pytest.mark.parametrize("stage", ["collect", "execute"])
def test_locks_release_and_partial_evidence_survives_interruption(project, monkeypatch, stage):
    lock = FakeLock(refused=("source_b",))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("operator interruption")

    target = "collect_inventory" if stage == "collect" else "execute_retention"
    monkeypatch.setattr(maintain, target, interrupt)
    with pytest.raises(KeyboardInterrupt, match="operator interruption"):
        _invoke(project, "--zone", "metadata", "--apply", lock=lock)
    record = _record(project)
    assert record["lock"] == lock.name
    expected = ["skipped"] if stage == "collect" else ["planned", "planned", "skipped"]
    assert [item["status"] for item in record["items"]] == expected
    assert set(lock.released) == {"source_a", "source_c"} and not lock.held


@pytest.mark.parametrize("zone", ["bronze", "runs-table", "lineage"])
@pytest.mark.parametrize("format", ["text", "json"])
def test_unscoped_zones_never_consult_lock_or_print_warning(project, monkeypatch, zone, format):
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    monkeypatch.setattr(maintain._Compute, "get_session", lambda self: session)
    lock = FakeLock(refused=SOURCE_IDS)
    result = _invoke(project, "--zone", zone, "--apply", "--format", format, lock=lock)
    assert result.exit_code == 0, result.output
    assert lock.acquired == lock.released == []
    assert _record(project)["lock"] == lock.name
    assert WARNING not in result.output


@pytest.mark.parametrize("zone", ["metadata", "raw", "lineage", "bronze", "runs-table"])
@pytest.mark.parametrize("format", ["text", "json"])
@pytest.mark.parametrize("apply", [False, True])
def test_null_lock_is_recorded_and_warning_tracks_selected_zones(
    project, monkeypatch, zone, format, apply
):
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    monkeypatch.setattr(maintain._Compute, "get_session", lambda self: session)
    result = _invoke(project, "--zone", zone, "--format", format, *(("--apply",) if apply else ()))
    assert result.exit_code == 0, result.output
    assert _record(project)["lock"] == "none"
    stream = result.stderr if format == "json" else result.stdout
    assert (WARNING in stream) is (zone in {"metadata", "raw"})


def test_source_filter_only_acquires_selected_source(project):
    lock = FakeLock()
    result = _invoke(project, "--zone", "metadata", "--source-id", "source_c", lock=lock)
    assert result.exit_code == 0, result.output
    assert lock.acquired == lock.released == ["source_c"]


def test_injected_lock_does_not_leak_into_next_invocation(project):
    result = _invoke(project, "--zone", "metadata", lock=FakeLock(refused=SOURCE_IDS))
    assert result.exit_code == 0, result.output
    result = _invoke(project, "--zone", "metadata", "--format", "json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["lock"] == "none"
    assert all(item["status"] == "planned" for item in json.loads(result.stdout)["items"])
    assert WARNING in result.stderr


def test_locked_source_preserves_shared_pipeline_summary(project, now):
    path = project / "data/metadata/pipelines/old-batch/summary.json"
    _write(path, json.dumps({"pipeline": {"started_at": (now - timedelta(days=200)).isoformat()}}))
    result = _invoke(project, "--zone", "metadata", "--apply", lock=FakeLock(refused=("source_b",)))
    assert result.exit_code == 0, result.output
    assert path.exists()
    assert {"zone": "metadata", "target": str(path), "reasons": ["source_filter"]} in _record(
        project
    )["protected"]


def test_contention_does_not_filter_bronze_sources_or_shared_zones(project, monkeypatch, now):
    lock = FakeLock(refused=("source_b",))
    collect = maintain.collect_inventory
    session = object()
    monkeypatch.setattr(maintain._Compute, "get_session", lambda self: session)
    calls = []

    def mixed_collect(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["session"] is session
        if "metadata" in kwargs["zones"]:
            assert kwargs["source_ids"] == frozenset({"source_a", "source_c"})
            return collect(*args, **kwargs)
        assert kwargs["source_ids"] is None
        assert kwargs["zones"] == frozenset({"bronze", "lineage", "runs-table"})
        return MaintenanceInventory(
            bronze=(
                BronzeTableInventory(
                    "semantics.source_b",
                    ("source_b",),
                    (
                        SnapshotEntry(1, now - timedelta(days=200), False),
                        SnapshotEntry(2, now, True),
                    ),
                ),
            )
        )

    monkeypatch.setattr(maintain, "collect_inventory", mixed_collect)
    result = _invoke(project, "--format", "json", lock=lock)
    assert result.exit_code == 0, result.output
    record = _record(project)
    bronze = [item for item in record["items"] if item["zone"] == "bronze"]
    assert len(bronze) == 1
    assert bronze[0]["target"] == "semantics.source_b" and bronze[0]["status"] == "planned"
    assert len(calls) == 2 and not lock.held


def test_documentation_states_no_overlap_requirement_and_cron_caveat():
    docs = Path(__file__).resolve().parents[3] / "docs/maintenance.md"
    text = docs.read_text()
    assert "Do not run `janus maintain` while a run of the same source is in flight" in text
    assert "Cron" in text and "serialise" in text
    assert 'lock: "none"' in text
    assert WARNING in text
