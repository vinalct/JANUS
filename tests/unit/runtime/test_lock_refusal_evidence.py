"""AC-1: refusing execution must create neither source evidence nor stdout."""

import json
import os
import subprocess
import sys

import pytest

from janus.lineage import MetadataZonePaths, RunObserver
from janus.runtime import (
    LOCK_FAILURE_PHASE,
    BatchExecutor,
    SourceExecutionService,
    SourceExecutor,
    SparkSessionProvider,
    execute_source_attempt,
)
from janus.runtime.locking import SourceLockedError, source_state_lock_path
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.utils.storage import StorageLayout
from tests.support.orchestration_capture import capture_case
from tests.support.parallel_execution import (
    REPO_ROOT,
    file_snapshot,
    held_lock,
    planned_project,
    probe_lock,
)


@pytest.mark.parametrize("flag", ["--execute", "--ingest-raw-to-bronze"])
def test_single_source_refusal_exits_two_and_writes_nothing(tmp_path, flag):
    from janus.runtime.locking import source_state_lock_path
    from janus.utils.storage import StorageLayout

    root, _registry, batch, config, _paths = planned_project(tmp_path)
    plan = batch.source("A").require_planned_run().plan
    path = source_state_lock_path(plan, StorageLayout.from_environment_config(config, root))
    with held_lock(path) as (_process, _release, record):
        before = file_snapshot(path.parent)
        replay_args = (
            ["--bronze-table", "bronze.replayed"] if flag == "--ingest-raw-to-bronze" else []
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "janus.main",
                "--project-root",
                str(root),
                "--source-id",
                "A",
                flag,
                *replay_args,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        assert result.returncode == 2 and result.stdout == ""
        for key in ("pid", "hostname", "run_id", "acquired_at"):
            assert str(record[key]) in result.stderr
        assert file_snapshot(path.parent) == before


@pytest.mark.parametrize("replay", [False, True], ids=["execute", "replay"])
def test_refusal_precedes_observer_wiring_and_spark(tmp_path, monkeypatch, replay):
    root, _registry, batch, config, paths = planned_project(tmp_path)
    planned_run = batch.source("A").require_planned_run()
    path = source_state_lock_path(
        planned_run.plan, StorageLayout.from_environment_config(config, root)
    )

    def reject_side_effect(*args, **kwargs):
        pytest.fail("a refused run must not wire its observer or start Spark")

    monkeypatch.setattr("janus.runtime.executor.wire_run_event_emitter", reject_side_effect)
    monkeypatch.setattr("janus.scripts.raw_to_bronze.wire_run_event_emitter", reject_side_effect)
    provider = SparkSessionProvider(config, paths, session_factory=reject_side_effect)
    with held_lock(path):
        before = file_snapshot(root / "data")
        with pytest.raises(SourceLockedError):
            if replay:
                RawToBronzeLoader().ingest(
                    planned_run, provider, config, bronze_table="bronze.replayed"
                )
            else:
                SourceExecutor().execute(planned_run, provider, config)
        assert provider.was_started is False
        assert file_snapshot(root / "data") == before


@pytest.mark.parametrize("case", ["independent", "failed", "empty", "replay"])
def test_lock_spans_observation_and_terminal_persistence(tmp_path, monkeypatch, case):
    checked_paths = []

    def guarded(method):
        def call(observer, plan, *args, **kwargs):
            path = MetadataZonePaths.from_plan(plan).base_dir / ".lock"
            before = probe_lock(path)
            assert before["acquired"] is False
            holder = before["holder"]
            assert holder["run_id"] == plan.run_context.run_id
            assert holder["pipeline_run_id"] == plan.run_context.attributes_as_dict().get(
                "pipeline_run_id"
            )
            assert holder["command"] == (
                "run --ingest-raw-to-bronze" if case == "replay" else "run --execute"
            )
            result = method(observer, plan, *args, **kwargs)
            assert probe_lock(path)["acquired"] is False
            checked_paths.append(path)
            return result

        return call

    for name in ("start_run", "record_success", "record_failure"):
        monkeypatch.setattr(RunObserver, name, guarded(getattr(RunObserver, name)))
    capture_case(tmp_path / case, case)
    assert len(checked_paths) == 2
    path = checked_paths[0]
    assert path.read_bytes() == b""
    assert probe_lock(path)["acquired"] is True


def test_shared_attempt_seam_classifies_lock_refusal(tmp_path, monkeypatch):
    root, _registry, plan, config, paths = planned_project(tmp_path)
    provider = SparkSessionProvider(
        config, paths, session_factory=lambda: pytest.fail("lock refusal started Spark")
    )
    service = SourceExecutionService(provider_factory=lambda *args: provider)

    def reject_observer(*args, **kwargs):
        pytest.fail("lock refusal wired an observer")

    monkeypatch.setattr("janus.runtime.executor.wire_run_event_emitter", reject_observer)
    with held_lock(root / "data/metadata/A/.lock") as (_process, _release, holder):
        before = file_snapshot(root / "data")
        attempt = execute_source_attempt(plan.source("A"), config, paths, source_execution=service)
        assert file_snapshot(root / "data") == before
    assert attempt.status == "failed"
    assert attempt.failure.phase == LOCK_FAILURE_PHASE
    assert attempt.failure.error_type == "SourceLockedError"
    for key in ("pid", "hostname", "run_id", "acquired_at"):
        assert str(holder[key]) in attempt.failure.reason
    assert dict(attempt.evidence) == {}
    assert provider.was_started is False


def test_shared_attempt_seam_classifies_unsupported_locking(tmp_path, monkeypatch):
    root, _registry, plan, config, paths = planned_project(tmp_path)
    provider = SparkSessionProvider(
        config, paths, session_factory=lambda: pytest.fail("unsupported locking started Spark")
    )
    service = SourceExecutionService(provider_factory=lambda *args: provider)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    before = file_snapshot(root / "data")

    attempt = execute_source_attempt(plan.source("A"), config, paths, source_execution=service)

    assert attempt.status == "failed"
    assert attempt.failure.phase == LOCK_FAILURE_PHASE
    assert attempt.failure.error_type == "LockUnsupportedError"
    assert "requires POSIX fcntl" in attempt.failure.reason
    assert dict(attempt.evidence) == {}
    assert provider.was_started is False
    assert file_snapshot(root / "data") == before


def test_locked_batch_source_skips_dependents_and_peer_succeeds(tmp_path, monkeypatch, capsys):
    from janus.cli import run_all
    from janus.main import main
    from janus.runtime.locking import hold_source_state_lock
    from janus.utils.storage import StorageLayout
    from tests.support.parallel_baseline import CannedSourceExecution

    root, _registry, plan, config, _paths = planned_project(tmp_path)

    class LockedExecution(CannedSourceExecution):
        def execute(self, planned_run, config, paths):
            with hold_source_state_lock(
                planned_run.plan,
                StorageLayout.from_environment_config(config, root),
                command="run-all",
                run_id=planned_run.plan.run_context.run_id,
                pipeline_run_id="parallel-fixture",
            ):
                return super().execute(planned_run, config, paths)

    monkeypatch.setattr(
        run_all,
        "_build_batch_executor",
        lambda logger: BatchExecutor(source_execution=LockedExecution("chain_and_peer")),
    )
    metadata = root / "data/metadata/A"
    with held_lock(metadata / ".lock") as (_process, _release, holder):
        before = file_snapshot(metadata)
        code = main(["run-all", "--project-root", str(root), "--pipeline-run-id", "locked"])
        assert file_snapshot(metadata) == before
    sources = {
        source["source_id"]: source for source in json.loads(capsys.readouterr().out)["sources"]
    }
    assert code == 1
    assert sources["A"]["status"] == "failed"
    assert sources["A"]["failure"]["phase"] == LOCK_FAILURE_PHASE
    assert sources["A"]["failure"]["error_type"] == "SourceLockedError"
    for key in ("pid", "hostname", "run_id", "acquired_at"):
        assert str(holder[key]) in sources["A"]["failure"]["reason"]
    assert sources["A"]["attempts"][0]["evidence"] == {}
    assert sources["B"]["status"] == "skipped"
    assert sources["B"]["skip"] == {
        "reason_code": "upstream_failed",
        "direct_blocking_upstream_ids": ["A"],
        "root_failed_source_ids": ["A"],
    }
    assert sources["C"]["status"] == "succeeded"
    assert set(file_snapshot(metadata)) == {".lock"}
    assert not any((metadata / name).exists() for name in ("runs", "lineage", "checkpoints"))
    assert plan.source_ids == ("A", "B", "C")
