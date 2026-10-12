"""AC-1: refusing execution must create neither source evidence nor stdout."""

import json
import os
import subprocess
import sys

import pytest

from janus.lineage import MetadataZonePaths, RunObserver
from janus.runtime import (
    BatchExecutor,
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
    lock_at,
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


@pytest.mark.xfail(strict=True, reason="attempts do not classify lock refusal")
def test_shared_attempt_seam_classifies_lock_refusal(tmp_path):
    from janus.runtime.locking import SourceLockedError

    root, _registry, plan, config, paths = planned_project(tmp_path)
    holder = lock_at(root / "data/metadata/A/.lock").holder

    class Refused:
        def execute(self, *args):
            raise SourceLockedError(root / "data/metadata/A/.lock", holder, wanted=holder)

    attempt = execute_source_attempt(plan.source("A"), config, paths, source_execution=Refused())
    assert attempt.status == "failed"
    assert attempt.failure.phase == "lock"
    assert attempt.failure.error_type == "SourceLockedError"
    assert dict(attempt.evidence) == {}


@pytest.mark.xfail(strict=True, reason="batch lock evidence and skip causes absent")
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
        lambda logger: BatchExecutor(source_execution=LockedExecution("chain")),
    )
    with held_lock(root / "data/metadata/A/.lock"):
        code = main(["run-all", "--project-root", str(root), "--pipeline-run-id", "locked"])
    sources = {
        source["source_id"]: source for source in json.loads(capsys.readouterr().out)["sources"]
    }
    assert code == 1
    assert sources["A"]["failure"]["phase"] == "lock"
    assert sources["A"]["attempts"][0]["evidence"] == {}
    assert sources["B"]["status"] == "skipped"
    assert sources["B"]["skip"] == {
        "reason_code": "upstream_failed",
        "direct_blocking_upstream_ids": ["A"],
        "root_failed_source_ids": ["A"],
    }
    assert sources["C"]["status"] == "succeeded"
    assert not list((root / "data/metadata/A/runs").glob("*.json"))
    assert plan.source_ids == ("A", "B", "C")
