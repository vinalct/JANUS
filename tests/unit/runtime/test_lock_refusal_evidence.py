"""AC-1: refusing execution must create neither source evidence nor stdout."""

import json
import os
import subprocess
import sys

import pytest

from janus.runtime import BatchExecutor, execute_source_attempt
from tests.support.parallel_execution import (
    REPO_ROOT,
    file_snapshot,
    held_lock,
    lock_at,
    planned_project,
)


@pytest.mark.xfail(strict=True, reason="single-source lock refusal is absent")
@pytest.mark.parametrize("flag", ["--execute", "--ingest-raw-to-bronze"])
def test_single_source_refusal_exits_two_and_writes_nothing(tmp_path, flag):
    from janus.runtime.locking import source_state_lock_path
    from janus.utils.storage import StorageLayout

    root, _registry, batch, config, _paths = planned_project(tmp_path)
    plan = batch.source("A").require_planned_run().plan
    path = source_state_lock_path(plan, StorageLayout.from_environment_config(config, root))
    with held_lock(path) as (_process, _release, record):
        before = file_snapshot(path.parent)
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
