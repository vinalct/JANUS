"""Malformed spawned results carry crash evidence and block consumers."""

import json
import multiprocessing
import os

import pytest

from janus.runtime.parallel import ParallelBatchExecutor, _WorkerHandle, run_worker
from tests.support.parallel_execution import FAKE_FACTORY, planned_project


def malformed_worker(task, connection):
    if task.source_id != "A":
        run_worker(task, connection)
        return
    messages = {
        "json": b"{",
        "utf8": b"\xff",
        "attempt": json.dumps(
            {
                "attempt": {},
                "worker": {
                    "pid": os.getpid(),
                    "spark_started": False,
                    "spark_application_id": None,
                },
            }
        ).encode(),
    }
    try:
        connection.send_bytes(messages[os.environ["JANUS_TEST_BAD_ENVELOPE"]])
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mode,error",
    [("json", "JSONDecodeError"), ("utf8", "UnicodeDecodeError"), ("attempt", "ValueError")],
)
def test_malformed_spawned_envelope_is_worker_crash_and_skips_dependent(
    tmp_path, monkeypatch, mode, error
):
    root, registry, plan, config, paths = planned_project(tmp_path)
    monkeypatch.setenv("JANUS_TEST_BAD_ENVELOPE", mode)

    def start(task):
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        child = context.Process(target=malformed_worker, args=(task, writer))
        child.start()
        writer.close()
        return _WorkerHandle(task, child, reader)

    outcome = ParallelBatchExecutor(
        max_parallel=3, process_starter=start, execution_factory=FAKE_FACTORY
    ).execute(plan, registry, config, paths)
    sources = {s.source_id: s for s in outcome.sources}
    attempt = sources["A"].attempts[0]
    assert attempt.failure.phase == "worker_crash"
    assert attempt.failure.error_type == "WorkerCrashed"
    assert error in attempt.failure.reason and "envelope refused" in attempt.failure.reason
    assert dict(attempt.evidence) == {}
    assert sources["B"].skip.root_failed_source_ids == ("A",)
    assert sources["C"].status == "succeeded"
    assert not list((root / "data/metadata/A/runs").glob("*.json"))
    (tmp_path / "crash-attempt.json").write_text(json.dumps(attempt.to_summary(), indent=2))
