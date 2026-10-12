import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from tests.support.parallel_baseline import (
    CAPTURE_ROOT,
    CASES,
    EXPECTED_EXITS,
    PIPELINE_ID,
    STARTED_AT,
    metadata_digests,
    write_capture_project,
)
from tests.support.parallel_execution import (
    REPO_ROOT,
    cli_process,
    download_barrier,
    normalized_terminal_contract,
    spark_project,
)
from tests.support.parallel_order24 import (
    assert_lazy_worker_events,
    lifecycle_spark_project,
    parse_log_events,
    source_events,
    spark_project_with_empty_source,
    write_lifecycle_evidence,
)

PARITY_RED = pytest.mark.xfail(
    strict=True, reason="parallel CLI execution and sequential parity are absent"
)
INTERRUPT_RED = pytest.mark.xfail(strict=True, reason="parallel CLI interrupt cleanup is absent")


def finish(process):
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=10)
        pytest.fail("parallel CLI exceeded bounded integration timeout")
    assert process.returncode in {0, 1}, stderr
    assert stdout.strip(), stderr
    return json.loads(stdout), parse_log_events(stderr)


@PARITY_RED
@pytest.mark.parametrize("case", CASES)
def test_max_parallel_one_matches_verbatim_baseline_goldens(case):
    pytest.importorskip("pyspark")
    from janus.runtime import BatchExecutor
    from janus.runtime.parallel import build_batch_executor

    assert type(build_batch_executor(max_parallel=1, logger=None)) is BatchExecutor
    root = CAPTURE_ROOT / case
    assert not root.exists(), f"golden replay requires an unused scratch root: {root}"
    root.mkdir(parents=True)
    try:
        write_capture_project(case, root)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.support.parallel_execution",
                "golden",
                case,
                "run-all",
                "--project-root",
                str(root),
                "--pipeline-run-id",
                PIPELINE_ID,
                "--started-at",
                STARTED_AT.isoformat(),
                "--max-parallel",
                "1",
            ],
            capture_output=True,
            check=False,
            timeout=60,
        )
        golden = REPO_ROOT / "tests/fixtures/orchestration/parallel_baseline" / case
        assert result.returncode == EXPECTED_EXITS[case]
        assert result.stdout == (golden / "stdout.txt").read_bytes()
        stored = root / "data/metadata/pipelines" / PIPELINE_ID / "summary.json"
        assert stored.read_bytes() == (golden / "summary.json").read_bytes()
        assert (
            metadata_digests(root / "data/metadata")
            == json.loads((golden / "capture.json").read_text())["metadata_zone"]
        )
    finally:
        shutil.rmtree(root)  # Only this test's newly created scratch fixture.


@PARITY_RED
def test_parallel_a_and_c_overlap_b_waits_for_a_and_summary_matches_sequential(tmp_path):
    pytest.importorskip("pyspark")
    from janus.runtime.parallel import ParallelBatchExecutor  # noqa: F401

    sequential_root = spark_project(tmp_path / "sequential")
    sequential, _events = finish(cli_process(sequential_root, max_parallel=1))
    root = spark_project(tmp_path / "concurrent")
    barrier_dir = tmp_path / "barrier"
    with download_barrier(barrier_dir) as (ready, release):
        process = cli_process(root, extra_env={"JANUS_TEST_BARRIER_DIR": str(barrier_dir)})
        try:
            assert ready.wait(30), "independent downloads never overlapped"
            release.set()
            summary, _events = finish(process)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    sources = {source["source_id"]: source for source in summary["sources"]}
    a, b, c = (sources[source]["attempts"][0]["timing"] for source in ("A", "B", "C"))
    assert max(a["started_at"], c["started_at"]) < min(a["ended_at"], c["ended_at"])
    assert b["started_at"] >= a["ended_at"]
    assert normalized_terminal_contract(summary) == normalized_terminal_contract(sequential)


@PARITY_RED
def test_failed_a_skips_b_without_fabricating_evidence_and_c_completes(tmp_path):
    pytest.importorskip("pyspark")
    from janus.runtime.parallel import ParallelBatchExecutor  # noqa: F401

    root = spark_project(tmp_path)
    process = cli_process(root, extra_env={"JANUS_TEST_FAIL_SOURCE": "A"})
    summary, _events = finish(process)
    assert process.returncode == 1
    sources = {source["source_id"]: source for source in summary["sources"]}
    assert sources["A"]["status"] == "failed"
    assert sources["B"]["status"] == "skipped" and sources["B"]["attempts"] == []
    assert sources["B"]["skip"]["root_failed_source_ids"] == ["A"]
    assert sources["C"]["status"] == "succeeded"
    assert not list((root / "data/metadata/B/runs").glob("*.json"))


@pytest.mark.spark
def test_worker_downloads_precede_spark_and_application_ids_are_distinct(tmp_path):
    pytest.importorskip("pyspark")
    from janus.runtime.parallel import ParallelBatchExecutor  # noqa: F401

    process = cli_process(
        lifecycle_spark_project(tmp_path), extra_env={"JANUS_TEST_CAPTURE_LIFECYCLE": "1"}
    )
    summary, events = finish(process)
    assert process.returncode == 0, summary
    assert all(source["status"] == "succeeded" for source in summary["sources"])
    workers = [event for event in events if event["event"] == "parallel_worker_finished"]
    assert len(workers) == 3
    assert len({event["worker_pid"] for event in workers}) == 3
    assert all(event["spark_started"] for event in workers)
    application_ids = [event["spark_application_id"] for event in workers]
    assert all(application_ids) and len(set(application_ids)) == 3
    for source in ("A", "C"):
        assert_lazy_worker_events(events, source)
    assert_lazy_worker_events(events, "B", scoped_lookup=True)
    spark_events = [event for event in events if event["event"].startswith("spark_session_")]
    assert spark_events
    assert all(
        event.get("worker_pid") is not None and event["worker_pid"] != process.pid
        for event in spark_events
    )
    assert {event["worker_pid"] for event in spark_events} == {
        event["worker_pid"] for event in workers
    }
    write_lifecycle_evidence(tmp_path / "lifecycle-evidence.json", process, summary, events)


@pytest.mark.spark
def test_empty_handoff_worker_stays_jvm_free(tmp_path):
    pytest.importorskip("pyspark")
    from janus.runtime.parallel import ParallelBatchExecutor  # noqa: F401

    process = cli_process(
        spark_project_with_empty_source(tmp_path),
        extra_env={"JANUS_TEST_EMPTY_SOURCE": "D", "JANUS_TEST_CAPTURE_LIFECYCLE": "1"},
    )
    summary, events = finish(process)
    assert process.returncode == 0, summary
    assert all(source["status"] == "succeeded" for source in summary["sources"])
    d = next(source for source in summary["sources"] if source["source_id"] == "D")
    assert d["attempts"][0]["status"] == "succeeded"
    workers = [event for event in events if event["event"] == "parallel_worker_finished"]
    assert len(workers) == 4
    d_events = source_events(events, "D")
    finished = next(event for event in d_events if event["event"] == "parallel_worker_finished")
    assert finished["spark_started"] is False and finished["spark_application_id"] is None
    envelope = next(
        event["envelope"] for event in d_events if event["event"] == "fixture_worker_envelope"
    )
    assert envelope["worker"] == {
        "pid": finished["worker_pid"],
        "spark_started": False,
        "spark_application_id": None,
    }
    assert envelope["attempt"]["status"] == "succeeded"
    handoff = next(
        event for event in d_events if event["event"] == "normalization_handoff_prepared"
    )
    assert handoff["is_empty"] is True
    # The established materializer emits this diagnostic without acquiring Spark.
    session_events = [
        event["event"] for event in d_events if event["event"].startswith("spark_session_")
    ]
    assert session_events == ["spark_session_skipped"]
    sample = next(event for event in d_events if event["event"] == "fixture_worker_process_sample")
    assert sample["cli_pid"] == process.pid and sample["worker_pid"] == finished["worker_pid"]
    assert not any(child["command"] == "java" for child in sample["children"])
    live_sample = next(
        event for event in d_events if event["event"] == "fixture_empty_worker_process_sample"
    )
    assert live_sample["worker_alive"] is True
    assert live_sample["worker_pid"] == finished["worker_pid"]
    assert not any(child["command"] == "java" for child in live_sample["children"])
    # Ordinary producers still use Spark in the same batch as empty D.
    assert all(event["spark_started"] for event in workers if event["source_id"] != "D")
    write_lifecycle_evidence(tmp_path / "lifecycle-evidence.json", process, summary, events)


@INTERRUPT_RED
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_terminates_workers_without_finalized_summary(tmp_path, sig):
    pytest.importorskip("pyspark")
    from janus.runtime.parallel import ParallelBatchExecutor  # noqa: F401

    root = spark_project(tmp_path)
    barrier_dir = tmp_path / "barrier"
    with download_barrier(barrier_dir) as (ready, _release):
        process = cli_process(root, extra_env={"JANUS_TEST_BARRIER_DIR": str(barrier_dir)})
        try:
            assert ready.wait(30)
            started = time.monotonic()
            process.send_signal(sig)
            stdout, stderr = process.communicate(timeout=15)
            assert time.monotonic() - started < 15
            assert process.returncode == 130 and stdout == ""
            assert not list((root / "data/metadata/pipelines").rglob("summary.json"))
            events = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
            workers = [
                event["worker_pid"]
                for event in events
                if event["event"] == "parallel_worker_started"
            ]
            assert len(workers) == 2
            for pid in workers:
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
