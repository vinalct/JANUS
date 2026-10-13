import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import pytest

from tests.support.orchestration import normalized_terminal_contract
from tests.support.parallel import (
    assert_lazy_worker_events,
    lifecycle_spark_project,
    parse_log_events,
    source_events,
    spark_project_with_empty_source,
    write_lifecycle_evidence,
)
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
    file_snapshot,
    spark_project,
)
from tests.support.parallel_interrupts import (
    session_processes,
    wait_for_events,
    wait_for_session_exit,
)
from tests.support.parallel_order24 import (
    PAGE_DELAY_SECONDS,
    SPAWN_COST_SECONDS,
    bronze_rows,
    example_project,
    fixture_service,
    run_example,
    write_parity_evidence,
)


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


@pytest.mark.spark
def test_parallel_summary_equals_sequential_summary(tmp_path):
    with fixture_service(page_delay_seconds=PAGE_DELAY_SECONDS) as server:
        sequential_root = example_project(tmp_path / "sequential", server)
        parallel_root = example_project(tmp_path / "parallel", server)
        sequential, _, sequential_process = run_example(sequential_root, max_parallel=1)
        parallel, events, parallel_process = run_example(parallel_root, max_parallel=3)
        assert sequential_process.returncode == parallel_process.returncode == 0
        assert normalized_terminal_contract(parallel) == normalized_terminal_contract(sequential)
        sources = {source["source_id"]: source for source in parallel["sources"]}
        a, b, c = (sources[source]["attempts"][0]["timing"] for source in ("A", "B", "C"))
        assert a["started_at"] < c["ended_at"] and c["started_at"] < a["ended_at"]
        assert b["started_at"] >= a["ended_at"]
        assert parallel["pipeline"]["max_parallel"] == 3
        assert "max_parallel" not in sequential["pipeline"]
        for source_id in ("A", "C"):
            source = next(s for s in sequential["sources"] if s["source_id"] == source_id)
            timing = source["attempts"][0]["timing"]
            assert _elapsed(timing) >= PAGE_DELAY_SECONDS >= 2 * SPAWN_COST_SECONDS
            extraction = [
                event
                for event in events
                if event.get("source_id") == source_id
                and event["event"] in {"source_extraction_started", "source_extraction_finished"}
            ]
            assert len(extraction) == 2
            assert (
                _elapsed(
                    {
                        "started_at": extraction[0]["timestamp"],
                        "ended_at": extraction[1]["timestamp"],
                    }
                )
                >= PAGE_DELAY_SECONDS
            )
        sequential_rows, parallel_rows = bronze_rows(sequential_root), bronze_rows(parallel_root)
        assert sequential_rows == parallel_rows
        assert parallel_rows["A"] == [("north", "North region"), ("south", "South region")]
        assert parallel_rows["B"] == [("north-001", "north", 10), ("south-001", "south", 20)]
        assert len(parallel_rows["C"]) == 1
        write_parity_evidence(tmp_path, sequential, parallel, bronze_rows=parallel_rows)


@pytest.mark.spark
def test_failed_upstream_skips_dependent_and_peer_completes_in_parallel(tmp_path):
    with fixture_service(fail_a=True) as server:
        sequential_root = example_project(tmp_path / "sequential", server)
        parallel_root = example_project(tmp_path / "parallel", server)
        b_directory = parallel_root / "runtime/data/metadata/B"
        # A pre-existing inode must survive unchanged; existence alone is not ownership.
        b_directory.mkdir(parents=True)
        (b_directory / ".lock").write_text("previous holder record\n")
        before = file_snapshot(b_directory)
        sequential, _, sequential_process = run_example(sequential_root, max_parallel=1)
        parallel, _, parallel_process = run_example(parallel_root, max_parallel=3)
        assert sequential_process.returncode == parallel_process.returncode == 1
        sources = {source["source_id"]: source for source in parallel["sources"]}
        assert sources["A"]["status"] == "failed"
        assert sources["A"]["failure"]["phase"] == "execution"
        assert sources["B"]["status"] == "skipped"
        assert sources["B"]["skip"] == {
            "reason_code": "upstream_failed",
            "direct_blocking_upstream_ids": ["A"],
            "root_failed_source_ids": ["A"],
        }
        assert sources["B"]["attempts"] == []
        assert sources["B"]["timing"] == {
            "started_at": None,
            "ended_at": None,
            "duration_seconds": 0.0,
        }
        assert sources["C"]["status"] == "succeeded"
        assert file_snapshot(b_directory) == before
        assert file_snapshot(sequential_root / "runtime/data/metadata/B") == {}
        assert not any(request.startswith("/details") for request in server.requests)
        assert sum(request.startswith("/reference") for request in server.requests) == 2
        assert normalized_terminal_contract(parallel) == normalized_terminal_contract(sequential)
        write_parity_evidence(
            tmp_path,
            sequential,
            parallel,
            b_directory_listing=sorted(before),
            b_run_id=sources["B"]["run_id"],
            b_directory_unchanged=True,
        )


@pytest.mark.spark
def test_max_parallel_one_is_the_sequential_runner(tmp_path):
    with fixture_service() as server:
        root = example_project(tmp_path / "sequential", server)
        summary, events, process = run_example(root, max_parallel=1)
    assert process.returncode == 0
    assert all(source["status"] == "succeeded" for source in summary["sources"])
    assert not any(event["event"].startswith("parallel_worker_") for event in events)
    starts = [event for event in events if event["event"] == "source_execution_started"]
    assert len(starts) == 3
    assert {event["pid"] for event in starts} == {process.pid}
    assert "max_parallel" not in summary["pipeline"]
    assert "max_parallel" not in summary


@pytest.mark.spark
def test_shared_metadata_directory_nodes_never_overlap(tmp_path):
    with fixture_service(page_delay_seconds=PAGE_DELAY_SECONDS) as server:
        root = example_project(tmp_path / "shared", server, shared_metadata=True)
        summary, _, process = run_example(root, max_parallel=3)
    assert process.returncode == 0
    assert all(source["status"] == "succeeded" for source in summary["sources"])
    assert all(source.get("failure") is None for source in summary["sources"])
    sources = {source["source_id"]: source for source in summary["sources"]}
    a, c = (sources[source]["attempts"][0]["timing"] for source in ("A", "C"))
    assert a["ended_at"] <= c["started_at"] or c["ended_at"] <= a["started_at"]
    assert sources["B"]["attempts"][0]["timing"]["started_at"] >= a["ended_at"]


def _elapsed(timing):
    return (
        datetime.fromisoformat(timing["ended_at"]) - datetime.fromisoformat(timing["started_at"])
    ).total_seconds()


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


@pytest.mark.spark
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
            before = session_processes(process.pid)
            started = time.monotonic()
            if sig == signal.SIGINT:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
            stdout, stderr = process.communicate(timeout=15)
            elapsed = time.monotonic() - started
            assert elapsed < 15
            assert process.returncode == 130 and stdout == ""
            assert "completed sources: 0, pending sources: 3" in stderr
            assert not list((root / "data/metadata/pipelines").rglob("summary.json"))
            events = parse_log_events(stderr)
            workers = [
                event["worker_pid"]
                for event in events
                if event["event"] == "parallel_worker_started"
            ]
            assert len(workers) == 2
            for pid in workers:
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
            after = wait_for_session_exit(process.pid)
            partial = next(event for event in events if event["event"] == "parallel_batch_partial")
            assert partial["completed_source_ids"] == []
            assert partial["pending_source_ids"] == ["A", "B", "C"]
            from tests.support.parallel_execution import lock_at

            for source_id in partial["pending_source_ids"]:
                with lock_at(root / "data/metadata" / source_id / ".lock", source_id=source_id):
                    pass
            (tmp_path / "interrupt-evidence.json").write_text(
                json.dumps(
                    {
                        "signal": sig.name,
                        "target": "process_group" if sig == signal.SIGINT else "parent",
                        "exit_code": process.returncode,
                        "elapsed_seconds": elapsed,
                        "grace_seconds": 2,
                        "before": before,
                        "after": after,
                        "partial": partial,
                    },
                    indent=2,
                )
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)


@pytest.mark.spark
def test_sigint_terminates_workers_and_exits_130(tmp_path):
    from uuid import uuid4

    from tests.support.parallel_execution import lock_at

    with fixture_service(page_delay_seconds=5) as server:
        root = example_project(tmp_path / "interrupted-example", server)
        pipeline_id = f"task16-{uuid4().hex}"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.support.parallel_order24",
                "run-all",
                "--project-root",
                str(root),
                "--environment",
                "example",
                "--pipeline-run-id",
                pipeline_id,
                "--max-parallel",
                "3",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))),
                "JANUS_SPARK_DRIVER_MEMORY": "1g",
            },
        )
        try:
            prefix = wait_for_events(process, "source_execution_started", count=2)
            before = session_processes(process.pid)
            started = time.monotonic()
            os.killpg(process.pid, signal.SIGINT)
            stdout, stderr = process.communicate(timeout=40)
            elapsed = time.monotonic() - started
            stderr = prefix + stderr
            events = parse_log_events(stderr)
            assert elapsed < 40 and process.returncode == 130 and stdout == "", stderr
            assert sum(e["event"] == "parallel_worker_dispatched" for e in events) == 2
            assert not (
                root / "runtime/data/metadata/pipelines" / pipeline_id / "summary.json"
            ).exists()
            assert "completed sources: 0, pending sources: 3" in stderr
            partial = next(e for e in events if e["event"] == "parallel_batch_partial")
            assert partial["pending_source_ids"] == ["A", "B", "C"]
            after = wait_for_session_exit(process.pid)
            for source_id in partial["pending_source_ids"]:
                with lock_at(
                    root / "runtime/data/metadata" / source_id / ".lock", source_id=source_id
                ):
                    pass
            (tmp_path / "interrupt-evidence.json").write_text(
                json.dumps(
                    {
                        "signal": "SIGINT",
                        "target": "process_group",
                        "exit_code": process.returncode,
                        "elapsed_seconds": elapsed,
                        "grace_seconds": 30,
                        "before": before,
                        "after": after,
                        "partial": partial,
                    },
                    indent=2,
                )
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)


@pytest.mark.spark
def test_sigterm_to_parent_reaps_workers_with_live_spark(tmp_path):
    from tests.support.parallel_execution import lock_at

    root = spark_project(tmp_path)
    process = cli_process(root)
    try:
        prefix = wait_for_events(process, "spark_session_started", timeout=60)
        before = session_processes(process.pid)
        assert any(row["command"] == "java" for row in before)
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=12)
        elapsed = time.monotonic() - started
        stderr = prefix + stderr
        assert elapsed < 12 and process.returncode == 130 and stdout == "", stderr
        assert not list((root / "data/metadata/pipelines").rglob("summary.json"))
        partial = next(
            e for e in parse_log_events(stderr) if e["event"] == "parallel_batch_partial"
        )
        assert {"A", "C"} <= set(partial["pending_source_ids"])
        after = wait_for_session_exit(process.pid)
        for source_id in partial["pending_source_ids"]:
            with lock_at(root / "data/metadata" / source_id / ".lock", source_id=source_id):
                pass
        (tmp_path / "interrupt-evidence.json").write_text(
            json.dumps(
                {
                    "signal": "SIGTERM",
                    "target": "parent",
                    "exit_code": process.returncode,
                    "elapsed_seconds": elapsed,
                    "grace_seconds": 2,
                    "before": before,
                    "after": after,
                    "partial": partial,
                },
                indent=2,
            )
        )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
