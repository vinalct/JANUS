"""Parent signals, late results, and sequential partial-evidence parity."""

import json
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import janus.runtime.parallel as parallel
from janus.orchestration.timing import PipelineClock
from janus.runtime.batch import BatchExecutionInterrupted, BatchExecutor, execute_source_attempt
from tests.support.parallel_baseline import STARTED_AT, CannedSourceExecution
from tests.support.parallel_execution import (
    FAKE_FACTORY,
    call_worker,
    planned_project,
    worker_request,
)
from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore


def fake_handle(task, *, envelope=None):
    handle = Mock(request=task, pid=123, exitcode=0, transport_error=None)
    handle.envelope = envelope
    handle.result_received = envelope is not None
    handle.is_alive.return_value = True
    return handle


@pytest.fixture
def keyboard_interrupt_sigint():
    # Earlier Spark tests can leave PySpark's handler installed after stopping its
    # context. The parallel parent creates no Spark context and starts with Python's
    # ordinary SIGINT handler; preserve the surrounding test process's handler.
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("linger", [False, True])
def test_grace_records_late_envelope_and_leaves_unanswered_sources_pending(
    tmp_path, monkeypatch, sig, linger, keyboard_interrupt_sigint
):
    root, registry, plan, config, paths = planned_project(tmp_path)
    envelope = call_worker(worker_request(plan, config, paths))
    handles = {}
    ticks = iter((0.0, 0.0, 2.0))
    events = []
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def start(task):
        handle = fake_handle(task)
        handle.terminate.side_effect = lambda: events.append(("terminate", task.source_id))
        handle.kill.side_effect = lambda: events.append(("kill", task.source_id))
        handles[task.source_id] = handle
        return handle

    def wait(handles_to_wait, **kwargs):
        if "timeout" not in kwargs:
            os.kill(os.getpid(), sig)
            pytest.fail("parent signal did not interrupt dispatch")
        assert events[:2] == [("terminate", "A"), ("terminate", "C")]
        assert 0 < kwargs["timeout"] <= 1.5
        handles["A"].envelope = envelope
        handles["A"].result_received = True
        handles["A"].is_alive.return_value = linger
        handles["C"].result_received = True
        handles["C"].is_alive.return_value = False
        return handles_to_wait

    monkeypatch.setattr(parallel, "_wait_any", wait)
    monkeypatch.setattr(parallel, "monotonic", lambda: next(ticks, 2.0))
    logger = Mock(context={})
    store = MemorySummaryStore()
    store.persist = Mock(side_effect=AssertionError("interrupted summary was persisted"))
    runner = parallel.ParallelBatchExecutor(
        max_parallel=3, process_starter=start, logger=logger, interrupt_grace_seconds=1.5
    )
    with pytest.raises(BatchExecutionInterrupted) as raised:
        runner.execute(plan, registry, config, paths, summary_store=store)
    partial = raised.value.partial
    assert [s.source_id for s in partial.source_outcomes] == ["A"]
    assert partial.source_outcomes[0].attempts[0].to_summary() == envelope["attempt"]
    assert partial.pending_source_ids == ("B", "C")
    assert list(handles) == ["A", "C"]  # Successful A never dispatches B during shutdown.
    if linger:
        handles["A"].kill.assert_called_once()
    else:
        handles["A"].kill.assert_not_called()
    handles["C"].kill.assert_not_called()
    for handle in handles.values():
        handle.close.assert_called_once()
    assert {s: signal.getsignal(s) for s in previous} == previous
    store.persist.assert_not_called()
    assert not list((root / "data/metadata/pipelines").rglob("summary.json"))
    (tmp_path / "partial.json").write_text(
        json.dumps(
            {
                "completed": [s.to_summary() for s in partial.source_outcomes],
                "pending": partial.pending_source_ids,
                "handlers_restored": True,
            },
            indent=2,
        )
    )


def test_second_interrupt_skips_grace_and_kills_every_survivor(tmp_path, monkeypatch):
    _root, registry, plan, config, paths = planned_project(tmp_path)
    handles = []

    def start(task):
        handle = fake_handle(task)
        handles.append(handle)
        return handle

    wait = Mock(side_effect=[KeyboardInterrupt(), KeyboardInterrupt()])
    logger = Mock(context={})
    monkeypatch.setattr(parallel, "_wait_any", wait)
    monkeypatch.setattr(parallel, "monotonic", lambda: 0.0)
    with pytest.raises(BatchExecutionInterrupted) as raised:
        parallel.ParallelBatchExecutor(
            max_parallel=3, process_starter=start, logger=logger
        ).execute(plan, registry, config, paths)
    assert raised.value.partial.pending_source_ids == ("A", "B", "C")
    assert wait.call_count == 2
    assert wait.call_args.kwargs["timeout"] == 30.0
    assert "parallel_batch_interrupted_again" in [c.args[0] for c in logger.info.call_args_list]
    for handle in handles:
        handle.terminate.assert_called_once()
        handle.kill.assert_called_once()
        handle.join.assert_called_once_with(timeout=5)
        handle.close.assert_called_once()


def test_shutdown_reaps_survivors_even_when_grace_collection_raises(tmp_path):
    _root, _registry, plan, config, paths = planned_project(tmp_path)
    handle = fake_handle(worker_request(plan, config, paths))
    running = {"A": handle}
    with pytest.raises(RuntimeError, match="collection failed"):
        parallel.ParallelBatchExecutor(max_parallel=3)._stop_workers(
            running, collect=Mock(side_effect=RuntimeError("collection failed"))
        )
    handle.terminate.assert_called_once()
    handle.kill.assert_called_once()
    handle.join.assert_called_once_with(timeout=5)
    handle.close.assert_called_once()
    assert running == {}


@pytest.mark.parametrize("termination", ["success", "error"])
def test_parent_handlers_are_restored_after_dispatch(tmp_path, monkeypatch, termination):
    _root, registry, plan, config, paths = planned_project(tmp_path)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def start(task):
        assert signal.getsignal(signal.SIGTERM) != previous[signal.SIGTERM]
        handle = fake_handle(task, envelope=call_worker(task))
        handle.is_alive.return_value = False
        return handle

    if termination == "error":
        monkeypatch.setattr(parallel, "_wait_any", Mock(side_effect=RuntimeError("wait failed")))
    else:
        monkeypatch.setattr(parallel, "_wait_any", lambda handles: handles)
    runner = parallel.ParallelBatchExecutor(
        max_parallel=3, process_starter=start, execution_factory=FAKE_FACTORY
    )
    if termination == "error":
        with pytest.raises(RuntimeError, match="wait failed"):
            runner.execute(plan, registry, config, paths)
    else:
        store = MemorySummaryStore()
        persist = store.persist

        def checked_persist(outcome):
            assert {s: signal.getsignal(s) for s in previous} == previous
            return persist(outcome)

        store.persist = checked_persist
        runner.execute(plan, registry, config, paths, summary_store=store)
    assert {s: signal.getsignal(s) for s in previous} == previous


def test_embedding_thread_keeps_parent_handlers_and_logs_once(tmp_path, monkeypatch):
    _root, registry, plan, config, paths = planned_project(tmp_path)
    envelopes = {
        s.source_id: call_worker(worker_request(plan, config, paths, source_id=s.source_id))
        for s in plan.sources
    }
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    logger = Mock(context={})

    def start(task):
        handle = fake_handle(task, envelope=envelopes[task.source_id])
        handle.is_alive.return_value = False
        return handle

    monkeypatch.setattr(parallel, "_wait_any", lambda handles: handles)
    runner = parallel.ParallelBatchExecutor(max_parallel=3, process_starter=start, logger=logger)
    with ThreadPoolExecutor(max_workers=1) as embedding:
        outcome = embedding.submit(
            runner.execute, plan, registry, config, paths, summary_store=MemorySummaryStore()
        ).result(timeout=5)
    assert all(source.status == "succeeded" for source in outcome.sources)
    assert {s: signal.getsignal(s) for s in previous} == previous
    assert [c.args[0] for c in logger.info.call_args_list].count(
        "parallel_signal_handlers_not_installed"
    ) == 1


def test_parallel_partial_evidence_equals_sequential_at_same_logical_interrupt(
    tmp_path, monkeypatch
):
    _root, registry, plan, config, paths = planned_project(tmp_path, "independent_three")
    clock = PipelineClock(wall_clock=lambda: STARTED_AT, monotonic_clock=lambda: 0.0)
    execution = CannedSourceExecution("independent_three")

    class InterruptAfterA:
        def execute(self, planned_run, config, paths):
            if planned_run.plan.source.source_id != "A":
                raise KeyboardInterrupt
            return execution.execute(planned_run, config, paths)

    with pytest.raises(BatchExecutionInterrupted) as sequential:
        BatchExecutor(source_execution=InterruptAfterA(), clock=clock).execute(
            plan, registry, config, paths, summary_store=MemorySummaryStore()
        )
    attempt = execute_source_attempt(
        plan.source("A"), config, paths, source_execution=execution, clock=clock
    )
    handles = []

    def start(task):
        handle = fake_handle(task)
        handle.is_alive.return_value = False
        if task.source_id == "A":
            handle.envelope = {"attempt": attempt.to_summary(), "worker": {}}
            handle.result_received = True
        else:
            handle.result_received = True  # Interrupted workers close without an envelope.
        handles.append(handle)
        return handle

    # Build the first readiness result after all three handles were dispatched.
    waits = iter(("A", "interrupt", "pending"))

    def wait(current, **kwargs):
        step = next(waits)
        if step == "interrupt":
            raise KeyboardInterrupt
        return [handles[0]] if step == "A" else current

    monkeypatch.setattr(parallel, "_wait_any", wait)
    with pytest.raises(BatchExecutionInterrupted) as spawned:
        parallel.ParallelBatchExecutor(max_parallel=3, process_starter=start, clock=clock).execute(
            plan, registry, config, paths, summary_store=MemorySummaryStore()
        )
    assert spawned.value.partial == sequential.value.partial
    assert spawned.value.partial.pending_source_ids == ("B", "C")


@pytest.mark.parametrize("grace", [0, -1, float("inf"), float("nan")])
def test_interrupt_grace_must_be_finite_and_positive(grace):
    with pytest.raises(ValueError, match="interrupt_grace_seconds"):
        parallel.ParallelBatchExecutor(max_parallel=3, interrupt_grace_seconds=grace)
