"""Worker bootstrap, per-node snapshots, crashes, bounded interrupts, and N=1."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest

from janus.orchestration.results import SourceAttempt
from janus.runtime import BatchExecutionInterrupted
from tests.support.operator_cli import arm_spark_tripwire
from tests.support.parallel_execution import (
    FAKE_FACTORY,
    REPO_ROOT,
    break_registry,
    call_worker,
    eager_bootstrap_stub,
    file_snapshot,
    normalized_terminal_contract,
    planned_project,
    rewrite_contract,
    rewrite_yaml,
    tripwire_worker,
    worker_request,
)

RUNNER_RED = pytest.mark.xfail(strict=True, reason="spawned parallel batch runner is absent")
INTERRUPT_RED = pytest.mark.xfail(
    strict=True,
    reason="parallel interruption and worker-crash handling are absent",
)


def test_worker_json_envelope_round_trips_without_starting_spark(tmp_path):
    _root, _registry, plan, config, paths = planned_project(tmp_path)
    envelope = call_worker(worker_request(plan, config, paths))
    attempt = SourceAttempt.from_summary(envelope["attempt"])
    assert attempt.source_id == "A" and attempt.status == "succeeded"
    assert attempt.to_summary() == envelope["attempt"]
    assert envelope["worker"]["pid"] == os.getpid()
    assert envelope["worker"]["spark_started"] is False
    assert envelope["worker"]["spark_application_id"] is None


def test_bootstrap_tripwire_proves_itself_then_reaches_shared_attempt_seam(tmp_path, monkeypatch):
    _root, _registry, plan, config, paths = planned_project(tmp_path)
    arm_spark_tripwire(monkeypatch)
    # Prove the tripwire catches an eager bootstrap before checking the real worker.
    with pytest.raises(pytest.fail.Exception, match="Spark session"):
        eager_bootstrap_stub(config, paths)
    import janus.runtime.parallel as parallel

    calls = []
    original = parallel.execute_source_attempt
    providers = []
    from janus.runtime import SparkSessionProvider

    construct = SparkSessionProvider.__init__

    def constructing(self, *args, **kwargs):
        construct(self, *args, **kwargs)
        providers.append(self)

    monkeypatch.setattr(SparkSessionProvider, "__init__", constructing)

    def recording(source, *args, **kwargs):
        calls.append(source.source_id)
        assert source.config_version == plan.source("A").config_version
        return original(source, *args, **kwargs)

    monkeypatch.setattr(parallel, "execute_source_attempt", recording)
    envelope = call_worker(worker_request(plan, config, paths))
    assert calls == ["A"]
    assert len(providers) == 1 and providers[0].was_started is False
    assert envelope["attempt"]["status"] == "succeeded"
    assert envelope["worker"]["spark_started"] is False


@pytest.mark.parametrize("mutate", [rewrite_yaml, rewrite_contract, break_registry])
def test_snapshot_drift_refuses_before_lock_or_run_evidence(tmp_path, mutate):
    from janus.lineage import compute_config_version
    from janus.models.source_config import SourceConfigValidationError
    from janus.registry import load_registry

    root, _registry, plan, config, paths = planned_project(tmp_path)
    request = worker_request(plan, config, paths)
    before = file_snapshot(root / "data/metadata")
    changed = mutate(root)
    envelope = call_worker(request)
    attempt = SourceAttempt.from_summary(envelope["attempt"])
    assert attempt.status == "failed" and attempt.failure.phase == "snapshot_drift"
    assert "A" in attempt.failure.reason
    assert dict(attempt.evidence) == {}
    assert file_snapshot(root / "data/metadata") == before
    assert not list((root / "data/metadata").rglob(".lock"))
    if mutate is rewrite_yaml:
        assert request.expected_config_version[:12] in attempt.failure.reason
        assert compute_config_version(changed)[:12] in attempt.failure.reason
    elif mutate is rewrite_contract:
        new_contract = load_registry(root).contracts.get("A")
        assert "schema_version" in attempt.failure.reason
        assert request.expected_schema_version[:12] in attempt.failure.reason
        assert new_contract.schema_version[:12] in attempt.failure.reason
    else:
        assert "invalid" in attempt.failure.reason.lower()
        with pytest.raises(SourceConfigValidationError) as invalid:
            load_registry(root)
        assert str(invalid.value).splitlines()[0] in attempt.failure.reason
        assert str(invalid.value).splitlines()[1] in attempt.failure.reason


def test_unrelated_yaml_comment_does_not_refuse_node(tmp_path):
    root, _registry, plan, config, paths = planned_project(tmp_path)
    request = worker_request(plan, config, paths)
    source_path = plan.source("C").config_path
    assert source_path != plan.source("A").config_path
    source_path.write_text(source_path.read_text() + "\n# unrelated source edit\n")
    envelope = call_worker(request)
    assert envelope["attempt"]["status"] == "succeeded"
    assert root.exists()


@RUNNER_RED
def test_one_slot_constructs_existing_runner_without_importing_multiprocessing():
    script = (
        "import sys; from janus.runtime.parallel import build_batch_executor; "
        "from janus.runtime import BatchExecutor; "
        "assert 'multiprocessing' not in sys.modules; "
        "runner=build_batch_executor(max_parallel=1,logger=None); "
        "assert type(runner) is BatchExecutor; assert 'multiprocessing' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr


@RUNNER_RED
def test_parent_never_starts_spark_and_spawned_fake_batch_matches_sequential(tmp_path, monkeypatch):
    from janus.runtime import BatchExecutor
    from janus.runtime.parallel import ParallelBatchExecutor
    from tests.support.parallel_baseline import CannedSourceExecution
    from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore

    _root, registry, plan, config, paths = planned_project(tmp_path)
    arm_spark_tripwire(monkeypatch)
    sequential = BatchExecutor(source_execution=CannedSourceExecution("chain_and_peer")).execute(
        plan, registry, config, paths, summary_store=MemorySummaryStore()
    )
    parallel = ParallelBatchExecutor(max_parallel=3, execution_factory=FAKE_FACTORY).execute(
        plan, registry, config, paths, summary_store=MemorySummaryStore()
    )
    assert normalized_terminal_contract(parallel.to_summary()) == normalized_terminal_contract(
        sequential.to_summary()
    )
    assert "max_parallel" not in sequential.to_summary()["pipeline"]
    assert parallel.to_summary()["pipeline"]["max_parallel"] == 3


@INTERRUPT_RED
def test_crashed_worker_returns_failure_and_skips_consumer(tmp_path):
    from janus.runtime.parallel import ParallelBatchExecutor

    _root, registry, plan, config, paths = planned_project(tmp_path)
    runner = ParallelBatchExecutor(
        max_parallel=3, execution_factory="tests.support.parallel_execution:crash_execution_factory"
    )
    outcome = runner.execute(plan, registry, config, paths)
    sources = {source.source_id: source for source in outcome.sources}
    assert sources["A"].failure.phase == "worker_crash"
    assert sources["A"].failure.error_type == "WorkerCrashed"
    assert "137" in sources["A"].failure.reason
    assert dict(sources["A"].attempts[0].evidence) == {}
    assert sources["B"].status == "skipped"
    assert sources["B"].skip.root_failed_source_ids == ("A",)


@INTERRUPT_RED
def test_interrupt_signals_all_workers_then_kills_after_fake_clock_grace(tmp_path, monkeypatch):
    import janus.runtime.parallel as parallel

    _root, registry, plan, config, paths = planned_project(tmp_path)
    events = []
    handles = []
    ticks = iter((0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0))

    def start(request):
        handle = Mock()
        handle.request = request
        handle.planned_source = plan.source(request.source_id)
        handle.envelope = None
        handle.exitcode = None
        handle.is_alive.return_value = True
        handle.terminate.side_effect = lambda: events.append(("SIGTERM", request.source_id))
        handle.kill.side_effect = lambda: events.append(("SIGKILL", request.source_id))
        handle.join.side_effect = lambda **kwargs: events.append(("join", request.source_id))
        handles.append(handle)
        return handle

    waited = False

    def wait_any(*args, **kwargs):
        nonlocal waited
        if not waited:
            waited = True
            assert len(handles) == 2
            raise KeyboardInterrupt
        return []

    monkeypatch.setattr(parallel, "_wait_any", wait_any)
    monkeypatch.setattr(parallel, "monotonic", lambda: next(ticks, 5.0))
    runner = parallel.ParallelBatchExecutor(
        max_parallel=3, process_starter=start, interrupt_grace_seconds=1.5
    )
    with pytest.raises(BatchExecutionInterrupted) as interrupted:
        runner.execute(plan, registry, config, paths)
    assert events[:2] == [("SIGTERM", "A"), ("SIGTERM", "C")]
    assert [(signal, node) for signal, node in events if signal == "SIGKILL"] == [
        ("SIGKILL", "A"),
        ("SIGKILL", "C"),
    ]
    assert interrupted.value.partial.pending_source_ids == ("A", "B", "C")
    assert interrupted.value.partial.source_outcomes == ()
    assert not list((plan.request.project_root / "data/metadata/pipelines").rglob("summary.json"))


@pytest.mark.parametrize("armed", [False, True])
def test_worker_bootstrap_is_spawn_safe_and_stdout_is_empty(tmp_path, capfd, armed):
    import multiprocessing

    from janus.runtime.parallel import run_worker

    _root, _registry, plan, config, paths = planned_project(tmp_path)
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=tripwire_worker if armed else run_worker,
        args=(worker_request(plan, config, paths), writer),
    )
    process.start()
    writer.close()
    try:
        assert reader.poll(15)
        envelope = json.loads(reader.recv_bytes())
        process.join(5)
        assert process.exitcode == 0
        assert envelope["attempt"]["status"] == "succeeded"
        assert envelope["worker"]["spark_started"] is False
        captured = capfd.readouterr()
        assert captured.out == ""
        records = [json.loads(line) for line in captured.err.splitlines()]
        assert [record["event"] for record in records] == ["parallel_worker_started"]
        fields = records[0]["fields"]
        assert fields["worker_pid"] == process.pid
        assert fields["source_id"] == "A"
        assert fields["run_id"] == plan.source("A").run_id
        assert fields["pipeline_run_id"] == plan.request.pipeline_run_id
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        reader.close()


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"run_id": "different-run"}, "run_id"),
        ({"attempt": 2}, "attempt"),
        ({"expected_upstream_ids": ("C",)}, "upstream_ids"),
        ({"expected_schema_version": None}, "schema_version"),
    ],
)
def test_worker_refuses_identity_and_contract_presence_mismatches(tmp_path, changes, field):
    root, _registry, plan, config, paths = planned_project(tmp_path)
    task = replace(worker_request(plan, config, paths), **changes)
    envelope = call_worker(task)
    failure = envelope["attempt"]["failure"]
    assert failure["phase"] == "snapshot_drift"
    assert field in failure["reason"]
    assert envelope["attempt"]["run_id"] == task.run_id
    assert envelope["attempt"]["attempt"] == task.attempt
    assert file_snapshot(root / "data/metadata") == {}


@pytest.mark.parametrize("error", [LookupError("source vanished"), OSError("file unreadable")])
@pytest.mark.parametrize("stage", ["load", "plan"])
def test_worker_translates_registry_and_planning_errors_before_factory(
    tmp_path, monkeypatch, error, stage
):
    import janus.runtime.parallel as parallel

    _root, _registry, plan, config, paths = planned_project(tmp_path)
    factory = Mock(side_effect=AssertionError("execution constructed before verification"))
    if stage == "load":
        monkeypatch.setattr(parallel, "load_registry", Mock(side_effect=error))
    else:
        monkeypatch.setattr(parallel.BatchPlanner, "plan", Mock(side_effect=error))
    monkeypatch.setattr(parallel, "_resolve_execution_factory", factory)
    attempt = call_worker(worker_request(plan, config, paths))["attempt"]
    assert attempt["failure"]["phase"] == "snapshot_drift"
    assert str(error) in attempt["failure"]["reason"]
    factory.assert_not_called()


def test_worker_refuses_new_node_planning_failure(tmp_path, monkeypatch):
    import janus.runtime.parallel as parallel
    from janus.orchestration.plans import SourcePlanFailure

    _root, _registry, plan, config, paths = planned_project(tmp_path)
    failed = replace(
        plan.source("A"),
        planned_run=None,
        failure=SourcePlanFailure(error_type="RuntimeError", reason="hook failed after planning"),
    )
    monkeypatch.setattr(
        parallel.BatchPlanner, "plan", lambda *args, **kwargs: Mock(source=lambda _: failed)
    )
    attempt = call_worker(worker_request(plan, config, paths))["attempt"]
    assert attempt["failure"]["phase"] == "snapshot_drift"
    assert "disagreed" in attempt["failure"]["reason"]
    assert "hook failed after planning" in attempt["failure"]["reason"]


@pytest.mark.parametrize(
    "reference",
    [
        "os:system",
        "janus_other.module:factory",
        "tests_other.module:factory",
        "janus.runtime.parallel",
        "janus.runtime.parallel:factory.extra",
        "janus..runtime:factory",
    ],
)
def test_factory_resolver_rejects_unapproved_imports(reference, monkeypatch):
    import janus.runtime.parallel as parallel

    importer = Mock(side_effect=AssertionError("unapproved module imported"))
    monkeypatch.setattr(parallel.importlib, "import_module", importer)
    with pytest.raises(ValueError, match="Unsupported worker execution factory"):
        parallel._resolve_execution_factory(reference)
    importer.assert_not_called()


def test_factory_resolver_requires_callable():
    from janus.runtime.parallel import _resolve_execution_factory

    with pytest.raises(TypeError, match="not callable"):
        _resolve_execution_factory("janus.runtime.parallel:SNAPSHOT_DRIFT_PHASE")


def test_default_factory_constructs_lazy_provider_and_passes_lock_wait(tmp_path, monkeypatch):
    from janus.runtime.parallel import (
        DEFAULT_EXECUTION_FACTORY,
        _ProviderRecorder,
        _resolve_execution_factory,
        _worker_logger,
    )

    _root, _registry, plan, config, paths = planned_project(tmp_path)
    arm_spark_tripwire(monkeypatch)
    task = worker_request(plan, config, paths)
    logger = _worker_logger(task)
    recorder = _ProviderRecorder()
    execution = _resolve_execution_factory(DEFAULT_EXECUTION_FACTORY)(logger, recorder, 1.5)
    assert execution.executor.logger is logger
    assert execution.executor.lock_wait_seconds == 1.5
    assert recorder.provider is None
    provider = execution.provider_factory(config, paths, logger)
    assert recorder.provider is provider and recorder.was_started is False
    assert recorder.application_id is None


def test_worker_interrupt_sends_no_envelope_and_closes_pipe(tmp_path, monkeypatch, capfd):
    import signal

    import janus.runtime.parallel as parallel
    from tests.support.parallel_execution import EnvelopeSink

    _root, _registry, plan, config, paths = planned_project(tmp_path)

    def terminate(*args, **kwargs):
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(parallel, "execute_source_attempt", terminate)
    sink = EnvelopeSink()
    sink.close = Mock()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        parallel.run_worker(worker_request(plan, config, paths), sink)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert sink.messages == []
    sink.close.assert_called_once()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "parallel_worker_interrupted" in captured.err
