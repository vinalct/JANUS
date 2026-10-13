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


def test_one_slot_constructs_existing_runner_without_importing_multiprocessing():
    script = """
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from janus.orchestration import BatchPlanner, BatchPlanRequest
from janus.registry import load_registry
from janus.runtime import BatchExecutor
from janus.runtime.parallel import build_batch_executor
from janus.utils.environment import load_environment_config, materialize_runtime_paths
from tests.support.parallel_baseline import CannedSourceExecution, write_capture_project, STARTED_AT
assert 'multiprocessing' not in sys.modules
with TemporaryDirectory() as temporary:
    root = Path(temporary)
    write_capture_project('chain_and_peer', root)
    registry = load_registry(root)
    request = BatchPlanRequest.create(environment='local', project_root=root, planned_at=STARTED_AT)
    plan = BatchPlanner().plan(request, registry=registry)
    config = load_environment_config('local', root)
    runner = build_batch_executor(max_parallel=1, logger=None)
    assert type(runner) is BatchExecutor
    runner.source_execution = CannedSourceExecution('chain_and_peer')
    outcome = runner.execute(plan, registry, config, materialize_runtime_paths(config, root))
    assert outcome.is_successful
    assert 'max_parallel' not in outcome.to_summary()['pipeline']
    assert 'multiprocessing' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr


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


@pytest.mark.parametrize(
    "mode,reason",
    [("137", "code 137"), ("SIGKILL", "SIGKILL"), ("0", "cleanly without sending a result")],
)
def test_crashed_worker_returns_failure_and_skips_consumer(tmp_path, monkeypatch, mode, reason):
    from janus.runtime.parallel import ParallelBatchExecutor
    from tests.support.parallel_execution import lock_at

    root, registry, plan, config, paths = planned_project(tmp_path)
    monkeypatch.setenv("JANUS_TEST_WORKER_EXIT", mode)
    runner = ParallelBatchExecutor(
        max_parallel=3, execution_factory="tests.support.parallel_execution:crash_execution_factory"
    )
    outcome = runner.execute(plan, registry, config, paths)
    sources = {source.source_id: source for source in outcome.sources}
    assert sources["A"].failure.phase == "worker_crash"
    assert sources["A"].failure.error_type == "WorkerCrashed"
    assert reason in sources["A"].failure.reason
    assert dict(sources["A"].attempts[0].evidence) == {}
    assert sources["B"].status == "skipped"
    assert sources["B"].skip.root_failed_source_ids == ("A",)
    assert sources["C"].status == "succeeded"
    lock_path = root / "data/metadata/A/.lock"
    assert lock_path.is_file()
    with lock_at(lock_path):
        pass
    (tmp_path / "crash-attempt.json").write_text(
        json.dumps(sources["A"].attempts[0].to_summary(), indent=2)
    )


def test_parent_tripwire_catches_planted_preflight_acquisition(tmp_path, monkeypatch):
    import janus.runtime.parallel as parallel
    from janus.runtime import SparkSessionProvider

    _root, registry, plan, config, paths = planned_project(tmp_path)
    arm_spark_tripwire(monkeypatch)
    preflight = parallel._preflight

    class EagerParent(parallel.ParallelBatchExecutor):
        def execute(self, *args, **kwargs):
            def eager_preflight(*args, **kwargs):
                SparkSessionProvider(config, paths).get()
                preflight(*args, **kwargs)

            monkeypatch.setattr(parallel, "_preflight", eager_preflight)
            return super().execute(*args, **kwargs)

    with pytest.raises(pytest.fail.Exception, match="Spark session"):
        EagerParent(max_parallel=3, execution_factory=FAKE_FACTORY).execute(
            plan, registry, config, paths
        )
    assert not list((_root / "data/metadata").rglob("summary.json"))


def test_parent_tripwire_covers_three_spawned_workers_and_summary_persistence(
    tmp_path, monkeypatch
):
    from janus.runtime.parallel import ParallelBatchExecutor
    from tests.support.parallel import BOOTSTRAP_FACTORY

    root, registry, plan, config, paths = planned_project(tmp_path, "independent_three")
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("JANUS_TEST_WORKER_REPORT_DIR", str(report_dir))
    arm_spark_tripwire(monkeypatch)
    outcome = ParallelBatchExecutor(max_parallel=3, execution_factory=BOOTSTRAP_FACTORY).execute(
        plan, registry, config, paths
    )
    assert outcome.is_successful
    stored = root / "data/metadata/pipelines" / plan.request.pipeline_run_id / "summary.json"
    assert json.loads(stored.read_text()) == outcome.to_summary()
    reports = [
        json.loads((report_dir / f"{source}.json").read_text()) for source in ("A", "B", "C")
    ]
    assert len({report["pid"] for report in reports}) == 3
    assert all(report["pid"] != os.getpid() for report in reports)
    assert all(report["pyspark_imported"] is False for report in reports)
    assert all(report["providers_started"] == [False] for report in reports)
    assert all(report["started_after_lock"] is False for report in reports)


@pytest.mark.parametrize("guarded,eager", [(False, False), (True, False), (True, True)])
def test_spawned_bootstrap_reports_no_import_or_session_and_detects_violation(
    tmp_path, monkeypatch, guarded, eager
):
    import multiprocessing

    from janus.runtime.parallel import run_worker
    from tests.support.parallel import BOOTSTRAP_FACTORY, bootstrap_tripwire_worker

    _root, _registry, plan, config, paths = planned_project(tmp_path)
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("JANUS_TEST_WORKER_REPORT_DIR", str(report_dir))
    task = worker_request(plan, config, paths, factory=BOOTSTRAP_FACTORY)
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    target = bootstrap_tripwire_worker if guarded else run_worker
    args = (task, writer, eager) if guarded else (task, writer)
    child = context.Process(target=target, args=args)
    child.start()
    writer.close()
    try:
        assert reader.poll(15), "bootstrap child did not return or close its pipe"
        if eager:
            with pytest.raises(EOFError):
                reader.recv_bytes()
        else:
            envelope = json.loads(reader.recv_bytes())
            assert envelope["attempt"]["status"] == "succeeded"
            assert envelope["worker"] == {
                "pid": child.pid,
                "spark_started": False,
                "spark_application_id": None,
            }
        child.join(timeout=5)
        assert child.exitcode == 0
        report = json.loads((report_dir / "A.json").read_text())
        assert report["pid"] == child.pid != os.getpid()
        assert report["pyspark_imported"] is False
        assert report["providers_started"] == [False]
        if eager:
            assert "Spark session" in report["tripwire"]
            assert "registry_loaded" not in report["events"]
        else:
            assert report["started_after_lock"] is False
            assert report["application_id"] is None
            if guarded:
                assert report["tripwire"] is None
                milestones = [
                    "parallel_worker_started",
                    "registry_loaded",
                    "planning_completed",
                    "snapshot_verified",
                    "provider_constructed",
                    "source_lock_acquired",
                ]
                assert [event for event in report["events"] if event in milestones] == milestones
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        child.close()
        reader.close()


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


@pytest.mark.parametrize("capacity", [-1, 0, 1])
def test_parallel_runner_requires_multiple_slots(capacity):
    from janus.runtime.parallel import ParallelBatchExecutor

    with pytest.raises(ValueError, match="use BatchExecutor for 1"):
        ParallelBatchExecutor(max_parallel=capacity)


def _contract_without_timings(value):
    """Keep all evidence; remove only measured timestamps/durations and the additive key."""
    if isinstance(value, dict):
        return {
            key: _contract_without_timings(item)
            for key, item in value.items()
            if key not in {"started_at", "ended_at", "duration_seconds", "max_parallel"}
        }
    if isinstance(value, list):
        return [_contract_without_timings(item) for item in value]
    return value


@pytest.mark.parametrize("case", [
    "independent_three", "chain_and_peer", "upstream_fails", "upstream_raises",
    "planning_failure", "diamond", "diamond_one_arm_fails", "shared_metadata_dir",
    "independent_seven",
])
def test_runner_uses_worker_results_with_bounded_deterministic_dispatch(
    tmp_path, monkeypatch, case
):
    import janus.runtime.parallel as parallel
    from janus.hooks import built_in_hooks
    from tests.support.parallel_baseline import CannedSourceExecution, FailingPlanHook
    from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore

    root, registry, plan, config, paths = planned_project(tmp_path, case)
    monkeypatch.setenv("JANUS_TEST_SCENARIO", case)
    hooks = (*built_in_hooks(), ("parallel_baseline.fail_plan", FailingPlanHook()))
    monkeypatch.setattr("janus.hooks.built_in_hooks", lambda: hooks)
    sequential = parallel.BatchExecutor(source_execution=CannedSourceExecution(case)).execute(
        plan, registry, config, paths, summary_store=MemorySummaryStore()
    )
    active = []
    dispatches = []
    attempts = {}
    peaks = []
    events = []
    real_preflight = parallel._preflight

    def preflight(*args):
        real_preflight(*args)
        events.append("preflight")

    store = MemorySummaryStore()
    store.assert_available = lambda _: events.append("available")

    def start(task):
        assert events[:2] == ["preflight", "available"]
        assert len(active) < 3
        if case == "shared_metadata_dir":
            assert active == []
        handle = Mock(request=task, pid=os.getpid(), exitcode=0)
        handle.envelope = call_worker(task)
        attempts[task.source_id] = handle.envelope["attempt"]
        handle.join.side_effect = lambda: active.remove(task.source_id)
        active.append(task.source_id)
        peaks.append(len(active))
        dispatches.append(task.source_id)
        return handle

    monkeypatch.setattr(parallel, "_preflight", preflight)
    # Complete in reverse order to prove that summaries keep the plan order.
    monkeypatch.setattr(parallel, "_wait_any", lambda handles: list(reversed(handles)))
    logger = Mock(context={})
    outcome = parallel.ParallelBatchExecutor(
        max_parallel=3, process_starter=start, execution_factory=FAKE_FACTORY, logger=logger
    ).execute(plan, registry, config, paths, summary_store=store)
    assert not active
    assert max(peaks) <= 3
    if case == "independent_seven":
        assert max(peaks) == 3
        assert dispatches == list(plan.source_ids)
    if case == "chain_and_peer":
        assert dispatches == ["A", "C", "B"]
    assert tuple(source.source_id for source in outcome.sources) == plan.source_ids
    assert outcome.to_summary()["pipeline"]["max_parallel"] == 3
    assert outcome.schema_version == 1
    assert _contract_without_timings(outcome.to_summary()) == _contract_without_timings(
        sequential.to_summary()
    )
    for source in outcome.sources:
        if source.attempts:
            assert source.attempts[0].to_summary() == attempts[source.source_id]
    assert len(logger.info.call_args_list) == 2 * len(dispatches)
    assert root.exists()


@pytest.mark.parametrize("failure", ["snapshot", "summary"])
def test_parent_preflight_refuses_before_starting_workers(tmp_path, failure):
    from janus.orchestration.errors import DuplicatePipelineRunError
    from janus.runtime import BatchExecutionPreflightError
    from janus.runtime.parallel import ParallelBatchExecutor
    from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore

    root, registry, plan, config, paths = planned_project(tmp_path)
    starter = Mock(side_effect=AssertionError("worker started before preflight"))
    store = MemorySummaryStore()
    if failure == "snapshot":
        registry = replace(registry, project_root=root / "different-root")
        expected = BatchExecutionPreflightError
    else:
        store.assert_available = Mock(side_effect=DuplicatePipelineRunError("already finalized"))
        expected = DuplicatePipelineRunError
    with pytest.raises(expected):
        ParallelBatchExecutor(max_parallel=3, process_starter=starter).execute(
            plan, registry, config, paths, summary_store=store
        )
    starter.assert_not_called()


def _send_large_worker_result(task, connection):
    try:
        connection.send_bytes(b"x" * (17 * 1024 * 1024))
    except BrokenPipeError:
        pass
    finally:
        connection.close()


def test_oversized_spawned_envelope_is_refused_without_pipe_deadlock(tmp_path):
    import multiprocessing

    from janus.runtime.parallel import ParallelBatchExecutor, _WorkerHandle
    from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore

    _root, registry, plan, config, paths = planned_project(tmp_path)
    handles = []

    def start(task):
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_send_large_worker_result, args=(task, writer))
        process.start()
        writer.close()
        handle = _WorkerHandle(task, process, reader)
        handles.append(handle)
        return handle

    try:
        outcome = ParallelBatchExecutor(max_parallel=3, process_starter=start).execute(
            plan, registry, config, paths, summary_store=MemorySummaryStore()
        )
        sources = {source.source_id: source for source in outcome.sources}
        for source_id in ("A", "C"):
            failure = sources[source_id].failure
            assert failure.phase == "worker_crash"
            assert "16777216" in failure.reason and "16 MiB" in failure.reason
            assert not sources[source_id].attempts[0].evidence
        assert sources["B"].status == "skipped"
    finally:
        for handle in handles:
            try:
                if handle.is_alive():
                    handle.kill()
                    handle.join(timeout=5)
                handle.close()
            except ValueError:  # The runner already closed its completed process.
                pass


def _run_fixture_worker(task, connection):
    """Make the fixture's custom planning hook available in a spawned interpreter."""
    from unittest.mock import patch

    from janus.hooks import built_in_hooks
    from janus.runtime.parallel import run_worker
    from tests.support.parallel_baseline import FailingPlanHook

    hooks = (*built_in_hooks(), ("parallel_baseline.fail_plan", FailingPlanHook()))
    with patch("janus.hooks.built_in_hooks", return_value=hooks):
        run_worker(task, connection)


@pytest.mark.parametrize("case", [
    "independent_three", "chain_and_peer", "upstream_fails", "upstream_raises",
    "planning_failure", "diamond", "diamond_one_arm_fails", "shared_metadata_dir",
])
def test_spawned_batches_preserve_every_baseline_case(tmp_path, monkeypatch, case):
    import multiprocessing

    from janus.runtime import BatchExecutor
    from janus.runtime.parallel import ParallelBatchExecutor, _spawn_worker, _WorkerHandle
    from tests.support.parallel_baseline import CannedSourceExecution
    from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore

    _root, registry, plan, config, paths = planned_project(tmp_path, case)
    monkeypatch.setenv("JANUS_TEST_SCENARIO", case)
    sequential = BatchExecutor(source_execution=CannedSourceExecution(case)).execute(
        plan, registry, config, paths, summary_store=MemorySummaryStore()
    )

    def fixture_starter(task):
        if case != "planning_failure":
            return _spawn_worker(task)
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_run_fixture_worker, args=(task, writer))
        process.start()
        writer.close()
        return _WorkerHandle(task, process, reader)

    outcome = ParallelBatchExecutor(
        max_parallel=3, execution_factory=FAKE_FACTORY, process_starter=fixture_starter
    ).execute(plan, registry, config, paths, summary_store=MemorySummaryStore())
    assert _contract_without_timings(outcome.to_summary()) == _contract_without_timings(
        sequential.to_summary()
    )


@pytest.mark.parametrize("fail_on_breach", [False, True])
def test_parent_applies_budget_policy_once_after_worker_result(
    tmp_path, monkeypatch, fail_on_breach
):
    import yaml

    import janus.runtime.parallel as parallel
    from janus.orchestration import BatchPlanner
    from janus.registry import load_registry
    from tests.unit.orchestration.test_ready_queue_scheduler import MemorySummaryStore

    root, _registry, plan, config, paths = planned_project(tmp_path)
    path = plan.source("A").config_path
    document = yaml.safe_load(path.read_text())
    document["operations"] = {"max_duration_seconds": 4}
    path.write_text(yaml.safe_dump(document))
    registry = load_registry(root)
    plan = BatchPlanner().plan(
        replace(plan.request, fail_on_budget_breach=fail_on_breach), registry=registry
    )
    received = []
    policy_calls = []
    original_policy = parallel._apply_budget_policy

    def policy(attempt, source, request):
        policy_calls.append(source.source_id)
        return original_policy(attempt, source, request)

    def start(task):
        handle = Mock(request=task, pid=os.getpid(), exitcode=0)
        handle.envelope = call_worker(task)
        received.append(handle.envelope["attempt"])
        return handle

    monkeypatch.setattr(parallel, "_wait_any", lambda handles: handles)
    monkeypatch.setattr(parallel, "_apply_budget_policy", policy)
    outcome = parallel.ParallelBatchExecutor(
        max_parallel=3, execution_factory=FAKE_FACTORY, process_starter=start
    ).execute(plan, registry, config, paths, summary_store=MemorySummaryStore())
    sources = {source.source_id: source for source in outcome.sources}
    assert received[0]["status"] == "succeeded"
    assert received[0]["evidence"]["duration_budget_outcome"] == "exceeded"
    assert len(policy_calls) == len(set(policy_calls)) == len(received)
    assert sources["A"].status == ("failed" if fail_on_breach else "succeeded")
    assert sources["B"].status == ("skipped" if fail_on_breach else "succeeded")
    if fail_on_breach:
        assert sources["A"].failure.phase == "budget"
        assert sources["B"].skip.root_failed_source_ids == ("A",)
    assert sources["C"].status == "succeeded"
    persisted = json.loads(next((root / "data/metadata/A/runs").glob("*.json")).read_text())
    assert persisted["status"] == "succeeded"


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("spawn failed"), KeyboardInterrupt()])
def test_spawn_failure_closes_pipe_and_reaps_any_started_child(
    tmp_path, monkeypatch, started, error
):
    import multiprocessing

    from janus.runtime.parallel import _spawn_worker

    _root, _registry, plan, config, paths = planned_project(tmp_path)
    reader, writer = Mock(), Mock()
    process = Mock(pid=123 if started else None)
    process.start.side_effect = error
    process.is_alive.return_value = True
    context = Mock()
    context.Pipe.return_value = reader, writer
    context.Process.return_value = process
    get_context = Mock(return_value=context)
    monkeypatch.setattr(multiprocessing, "get_context", get_context)
    with pytest.raises(type(error)) as raised:
        _spawn_worker(worker_request(plan, config, paths))
    assert raised.value is error
    get_context.assert_called_once_with("spawn")
    reader.close.assert_called_once()
    writer.close.assert_called_once()
    process.close.assert_called_once()
    if started:
        process.kill.assert_called_once()
        process.join.assert_called_once()
    else:
        process.kill.assert_not_called()
        process.join.assert_not_called()
