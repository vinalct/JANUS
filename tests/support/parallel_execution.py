from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

from janus.orchestration import BatchPlanner, BatchPlanRequest
from janus.registry import load_registry
from janus.utils.environment import load_environment_config, materialize_runtime_paths
from tests.support.contracts import DECLARED_CONTRACT_PATH
from tests.support.orchestration import GraphCase, SourceSpec
from tests.support.parallel_baseline import CASES, STARTED_AT, CannedSourceExecution

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_FACTORY = "tests.support.parallel_execution:fake_execution_factory"
SPARK_FACTORY = "tests.support.parallel_execution:spark_execution_factory"
GRAPH_CASES = CASES | {"independent_seven": GraphCase(tuple(SourceSpec(f"S{i}") for i in range(7)))}


def graph_project(tmp_path, case="chain_and_peer", *, grouped=False):
    from tests.support.orchestration import source_documents, write_project

    root = tmp_path / case
    documents = source_documents(GRAPH_CASES[case])
    if case == "planning_failure":
        next(document for document in documents if document["source_id"] == "A")["source_hook"] = (
            "parallel_baseline.fail_plan"
        )
    if case == "shared_metadata_dir":
        for document in documents:
            document["outputs"]["metadata"]["path"] = "data/metadata/shared"
    write_project(root, documents, grouped=grouped)
    profiles = root / "conf/environments"
    profiles.mkdir(exist_ok=True)
    (profiles / "local.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "local",
                "runtime": {"log_level": "INFO"},
                "spark": {
                    "app_name": "parallel-execution-fixtures",
                    "master": "local[1]",
                    "warehouse_dir": "data/metadata/spark-warehouse",
                    "config": {},
                },
                "storage": {
                    "root_dir": "data",
                    "raw_dir": "data/raw",
                    "bronze_dir": "data/bronze",
                    "metadata_dir": "data/metadata",
                },
                "observability": {"metrics": {"transport": "disabled"}},
            }
        )
    )
    return root


def planned_project(tmp_path, case="chain_and_peer", *, pipeline_id="parallel-fixture"):
    from unittest.mock import patch

    from janus.hooks import built_in_hooks
    from tests.support.parallel_baseline import FailingPlanHook

    root = graph_project(tmp_path, case)
    request = BatchPlanRequest.create(
        environment="local", project_root=root, pipeline_run_id=pipeline_id, planned_at=STARTED_AT
    )
    hooks = (*built_in_hooks(), ("parallel_baseline.fail_plan", FailingPlanHook()))
    with patch("janus.hooks.built_in_hooks", return_value=hooks):
        registry = load_registry(root)
        plan = BatchPlanner().plan(request, registry=registry)
    config = load_environment_config("local", root)
    return root, registry, plan, config, materialize_runtime_paths(config, root)


def lock_at(path, *, source_id="A", run_id="holder-run", wait_seconds=0, **kwargs):
    from janus.runtime.locking import LockHolder, SourceStateLock

    holder = LockHolder.now(
        source_id=source_id, command="run --execute", run_id=run_id, pipeline_run_id=None
    )
    return SourceStateLock(Path(path), holder, wait_seconds=wait_seconds, **kwargs)


def hold_lock(path, ready, release, connection, source_id="A"):
    """Signal only after the kernel lock and holder record have been acquired."""
    try:
        with lock_at(path, source_id=source_id) as lock:
            connection.send(json.loads(lock.holder.to_json()))
            ready.set()
            if not release.wait(30):
                raise TimeoutError("holder was never released")
    except BaseException as error:
        connection.send({"helper_error": f"{type(error).__name__}: {error}"})
        ready.set()
        raise
    finally:
        connection.close()


@contextlib.contextmanager
def held_lock(path, *, source_id="A"):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=hold_lock, args=(str(path), ready, release, writer, source_id))
    process.start()
    writer.close()
    try:
        assert ready.wait(15), "lock holder failed to become ready"
        assert reader.poll(1), "holder did not send diagnostics"
        holder = reader.recv()
        assert "helper_error" not in holder, holder
        yield process, release, holder
    finally:
        # A killed Event waiter leaves notify's acknowledgement semaphore orphaned.
        # Do not notify a holder that is already dead: Event.set() can block forever.
        if process.is_alive():
            release.set()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        reader.close()
        assert not process.is_alive()


def try_lock(path, connection, wait_seconds=0, attempting=None):
    from janus.runtime.locking import SourceLockedError

    started = time.monotonic()
    if attempting is not None:
        attempting.set()
    try:
        with lock_at(path, run_id="contender", wait_seconds=wait_seconds):
            result = {"acquired": True}
    except SourceLockedError as error:
        result = {
            "acquired": False,
            "message": str(error),
            "holder": json.loads(error.holder.to_json()) if error.holder else None,
        }
    result["elapsed"] = time.monotonic() - started
    connection.send(result)
    connection.close()


def probe_lock(path, *, wait_seconds=0):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=try_lock, args=(str(path), writer, wait_seconds))
    process.start()
    writer.close()
    try:
        assert reader.poll(15), "lock contender did not respond"
        result = reader.recv()
        process.join(5)
        assert process.exitcode == 0
        return result
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        reader.close()


def exec_lock_holder(path, ready, proceed, connection):
    """exec bypasses Python cleanup; CLOEXEC must release the live descriptor."""
    lock = lock_at(path)
    lock.acquire()
    # multiprocessing makes its passed pipe inheritable during spawn. EOF is the
    # exec handshake, so only that diagnostic descriptor needs its default restored.
    os.set_inheritable(connection.fileno(), False)
    connection.send(os.getpid())
    ready.set()
    assert proceed.wait(15)
    os.execlp("sleep", "sleep", "30")


def crash_execution_factory(logger, recorder, lock_wait_seconds):
    class Crash:
        def execute(self, *args):
            os._exit(137)

    return Crash()


def fake_execution_factory(logger, recorder, lock_wait_seconds):
    execution = CannedSourceExecution(os.environ.get("JANUS_TEST_SCENARIO", "chain_and_peer"))

    class LazyFakeExecution:
        def execute(self, planned_run, config, paths):
            provider = recorder.factory(config, paths, logger)
            try:
                assert provider.was_started is False
                return execution.execute(planned_run, config, paths)
            finally:
                provider.stop()

    return LazyFakeExecution()


def rewrite_yaml(root, source_id="A"):
    for path in sorted((root / "conf/sources").glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        sources = data.get("sources", [data])
        for source in sources:
            if source["source_id"] == source_id:
                source["name"] += " changed after planning"
                path.write_text(yaml.safe_dump(data))
                return path
    raise LookupError(source_id)


def rewrite_contract(root):
    path = root / DECLARED_CONTRACT_PATH
    data = yaml.safe_load(path.read_text())
    data["name"] += " changed after planning"
    path.write_text(yaml.safe_dump(data))
    return path


def break_registry(root):
    path = root / "conf/sources/invalid.yaml"
    path.write_text("sources: [not-a-source]\n")
    return path


def file_snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def worker_request(plan, config, paths, *, source_id="A", factory=FAKE_FACTORY):
    from janus.runtime.parallel import WorkerTask

    source = plan.source(source_id)
    contract = source.require_planned_run().plan.data_contract
    return WorkerTask(
        request=plan.request,
        source_id=source_id,
        run_id=source.run_id,
        attempt=source.attempt,
        expected_config_version=source.config_version,
        expected_schema_version=contract.schema_version if contract else None,
        expected_upstream_ids=source.upstream_ids,
        environment_config=config,
        resolved_paths=paths,
        log_level="INFO",
        log_bindings=(("pipeline_run_id", plan.request.pipeline_run_id),),
        lock_wait_seconds=0,
        execution_factory=factory,
    )


class EnvelopeSink:
    def __init__(self):
        self.messages = []

    def send_bytes(self, message):
        self.messages.append(json.loads(message))

    def close(self):
        pass


def call_worker(request):
    from janus.runtime.parallel import run_worker

    sink = EnvelopeSink()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_worker(request, sink)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert len(sink.messages) == 1
    return sink.messages[0]


def eager_bootstrap_stub(config, paths):
    from janus.runtime import SparkSessionProvider

    SparkSessionProvider(config, paths).get()


def tripwire_worker(request, connection):
    import pytest

    from janus.runtime.parallel import run_worker
    from tests.support.operator_cli import arm_spark_tripwire

    with pytest.MonkeyPatch.context() as patch:
        arm_spark_tripwire(patch)
        run_worker(request, connection)


def normalized_terminal_contract(summary):
    """Use the CLI/adapter parity mask, preserving failures and graph identities."""
    return {
        key: value
        for key, value in summary.items()
        if key in {"schema_version", "selection", "graph", "config_versions"}
    } | {
        "environment": summary["pipeline"]["environment"],
        "sources": [
            {
                key: value
                for key, value in source.items()
                if key not in {"run_id", "timing", "attempts"}
            }
            | {
                "attempts": [
                    {
                        key: attempt[key]
                        for key in ("attempt", "status", "failure")
                        if key in attempt
                    }
                    for attempt in source["attempts"]
                ]
            }
            for source in summary["sources"]
        ],
        "totals": {
            key: value for key, value in summary["totals"].items() if key != "duration_seconds"
        },
    }


class FixtureTransport:
    """Offline ordinary downloads; the executor still owns extraction and Spark."""

    def __init__(self, source_id, recorder, logger):
        self.source_id = source_id
        self.recorder = recorder
        self.logger = logger

    def open(self):
        pass

    def close(self):
        pass

    def send(self, request):
        from janus.strategies.api import ApiResponse

        assert self.recorder.provider._session is None, "download has a live Spark session"
        self.logger.info("fixture_http_request", source_id=self.source_id)
        if os.environ.get("JANUS_TEST_FAIL_SOURCE") == self.source_id:
            raise RuntimeError("scripted extraction failure")
        barrier = os.environ.get("JANUS_TEST_BARRIER_DIR")
        if barrier and self.source_id in {"A", "C"}:
            with socket.socket(socket.AF_UNIX) as stream:
                stream.settimeout(30)
                stream.connect(str(Path(barrier) / "downloads.sock"))
                stream.sendall(self.source_id.encode())
                assert stream.recv(1) == b"1"
        return ApiResponse(
            request=request,
            status_code=200,
            body=json.dumps(
                [] if os.environ.get("JANUS_TEST_EMPTY_SOURCE") == self.source_id else [{"id": "1"}]
            ).encode(),
            received_at=STARTED_AT,
        )


def spark_execution_factory(logger, recorder, lock_wait_seconds):
    from janus.runtime import SourceExecutionService, SourceExecutor
    from janus.strategies.api import ApiStrategy
    from tests.support.orchestration_capture import EmptyHandoffHook

    service = SourceExecutionService(
        executor=SourceExecutor(logger=logger), provider_factory=recorder.factory
    )

    class OfflineExecution:
        def execute(self, planned_run, config, paths):
            source_id = planned_run.plan.source.source_id
            offline = replace(
                planned_run,
                strategy=ApiStrategy(
                    transport_factory=lambda **kwargs: FixtureTransport(source_id, recorder, logger)
                ),
                hook=(
                    EmptyHandoffHook()
                    if os.environ.get("JANUS_TEST_EMPTY")
                    or os.environ.get("JANUS_TEST_EMPTY_SOURCE") == source_id
                    else None
                ),
            )
            result = service.execute(offline, config, paths)
            if os.environ.get("JANUS_TEST_CAPTURE_LIFECYCLE") and source_id == "D":
                from tests.support.parallel_order24 import process_children

                logger.info(
                    "fixture_empty_worker_process_sample",
                    children=process_children(os.getpid()),
                    worker_alive=True,
                )
            return result

    return OfflineExecution()


def run_parallel_cli(case, argv):
    from janus.cli import run_all
    from janus.main import main
    from janus.runtime.parallel import ParallelBatchExecutor

    def executor(logger, *args, **kwargs):
        from janus.runtime import BatchExecutor

        if "--max-parallel" in argv and argv[argv.index("--max-parallel") + 1] == "1":

            class Recorder:
                def factory(self, config, paths, logger):
                    from janus.runtime import SparkSessionProvider

                    self.provider = SparkSessionProvider(config, paths, logger)
                    return self.provider

            return BatchExecutor(
                source_execution=spark_execution_factory(logger, Recorder(), 0), logger=logger
            )
        if os.environ.get("JANUS_TEST_CAPTURE_LIFECYCLE"):
            from tests.support.parallel_order24 import install_worker_evidence_capture

            install_worker_evidence_capture(logger)
        return ParallelBatchExecutor(
            max_parallel=3,
            execution_factory=SPARK_FACTORY,
            logger=logger,
            interrupt_grace_seconds=2,
        )

    run_all._build_batch_executor = executor
    cli_argv = list(argv)
    if (
        os.environ.get("JANUS_TEST_CAPTURE_LIFECYCLE")
        and "--max-parallel" not in run_all.build_parser()._option_string_actions
    ):
        index = cli_argv.index("--max-parallel")
        del cli_argv[index : index + 2]
    return main(cli_argv)


def cli_process(root, *, max_parallel=3, extra_env=None):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("JANUS_")}
    environment.update(extra_env or {})
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support.parallel_execution",
            "spark",
            "run-all",
            "--project-root",
            str(root),
            "--pipeline-run-id",
            "parallel",
            "--max-parallel",
            str(max_parallel),
            "--started-at",
            STARTED_AT.isoformat(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def example_multiprocess_job():
    root = Path(os.environ["JANUS_TEST_DAGSTER_ROOT"])
    return parity_adapter(root, executor="multiprocess").job


def adapter_execution_factory(logger):
    from janus.runtime import SourceExecutionService, SourceExecutor, SparkSessionProvider
    from janus.runtime.contract_preflight import LiveTable
    from janus.runtime.locking import hold_source_state_lock
    from janus.utils.storage import StorageLayout
    from janus.writers.evolution import LiveColumn
    from tests.integration.dagster.test_cli_adapter_parity import _ScenarioExecution

    scenario = os.environ.get("JANUS_TEST_DAGSTER_SCENARIO", "success")
    if scenario == "preflight":
        return SourceExecutionService(
            executor=SourceExecutor(
                logger=logger,
                preflight_loader=lambda *args: LiveTable(
                    (LiveColumn("id", "string", False), LiveColumn("stray", "string", False)),
                    "1.0.0",
                ),
            ),
            provider_factory=lambda config, paths, logger: SparkSessionProvider(
                config,
                paths,
                logger,
                session_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("strict preflight started Spark")
                ),
            ),
        )

    execution = _ScenarioExecution(
        fail_once={"A"} if scenario == "retry" else set(),
        fail_always={"A"} if scenario == "failure" else set(),
        budget_exceeded={"A"} if scenario == "budget" else set(),
    )

    class LockedScenario:
        def execute(self, planned_run, config, paths):
            plan = planned_run.plan
            with hold_source_state_lock(
                plan,
                StorageLayout.from_environment_config(config, plan.run_context.project_root),
                command="run-all",
                run_id=plan.run_context.run_id,
                pipeline_run_id=plan.run_context.attributes_as_dict().get("pipeline_run_id"),
            ):
                return execution.execute(planned_run, config, paths)

    return LockedScenario()


def parity_adapter(root, *, executor="in_process"):
    from dagster import RetryPolicy

    from janus.adapters.dagster import DagsterAdapterServices, build_dagster_adapter
    from tests.integration.dagster.test_cli_adapter_parity import _BudgetPolicyPlanner

    scenario = os.environ.get("JANUS_TEST_DAGSTER_SCENARIO", "success")
    return build_dagster_adapter(
        root,
        executor=executor,
        **({"max_concurrent": 3} if executor == "multiprocess" else {}),
        services=DagsterAdapterServices(
            batch_planner=_BudgetPolicyPlanner() if scenario == "budget" else BatchPlanner(),
            source_execution_factory=adapter_execution_factory,
        ),
        retry_policy=RetryPolicy(max_retries=1) if scenario == "retry" else None,
    )


def spark_project(tmp_path, case="chain_and_peer"):
    root = graph_project(tmp_path, case)
    from tests.support.spark_sessions import checked_in_iceberg_block

    path = root / "conf/environments/local.yaml"
    config = yaml.safe_load(path.read_text())
    config["spark"]["iceberg"] = checked_in_iceberg_block()
    config["spark"]["iceberg"]["warehouse_dir"] = str(root / "data/bronze/iceberg")
    config["spark"]["iceberg"]["uri"] = (
        f"jdbc:sqlite:{root}/data/metadata/catalog.sqlite?journal_mode=WAL&busy_timeout=30000"
    )
    config["spark"]["ivy_dir"] = str(REPO_ROOT / "data/metadata/ivy")
    config["spark"]["config"] = {
        "spark.driver.memory": "1g",
        "spark.ui.enabled": "false",
        "spark.sql.shuffle.partitions": "1",
    }
    path.write_text(yaml.safe_dump(config))
    return root


def download_barrier_server(directory, listening, downloads_ready, release):
    connections = []
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(Path(directory) / "downloads.sock"))
        server.listen(2)
        server.settimeout(30)
        listening.set()
        try:
            sources = set()
            for _ in range(2):
                connection, _address = server.accept()
                connections.append(connection)
                sources.add(connection.recv(1).decode())
            assert sources == {"A", "C"}
            downloads_ready.set()
            assert release.wait(30)
            for connection in connections:
                connection.sendall(b"1")
        finally:
            for connection in connections:
                connection.close()


@contextlib.contextmanager
def download_barrier(directory):
    directory.mkdir()
    context = multiprocessing.get_context("spawn")
    listening, downloads_ready, release = (context.Event() for _ in range(3))
    server = context.Process(
        target=download_barrier_server, args=(directory, listening, downloads_ready, release)
    )
    server.start()
    try:
        assert listening.wait(10)
        yield downloads_ready, release
    finally:
        release.set()
        server.join(5)
        if server.is_alive():
            server.kill()
            server.join(5)


if __name__ == "__main__":
    if sys.argv[1] == "golden":
        from tests.support.parallel_baseline import launch

        raise SystemExit(launch(sys.argv[2], sys.argv[3:]))
    raise SystemExit(run_parallel_cli(sys.argv[1], sys.argv[2:]))
