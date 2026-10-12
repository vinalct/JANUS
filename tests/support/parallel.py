"""Process and ordered-log evidence for the lazy parallel Spark lifecycle."""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

BOOTSTRAP_FACTORY = "tests.support.parallel:bootstrap_execution_factory"


def _write_report(source_id, **fields):
    directory = Path(os.environ["JANUS_TEST_WORKER_REPORT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{source_id}.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(previous | fields, sort_keys=True))


def bootstrap_execution_factory(logger, recorder, lock_wait_seconds):
    """Check bootstrap inside the child, without relying on parent monkeypatches."""
    from janus.runtime import SparkSessionProvider
    from janus.runtime.locking import hold_source_state_lock
    from janus.utils.storage import StorageLayout
    from tests.support.parallel_baseline import CannedSourceExecution

    provider = recorder.factory({}, {}, logger)
    source_id = logger.context["source_id"]
    started = [
        item.was_started for item in gc.get_objects() if isinstance(item, SparkSessionProvider)
    ]
    _write_report(
        source_id,
        pid=os.getpid(),
        pyspark_imported="pyspark" in sys.modules,
        providers_started=started,
        application_id=provider.application_id,
    )
    execution = CannedSourceExecution("chain_and_peer")

    class LockedFakeExecution:
        def execute(self, planned_run, config, paths):
            plan = planned_run.plan
            try:
                with hold_source_state_lock(
                    plan,
                    StorageLayout.from_environment_config(config, plan.run_context.project_root),
                    command="run-all",
                    run_id=plan.run_context.run_id,
                    pipeline_run_id=logger.context["pipeline_run_id"],
                    wait_seconds=lock_wait_seconds,
                    logger=logger,
                ):
                    _write_report(source_id, started_after_lock=provider.was_started)
                    return execution.execute(planned_run, config, paths)
            finally:
                provider.stop()

    return LockedFakeExecution()


def bootstrap_tripwire_worker(task, connection, eager=False):
    """Arm in the child before registry reads; persist trips from planted violations."""
    import pytest

    import janus.runtime.parallel as parallel
    from janus.runtime import SparkSessionProvider
    from janus.utils.logging import StructuredLogger
    from tests.support.operator_cli import arm_spark_tripwire

    events = []
    providers = []
    load = parallel.load_registry
    plan = parallel.BatchPlanner.plan
    verify = parallel._verify_source
    construct = SparkSessionProvider.__init__
    log = StructuredLogger.info

    def loading(*args, **kwargs):
        if eager:
            SparkSessionProvider(task.environment_config, task.resolved_paths).get()
        result = load(*args, **kwargs)
        events.append("registry_loaded")
        return result

    def planning(*args, **kwargs):
        result = plan(*args, **kwargs)
        events.append("planning_completed")
        return result

    def verifying(*args, **kwargs):
        verify(*args, **kwargs)
        events.append("snapshot_verified")

    def constructing(self, *args, **kwargs):
        construct(self, *args, **kwargs)
        providers.append(self)
        events.append("provider_constructed")

    def logging(self, event, **fields):
        events.append(event)
        log(self, event, **fields)

    trip = None
    with pytest.MonkeyPatch.context() as patch:
        arm_spark_tripwire(patch)
        patch.setattr(parallel, "load_registry", loading)
        patch.setattr(parallel.BatchPlanner, "plan", planning)
        patch.setattr(parallel, "_verify_source", verifying)
        patch.setattr(SparkSessionProvider, "__init__", constructing)
        patch.setattr(StructuredLogger, "info", logging)
        try:
            parallel.run_worker(task, connection)
        except pytest.fail.Exception as error:
            trip = str(error)
        finally:
            _write_report(
                task.source_id,
                pid=os.getpid(),
                events=events,
                tripwire=trip,
                providers_started=[provider.was_started for provider in providers],
                pyspark_imported="pyspark" in sys.modules,
            )


def parse_log_events(stderr):
    """Flatten structured fields once; Spark also writes non-JSON stderr lines."""
    events = []
    for line in stderr.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        events.append(event | event.get("fields", {}))
    return events


def source_events(events, source_id):
    return [event for event in events if event.get("source_id") == source_id]


def assert_lazy_worker_events(events, source_id, *, scoped_lookup=False):
    """Prove acquisition/release order and the lack of live Spark during requests."""
    sequence = [event["event"] for event in source_events(events, source_id)]
    worker_start = sequence.index("parallel_worker_started")
    extraction_end = sequence.index("source_extraction_finished")
    starts = [index for index, name in enumerate(sequence) if name == "spark_session_starting"]
    stops = [index for index, name in enumerate(sequence) if name == "spark_session_stopped"]
    requests = [index for index, name in enumerate(sequence) if name == "fixture_http_request"]
    assert requests and all(worker_start < index < extraction_end for index in requests)
    if scoped_lookup:
        assert len(starts) == len(stops) == 2
        assert worker_start < starts[0] < stops[0] < requests[0]
        assert extraction_end < starts[1] < stops[1]
    else:
        assert len(starts) == len(stops) == 1
        assert worker_start < extraction_end < starts[0] < stops[0]
    for request in requests:
        assert sum(index < request for index in starts) == sum(index < request for index in stops)


def process_children(worker_pid):
    """Read one bounded ps sample in the current process namespace."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,comm="],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return [
        {"pid": int(pid), "ppid": int(ppid), "command": command}
        for line in result.stdout.splitlines()
        for pid, ppid, command in [line.split(maxsplit=2)]
        if int(ppid) == worker_pid
    ]


def sample_finished_worker(logger, event, fields):
    """Sample ps synchronously in the CLI parent's finished-event handler."""
    if event != "parallel_worker_finished":
        return
    children = process_children(fields["worker_pid"])
    logger.info("fixture_worker_process_sample", **fields, children=children, cli_pid=os.getpid())


def install_worker_evidence_capture(logger):
    """Keep actual envelopes and ps evidence in test code in the CLI parent."""
    import janus.runtime.parallel as parallel

    read = parallel._read_envelope

    def reading(handle):
        read(handle)
        if handle.envelope is not None:
            logger.info(
                "fixture_worker_envelope",
                source_id=handle.request.source_id,
                envelope=handle.envelope,
            )

    parallel._read_envelope = reading

    class ProcessSampler(logging.Handler):
        def emit(self, record):
            sample_finished_worker(logger, record.getMessage(), record.event_fields)

    logger.logger.addHandler(ProcessSampler())


def spark_project_with_empty_source(tmp_path):
    import yaml

    from tests.support.orchestration import SourceSpec, source_payload

    root = lifecycle_spark_project(tmp_path)
    (root / "conf/sources/D.yaml").write_text(yaml.safe_dump(source_payload(SourceSpec("D"))))
    return root


def lifecycle_spark_project(tmp_path):
    """Prepare an existing shared catalog; concurrent first-time DDL is a separate gate."""
    import yaml
    from pyiceberg.catalog import load_catalog

    from janus.utils.catalog_properties import (
        derive_pyiceberg_catalog_name,
        derive_pyiceberg_catalog_properties,
    )
    from janus.utils.environment import load_environment_config, materialize_runtime_paths
    from tests.support.parallel_execution import spark_project

    root = spark_project(tmp_path)
    profile = root / "conf/environments/local.yaml"
    document = yaml.safe_load(profile.read_text())
    # The fixture provisions its catalog before dispatch, without starting Spark.
    # Workers need no schema bootstrap while concurrently writing their tables.
    catalog_name = document["spark"]["iceberg"]["catalog_name"]
    document["spark"]["config"][
        f"spark.sql.catalog.{catalog_name}.jdbc.init-catalog-tables"
    ] = "false"
    profile.write_text(yaml.safe_dump(document))
    config = load_environment_config("local", root)
    paths = materialize_runtime_paths(config, root)
    catalog = load_catalog(
        derive_pyiceberg_catalog_name(config),
        **(derive_pyiceberg_catalog_properties(config, paths) | {"init_catalog_tables": "true"}),
    )
    catalog.create_namespace_if_not_exists("bronze")
    catalog.engine.dispose()
    return root


def write_lifecycle_evidence(path, process, summary, events):
    selected = [
        event
        for event in events
        if event["event"].startswith(("parallel_worker_", "spark_session_", "fixture_"))
        or event["event"]
        in {
            "source_extraction_started",
            "source_extraction_finished",
            "normalization_handoff_prepared",
        }
    ]
    path.write_text(
        json.dumps(
            {
                "cli_pid": process.pid,
                "exit_code": process.returncode,
                "summary": summary,
                "events": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )
