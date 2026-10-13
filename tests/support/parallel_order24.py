from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
import yaml
from examples.orchestration.fixture_service import ExampleFixtureHandler, build_server

from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
)
from janus.utils.environment import (
    ICEBERG_CATALOG_DB_PATH_KEY,
    load_environment_config,
    materialize_runtime_paths,
)
from tests.integration.orchestration.test_dependency_execution import _table_rows
from tests.support.parallel import parse_log_events
from tests.support.parallel_execution import REPO_ROOT
from tests.support.spark_sessions import require_iceberg_runtime, start_session

SPAWN_COST_SECONDS = 0.312488
PAGE_DELAY_SECONDS = 4 * SPAWN_COST_SECONDS
BRONZE_COLUMNS = {
    "A": ("reference_id", "label"),
    "B": ("detail_id", "reference_id", "amount"),
    "C": ("event_id", "window_start", "window_end"),
}
BRONZE_TABLES = {"A": "reference_a", "B": "details_b", "C": "independent_c"}


class DelayedFixtureHandler(ExampleFixtureHandler):
    """Delay real A/C pages; fail A once per request with a non-retryable 500."""

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in {"/reference", "/independent"}:
            time.sleep(self.server.page_delay_seconds)
        with self.server.request_lock:
            self.server.requests.append(self.path)
        if path == "/reference" and self.server.fail_a:
            self._send(500, {"error": "deliberate A failure"})
            return
        super().do_GET()

    def log_message(self, format, *args):
        pass


@contextlib.contextmanager
def fixture_service(*, fail_a=False, page_delay_seconds=0):
    server = build_server(port=0)
    server.RequestHandlerClass = DelayedFixtureHandler
    server.fail_a = fail_a
    server.page_delay_seconds = page_delay_seconds
    server.requests = []
    server.request_lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def example_project(root, server, *, shared_metadata=False):
    """Copy tracked example inputs and provision the shared JDBC catalog, session-free."""
    require_iceberg_runtime()
    from pyiceberg.catalog import load_catalog

    shutil.copytree(REPO_ROOT / "examples/orchestration/conf", root / "conf")
    source_path = root / "conf/sources/sources.yaml"
    document = yaml.safe_load(source_path.read_text())
    host, port = server.server_address
    for source in document["sources"]:
        source["access"]["base_url"] = f"http://{host}:{port}"
        source["access"]["timeout_seconds"] = max(5, 4 * PAGE_DELAY_SECONDS)
        source["extraction"]["retry"]["retryable_status_codes"] = [408, 429, 502, 503, 504]
        if shared_metadata and source["source_id"] == "C":
            source["outputs"]["metadata"]["path"] = "data/metadata/A"
    source_path.write_text(yaml.safe_dump(document, sort_keys=False))
    profile = root / "conf/environments/example.yaml"
    document = yaml.safe_load(profile.read_text())
    document["runtime"]["log_level"] = "INFO"
    document["spark"]["ivy_dir"] = str(REPO_ROOT / "data/metadata/ivy")
    document["spark"]["config"]["spark.sql.catalog.janus.jdbc.init-catalog-tables"] = "false"
    profile.write_text(yaml.safe_dump(document))
    config = load_environment_config("example", root)
    paths = materialize_runtime_paths(config, root)
    paths[ICEBERG_CATALOG_DB_PATH_KEY].parent.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(
        derive_pyiceberg_catalog_name(config),
        **(derive_pyiceberg_catalog_properties(config, paths) | {"init_catalog_tables": "true"}),
    )
    catalog.create_namespace_if_not_exists("orchestration_example")
    catalog.engine.dispose()
    return root


def run_example(root, *, max_parallel):
    """Execute and drain subprocess pipes together, keeping the actual pid and evidence."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("JANUS_")}
    environment.update(
        JANUS_SPARK_DRIVER_MEMORY="1g",
        PYTHONPATH=os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))),
    )
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
            f"task15-{uuid4().hex}",
            "--max-parallel",
            str(max_parallel),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=10)
        pytest.fail("example CLI exceeded bounded integration timeout")
    assert process.returncode in {0, 1}, stderr
    assert stdout.strip(), stderr
    summary = json.loads(stdout)
    stored = root / "runtime/data/metadata/pipelines" / summary["pipeline"]["pipeline_run_id"]
    assert json.loads((stored / "summary.json").read_text()) == summary
    (root / "cli-evidence.json").write_text(
        json.dumps(
            {
                "pid": process.pid,
                "exit_code": process.returncode,
                "summary": summary,
                "events": parse_log_events(stderr),
                "driver_memory": "1g",
                "page_delay_seconds": PAGE_DELAY_SECONDS,
                "spawn_cost_seconds": SPAWN_COST_SECONDS,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (root / "cli-stderr.log").write_text(stderr)
    return summary, parse_log_events(stderr), process


def bronze_rows(root):
    """Reuse the dependency suite's Spark reader for every business column."""
    from janus.utils.environment import build_spark_options

    config = load_environment_config("example", root)
    paths = materialize_runtime_paths(config, root)
    options = build_spark_options(config, paths)
    options.pop("spark.jars.packages", None)
    options["spark.jars"] = ",".join(
        str(path) for path in sorted((REPO_ROOT / "data/metadata/ivy/jars").glob("*.jar"))
    )
    rows = {
        source: _table_rows(
            lambda: start_session("janus-bronze-reader", options),
            f"janus.orchestration_example.{BRONZE_TABLES[source]}",
            *columns,
        )
        for source, columns in BRONZE_COLUMNS.items()
    }
    (root / "bronze-rows.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows


def write_parity_evidence(directory, sequential, parallel, **fields):
    from tests.support.orchestration import normalized_terminal_contract

    (directory / "parity-evidence.json").write_text(
        json.dumps(
            {
                "sequential": normalized_terminal_contract(sequential),
                "parallel": normalized_terminal_contract(parallel),
                "intervals": {
                    source["source_id"]: source["timing"] for source in parallel["sources"]
                },
                **fields,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def run_cli(argv):
    from janus.cli import run_all
    from janus.main import main
    from janus.runtime.parallel import build_batch_executor

    original = run_all._build_batch_executor
    if "--max-parallel" not in run_all.build_parser()._option_string_actions:
        argv = list(argv)
        index = argv.index("--max-parallel")
        max_parallel = int(argv[index + 1])
        del argv[index : index + 2]

        def executor(logger):
            logger.context["pid"] = os.getpid()
            if max_parallel == 1:
                return original(logger)
            return build_batch_executor(max_parallel=max_parallel, logger=logger)

        run_all._build_batch_executor = executor
    else:

        def executor(logger, *args, **kwargs):
            logger.context["pid"] = os.getpid()
            return original(logger, *args, **kwargs)

        run_all._build_batch_executor = executor
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
