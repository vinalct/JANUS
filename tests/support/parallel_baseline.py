from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatchcase
from hashlib import sha256
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from janus.hooks import built_in_hooks
from janus.lineage import MetadataZonePaths, RunObserver
from janus.models import ExecutionPlan, ExtractionResult, RunContext, SourceConfig, WriteResult
from janus.orchestration import PipelineClock
from janus.runtime import BatchExecutor, ExecutedRun
from janus.strategies.api import ApiHook
from janus.utils.environment import load_environment_config
from janus.utils.storage import StorageLayout, bronze_table_identifier
from tests.support.orchestration import (
    GRAPH_CASES,
    GraphCase,
    SourceSpec,
    source_documents,
    write_project,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = Path("/tmp/janus-parallel-baseline")
PIPELINE_ID = "parallel-baseline"
STARTED_AT = datetime(2026, 1, 1, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(seconds=5)
CHAIN = GraphCase((SourceSpec("A"), SourceSpec("B", ("A",)), SourceSpec("C")))
CASES = {
    "independent_three": GraphCase(tuple(SourceSpec(s) for s in ("A", "B", "C"))),
    "chain_and_peer": CHAIN,
    "upstream_fails": CHAIN,
    "upstream_raises": CHAIN,
    "planning_failure": CHAIN,
    "diamond": GRAPH_CASES["diamond"],
    "diamond_one_arm_fails": GRAPH_CASES["diamond"],
    "shared_metadata_dir": GraphCase((SourceSpec("A"), SourceSpec("C"))),
}
EXPECTED_EXITS = dict.fromkeys(CASES, 0) | dict.fromkeys(
    ("upstream_fails", "upstream_raises", "planning_failure", "diamond_one_arm_fails"), 1
)

VOLATILE_FILES = {
    "logs/*.log": "Structured log files contain real emission timestamps.",
    "lineage/openlineage/*.ndjson": "OpenLineage day files contain real eventTime values.",
}


class FailingPlanHook(ApiHook):
    def on_plan(self, plan):
        raise RuntimeError("scripted planning failure")


class CannedSourceExecution:
    """SourceExecution test double; the observer still writes its real contracts."""

    def __init__(self, case):
        self.case = case

    def execute(self, planned_run, environment_config, resolved_paths):
        plan = planned_run.plan
        source_id = plan.source.source_id
        failure = (source_id == "A" and self.case in {"upstream_fails", "upstream_raises"}) or (
            source_id == "B" and self.case == "diamond_one_arm_fails"
        )
        observer = RunObserver()
        observer.start_run(plan)
        extraction = ExtractionResult.from_plan(plan, (), records_extracted=1, checkpoint_value="1")
        error = RuntimeError("scripted execution failure") if failure else None
        writes = (
            ()
            if failure
            else (
                WriteResult.from_plan(
                    plan,
                    "bronze",
                    path=bronze_table_identifier(plan.bronze_output.path, fallback_name=source_id),
                    format_name="iceberg",
                    mode="overwrite",
                    records_written=1,
                ),
            )
        )
        if failure:
            persisted = observer.record_failure(plan, error, extraction, finished_at=FINISHED_AT)
        else:
            persisted = observer.record_success(plan, extraction, writes, finished_at=FINISHED_AT)
        if self.case == "upstream_raises" and source_id == "A":
            raise error
        checkpoint = persisted.checkpoint_result
        return ExecutedRun(
            planned_run=planned_run,
            status="failed" if failure else "succeeded",
            extraction_result=extraction,
            write_results=writes,
            run_metadata_path=persisted.run_metadata_path,
            lineage_path=persisted.lineage_path,
            checkpoint_state_path=checkpoint.current_path if checkpoint else None,
            checkpoint_history_path=checkpoint.history_path if checkpoint else None,
            failure_reason=str(error) if error else None,
            error_type=type(error).__name__ if error else None,
            duration_budget_outcome=persisted.run_metadata.duration_budget_outcome,
        )


def write_capture_project(case, root):
    documents = source_documents(CASES[case])
    for doc in documents:
        doc["extraction"].update(checkpoint_strategy="max_value", checkpoint_field="id")
        if case == "planning_failure" and doc["source_id"] == "A":
            doc["source_hook"] = "parallel_baseline.fail_plan"
        if case == "shared_metadata_dir":
            doc["outputs"]["metadata"]["path"] = "data/metadata/shared"
            doc["extraction"].update(checkpoint_strategy="none")
    write_project(root, documents)
    profiles = root / "conf/environments"
    profiles.mkdir()
    (profiles / "local.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "local",
                "runtime": {"log_level": "INFO"},
                "spark": {
                    "app_name": "parallel-baseline",
                    "master": "local[1]",
                    "warehouse_dir": "data/metadata/spark-warehouse",
                },
                "storage": {
                    "root_dir": "data",
                    **{f"{zone}_dir": f"data/{zone}" for zone in ("raw", "bronze", "metadata")},
                },
                "observability": {"metrics": {"transport": "disabled"}},
            }
        ),
        encoding="utf-8",
    )


def launch(case, argv):
    from janus.cli import dispatch, run_all

    ticks = count()
    clock = PipelineClock(
        wall_clock=lambda: STARTED_AT + timedelta(seconds=next(ticks)),
        monotonic_clock=lambda: float(next(ticks)),
    )
    hooks = (*built_in_hooks(), ("parallel_baseline.fail_plan", FailingPlanHook()))
    # The hook catalog is a fixture input, before planning. No planner, scheduler,
    # attempt translator, persistence method, or worker implementation is patched.
    with (
        patch("janus.hooks.built_in_hooks", return_value=hooks),
        patch.object(
            run_all,
            "_build_batch_executor",
            side_effect=lambda logger: sequential_launcher(case, clock, logger),
        ),
    ):
        return dispatch.main(argv)


def sequential_launcher(case, clock, logger):
    from janus.runtime.parallel import build_batch_executor

    runner = build_batch_executor(max_parallel=1, logger=logger)
    assert type(runner) is BatchExecutor
    runner.source_execution = CannedSourceExecution(case)
    runner.clock = clock
    runner.logger = logger
    return runner


def metadata_digests(root):
    """Hash every nonvolatile metadata file exactly as persisted."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if not any(fnmatchcase(relative, pattern) for pattern in VOLATILE_FILES):
                result[relative] = sha256(path.read_bytes()).hexdigest()
    return result


def capture_run_all(case: str, root: Path) -> dict:
    """Run an isolated real CLI subprocess; root must not already exist."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_capture_project(case, root)
    env = {key: value for key, value in os.environ.items() if not key.startswith("JANUS_")}
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.support.parallel_baseline",
            "launch",
            case,
            "--",
            "run-all",
            "--project-root",
            str(root),
            "--environment",
            "local",
            "--pipeline-run-id",
            PIPELINE_ID,
            "--started-at",
            STARTED_AT.isoformat(),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )
    events = []
    for line in completed.stderr.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"{case}: unstructured stderr: {completed.stderr}") from error
        events.append(record["event"])
    summary = root / "data/metadata/pipelines" / PIPELINE_ID / "summary.json"
    if completed.returncode != EXPECTED_EXITS[case] or not summary.is_file():
        raise AssertionError(f"{case}: exit {completed.returncode}; {completed.stderr}")
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr_events": events,
        "summary_json": summary.read_text(encoding="utf-8"),
        "metadata_zone": metadata_digests(root / "data/metadata"),
    }


def capture_goldens(output):
    output.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        root = CAPTURE_ROOT / case
        if root.exists():
            raise FileExistsError(f"Capture requires a clean scratch root: {root}")
        try:
            captured = capture_run_all(case, root)
            directory = output / case
            directory.mkdir()
            (directory / "stdout.txt").write_text(captured.pop("stdout"), encoding="utf-8")
            (directory / "summary.json").write_text(captured.pop("summary_json"), encoding="utf-8")
            (directory / "capture.json").write_text(
                json.dumps(captured, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        finally:
            if root.exists():
                shutil.rmtree(root)  # Only this invocation's newly created scratch project.


def metadata_resolutions(sources, root, config):
    """Use both production derivations, including disabled/invalid-graph sources."""
    layout = StorageLayout.from_environment_config(config, root)
    rows = []
    for source in sources:
        context = RunContext.create(
            "parallel-inventory", config["name"], root, started_at=STARTED_AT
        )
        plan = ExecutionPlan.from_source_config(source, context)
        executor = layout.resolve_output(plan, "metadata").resolved_path
        operator = MetadataZonePaths.from_plan(plan).base_dir

        def display(path):
            return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

        rows.append(
            {
                "source_id": source.source_id,
                "enabled": source.enabled,
                "executor": display(executor),
                "operator": display(operator),
                "agrees": executor == operator,
            }
        )
    return sorted(rows, key=lambda row: row["source_id"])


def directory_collisions(rows):
    shared, nested = [], []
    by_directory = {}
    for row in rows:
        by_directory.setdefault(row["executor"], []).append(row["source_id"])
    for directory, sources in sorted(by_directory.items()):
        if len(sources) > 1:
            shared.append({"directory": directory, "sources": sorted(sources)})
        for child, children in sorted(by_directory.items()):
            if child != directory and Path(child).is_relative_to(directory):
                nested.append(
                    {
                        "ancestor": directory,
                        "ancestor_sources": sorted(sources),
                        "descendant": child,
                        "descendant_sources": sorted(children),
                    }
                )
    return {"shared": shared, "nested": nested}


def _sources(root):
    sources = []
    for path in sorted((root / "conf/sources").rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        sources.extend(
            SourceConfig.from_mapping(item, path) for item in document.get("sources", [document])
        )
    return sources


def inventory(project_root):
    """Read only: no registry validation or Spark needed for path resolution."""
    result = {}
    with TemporaryDirectory(prefix="janus-parallel-inventory-") as temporary:
        projects = {"checked_in": project_root, "example": project_root / "examples/orchestration"}
        for name, case in GRAPH_CASES.items():
            for grouped in (True, False):
                key = f"fixture:{name}:{'grouped' if grouped else 'individual'}"
                projects[key] = write_project(
                    Path(temporary) / key.replace(":", "-"),
                    source_documents(case),
                    grouped=grouped,
                )
        for name in CASES:
            key = f"baseline:{name}"
            root = Path(temporary) / key.replace(":", "-")
            write_capture_project(name, root)
            projects[key] = root
        for name, root in projects.items():
            result[name] = {}
            for profile in ("local", "cluster"):
                with patch.dict(os.environ, {}, clear=True):
                    config = load_environment_config(profile, project_root)
                    rows = metadata_resolutions(_sources(root), root, config)
                    with patch.dict(os.environ, {"JANUS_METADATA_DIR": "/tmp/elsewhere"}):
                        override = load_environment_config(profile, project_root)
                        custom = metadata_resolutions(_sources(root), root, override)
                result[name][profile] = {
                    "rows": rows,
                    "collisions": directory_collisions(rows),
                    "custom_metadata_rows": custom,
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Explicitly generate eight golden directories")
    capture.add_argument("output", type=Path)
    launch_parser = commands.add_parser("launch", help="Internal CLI subprocess launcher")
    launch_parser.add_argument("case", choices=CASES)
    launch_parser.add_argument("argv", nargs=argparse.REMAINDER)
    inventory_parser = commands.add_parser(
        "inventory", help="Resolve every source under both profiles"
    )
    inventory_parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    for name in ("ivy", "orphan-jvm", "spawn-cost", "environment"):
        subparser = commands.add_parser(name, help="Explicit container-only experiment")
        subparser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if args.command == "launch":
        return launch(args.case, args.argv[1:] if args.argv[:1] == ["--"] else args.argv)
    if args.command == "capture":
        capture_goldens(args.output.resolve())
        return 0
    if args.command == "inventory":
        result = inventory(args.project_root.resolve())
    else:
        # A minimal -c parent prevents spawn from reimporting this capture module
        # (and its Janus/fixture imports) during the bare-spawn measurement.
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; "
                "from tests.support.parallel_baseline_experiments import run_experiment; "
                "print(json.dumps(run_experiment(sys.argv[1],Path(sys.argv[2])),"
                "indent=2,sort_keys=True))",
                args.command,
                str(args.project_root.resolve()),
            ],
            check=False,
        ).returncode
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
