"""Entry-point coverage for the one-shot run-all command."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import janus.cli.run_all as run_all_module
from janus.main import main
from janus.orchestration import PipelineSummaryStore
from janus.runtime import BatchExecutor as CoreBatchExecutor
from tests.support.orchestration import build_graph_project

STARTED_AT = "2026-09-13T12:00:00Z"
PIPELINE_ID = "cli-batch"


class FakeExecutedRun:
    """The established execution result surface without HTTP or compute."""

    def __init__(self, planned_run: Any, status: str = "succeeded") -> None:
        self.planned_run = planned_run
        self.status = status

    @property
    def is_successful(self) -> bool:
        return self.status == "succeeded"

    def to_summary(self) -> dict[str, Any]:
        source_id = self.planned_run.plan.source.source_id
        summary: dict[str, Any] = {
            "status": self.status,
            "strategy_metadata": {"source": source_id},
            "materialized_outputs": [],
            "metadata_outputs": {
                "run_metadata_path": f"/metadata/runs/{source_id}.json",
                "lineage_path": f"/metadata/lineage/{source_id}.json",
                "checkpoint_state_path": None,
                "checkpoint_history_path": None,
                "validation_report_path": None,
            },
        }
        if not self.is_successful:
            summary.update(
                {
                    "failure_reason": f"{source_id} returned a failed extraction",
                    "error_type": "ExtractionFailure",
                }
            )
        return summary


@dataclass
class ExecutionSpy:
    failures: set[str] = field(default_factory=set)
    interrupts: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    planned_runs: list[Any] = field(default_factory=list)

    def execute(self, planned_run, environment_config, resolved_paths):
        source_id = planned_run.plan.source.source_id
        assert environment_config["name"] == "local"
        assert "metadata_dir" in resolved_paths
        self.calls.append(source_id)
        self.planned_runs.append(planned_run)
        if source_id in self.interrupts:
            raise KeyboardInterrupt
        status = "failed" if source_id in self.failures else "succeeded"
        return FakeExecutedRun(planned_run, status)


def _project(tmp_path: Path, graph: str = "chain_and_peer") -> Path:
    root = build_graph_project(tmp_path, graph)
    environments = root / "conf" / "environments"
    environments.mkdir(parents=True)
    (environments / "local.yaml").write_text(
        """
name: local
runtime:
  log_level: INFO
spark:
  app_name: janus-run-all-test
  master: local[1]
  warehouse_dir: data/metadata/spark-warehouse
  config: {}
storage:
  root_dir: data
  raw_dir: data/raw
  bronze_dir: data/bronze
  metadata_dir: data/metadata
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _arguments(root: Path, *extra: str, pipeline_id: str = PIPELINE_ID) -> list[str]:
    return [
        "run-all",
        "--project-root",
        str(root),
        "--pipeline-run-id",
        pipeline_id,
        "--started-at",
        STARTED_AT,
        *extra,
    ]


def _install_runner(monkeypatch, execution: ExecutionSpy, **kwargs: Any) -> CoreBatchExecutor:
    runner = CoreBatchExecutor(source_execution=execution, **kwargs)
    monkeypatch.setattr(run_all_module, "_build_batch_executor", lambda _logger: runner)
    return runner


@pytest.mark.parametrize(
    ("filters", "root_ids", "source_ids", "included_upstreams"),
    (
        ((), ["A", "B", "C", "D"], ["A", "B", "C", "D"], []),
        (
            ("--tag", "terminal", "--tag", "independent", "--tag", "terminal"),
            ["C", "D"],
            ["A", "B", "C", "D"],
            ["A", "B"],
        ),
        (
            ("--domain", "reporting"),
            ["B", "D"],
            ["A", "B", "D"],
            ["A"],
        ),
    ),
)
def test_selection_executes_in_order_and_explains_upstream_closure(
    tmp_path,
    monkeypatch,
    capsys,
    filters,
    root_ids,
    source_ids,
    included_upstreams,
):
    root = _project(tmp_path)
    execution = ExecutionSpy()
    _install_runner(monkeypatch, execution)

    exit_code = main(_arguments(root, *filters))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert execution.calls == source_ids
    assert payload["selection"] == {
        "requested": {
            "tags": sorted(set(filters[1::2])) if "--tag" in filters else [],
            "domains": sorted(set(filters[1::2])) if "--domain" in filters else [],
        },
        "root_ids": root_ids,
        "included_upstream_ids": included_upstreams,
        "source_ids": source_ids,
    }
    assert payload["totals"]["status"] == "succeeded"
    summary_path = Path(payload["summary_persistence"]["path"])
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload


def test_a_failed_source_skips_descendants_but_independent_work_continues(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _project(tmp_path)
    execution = ExecutionSpy(failures={"A"})
    _install_runner(monkeypatch, execution)

    exit_code = main(_arguments(root))
    payload = json.loads(capsys.readouterr().out)
    sources = {source["source_id"]: source for source in payload["sources"]}

    assert exit_code == 1
    assert execution.calls == ["A", "C"]
    assert {source_id: source["status"] for source_id, source in sources.items()} == {
        "A": "failed",
        "B": "skipped",
        "C": "succeeded",
        "D": "skipped",
    }
    assert sources["B"]["skip"]["root_failed_source_ids"] == ["A"]
    assert sources["D"]["skip"]["root_failed_source_ids"] == ["A"]
    assert payload["summary_persistence"]["status"] == "succeeded"
    assert payload["totals"]["status"] == "failed"


def test_identity_timestamp_and_resume_reach_each_source_plan(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _project(tmp_path, "chain")
    execution = ExecutionSpy()
    _install_runner(monkeypatch, execution)

    exit_code = main(
        _arguments(
            root,
            "--domain",
            "reporting",
            "--resume",
            pipeline_id="demo-20260913",
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["pipeline"]["pipeline_run_id"] == "demo-20260913"
    assert payload["pipeline"]["planned_at"] == "2026-09-13T12:00:00+00:00"
    assert execution.calls == ["A", "B"]
    for planned_run in execution.planned_runs:
        context = planned_run.plan.run_context
        assert context.started_at.isoformat() == "2026-09-13T12:00:00+00:00"
        assert context.run_id != "demo-20260913"
        attributes = context.attributes_as_dict()
        assert {
            key: attributes[key]
            for key in ("pipeline_attempt", "pipeline_run_id", "resume", "trigger")
        } == {
            "pipeline_attempt": "1",
            "pipeline_run_id": "demo-20260913",
            "resume": "true",
            "trigger": "run-all",
        }


@pytest.mark.parametrize(
    ("graph", "message"),
    (
        ("cycle", "cycle"),
        ("missing_producer", "missing from the registry"),
        ("disabled_producer", "disabled"),
    ),
)
def test_invalid_graphs_return_two_before_runner_construction(
    tmp_path,
    monkeypatch,
    capsys,
    graph,
    message,
):
    root = _project(tmp_path, graph)

    def fail_if_built(_logger):
        raise AssertionError("the runner must not be constructed for an invalid graph")

    monkeypatch.setattr(run_all_module, "_build_batch_executor", fail_if_built)

    exit_code = main(_arguments(root))
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert message in captured.err.lower()


def test_no_matching_selection_returns_two_without_execution(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    execution = ExecutionSpy()

    def fail_if_built(_logger):
        raise AssertionError("the runner must not be constructed for an empty selection")

    monkeypatch.setattr(run_all_module, "_build_batch_executor", fail_if_built)

    exit_code = main(_arguments(root, "--tag", "absent"))
    captured = capsys.readouterr()

    assert exit_code == 2
    assert execution.calls == []
    assert captured.out == ""
    assert "No enabled source matches tag in (absent)" in captured.err


def test_summary_persistence_failure_prints_the_complete_failed_aggregate(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _project(tmp_path, "independent")
    execution = ExecutionSpy()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("atomic replace unavailable")

    _install_runner(
        monkeypatch,
        execution,
        summary_store_factory=lambda layout: PipelineSummaryStore(
            layout,
            atomic_replace=fail_replace,
        ),
    )

    exit_code = main(_arguments(root))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert execution.calls == ["A", "C"]
    assert payload["totals"]["status"] == "failed"
    assert payload["summary_persistence"]["status"] == "failed"
    assert payload["summary_persistence"]["path"].endswith(
        "/data/metadata/pipelines/cli-batch/summary.json"
    )
    assert "Could not persist pipeline summary" in captured.err


def test_unexpected_batch_operation_returns_one_without_claiming_completion(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _project(tmp_path, "independent")

    class BrokenRunner:
        def execute(self, *_args):
            raise OSError("metadata storage unavailable")

    monkeypatch.setattr(run_all_module, "_build_batch_executor", lambda _logger: BrokenRunner())

    exit_code = main(_arguments(root))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Batch execution failed: metadata storage unavailable" in captured.err


def test_interruption_uses_shell_interrupt_status_and_no_completed_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _project(tmp_path, "chain")
    execution = ExecutionSpy(interrupts={"A"})
    _install_runner(monkeypatch, execution)

    exit_code = main(_arguments(root))
    captured = capsys.readouterr()

    assert exit_code == 130
    assert execution.calls == ["A"]
    assert captured.out == ""
    assert "was interrupted" in captured.err
    assert "pending sources: 2" in captured.err


@pytest.mark.parametrize(
    "arguments",
    (
        ("--source-id", "A"),
        ("--execute",),
        ("--run-id", "source-run"),
        ("--include-disabled",),
        ("--bronze-table", "bronze.a"),
        ("--ingest-raw-to-bronze",),
        ("--with-spark",),
    ),
)
def test_single_source_only_flags_are_explicitly_rejected(arguments, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run-all", *arguments])

    assert exc_info.value.code == 2
    assert "cannot be used with run-all" in capsys.readouterr().err


def test_tag_and_domain_filters_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run-all", "--tag", "reference", "--domain", "reporting"])

    assert exc_info.value.code == 2
    assert "not allowed with argument --tag" in capsys.readouterr().err


def test_shared_options_must_follow_run_all(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--environment", "local", "run-all"])

    assert exc_info.value.code == 2
    assert "unrecognized arguments: run-all" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", (["--help"], ["run-all", "--help"]))
def test_help_paths_do_not_load_configuration_or_compute(monkeypatch, capsys, arguments):
    monkeypatch.setattr(
        run_all_module,
        "load_environment_config",
        lambda *_args: pytest.fail("help must not load an environment"),
    )
    monkeypatch.setattr(
        run_all_module,
        "_build_batch_executor",
        lambda _logger: pytest.fail("help must not construct a runner"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(arguments)

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    if arguments[0] == "run-all":
        assert "required upstreams are included automatically" in help_text
        assert "does not schedule future runs" in help_text
    else:
        assert "janus run-all [options]" in help_text
