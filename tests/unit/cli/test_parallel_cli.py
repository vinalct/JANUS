"""Operator concurrency choices and warnings through the real batch CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import yaml

from janus.cli import run_all
from janus.main import main
from janus.orchestration import PipelineClock
from janus.runtime import BatchExecutor
from janus.runtime.parallel import ParallelBatchExecutor
from tests.unit.cli.test_run_all_cli import ExecutionSpy, _arguments, _project


def _profile(root, settings):
    path = root / "conf/environments/local.yaml"
    config = yaml.safe_load(path.read_text())
    if settings is not None:
        config["orchestration"] = settings
    path.write_text(yaml.safe_dump(config))


@pytest.mark.parametrize(
    "settings,flag,expected",
    [
        (None, (), 1),
        ({"max_parallel_default": 2, "lock_wait_seconds": 3}, (), 2),
        ({"max_parallel_default": 2, "lock_wait_seconds": 3}, ("--max-parallel", "1"), 1),
        ({"max_parallel_default": 1, "lock_wait_seconds": 3}, ("--max-parallel", "4"), 4),
    ],
)
def test_cli_flag_overrides_profile_and_selects_production_runner(
    tmp_path, monkeypatch, capsys, settings, flag, expected
):
    root = _project(tmp_path)
    _profile(root, settings)
    monkeypatch.setattr(run_all.os, "process_cpu_count", lambda: 8)
    actual_factory = run_all._build_batch_executor
    constructed = []
    execution = ExecutionSpy()

    def recording_factory(logger, **resolved):
        runner = actual_factory(logger, **resolved)
        constructed.append(runner)
        # Source evidence is canned; resolution and production composition remain real.
        return BatchExecutor(source_execution=execution)

    monkeypatch.setattr(run_all, "_build_batch_executor", recording_factory)
    assert main(_arguments(root, *flag)) == 0
    captured = capsys.readouterr()
    assert execution.calls == ["A", "B", "C", "D"]
    assert json.loads(captured.out)["totals"]["status"] == "succeeded"
    assert len(constructed) == 1
    runner = constructed[0]
    wait = 0 if settings is None else settings["lock_wait_seconds"]
    if expected == 1:
        assert type(runner) is BatchExecutor
        assert runner.source_execution.executor.lock_wait_seconds == wait
    else:
        assert type(runner) is ParallelBatchExecutor
        assert runner.max_parallel == expected
        assert runner.lock_wait_seconds == wait
    assert runner.logger is not None
    assert "max_parallel_exceeds_cpu_count" not in captured.err


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "private-value"])
def test_invalid_flag_refuses_without_loading_profile(monkeypatch, capsys, value):
    monkeypatch.setattr(
        run_all, "load_environment_config", lambda *args: pytest.fail("invalid flag loaded profile")
    )
    with pytest.raises(SystemExit) as error:
        main(["run-all", "--max-parallel", value])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "argument --max-parallel: must be an integer >= 1" in captured.err
    assert "private-value" not in captured.err


@pytest.mark.parametrize("prefix", [[], ["run"]])
@pytest.mark.parametrize("flag", [["--max-parallel", "2"], ["--max-parallel=2"]])
def test_single_source_rejects_parallel_flag_by_name(monkeypatch, capsys, prefix, flag):
    from janus.cli import run

    monkeypatch.setattr(
        run, "load_environment_config", lambda *args: pytest.fail("refused flag loaded profile")
    )
    with pytest.raises(SystemExit) as error:
        main([*prefix, "--source-id", "X", "--execute", *flag])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--max-parallel applies to run-all; a single source runs in this process" in captured.err


@pytest.mark.parametrize(
    "process_count,host_count,parallel,available,warns",
    [
        (2, 64, 3, 2, True),
        (2, 64, 2, 2, False),
        (None, 2, 3, 2, True),
        (None, None, 2, 1, True),
        (None, None, 1, 1, False),
    ],
)
def test_cpu_warning_once_with_fallbacks_and_unchanged_stdout(
    tmp_path, monkeypatch, capsys, process_count, host_count, parallel, available, warns
):
    root = _project(tmp_path)
    clock = PipelineClock(
        wall_clock=lambda: datetime(2026, 10, 12, tzinfo=UTC), monotonic_clock=lambda: 0.0
    )
    monkeypatch.setattr(
        run_all,
        "_build_batch_executor",
        lambda _logger, **_settings: BatchExecutor(source_execution=ExecutionSpy(), clock=clock),
    )
    monkeypatch.setattr(run_all.os, "process_cpu_count", lambda: process_count)
    monkeypatch.setattr(run_all.os, "cpu_count", lambda: host_count)
    assert main(_arguments(root)) == 0
    baseline = capsys.readouterr()
    assert "max_parallel_exceeds_cpu_count" not in baseline.err
    assert "warning: --max-parallel" not in baseline.err
    path = root / "data/metadata/pipelines/cli-batch/summary.json"
    baseline_summary = path.read_bytes()
    path.unlink()

    assert main(_arguments(root, "--max-parallel", str(parallel))) == 0
    captured = capsys.readouterr()
    assert captured.out == baseline.out
    assert path.read_bytes() == baseline_summary
    assert captured.err.count('"event": "max_parallel_exceeds_cpu_count"') == int(warns)
    warning = (
        f"warning: --max-parallel {parallel} exceeds the {available} CPUs available to this "
        "process; each worker may start its own Spark driver"
    )
    assert captured.err.count(warning) == int(warns)
    events = [json.loads(line) for line in captured.err.splitlines() if line.startswith("{")]
    cpu_events = [event for event in events if event["event"] == "max_parallel_exceeds_cpu_count"]
    if warns:
        assert cpu_events[0]["level"] == "WARNING"
        assert cpu_events[0]["fields"] == {
            "max_parallel": parallel,
            "cpu_count": available,
            "environment": "local",
            "pipeline_run_id": "cli-batch",
            "project_root": str(root),
        }


@pytest.mark.parametrize("explicit", [False, True])
def test_real_cli_runs_spawned_workers_and_reports_resolved_parallelism(
    tmp_path, monkeypatch, capsys, explicit
):
    from tests.support.operator_cli import arm_spark_tripwire
    from tests.support.parallel_execution import FAKE_FACTORY, graph_project

    root = graph_project(tmp_path)
    _profile(root, {"max_parallel_default": 2, "lock_wait_seconds": 3})
    arm_spark_tripwire(monkeypatch)
    actual_factory = run_all._build_batch_executor

    def offline_factory(logger, **settings):
        runner = actual_factory(logger, **settings)
        assert type(runner) is ParallelBatchExecutor
        # Workers use the existing engine-free source-evidence fixture.
        runner.execution_factory = FAKE_FACTORY
        return runner

    monkeypatch.setattr(run_all, "_build_batch_executor", offline_factory)
    flag = ("--max-parallel", "3") if explicit else ()
    assert main(_arguments(root, *flag)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pipeline"]["max_parallel"] == (3 if explicit else 2)
    assert {row["source_id"]: row["status"] for row in summary["sources"]} == {
        "A": "succeeded",
        "B": "succeeded",
        "C": "succeeded",
    }
    path = root / "data/metadata/pipelines/cli-batch/summary.json"
    assert json.loads(path.read_text()) == summary


@pytest.mark.parametrize("replay", [False, True])
def test_single_source_profile_wait_reaches_live_and_raw_replay_locks(
    tmp_path, monkeypatch, capsys, replay
):
    from contextlib import contextmanager

    from janus.runtime import executor
    from janus.scripts import raw_to_bronze
    from tests.support.operator_cli import arm_spark_tripwire
    from tests.unit.cli.test_run_all_cli import FakeExecutedRun

    root = _project(tmp_path)
    _profile(root, {"lock_wait_seconds": 7})
    arm_spark_tripwire(monkeypatch)
    module = raw_to_bronze if replay else executor
    original_lock = module.hold_source_state_lock
    waits = []

    @contextmanager
    def observed_lock(*args, **kwargs):
        waits.append(kwargs["wait_seconds"])
        with original_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(module, "hold_source_state_lock", observed_lock)
    if replay:
        monkeypatch.setattr(
            raw_to_bronze.RawToBronzeLoader,
            "_ingest_locked",
            lambda self, planned, *args, **kwargs: FakeExecutedRun(planned),
        )
        flag = ("--ingest-raw-to-bronze", "--bronze-table", "bronze.a")
    else:
        monkeypatch.setattr(
            executor.SourceExecutor,
            "_execute_locked",
            lambda self, planned, *args: FakeExecutedRun(planned),
        )
        flag = ("--execute",)
    assert main(["--project-root", str(root), "--source-id", "A", *flag]) == 0
    assert json.loads(capsys.readouterr().out)
    assert waits == [7]
