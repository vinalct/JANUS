import json
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from janus.cli.run_all import build_parser
from janus.lineage import RunObserver
from janus.main import main
from janus.models import ExtractionResult
from janus.observability import GuardedRunEventEmitter, IcebergAppendOutcome, IcebergAppendResult
from janus.orchestration import BatchPlanRequest
from janus.runtime.batch import SourceCleanupError
from janus.runtime.executor import ExecutedRun
from tests.support.metrics_order23 import exit_code
from tests.unit.cli.test_run_all_cli import (
    ExecutionSpy,
    _arguments,
    _install_runner,
    _project,
)


class BudgetExecution(ExecutionSpy):
    def __init__(self, outcome, *, failures=(), cleanup_failure=False):
        super().__init__(failures=set(failures))
        self.budget_outcome = outcome
        self.cleanup_failure = cleanup_failure
        self.returned = []

    def execute(self, planned_run, environment_config, resolved_paths):
        result = super().execute(planned_run, environment_config, resolved_paths)
        source_id = planned_run.plan.source.source_id
        executed = ExecutedRun(
            planned_run=planned_run,
            status=result.status,
            duration_budget_outcome=self.budget_outcome if source_id == "A" else "undeclared",
            failure_reason=f"{source_id} returned a failed extraction"
            if not result.is_successful
            else None,
            error_type="ExtractionFailure" if not result.is_successful else None,
        )
        self.returned.append(executed)
        if self.cleanup_failure and source_id == "A":
            raise SourceCleanupError(RuntimeError("cleanup failed"), executed)
        return executed


def _declare_budget(root, budget):
    source_path = root / "conf/sources/sources.yaml"
    document = yaml.safe_load(source_path.read_text())
    for source in document["sources"]:
        if source["source_id"] == "A" and budget is not None:
            source["operations"] = {"max_duration_seconds": budget}
    source_path.write_text(yaml.safe_dump(document))


def _run(tmp_path, monkeypatch, capsys, *, outcome="exceeded", fail=False, **kwargs):
    assert "--fail-on-budget-breach" in build_parser().format_help()
    root = _project(tmp_path)
    _declare_budget(root, None if outcome == "undeclared" else 4)
    execution = BudgetExecution(outcome, **kwargs)
    _install_runner(monkeypatch, execution)
    argv = _arguments(root, *(["--fail-on-budget-breach"] if fail else []))
    code = exit_code(lambda: main(argv))
    summary = json.loads(capsys.readouterr().out)
    return code, summary, execution


def test_default_exceeded_run_succeeds_and_dependents_execute(tmp_path, monkeypatch, capsys):
    code, summary, execution = _run(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert execution.calls == ["A", "B", "C", "D"]
    sources = {source["source_id"]: source for source in summary["sources"]}
    assert sources
    assert all(source["status"] == "succeeded" for source in sources.values())
    assert sources["A"]["attempts"][0]["evidence"]["duration_budget_outcome"] == "exceeded"


def test_opt_in_fails_the_attempt_and_skips_only_dependents_with_aggregate(
    tmp_path, monkeypatch, capsys
):
    code, summary, execution = _run(tmp_path, monkeypatch, capsys, fail=True)
    assert code == 1
    assert execution.calls == ["A", "C"]
    sources = {source["source_id"]: source for source in summary["sources"]}
    assert sources["A"]["status"] == "failed"
    assert sources["A"]["attempts"][0]["failure"]["phase"] == "budget"
    assert sources["A"]["attempts"][0]["failure"]["error_type"] == "RunDurationBudgetExceeded"
    assert sources["A"]["attempts"][0]["failure"]["reason"] == (
        "Run exceeded its declared duration budget of 4s"
    )
    assert sources["A"]["attempts"][0]["evidence"]["status"] == "succeeded"
    assert sources["A"]["attempts"][0]["evidence"] == execution.returned[0].to_summary()
    assert sources["C"]["status"] == "succeeded"
    for source, upstream in (("B", "A"), ("D", "B")):
        assert sources[source]["status"] == "skipped"
        assert sources[source]["skip"]["reason_code"] == "upstream_failed"
        assert sources[source]["skip"]["direct_blocking_upstream_ids"] == [upstream]
        assert sources[source]["skip"]["root_failed_source_ids"] == ["A"]
    assert summary["totals"]["failed"] == 1
    assert summary["totals"]["skipped"] == 2


@pytest.mark.parametrize("outcome", ["within", "undeclared"])
def test_within_and_undeclared_are_unchanged_by_opt_in(outcome, tmp_path, monkeypatch, capsys):
    code, summary, execution = _run(tmp_path, monkeypatch, capsys, outcome=outcome, fail=True)
    assert code == 0
    assert execution.calls == ["A", "B", "C", "D"]
    assert summary["totals"]["failed"] == summary["totals"]["skipped"] == 0


def test_single_source_entry_point_rejects_the_batch_flag_by_name(capsys):
    assert "--fail-on-budget-breach" in build_parser().format_help()
    assert (
        exit_code(lambda: main(["run", "--source-id", "A", "--execute", "--fail-on-budget-breach"]))
        == 2
    )
    assert "--fail-on-budget-breach" in capsys.readouterr().err


def test_implicit_single_source_entry_point_rejects_the_batch_flag_by_name(capsys):
    assert (
        exit_code(lambda: main(["--source-id", "A", "--execute", "--fail-on-budget-breach"])) == 2
    )
    assert "--fail-on-budget-breach" in capsys.readouterr().err


@pytest.mark.parametrize("cleanup_failure", [False, True], ids=["execution", "cleanup"])
def test_failed_exceeded_run_keeps_its_original_failure(
    tmp_path, monkeypatch, capsys, cleanup_failure
):
    code, summary, execution = _run(
        tmp_path,
        monkeypatch,
        capsys,
        fail=True,
        failures=() if cleanup_failure else ("A",),
        cleanup_failure=cleanup_failure,
    )
    attempt = summary["sources"][0]["attempts"][0]
    assert code == 1
    assert execution.calls == ["A", "C"]
    assert attempt["evidence"]["duration_budget_outcome"] == "exceeded"
    assert attempt["failure"]["phase"] == ("cleanup" if cleanup_failure else "execution")
    assert attempt["failure"]["error_type"] == (
        "RuntimeError" if cleanup_failure else "ExtractionFailure"
    )


def test_missing_budget_outcome_is_unchanged_by_opt_in(tmp_path, monkeypatch, capsys):
    code, summary, execution = _run(tmp_path, monkeypatch, capsys, outcome=None, fail=True)
    assert code == 0
    assert execution.calls == ["A", "B", "C", "D"]
    assert summary["totals"]["failed"] == summary["totals"]["skipped"] == 0


def test_budget_policy_is_a_frozen_request_fact(tmp_path):
    request = BatchPlanRequest.create(environment="local", project_root=tmp_path)
    assert request.fail_on_budget_breach is False
    flagged = replace(request, fail_on_budget_breach=True)
    assert flagged.fail_on_budget_breach is True
    with pytest.raises(FrozenInstanceError):
        flagged.fail_on_budget_breach = False


class ObservedBudgetExecution(ExecutionSpy):
    """Persist through the real observer and count the guarded runs-table append."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.persisted = []
        self.artifact_bytes = {}
        self.terminal_calls = []

    def execute(self, planned_run, environment_config, resolved_paths):
        super().execute(planned_run, environment_config, resolved_paths)
        plan = planned_run.plan

        def count_append(record, *_args, **_kwargs):
            self.rows.append(record)
            return IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")

        emitter = GuardedRunEventEmitter(
            environment_config, resolved_paths, runs_table_sink=count_append
        )
        terminal_calls = self.terminal_calls

        class CountingObserver(RunObserver):
            def record_success(self, *args, **kwargs):
                terminal_calls.append(plan.source.source_id)
                return super().record_success(*args, **kwargs)

        observer = CountingObserver(emitter=emitter)
        observer.start_run(plan)
        persisted = observer.record_success(
            plan,
            ExtractionResult.from_plan(plan, ()),
            finished_at=plan.run_context.started_at + timedelta(seconds=5),
        )
        self.persisted.append(persisted)
        for path in (persisted.run_metadata_path, persisted.lineage_path):
            self.artifact_bytes[path] = path.read_bytes()
        return ExecutedRun(
            planned_run=planned_run,
            status="succeeded",
            run_metadata_path=persisted.run_metadata_path,
            lineage_path=persisted.lineage_path,
            duration_budget_outcome=persisted.run_metadata.duration_budget_outcome,
        )


@pytest.mark.parametrize("fail", [False, True], ids=["default", "flagged"])
def test_persisted_success_and_one_emission_survive_batch_policy(
    tmp_path, monkeypatch, capsys, fail
):
    root = _project(tmp_path)
    _declare_budget(root, 4)
    execution = ObservedBudgetExecution()
    _install_runner(monkeypatch, execution)
    code = main(_arguments(root, *(["--fail-on-budget-breach"] if fail else [])))
    summary = json.loads(capsys.readouterr().out)
    expected = ["A", "C"] if fail else ["A", "B", "C", "D"]
    assert code == (1 if fail else 0)
    assert execution.calls == execution.terminal_calls == expected
    assert [row.source_id for row in execution.rows] == expected
    assert all(row.status == "succeeded" for row in execution.rows)
    assert execution.rows[0].duration_budget_outcome == "exceeded"
    assert summary["sources"][0]["status"] == ("failed" if fail else "succeeded")
    for path, content in execution.artifact_bytes.items():
        assert path.read_bytes() == content
        assert json.loads(content)["status"] == "succeeded"
    persisted = execution.persisted[0]
    for path in (persisted.run_metadata_path, persisted.lineage_path):
        payload = json.loads(path.read_bytes())
        assert payload["duration_budget_outcome"] == "exceeded"
        assert payload["duration_budget_seconds"] == 4
    assert json.loads(Path(summary["summary_persistence"]["path"]).read_bytes()) == summary
