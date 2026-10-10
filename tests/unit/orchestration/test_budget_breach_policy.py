import json

import pytest

from janus.cli.run_all import build_parser
from janus.main import main
from tests.support.metrics_order23 import exit_code
from tests.unit.cli.test_run_all_cli import (
    ExecutionSpy,
    FakeExecutedRun,
    _arguments,
    _install_runner,
    _project,
)

pytestmark = pytest.mark.xfail(
    strict=True, reason="opt-in duration budget breach policy is absent"
)


class BudgetExecution(ExecutionSpy):
    def __init__(self, outcome):
        super().__init__()
        self.budget_outcome = outcome

    def execute(self, planned_run, environment_config, resolved_paths):
        super().execute(planned_run, environment_config, resolved_paths)
        result = FakeExecutedRun(planned_run)
        result.duration_budget_outcome = (
            self.budget_outcome if planned_run.plan.source.source_id == "A" else "within"
        )
        original = result.to_summary
        result.to_summary = lambda: {
            **original(),
            "duration_budget_outcome": result.duration_budget_outcome,
        }
        return result


def _run(tmp_path, monkeypatch, capsys, *, outcome="exceeded", fail=False):
    assert "--fail-on-budget-breach" in build_parser().format_help()
    root = _project(tmp_path)
    execution = BudgetExecution(outcome)
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
