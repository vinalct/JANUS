"""Pure readiness, terminal-only skips, lock domains, and completion-order parity."""

import ast
import itertools
import random
from dataclasses import FrozenInstanceError, replace
from math import factorial

import pytest

from janus.orchestration.results import PipelineOutcome, SourceAttempt, SourceOutcome
from janus.orchestration.scheduling import NodeDecision, ReadyQueue, decide
from janus.orchestration.timing import ExecutionTiming
from janus.runtime import BatchExecutor
from tests.support.orchestration import GraphCase, SourceSpec
from tests.support.parallel_baseline import CASES, STARTED_AT, CannedSourceExecution
from tests.support.parallel_execution import (
    GRAPH_CASES,
    REPO_ROOT,
    normalized_terminal_contract,
    planned_project,
)

COMPLETION_CASES = GRAPH_CASES | {
    "independent_six": GraphCase(tuple(SourceSpec(f"S{i}") for i in range(6))),
    "dependent_ten": GraphCase(
        (
            SourceSpec("A"),
            SourceSpec("B", ("A",)),
            SourceSpec("C", ("A",)),
            SourceSpec("D", ("B", "C")),
            SourceSpec("E"),
            SourceSpec("F", ("D", "E")),
            SourceSpec("G"),
            SourceSpec("H", ("F", "G")),
            SourceSpec("I"),
            SourceSpec("J", ("H", "I")),
        )
    ),
}


class MemorySummaryStore:
    def assert_available(self, pipeline_id):
        pass

    def persist(self, outcome):
        return outcome


def terminal(source, *, failed=False):
    attempt = (
        SourceAttempt.failed_from_exception(
            source_id=source.source_id,
            run_id=source.run_id,
            attempt=source.attempt,
            timing=ExecutionTiming(STARTED_AT, STARTED_AT, 0),
            error=RuntimeError("scripted failure"),
        )
        if failed
        else SourceAttempt(
            source.source_id,
            source.run_id,
            source.attempt,
            "succeeded",
            ExecutionTiming(STARTED_AT, STARTED_AT, 0),
        )
    )
    return SourceOutcome.from_attempts(source, (attempt,))


def simulate(plan, expected, priority, *, capacity=3, keys=None, decisions=None):
    keys = keys or {source: source for source in plan.source_ids}
    queue = ReadyQueue(plan, capacity, keys)
    outcomes = dict.fromkeys(plan.source_ids)
    running = []
    dispatches = []
    while not queue.is_finished(outcomes):
        while (decision := queue.next_decision(outcomes, running)) is not None:
            if decisions is not None:
                decisions.append(decision)
            source = decision.planned_source
            if decision.kind == "dispatch":
                assert len(running) < capacity
                assert keys[source.source_id] not in {keys[node] for node in running}
                assert all(
                    outcomes[upstream].status == "succeeded" for upstream in source.upstream_ids
                )
                running.append(source.source_id)
                dispatches.append(source.source_id)
            elif decision.kind == "skip":
                outcomes[source.source_id] = SourceOutcome.skipped(
                    source,
                    direct_blocking_upstream_ids=decision.blockers,
                    root_failed_source_ids=decision.roots,
                )
            else:
                assert decision.kind == "planning_failure"
                outcomes[source.source_id] = SourceOutcome.from_planning_failure(source)
        if running:
            finished = min(running, key=priority.index)
            running.remove(finished)
            outcomes[finished] = expected[finished]
        else:
            assert queue.is_finished(outcomes), "scheduler deadlocked with undecided nodes"
    return outcomes, dispatches


@pytest.mark.parametrize("case", CASES)
def test_dispatch_and_skip_contract_equal_sequential_for_every_baseline_case(
    tmp_path, monkeypatch, case
):
    _root, registry, plan, config, paths = planned_project(tmp_path, case)
    sequential_decisions = []

    def record_decision(*args):
        decision = decide(*args)
        sequential_decisions.append(decision)
        return decision

    monkeypatch.setattr("janus.runtime.batch.decide", record_decision)
    sequential = BatchExecutor(source_execution=CannedSourceExecution(case)).execute(
        plan, registry, config, paths, summary_store=MemorySummaryStore()
    )
    expected = {source.source_id: source for source in sequential.sources}
    queued_decisions = []
    actual, dispatches = simulate(
        plan, expected, list(plan.source_ids), capacity=1, decisions=queued_decisions
    )
    assert queued_decisions == sequential_decisions
    assert dispatches == [source.source_id for source in sequential.sources if source.attempts]
    for source_id, outcome in actual.items():
        assert outcome.to_summary() == expected[source_id].to_summary()


@pytest.mark.parametrize("c_failed", [False, True], ids=["C-succeeded", "C-failed"])
def test_diamond_waits_for_all_terminal_upstreams_before_skipping(tmp_path, c_failed):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path, "diamond")
    outcomes = dict.fromkeys(plan.source_ids)
    outcomes["A"] = terminal(plan.source("A"))
    outcomes["B"] = terminal(plan.source("B"), failed=True)
    queue = ReadyQueue(plan, 3, {source: source for source in plan.source_ids})
    assert queue.next_decision(outcomes, {"C"}) is None
    outcomes["C"] = terminal(plan.source("C"), failed=c_failed)
    decision = queue.next_decision(outcomes, set())
    assert decision.kind == "skip" and decision.planned_source.source_id == "D"
    expected = ("B", "C") if c_failed else ("B",)
    assert decision.blockers == decision.roots == expected


@pytest.mark.parametrize("case", COMPLETION_CASES)
def test_every_completion_permutation_has_same_final_map_and_summary(tmp_path, monkeypatch, case):
    monkeypatch.setitem(GRAPH_CASES, case, COMPLETION_CASES[case])
    _root, registry, plan, config, paths = planned_project(tmp_path, case)
    execution_case = "diamond_one_arm_fails" if case == "dependent_ten" else case
    sequential = BatchExecutor(source_execution=CannedSourceExecution(execution_case)).execute(
        plan, registry, config, paths, summary_store=MemorySummaryStore()
    )
    expected = {source.source_id: source for source in sequential.sources}
    orders = itertools.permutations(plan.source_ids)
    if len(plan.source_ids) > 6:
        generator = random.Random(24)
        orders = [generator.sample(plan.source_ids, len(plan.source_ids)) for _ in range(200)]
    count = 0
    for order in orders:
        actual, _dispatches = simulate(plan, expected, list(order))
        assert actual == expected
        summary = PipelineOutcome.from_plan(
            plan,
            timing=sequential.timing,
            sources=tuple(actual[source_id] for source_id in plan.source_ids),
        ).to_summary()
        assert normalized_terminal_contract(summary) == normalized_terminal_contract(
            sequential.to_summary()
        )
        count += 1
    assert count == (factorial(len(plan.source_ids)) if len(plan.source_ids) <= 6 else 200)


def test_shared_lock_key_holds_back_peer_without_blocking_other_domains(tmp_path):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path, "independent_three")
    queue = ReadyQueue(plan, 3, {"A": "shared", "B": "shared", "C": "other"})
    outcomes = dict.fromkeys(plan.source_ids)
    assert queue.next_decision(outcomes, set()).planned_source.source_id == "A"
    assert queue.next_decision(outcomes, {"A"}).planned_source.source_id == "C"
    assert queue.next_decision(outcomes, {"A", "C"}) is None
    outcomes["A"] = terminal(plan.source("A"))
    assert queue.next_decision(outcomes, {"C"}).planned_source.source_id == "B"


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_capacity_never_exceeded(tmp_path, capacity):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path, "independent_three")
    queue = ReadyQueue(plan, capacity, {source: source for source in plan.source_ids})
    outcomes = dict.fromkeys(plan.source_ids)
    running = set()
    for _ in range(capacity):
        decision = queue.next_decision(outcomes, running)
        assert decision.kind == "dispatch"
        running.add(decision.planned_source.source_id)
    assert queue.next_decision(outcomes, running) is None


def test_skip_and_planning_failure_can_be_decided_at_full_capacity(tmp_path):
    from janus.orchestration.plans import SourcePlanFailure

    _root, _registry, plan, _config, _paths = planned_project(tmp_path)
    outcomes = dict.fromkeys(plan.source_ids)
    outcomes["A"] = terminal(plan.source("A"), failed=True)
    queue = ReadyQueue(plan, 1, {source: source for source in plan.source_ids})
    assert queue.next_decision(outcomes, {"C"}).kind == "skip"
    failed = replace(
        plan.source("B"),
        planned_run=None,
        failure=SourcePlanFailure(error_type="RuntimeError", reason="failure"),
    )
    changed = replace(
        plan,
        sources=tuple(failed if source.source_id == "B" else source for source in plan.sources),
    )
    outcomes["A"] = terminal(plan.source("A"))
    assert (
        ReadyQueue(changed, 1, {"A": "a", "C": "c"}).next_decision(outcomes, {"C"}).kind
        == "planning_failure"
    )


@pytest.mark.parametrize("capacity", [1, 3], ids=["capacity-full", "lock-conflict"])
def test_held_back_dispatch_does_not_hide_later_planning_failure(tmp_path, capacity):
    from janus.orchestration.plans import SourcePlanFailure

    _root, _registry, plan, _config, _paths = planned_project(tmp_path, "independent_three")
    failed = replace(
        plan.source("C"),
        planned_run=None,
        failure=SourcePlanFailure(error_type="RuntimeError", reason="cannot plan"),
    )
    changed = replace(plan, sources=(*plan.sources[:2], failed))
    queue = ReadyQueue(changed, capacity, {"A": "shared", "B": "shared"})
    outcomes = dict.fromkeys(plan.source_ids)
    assert queue.next_decision(outcomes, {"A"}) == NodeDecision("planning_failure", failed)
    assert outcomes == dict.fromkeys(plan.source_ids)
    outcomes["C"] = SourceOutcome.from_planning_failure(failed)
    assert queue.next_decision(outcomes, {"A"}) is None


def test_lock_keys_are_opaque_and_never_treated_as_directory_ancestry(tmp_path):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path, "independent_three")
    queue = ReadyQueue(plan, 3, {"A": "key", "B": "key/child", "C": "other"})
    outcomes = dict.fromkeys(plan.source_ids)
    assert queue.next_decision(outcomes, {"A"}) == NodeDecision("dispatch", plan.source("B"))


def test_skip_rule_bodies_are_moved_once_and_not_copied():
    paths = sorted((REPO_ROOT / "src/janus").rglob("*.py"))
    assert paths
    definitions = {name: [] for name in ("blocking_upstreams", "root_failures")}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name.lstrip("_") in definitions:
                definitions[node.name.lstrip("_")].append(path.relative_to(REPO_ROOT).as_posix())
    assert definitions == {name: ["src/janus/orchestration/scheduling.py"] for name in definitions}


@pytest.mark.parametrize("capacity", [0, -1])
def test_nonpositive_capacity_is_refused_at_construction(tmp_path, capacity):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path)
    with pytest.raises(ValueError, match="max_parallel must be 1 or greater"):
        ReadyQueue(plan, capacity, {source: source for source in plan.source_ids})


def test_missing_runnable_lock_key_is_refused_even_if_a_dependency_will_fail(tmp_path):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path)
    with pytest.raises(ValueError, match=r"Missing lock keys for planned sources: \['B'\]"):
        ReadyQueue(plan, 3, {"A": "a", "C": "c"})


def test_unfinished_earlier_dependency_does_not_hold_back_independent_peer(tmp_path):
    _root, _registry, plan, _config, _paths = planned_project(tmp_path)
    queue = ReadyQueue(plan, 3, {source: source for source in plan.source_ids})
    outcomes = dict.fromkeys(plan.source_ids)
    decision = queue.next_decision(outcomes, {"A"})
    assert decision == NodeDecision("dispatch", plan.source("C"))
    assert queue.is_finished(outcomes) is False
    assert queue.next_decision(outcomes, {"A", "C"}) is None
    outcomes.update({source.source_id: terminal(source) for source in plan.sources})
    outcomes["unrelated"] = None
    assert queue.is_finished(outcomes) is True
    assert queue.next_decision(outcomes, ()) is None


def test_skip_takes_precedence_over_planning_failure_and_deduplicates_roots(tmp_path):
    from janus.orchestration.plans import SourcePlanFailure

    _root, _registry, plan, _config, _paths = planned_project(tmp_path, "diamond")
    outcomes = dict.fromkeys(plan.source_ids)
    outcomes["A"] = terminal(plan.source("A"), failed=True)
    for source_id in ("B", "C"):
        outcomes[source_id] = SourceOutcome.skipped(
            plan.source(source_id),
            direct_blocking_upstream_ids=("A",),
            root_failed_source_ids=("A",),
        )
    failed = replace(
        plan.source("D"),
        planned_run=None,
        failure=SourcePlanFailure(error_type="RuntimeError", reason="cannot plan"),
    )
    decision = decide(plan, failed, outcomes)
    assert decision == NodeDecision("skip", failed, ("B", "C"), ("A",))
    with pytest.raises(FrozenInstanceError):
        decision.kind = "dispatch"


def test_deciding_before_upstream_is_terminal_raises_pure_order_error(tmp_path):
    from janus.orchestration import BatchPlanningError, SchedulingOrderError

    _root, _registry, plan, _config, _paths = planned_project(tmp_path)
    with pytest.raises(SchedulingOrderError) as exc_info:
        decide(plan, plan.source("B"), dict.fromkeys(plan.source_ids))
    assert isinstance(exc_info.value, BatchPlanningError)
    assert str(exc_info.value) == (
        "Source 'B' was reached before upstream 'A'; nothing further was scheduled"
    )
