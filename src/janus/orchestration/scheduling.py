"""Pure batch decisions and readiness in deterministic plan order."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from janus.orchestration.errors import SchedulingOrderError
from janus.orchestration.plans import BatchPlan, PlannedSource
from janus.orchestration.results import SourceOutcome


def blocking_upstreams(
    plan: BatchPlan,
    planned_source: PlannedSource,
    outcomes: Mapping[str, SourceOutcome | None],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for upstream_id in plan.upstreams_of(planned_source.source_id):
        upstream = outcomes[upstream_id]
        if upstream is None:
            raise SchedulingOrderError(
                f"Source {planned_source.source_id!r} was reached before upstream "
                f"{upstream_id!r}; nothing further was scheduled"
            )
        if upstream.status in {"failed", "skipped"}:
            blockers.append(upstream_id)
    return tuple(blockers)


def root_failures(
    blockers: tuple[str, ...],
    outcomes: Mapping[str, SourceOutcome | None],
) -> tuple[str, ...]:
    root_ids: set[str] = set()
    for source_id in blockers:
        outcome = outcomes[source_id]
        assert outcome is not None
        if outcome.status == "failed":
            root_ids.add(source_id)
            continue
        assert outcome.skip is not None
        root_ids.update(outcome.skip.root_failed_source_ids)
    return tuple(sorted(root_ids))


@dataclass(frozen=True, slots=True)
class NodeDecision:
    """One node's next action, with the complete explanation when it is skipped."""

    kind: Literal["skip", "planning_failure", "dispatch"]
    planned_source: PlannedSource
    blockers: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()


def decide(
    plan: BatchPlan,
    planned_source: PlannedSource,
    outcomes: Mapping[str, SourceOutcome | None],
) -> NodeDecision:
    """Decide in the sequential loop's order: skip, planning failure, dispatch."""
    blockers = blocking_upstreams(plan, planned_source, outcomes)
    if blockers:
        return NodeDecision("skip", planned_source, blockers, root_failures(blockers, outcomes))
    if planned_source.failure is not None:
        return NodeDecision("planning_failure", planned_source)
    return NodeDecision("dispatch", planned_source)


@dataclass(slots=True)
class ReadyQueue:
    """Choose a decidable node in plan order without starting any work.

    Every upstream must be terminal before a decision, preserving all skip causes.
    Dispatch additionally requires capacity and a free opaque lock domain. Planning
    failures need no lock key because they never start a source execution.
    """

    plan: BatchPlan
    max_parallel: int
    lock_keys: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.max_parallel < 1:
            raise ValueError(f"max_parallel must be 1 or greater, got {self.max_parallel!r}")
        missing = tuple(
            source.source_id
            for source in self.plan.sources
            if source.is_planned and source.source_id not in self.lock_keys
        )
        if missing:
            raise ValueError(f"Missing lock keys for planned sources: {list(missing)}")

    def next_decision(
        self,
        outcomes: Mapping[str, SourceOutcome | None],
        running: Collection[str],
    ) -> NodeDecision | None:
        """Return the first decision allowed by dependencies, capacity and lock keys."""
        running_keys = {self.lock_keys[source_id] for source_id in running}
        for source in self.plan.sources:
            if outcomes[source.source_id] is not None or source.source_id in running:
                continue
            if any(
                outcomes[upstream] is None
                for upstream in self.plan.upstreams_of(source.source_id)
            ):
                continue
            decision = decide(self.plan, source, outcomes)
            if decision.kind != "dispatch":
                return decision
            if (
                len(running) < self.max_parallel
                and self.lock_keys[source.source_id] not in running_keys
            ):
                return decision
        return None

    def is_finished(self, outcomes: Mapping[str, SourceOutcome | None]) -> bool:
        """Return whether every node in this plan has a terminal outcome."""
        return all(outcomes[source_id] is not None for source_id in self.plan.source_ids)
