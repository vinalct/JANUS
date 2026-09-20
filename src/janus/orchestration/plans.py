"""The frozen records a batch is requested and answered with.

Everything here is orchestrator-neutral and JSON-renderable: a Dagster op, an Airflow
task and ``janus run-all`` are handed the same plan, and none of their types appear in
it. A plan is also *complete before anything executes* — every source already has its
run id, its config version and either a planned run or a recorded planning failure — so
the runner's only remaining decisions are about outcomes, never about identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from janus.models.dependencies import SourceDependencyEdge
from janus.orchestration.errors import BatchPlanningError, PipelineIdentityError
from janus.orchestration.identity import (
    default_pipeline_run_id,
    source_attempt_run_id,
    validate_pipeline_run_id,
)
from janus.orchestration.selection import NO_SELECTION, BatchSelection, SourceSelection
from janus.planner import PlannedRun

DEFAULT_TRIGGER = "run-all"
MAX_FAILURE_REASON_LENGTH = 2000
PIPELINE_ATTRIBUTE_KEYS = ("pipeline_run_id", "pipeline_attempt", "trigger")


@dataclass(frozen=True, slots=True)
class BatchPlanRequest:
    """What was asked of one batch: identity, scope, and the logical planning instant.

    ``planned_at`` is the *logical* timestamp. It pins run ids and run contexts so a
    replanned batch is comparable, and it is deliberately not the clock a later run's
    durations are measured against — a pipeline pinned to yesterday's planning instant
    still takes the time it takes.
    """

    environment: str
    project_root: Path
    pipeline_run_id: str
    planned_at: datetime
    selection: BatchSelection = NO_SELECTION
    attempt: int = 1
    trigger: str = DEFAULT_TRIGGER
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and scope here, so no later layer has to re-check them."""
        if not self.environment.strip():
            raise BatchPlanningError("environment must not be empty")
        if not self.trigger.strip():
            raise BatchPlanningError("trigger must not be empty")
        validate_pipeline_run_id(self.pipeline_run_id)
        if self.attempt < 1:
            raise PipelineIdentityError(f"attempt must be 1 or greater, got {self.attempt!r}")
        if self.planned_at.tzinfo is None or self.planned_at.utcoffset() is None:
            raise BatchPlanningError("planned_at must be timezone-aware")

    @classmethod
    def create(
        cls,
        *,
        environment: str,
        project_root: Path,
        pipeline_run_id: str | None = None,
        planned_at: datetime | None = None,
        selection: BatchSelection | None = None,
        attempt: int = 1,
        trigger: str = DEFAULT_TRIGGER,
        attributes: Mapping[str, str] | None = None,
    ) -> Self:
        """Normalize the inputs an entry point collects into one frozen request."""
        resolved_planned_at = planned_at or datetime.now(tz=UTC)
        resolved_pipeline_run_id = (
            validate_pipeline_run_id(pipeline_run_id.strip())
            if pipeline_run_id is not None
            else default_pipeline_run_id(environment, resolved_planned_at)
        )
        return cls(
            environment=environment,
            project_root=project_root.resolve(),
            pipeline_run_id=resolved_pipeline_run_id,
            planned_at=resolved_planned_at,
            selection=selection if selection is not None else NO_SELECTION,
            attempt=attempt,
            trigger=trigger,
            attributes=_freeze_string_mapping(attributes),
        )

    def source_run_id(self, source_id: str) -> str:
        """Return the collision-safe run id this batch gives one source's attempt."""
        return source_attempt_run_id(
            pipeline_run_id=self.pipeline_run_id,
            source_id=source_id,
            attempt=self.attempt,
        )

    def run_attributes(self) -> dict[str, str]:
        """Return the run-context attributes every source in this batch carries.

        Correlation travels through the attributes a run context already has, rather than
        through a parallel channel: whatever reads a single-source run's attributes today
        can see which pipeline it belonged to without learning a new format.
        """
        return {
            **dict(self.attributes),
            "pipeline_run_id": self.pipeline_run_id,
            "pipeline_attempt": str(self.attempt),
            "trigger": self.trigger,
        }

    def to_summary(self) -> dict[str, Any]:
        """Render the request as the identity block of a pipeline summary."""
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "attempt": self.attempt,
            "trigger": self.trigger,
            "environment": self.environment,
            "project_root": str(self.project_root),
            "planned_at": self.planned_at.isoformat(),
            "selection": {
                "tags": list(self.selection.tags),
                "domains": list(self.selection.domains),
                "description": self.selection.describe(),
            },
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class SourcePlanFailure:
    """One source that could not be planned, recorded instead of raised.

    A strategy that cannot be bound or a hook that raises is that *source's* problem. The
    batch keeps it, runs everything that does not depend on it, and lets the runner turn
    it into a failed result with its independent peers intact.
    """

    error_type: str
    reason: str
    phase: str = "planning"

    def __post_init__(self) -> None:
        """Reject an unattributable failure: a blank type or reason explains nothing."""
        if not self.error_type.strip():
            raise BatchPlanningError("error_type must not be empty")
        if not self.reason.strip():
            raise BatchPlanningError("reason must not be empty")
        if not self.phase.strip():
            raise BatchPlanningError("phase must not be empty")

    @classmethod
    def from_exception(cls, exc: BaseException) -> Self:
        """Capture one planning exception as a bounded, serializable record."""
        reason = str(exc).strip() or type(exc).__name__
        if len(reason) > MAX_FAILURE_REASON_LENGTH:
            reason = f"{reason[:MAX_FAILURE_REASON_LENGTH]}… (truncated)"
        return cls(error_type=type(exc).__name__, reason=reason)

    def to_summary(self) -> dict[str, Any]:
        """Render the failure the way a pipeline summary reports it."""
        return {"phase": self.phase, "error_type": self.error_type, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class PlannedSource:
    """One source's place in a batch: identity, dependencies, and its plan or its failure."""

    source_id: str
    run_id: str
    attempt: int
    selected_directly: bool
    upstream_ids: tuple[str, ...]
    config_path: Path
    config_version: str
    planned_run: PlannedRun | None = None
    failure: SourcePlanFailure | None = None

    def __post_init__(self) -> None:
        """Hold every node to exactly one of the two states a planned source can be in."""
        if not self.source_id.strip():
            raise BatchPlanningError("source_id must not be empty")
        if not self.run_id.strip():
            raise BatchPlanningError("run_id must not be empty")
        if not self.config_version.strip():
            raise BatchPlanningError("config_version must not be empty")
        if (self.planned_run is None) == (self.failure is None):
            raise BatchPlanningError(
                f"Source {self.source_id!r} must carry either a planned run or a planning "
                "failure, never both and never neither"
            )
        if self.planned_run is not None:
            plan = self.planned_run.plan
            if plan.source.source_id != self.source_id:
                raise BatchPlanningError(
                    f"Planned run for {self.source_id!r} carries source "
                    f"{plan.source.source_id!r}"
                )
            if plan.run_context.run_id != self.run_id:
                raise BatchPlanningError(
                    f"Planned run for {self.source_id!r} carries run id "
                    f"{plan.run_context.run_id!r}, not the batch-derived {self.run_id!r}"
                )

    @property
    def is_planned(self) -> bool:
        """Return True when this source has a plan the runner can execute."""
        return self.planned_run is not None

    def require_planned_run(self) -> PlannedRun:
        """Return the planned run, or raise if this node only holds a failure."""
        if self.planned_run is None:
            raise BatchPlanningError(
                f"Source {self.source_id!r} failed to plan and has no executable run"
            )
        return self.planned_run

    def to_summary(self) -> dict[str, Any]:
        """Render this node's identity evidence, without its full plan."""
        summary: dict[str, Any] = {
            "source_id": self.source_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "selected_directly": self.selected_directly,
            "upstream_ids": list(self.upstream_ids),
            "config_path": str(self.config_path),
            "config_version": self.config_version,
            "planned": self.is_planned,
        }
        if self.failure is not None:
            summary["planning_failure"] = self.failure.to_summary()
        return summary


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """One reproducible batch: what was asked, who is in it, and in which order."""

    request: BatchPlanRequest
    selection: SourceSelection
    sources: tuple[PlannedSource, ...]

    def __post_init__(self) -> None:
        """Keep the planned nodes and the resolved selection the same set, in one order."""
        planned_ids = tuple(source.source_id for source in self.sources)
        if planned_ids != self.selection.source_ids:
            raise BatchPlanningError(
                "The planned sources must be exactly the selected sources in topological "
                f"order: planned {list(planned_ids)}, selected {list(self.selection.source_ids)}"
            )

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Return every source in the batch, in execution order."""
        return self.selection.source_ids

    @property
    def root_ids(self) -> tuple[str, ...]:
        """Return the sources the filter selected directly."""
        return self.selection.root_ids

    @property
    def included_upstream_ids(self) -> tuple[str, ...]:
        """Return the sources included because something selected depends on them."""
        return self.selection.included_upstream_ids

    @property
    def edges(self) -> tuple[SourceDependencyEdge, ...]:
        """Return the dependency edges inside this batch."""
        return self.selection.graph.edges

    def source(self, source_id: str) -> PlannedSource:
        """Return one planned source, or raise ``LookupError`` when it is not in the batch."""
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise LookupError(f"Source {source_id!r} is not part of this batch plan")

    def upstreams_of(self, source_id: str) -> tuple[str, ...]:
        """Return the direct producers of one source *within this batch*."""
        return self.selection.graph.upstreams(source_id)

    def downstreams_of(self, source_id: str) -> tuple[str, ...]:
        """Return the direct consumers of one source *within this batch*."""
        return self.selection.graph.downstreams(source_id)

    def planning_failures(self) -> tuple[PlannedSource, ...]:
        """Return the nodes that could not be planned, in execution order."""
        return tuple(source for source in self.sources if not source.is_planned)

    def config_versions(self) -> dict[str, str]:
        """Return the config version pinned for every source in the batch."""
        return {source.source_id: source.config_version for source in self.sources}

    def to_summary(self) -> dict[str, Any]:
        """Render the plan as the evidence a pipeline summary is built on."""
        return {
            "pipeline": self.request.to_summary(),
            "selection": {
                "root_ids": list(self.root_ids),
                "included_upstream_ids": list(self.included_upstream_ids),
                "source_ids": list(self.source_ids),
            },
            "graph": {
                "edges": [
                    {
                        "producer_id": edge.producer_id,
                        "consumer_id": edge.consumer_id,
                        "table": edge.table,
                        "input_paths": list(edge.input_paths),
                    }
                    for edge in self.edges
                ]
            },
            "sources": [source.to_summary() for source in self.sources],
            "config_versions": self.config_versions(),
        }


def _freeze_string_mapping(values: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Freeze caller attributes, refusing the keys the batch owns."""
    if not values:
        return ()

    frozen: list[tuple[str, str]] = []
    for key, value in values.items():
        normalized_key = key.strip()
        if not normalized_key:
            raise BatchPlanningError("attribute keys must be non-empty strings")
        if normalized_key in PIPELINE_ATTRIBUTE_KEYS:
            raise BatchPlanningError(
                f"attribute {normalized_key!r} is derived from the batch request and cannot "
                "be supplied as a caller attribute"
            )
        if not value.strip():
            raise BatchPlanningError(f"attribute value for {normalized_key!r} must not be empty")
        frozen.append((normalized_key, value.strip()))
    return tuple(sorted(frozen))
