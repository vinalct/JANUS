"""Plan a whole batch against one registry snapshot, before anything executes.

Three properties this module exists to hold:

**One snapshot.** The registry is loaded and validated once and every source is planned
against that same object. Re-reading YAML between nodes would let a mid-batch edit split
one pipeline across two versions of the truth, and the graph the order came from would
no longer describe the sources being run.

**One planning implementation.** Every source goes through ``Planner.plan`` — the same
dispatch resolution, hook resolution, run context and dispatch validation a single-source
run gets. The batch adds identity and order; it does not add a second way to plan.

**Plan first, then execute.** Every plan is built before the first source runs, so a hook
that moves a source's bronze table or its request inputs is caught while the batch is
still a document. After the first write it would be too late: the DAG would already have
been ordered by a graph that no longer described the work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from janus.lineage import compute_config_version
from janus.models.dependencies import iter_iceberg_input_references
from janus.models.source_config import SourceConfig
from janus.orchestration.errors import GraphDriftError
from janus.orchestration.plans import (
    BatchPlan,
    BatchPlanRequest,
    PlannedSource,
    SourcePlanFailure,
)
from janus.orchestration.selection import SourceSelection, select_sources
from janus.planner import Planner, PlanningRequest
from janus.registry import (
    SourceRegistry,
    bronze_output_table_identifier,
    load_registry,
)

_ReferenceKeys = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class BatchPlanner:
    """Turn a batch request into an ordered, fully planned batch — or refuse it."""

    planner: Planner = field(default_factory=Planner)
    registry_loader: Callable[[Path], SourceRegistry] = load_registry

    def plan(
        self,
        request: BatchPlanRequest,
        *,
        registry: SourceRegistry | None = None,
    ) -> BatchPlan:
        """Resolve selection and plan every selected source against one snapshot."""
        snapshot = registry if registry is not None else self.registry_loader(request.project_root)
        selection = select_sources(snapshot.graph, snapshot.sources, request.selection)

        config_versions: dict[Path, str] = {}
        planned = tuple(
            self._plan_source(request, snapshot, selection, source_id, config_versions)
            for source_id in selection.source_ids
        )
        _reject_graph_drift(snapshot, planned)
        return BatchPlan(request=request, selection=selection, sources=planned)

    def _plan_source(
        self,
        request: BatchPlanRequest,
        snapshot: SourceRegistry,
        selection: SourceSelection,
        source_id: str,
        config_versions: dict[Path, str],
    ) -> PlannedSource:
        """Plan one source, keeping its own planning failure attributable to it."""
        source_config = snapshot.get_source(source_id)
        run_id = request.source_run_id(source_id)
        planning_request = PlanningRequest.create(
            source_id=source_id,
            environment=request.environment,
            project_root=request.project_root,
            run_id=run_id,
            started_at=request.planned_at,
            attributes=request.run_attributes(),
        )

        planned_run = None
        failure = None
        try:
            planned_run = self.planner.plan(planning_request, registry=snapshot)
        except Exception as exc: 
            failure = SourcePlanFailure.from_exception(exc)

        return PlannedSource(
            source_id=source_id,
            run_id=run_id,
            attempt=request.attempt,
            selected_directly=selection.is_root(source_id),
            upstream_ids=selection.graph.upstreams(source_id),
            config_path=source_config.config_path,
            config_version=_config_version(source_config, config_versions),
            planned_run=planned_run,
            failure=failure,
        )


def _config_version(source_config: SourceConfig, cache: dict[Path, str]) -> str:
    """Hash each config file once per batch, with the calculation lineage already uses."""
    config_path = source_config.config_path
    if config_path not in cache:
        cache[config_path] = compute_config_version(config_path)
    return cache[config_path]


def _reject_graph_drift(
    snapshot: SourceRegistry,
    planned: tuple[PlannedSource, ...],
) -> None:
    """Refuse a batch whose plans no longer match the graph it was ordered by."""
    issues: list[str] = []
    for source in planned:
        if source.planned_run is None:
            continue
        plan = source.planned_run.plan
        configured = snapshot.get_source(source.source_id)

        if plan.source.source_id != configured.source_id:
            issues.append(
                f"{source.source_id}: planning returned a plan for "
                f"{plan.source.source_id!r}"
            )
            continue

        planned_table = bronze_output_table_identifier(
            plan.bronze_output,
            fallback_name=plan.source.source_id,
        )
        configured_table = snapshot.graph.node(source.source_id).bronze_table
        if planned_table != configured_table:
            issues.append(
                f"{source.source_id}: the graph was built from bronze table "
                f"{configured_table!r}, but planning produced {planned_table!r}"
            )

        planned_references = _reference_keys(plan.source_config)
        configured_references = _reference_keys(configured)
        if planned_references != configured_references:
            issues.append(
                f"{source.source_id}: the graph was built from upstream references "
                f"{[list(key) for key in configured_references]}, but planning produced "
                f"{[list(key) for key in planned_references]}"
            )

    if issues:
        raise GraphDriftError(
            "Planning changed the dependency graph this batch was ordered by, so the order "
            "no longer describes the work:\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\nA source hook may shape a plan, but not the bronze table its source "
            "produces or the tables it reads. Nothing was executed."
        )


def _reference_keys(source_config: SourceConfig) -> _ReferenceKeys:
    """Return one source's declared dependencies as comparable (producer, table) pairs."""
    return tuple(
        sorted(
            (reference.upstream_source_id, reference.table_reference)
            for reference in iter_iceberg_input_references(source_config.access.request_inputs)
        )
    )
