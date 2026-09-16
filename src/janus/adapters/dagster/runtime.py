"""Run-scoped Dagster resource backed by JANUS planning and execution services."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dagster import InitResourceContext

from janus.lineage import compute_config_version
from janus.orchestration import BatchPlan, BatchPlanner, BatchPlanRequest, SourceSelection
from janus.registry import SourceRegistry
from janus.runtime import SourceExecutionService, SourceExecutor, execute_source_attempt
from janus.runtime.batch import SourceExecution
from janus.utils.environment import RuntimeLocation, prepare_runtime
from janus.utils.logging import StructuredLogger, build_structured_logger

from .errors import DagsterConfigurationDriftError
from .manifest import ADAPTER_NAME, ADAPTER_TAG, MANIFEST_TAG, SNAPSHOT_TAG, DagsterRunManifest

SourceExecutionFactory = Callable[[StructuredLogger], SourceExecution]


@dataclass(frozen=True, slots=True)
class DagsterDefinitionSnapshot:
    """Validated definition-time state transported into the run resource."""

    registry: SourceRegistry
    selection: SourceSelection
    environment_config: Mapping[str, Any]
    manifest: DagsterRunManifest

    def assert_run_tags(self, tags: Mapping[str, str]) -> None:
        """Reject a worker whose reconstructed definitions differ from the launched run."""
        if tags.get(ADAPTER_TAG) != ADAPTER_NAME:
            raise DagsterConfigurationDriftError(
                f"Dagster run is missing {ADAPTER_TAG!r}={ADAPTER_NAME!r}; nothing was executed"
            )
        if tags.get(SNAPSHOT_TAG) != self.manifest.snapshot_id:
            raise DagsterConfigurationDriftError(
                "Dagster definitions changed between launch and worker execution; "
                "the run snapshot no longer matches, so nothing was executed"
            )
        if tags.get(MANIFEST_TAG) != self.manifest.to_json():
            raise DagsterConfigurationDriftError(
                "Dagster run manifest differs from the worker definition snapshot; "
                "nothing was executed"
            )

    def assert_source_versions(self) -> None:
        """Reject edits made after this in-memory definition snapshot was constructed."""
        expected = dict(self.manifest.config_versions)
        actual: dict[str, str] = {}
        cached: dict[object, str] = {}
        for source_id in self.manifest.source_order:
            path = self.registry.get_source(source_id).config_path
            try:
                if path not in cached:
                    cached[path] = compute_config_version(path)
                version = cached[path]
            except OSError as exc:
                raise DagsterConfigurationDriftError(
                    f"Could not verify source {source_id!r} at {path}: {exc}; "
                    "nothing was executed"
                ) from exc
            actual[source_id] = version
        if actual != expected:
            changed = sorted(
                source_id
                for source_id in self.manifest.source_order
                if actual.get(source_id) != expected.get(source_id)
            )
            raise DagsterConfigurationDriftError(
                f"Source configuration changed after Dagster definitions were loaded: "
                f"{changed}; reload definitions and launch a new run. Nothing was executed"
            )


@dataclass(slots=True)
class DagsterRunRuntime:
    """One Dagster run's shared snapshot, paths, planner, and source execution seam."""

    snapshot: DagsterDefinitionSnapshot
    run_id: str
    planned_at: datetime
    environment_config: dict[str, Any]
    resolved_paths: Mapping[str, RuntimeLocation]
    batch_planner: BatchPlanner
    source_execution: SourceExecution
    _plans: dict[int, BatchPlan] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        context: InitResourceContext,
        snapshot: DagsterDefinitionSnapshot,
        *,
        batch_planner: BatchPlanner,
        source_execution_factory: SourceExecutionFactory,
    ) -> DagsterRunRuntime:
        """Validate parity and plan the entire batch before preparing runtime paths."""
        dagster_run = context.run
        if dagster_run is None:
            raise DagsterConfigurationDriftError(
                "The JANUS Dagster resource requires a run context; nothing was executed"
            )
        instance = context.instance
        run_id = dagster_run.run_id
        if instance is None or run_id is None:
            raise DagsterConfigurationDriftError(
                "The JANUS Dagster resource requires an instance and run id; "
                "nothing was executed"
            )
        snapshot.assert_run_tags(dagster_run.tags)
        snapshot.assert_source_versions()

        stats = instance.get_run_stats(run_id)
        if stats.start_time is None:
            raise DagsterConfigurationDriftError(
                "Dagster did not expose a run start time for pipeline identity; "
                "nothing was executed"
            )
        planned_at = datetime.fromtimestamp(stats.start_time, tz=UTC)
        initial_plan = _plan_attempt(
            snapshot,
            batch_planner,
            run_id,
            planned_at,
            attempt=1,
        )

        environment_config = copy.deepcopy(dict(snapshot.environment_config))
        resolved_paths = prepare_runtime(environment_config, snapshot.registry.project_root)
        logger = build_structured_logger(
            "janus.dagster",
            level=environment_config.get("runtime", {}).get("log_level", "INFO"),
        ).bind(
            environment=snapshot.manifest.environment,
            project_root=str(snapshot.registry.project_root),
            pipeline_run_id=run_id,
            dagster_job=snapshot.manifest.job_name,
        )
        return cls(
            snapshot=snapshot,
            run_id=run_id,
            planned_at=planned_at,
            environment_config=environment_config,
            resolved_paths=resolved_paths,
            batch_planner=batch_planner,
            source_execution=source_execution_factory(logger),
            _plans={1: initial_plan},
        )

    def execute(self, source_id: str, retry_number: int):
        """Execute one source once; Dagster alone decides whether to retry the op."""
        attempt = retry_number + 1
        plan = self._plans.get(attempt)
        if plan is None:
            self.snapshot.assert_source_versions()
            plan = _plan_attempt(
                self.snapshot,
                self.batch_planner,
                self.run_id,
                self.planned_at,
                attempt=attempt,
            )
            self._plans[attempt] = plan
        planned_source = plan.source(source_id)
        if planned_source.failure is not None:
            return planned_source, None
        return planned_source, execute_source_attempt(
            planned_source,
            self.environment_config,
            self.resolved_paths,
            source_execution=self.source_execution,
        )


def default_source_execution_factory(logger: StructuredLogger) -> SourceExecution:
    """Build the established executor/provider service for one Dagster run."""
    return SourceExecutionService(executor=SourceExecutor(logger=logger))


def _plan_attempt(
    snapshot: DagsterDefinitionSnapshot,
    planner: BatchPlanner,
    run_id: str,
    planned_at: datetime,
    *,
    attempt: int,
) -> BatchPlan:
    request = BatchPlanRequest.create(
        environment=snapshot.manifest.environment,
        project_root=snapshot.registry.project_root,
        pipeline_run_id=run_id,
        planned_at=planned_at,
        selection=snapshot.selection.selection,
        attempt=attempt,
        trigger=ADAPTER_NAME,
        attributes={
            "dagster_job_name": snapshot.manifest.job_name,
            "dagster_run_id": run_id,
        },
    )
    plan = planner.plan(request, registry=snapshot.registry)
    if plan.selection != snapshot.selection:
        raise DagsterConfigurationDriftError(
            "Dagster execution planning produced a different source selection or graph; "
            "nothing was executed"
        )
    if tuple(plan.config_versions().items()) != snapshot.manifest.config_versions:
        raise DagsterConfigurationDriftError(
            "Dagster execution planning produced different source configuration versions; "
            "nothing was executed"
        )
    return plan
