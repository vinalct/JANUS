"""Dagster definitions generated from JANUS's validated source graph."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dagster import (
    DagsterInstance,
    DagsterRunStatus,
    DefaultSensorStatus,
    Definitions,
    DependencyDefinition,
    Failure,
    GraphDefinition,
    In,
    Nothing,
    OpDefinition,
    Out,
    ResourceDefinition,
    RetryPolicy,
    in_process_executor,
    op,
    run_status_sensor,
)

from janus.orchestration import BatchPlanner, BatchSelection, PipelineOutcome, select_sources
from janus.registry import SourceRegistry, load_registry
from janus.utils.environment import load_environment_config

from .collector import SummaryStoreFactory, collect_dagster_run
from .events import attempt_event, planning_failure_event
from .manifest import ADAPTER_NAME, ADAPTER_TAG, MANIFEST_TAG, SNAPSHOT_TAG, DagsterRunManifest
from .names import source_op_name
from .runtime import (
    DagsterDefinitionSnapshot,
    DagsterRunRuntime,
    SourceExecutionFactory,
    default_source_execution_factory,
)

JANUS_RESOURCE_KEY = "janus"
DEFAULT_JOB_NAME = "janus_sources"
SOURCE_EXECUTION_POOL = "janus_source_execution"


@dataclass(frozen=True, slots=True)
class DagsterAdapter:
    """The generated definitions plus local execution and collection conveniences."""

    definitions: Definitions
    job: Any
    manifest: DagsterRunManifest
    source_op_names: Mapping[str, str]
    summary_store_factory: SummaryStoreFactory

    def collect(self, instance: DagsterInstance, run_id: str) -> PipelineOutcome:
        """Idempotently translate and persist one terminal run."""
        return collect_dagster_run(
            instance,
            run_id,
            summary_store_factory=self.summary_store_factory,
        )

    def execute_in_process(
        self,
        *,
        instance: DagsterInstance | None = None,
        raise_on_error: bool = False,
        run_config: Mapping[str, Any] | None = None,
    ) -> tuple[Any, PipelineOutcome]:
        """Run locally, then collect outside the job's dependency-success gating."""
        resolved_instance = instance or DagsterInstance.ephemeral()
        result = self.job.execute_in_process(
            instance=resolved_instance,
            raise_on_error=raise_on_error,
            run_config=run_config,
        )
        return result, self.collect(resolved_instance, result.run_id)


def _default_summary_store_factory(layout):
    from janus.orchestration import PipelineSummaryStore

    return PipelineSummaryStore(layout)


@dataclass(frozen=True, slots=True)
class DagsterAdapterServices:
    """Injectable JANUS seams used when Dagster starts or finishes a run."""

    batch_planner: BatchPlanner = field(default_factory=BatchPlanner)
    source_execution_factory: SourceExecutionFactory = default_source_execution_factory
    summary_store_factory: SummaryStoreFactory = _default_summary_store_factory


def build_dagster_adapter(
    project_root: Path,
    *,
    environment: str = "local",
    selection: BatchSelection | None = None,
    registry: SourceRegistry | None = None,
    environment_config: Mapping[str, Any] | None = None,
    services: DagsterAdapterServices | None = None,
    retry_policy: RetryPolicy | None = None,
    job_name: str = DEFAULT_JOB_NAME,
) -> DagsterAdapter:
    """Build one source op per expanded id and wire the shared graph's exact edges.

    Loading definitions reads and validates configuration only. Runtime directories,
    Spark providers, source execution, and summary writes begin only after Dagster starts
    a run or invokes a terminal status sensor.
    """
    resolved_root = project_root.resolve()
    resolved_registry = registry or load_registry(resolved_root)
    if resolved_registry.project_root != resolved_root:
        raise ValueError(
            f"Registry was loaded from {resolved_registry.project_root}, not {resolved_root}"
        )
    resolved_selection = select_sources(
        resolved_registry.graph,
        resolved_registry.sources,
        selection or BatchSelection(),
    )
    resolved_config = copy.deepcopy(
        dict(environment_config)
        if environment_config is not None
        else load_environment_config(environment, resolved_root)
    )
    op_names = {
        source_id: source_op_name(source_id) for source_id in resolved_selection.source_ids
    }
    if len(set(op_names.values())) != len(op_names):
        raise ValueError("Dagster op-name generation collided for distinct JANUS source ids")

    manifest = DagsterRunManifest.create(
        job_name=job_name,
        environment=environment,
        registry=resolved_registry,
        selection=resolved_selection,
        environment_config=resolved_config,
        op_names=op_names,
    )
    snapshot = DagsterDefinitionSnapshot(
        registry=resolved_registry,
        selection=resolved_selection,
        environment_config=resolved_config,
        manifest=manifest,
    )
    resolved_services = services or DagsterAdapterServices()
    resource = ResourceDefinition(
        lambda context: DagsterRunRuntime.create(
            context,
            snapshot,
            batch_planner=resolved_services.batch_planner,
            source_execution_factory=resolved_services.source_execution_factory,
        ),
        description="One run-scoped JANUS planning and source-execution service.",
    )
    source_ops = tuple(
        _source_op(
            source_id,
            resolved_registry,
            resolved_selection.graph.edges,
            op_names,
            retry_policy,
        )
        for source_id in resolved_selection.source_ids
    )
    dependencies = _dependencies(resolved_selection.source_ids, resolved_selection.graph, op_names)
    graph = GraphDefinition(
        name=f"{job_name}_graph",
        node_defs=source_ops,
        dependencies=dependencies,
        description="JANUS sources wired by validated Iceberg producer dependencies.",
    )
    job = graph.to_job(
        name=job_name,
        resource_defs={JANUS_RESOURCE_KEY: resource},
        executor_def=in_process_executor,
        tags={
            ADAPTER_TAG: ADAPTER_NAME,
            SNAPSHOT_TAG: manifest.snapshot_id,
            MANIFEST_TAG: manifest.to_json(),
            "janus/concurrency": "single-process",
        },
        description=(
            "Execute JANUS sources through the shared planner and SourceExecutor. "
            "The reference executor is intentionally single-process."
        ),
    )
    store_factory = resolved_services.summary_store_factory
    sensors = tuple(
        _terminal_sensor(status, job, store_factory)
        for status in (
            DagsterRunStatus.SUCCESS,
            DagsterRunStatus.FAILURE,
            DagsterRunStatus.CANCELED,
        )
    )
    definitions = Definitions(
        jobs=[job],
        sensors=sensors,
        metadata={
            "janus/snapshot_id": manifest.snapshot_id,
            "janus/source_count": len(manifest.source_order),
        },
    )
    return DagsterAdapter(
        definitions=definitions,
        job=job,
        manifest=manifest,
        source_op_names=op_names,
        summary_store_factory=store_factory,
    )


def build_definitions(project_root: Path, **kwargs: Any) -> Definitions:
    """Return loadable Dagster Definitions for a JANUS project."""
    return build_dagster_adapter(project_root, **kwargs).definitions


def build_job(project_root: Path, **kwargs: Any):
    """Return only the generated Dagster job for embedding in larger definitions."""
    return build_dagster_adapter(project_root, **kwargs).job


def _source_op(
    source_id: str,
    registry: SourceRegistry,
    edges: tuple[Any, ...],
    op_names: Mapping[str, str],
    retry_policy: RetryPolicy | None,
) -> OpDefinition:
    upstream_ids = tuple(
        sorted(edge.producer_id for edge in edges if edge.consumer_id == source_id)
    )
    ins = {
        f"upstream_{index}": In(
            Nothing,
            description=f"Dependency-only completion handle for {upstream_id!r}.",
        )
        for index, upstream_id in enumerate(upstream_ids)
    }
    metadata = _source_metadata(source_id, registry, edges)

    @op(
        name=op_names[source_id],
        description=f"Execute JANUS source {source_id!r} without transporting bronze data.",
        ins=ins,
        out=Out(Nothing, metadata=metadata),
        required_resource_keys={JANUS_RESOURCE_KEY},
        tags={key: _tag_value(value) for key, value in metadata.items()},
        retry_policy=retry_policy,
        pool=SOURCE_EXECUTION_POOL,
    )
    def execute_source(context):
        runtime = getattr(context.resources, JANUS_RESOURCE_KEY)
        planned_source, attempt = runtime.execute(source_id, context.retry_number)
        if attempt is None:
            context.log_event(planning_failure_event(context.run_id, planned_source))
            assert planned_source.failure is not None
            raise Failure(
                description=planned_source.failure.reason,
                metadata={
                    "janus/source_id": source_id,
                    "janus/failure": planned_source.failure.to_summary(),
                },
                allow_retries=True,
            )

        context.log_event(attempt_event(context.run_id, attempt))
        if attempt.status == "failed":
            assert attempt.failure is not None
            raise Failure(
                description=attempt.failure.reason,
                metadata={
                    "janus/source_id": source_id,
                    "janus/run_id": attempt.run_id,
                    "janus/attempt": attempt.attempt,
                    "janus/failure": attempt.failure.to_summary(),
                },
                allow_retries=True,
            )

    return execute_source


def _source_metadata(
    source_id: str,
    registry: SourceRegistry,
    edges: tuple[Any, ...],
) -> dict[str, Any]:
    source = registry.get_source(source_id)
    provenance = [
        {
            "producer_id": edge.producer_id,
            "table": edge.table,
            "input_paths": list(edge.input_paths),
        }
        for edge in edges
        if edge.consumer_id == source_id
    ]
    return {
        "janus/source_id": source.source_id,
        "janus/domain": source.domain,
        "janus/source_tags": list(source.tags),
        "janus/bronze_table": registry.graph.node(source_id).bronze_table,
        "janus/dependency_provenance": provenance,
    }


def _dependencies(source_ids, graph, op_names):
    dependencies: dict[str, dict[str, DependencyDefinition]] = {}
    for source_id in source_ids:
        upstream_ids = graph.upstreams(source_id)
        if upstream_ids:
            dependencies[op_names[source_id]] = {
                f"upstream_{index}": DependencyDefinition(op_names[upstream_id])
                for index, upstream_id in enumerate(upstream_ids)
            }
    return dependencies


def _terminal_sensor(status, job, summary_store_factory):
    @run_status_sensor(
        run_status=status,
        name=f"{job.name}_{status.value.lower()}_summary",
        monitored_jobs=[job],
        default_status=DefaultSensorStatus.RUNNING,
        description="Persist JANUS's shared pipeline summary after a terminal Dagster run.",
    )
    def collect_terminal_run(context):
        collect_dagster_run(
            context.instance,
            context.dagster_run.run_id,
            summary_store_factory=summary_store_factory,
        )

    return collect_terminal_run


def _tag_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
