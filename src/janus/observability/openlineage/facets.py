"""Pure JANUS-record to OpenLineage 2-0-2 ``RunEvent`` mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any
from urllib.parse import quote, urlsplit

from janus.lineage.models import LineageRecord, MaterializedOutput, RunMetadata
from janus.models.dependencies import SourceDependencyEdge, SourceDependencyGraph
from janus.observability.openlineage.constants import (
    DOCUMENTATION_JOB_FACET_SCHEMA_URL,
    ERROR_MESSAGE_RUN_FACET_SCHEMA_URL,
    JANUS_RUN_FACET_SCHEMA_URL,
    JOB_TYPE_JOB_FACET_SCHEMA_URL,
    OPENLINEAGE_PRODUCER,
    OPENLINEAGE_SCHEMA_URL,
    OUTPUT_STATISTICS_FACET_SCHEMA_URL,
    SOURCE_CODE_LOCATION_JOB_FACET_SCHEMA_URL,
    facet_base,
    openlineage_run_id,
)
from janus.observability.records import RunRecord

STATUS_TO_EVENT_TYPE = {
    "running": "START",
    "succeeded": "COMPLETE",
    "failed": "FAIL",
}

LINEAGE_FIELD_MAPPING: dict[str, str] = {
    "run_id": "run.runId (UUIDv5) and run.facets.janusRun.run_id",
    "source_id": "job.name and run.facets.janusRun.source_id",
    "source_name": "job.facets.documentation and run.facets.janusRun.source_name",
    "environment": "job.namespace and run.facets.janusRun.environment",
    "strategy_family": "run.facets.janusRun.strategy_family",
    "strategy_variant": "run.facets.janusRun.strategy_variant",
    "extraction_mode": "run.facets.janusRun.extraction_mode",
    "checkpoint_strategy": "run.facets.janusRun.checkpoint_strategy",
    "status": "eventType and run.facets.janusRun.status",
    "emitted_at": "eventTime and run.facets.janusRun.emitted_at",
    "source_config_path": "job.facets.sourceCodeLocation and janusRun.source_config_path",
    "config_version": "job.facets.sourceCodeLocation.version and janusRun.config_version",
    "configured_outputs": "run.facets.janusRun.configured_outputs",
    "materialized_outputs": "outputs and run.facets.janusRun.materialized_outputs",
    "artifacts": "outputs and run.facets.janusRun.artifacts",
    "checkpoint_field": "run.facets.janusRun.checkpoint_field",
    "source_hook": "run.facets.janusRun.source_hook",
    "records_extracted": "raw outputStatistics when exact, and janusRun.records_extracted",
    "checkpoint_value": "run.facets.janusRun.checkpoint_value",
    "failure_reason": "run.facets.errorMessage and run.facets.janusRun.failure_reason",
    "error_type": "run.facets.janusRun.error_type",
    "run_attributes": "run.facets.janusRun.run_attributes",
    "plan_notes": "run.facets.janusRun.plan_notes",
    "extraction_metadata": "run.facets.janusRun.extraction_metadata",
    "metadata": "run.facets.janusRun.metadata",
}
DELIBERATELY_DROPPED_LINEAGE_FIELDS: frozenset[str] = frozenset()

CUSTOM_ONLY_LINEAGE_FIELDS = frozenset(
    {
        "strategy_family",
        "strategy_variant",
        "extraction_mode",
        "checkpoint_strategy",
        "configured_outputs",
        "materialized_outputs",
        "artifacts",
        "config_version",
        "checkpoint_field",
        "source_hook",
        "checkpoint_value",
        "error_type",
        "run_attributes",
        "plan_notes",
        "extraction_metadata",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class OpenLineageDatasetContext:
    """Already-resolved catalog identity used to name datasets without doing I/O."""

    catalog_name: str
    warehouse: str

    def __post_init__(self) -> None:
        if not self.catalog_name.strip():
            raise ValueError("catalog_name must not be empty")
        if not self.warehouse.strip():
            raise ValueError("warehouse must not be empty")

    @property
    def iceberg_namespace(self) -> str:
        catalog = quote(self.catalog_name.strip(), safe="")
        warehouse = quote(self.warehouse.strip(), safe="")
        return f"iceberg://{catalog}/{warehouse}"


def build_openlineage_run_event(
    run_metadata: RunMetadata,
    dataset_context: OpenLineageDatasetContext,
    *,
    lineage_record: LineageRecord | None = None,
    run_record: RunRecord | None = None,
    graph: SourceDependencyGraph | None = None,
) -> dict[str, Any]:
    """Map one lifecycle record to a deterministic, JSON-serialisable ``RunEvent``."""

    _validate_records(run_metadata, lineage_record, run_record)
    event_type = STATUS_TO_EVENT_TYPE[run_metadata.status]
    dependencies = _declared_input_edges(graph, run_metadata.source_id)
    event_time = (
        run_metadata.started_at if event_type == "START" else _lineage(lineage_record).emitted_at
    )

    run_facets: dict[str, dict[str, Any]] = {
        "janusRun": _janus_run_facet(
            run_metadata,
            lineage_record=lineage_record,
            run_record=run_record,
            dependencies=dependencies,
        )
    }
    if run_metadata.failure_reason is not None:
        run_facets["errorMessage"] = {
            **facet_base(ERROR_MESSAGE_RUN_FACET_SCHEMA_URL),
            "message": run_metadata.failure_reason,
            "programmingLanguage": "Python",
        }

    return {
        "eventTime": event_time.isoformat(),
        "eventType": event_type,
        "run": {
            "runId": openlineage_run_id(run_metadata.run_id),
            "facets": run_facets,
        },
        "job": {
            "namespace": _job_namespace(run_metadata.environment),
            "name": run_metadata.source_id,
            "facets": _job_facets(run_metadata, lineage_record),
        },
        "inputs": [
            {
                "namespace": dataset_context.iceberg_namespace,
                "name": edge.table,
            }
            for edge in dependencies
        ],
        "outputs": _output_datasets(lineage_record, dataset_context),
        "producer": OPENLINEAGE_PRODUCER,
        "schemaURL": OPENLINEAGE_SCHEMA_URL,
    }


def _validate_records(
    run_metadata: RunMetadata,
    lineage_record: LineageRecord | None,
    run_record: RunRecord | None,
) -> None:
    if run_metadata.status == "running":
        if lineage_record is not None or run_record is not None:
            raise ValueError("START mapping accepts only the running RunMetadata")
        return
    if lineage_record is None or run_record is None:
        raise ValueError("terminal mapping requires both lineage_record and run_record")
    for label, candidate in (("lineage_record", lineage_record), ("run_record", run_record)):
        if candidate.run_id != run_metadata.run_id:
            raise ValueError(f"{label} must describe run {run_metadata.run_id!r}")
        if candidate.status != run_metadata.status:
            raise ValueError(f"{label} must agree with status {run_metadata.status!r}")
    if run_record.started_at != run_metadata.started_at:
        raise ValueError("run_record and run_metadata must agree on started_at")
    if run_record.emitted_at != lineage_record.emitted_at:
        raise ValueError("run_record and lineage_record must agree on emitted_at")


def _lineage(record: LineageRecord | None) -> LineageRecord:
    if record is None:
        raise ValueError("terminal mapping requires a lineage_record")
    return record


def _declared_input_edges(
    graph: SourceDependencyGraph | None,
    source_id: str,
) -> tuple[SourceDependencyEdge, ...]:
    if graph is None:
        return ()
    graph.node(source_id)
    return tuple(edge for edge in graph.edges if edge.consumer_id == source_id)


def _job_namespace(environment: str) -> str:
    return f"janus://{quote(environment.strip(), safe='')}"


def _job_facets(
    run_metadata: RunMetadata,
    lineage_record: LineageRecord | None,
) -> dict[str, dict[str, Any]]:
    source_location: dict[str, Any] = {
        **facet_base(SOURCE_CODE_LOCATION_JOB_FACET_SCHEMA_URL),
        "type": "file",
        "url": _file_uri(run_metadata.source_config_path),
        "path": run_metadata.source_config_path,
    }
    if lineage_record is not None:
        source_location["version"] = lineage_record.config_version
    return {
        "documentation": {
            **facet_base(DOCUMENTATION_JOB_FACET_SCHEMA_URL),
            "description": run_metadata.source_name,
            "contentType": "text/plain",
        },
        "jobType": {
            **facet_base(JOB_TYPE_JOB_FACET_SCHEMA_URL),
            "processingType": "BATCH",
            "integration": "JANUS",
            "jobType": "INGESTION",
        },
        "sourceCodeLocation": source_location,
    }


def _file_uri(path: str) -> str:
    encoded = quote(path, safe="/:")
    return f"file://{encoded}" if PurePath(path).is_absolute() else f"file:{encoded}"


def _janus_run_facet(
    run_metadata: RunMetadata,
    *,
    lineage_record: LineageRecord | None,
    run_record: RunRecord | None,
    dependencies: tuple[SourceDependencyEdge, ...],
) -> dict[str, Any]:
    fields = _lineage_fields(run_metadata, lineage_record)
    fields.update(
        {
            "_producer": OPENLINEAGE_PRODUCER,
            "_schemaURL": JANUS_RUN_FACET_SCHEMA_URL,
            "started_at": run_metadata.started_at.isoformat(),
            "checkpoint_decision": (
                run_record.checkpoint_decision if run_record is not None else None
            ),
            "checkpoint_advanced": (
                run_record.checkpoint_advanced if run_record is not None else None
            ),
            "quality": _quality_payload(run_record),
            "metadata_zone_paths": _metadata_zone_paths(run_record),
            "declared_inputs": [
                {
                    "producer_id": edge.producer_id,
                    "table": edge.table,
                    "input_paths": list(edge.input_paths),
                }
                for edge in dependencies
            ],
        }
    )
    return fields


def _lineage_fields(
    run_metadata: RunMetadata,
    lineage_record: LineageRecord | None,
) -> dict[str, Any]:
    payload = lineage_record.to_dict() if lineage_record is not None else run_metadata.to_dict()
    defaults: dict[str, Any] = {
        "emitted_at": None,
        "config_version": None,
        "artifacts": [],
        "source_hook": None,
        "extraction_metadata": {},
    }
    for name in LINEAGE_FIELD_MAPPING:
        payload.setdefault(name, defaults.get(name))
    return {name: payload[name] for name in LINEAGE_FIELD_MAPPING}


def _quality_payload(run_record: RunRecord | None) -> dict[str, Any] | None:
    if run_record is None:
        return None
    return {
        "outcome": run_record.quality_outcome,
        "checks_passed": run_record.quality_checks_passed,
        "checks_failed": run_record.quality_checks_failed,
        "checks_skipped": run_record.quality_checks_skipped,
        "failed_checks": (
            None
            if run_record.quality_failed_checks is None
            else list(run_record.quality_failed_checks)
        ),
    }


def _metadata_zone_paths(run_record: RunRecord | None) -> dict[str, str | None]:
    return {
        "run_metadata": run_record.run_metadata_path if run_record is not None else None,
        "lineage": run_record.lineage_path if run_record is not None else None,
        "checkpoint_history": (
            run_record.checkpoint_history_path if run_record is not None else None
        ),
        "validation_report": (
            run_record.validation_report_path if run_record is not None else None
        ),
    }


def _output_datasets(
    lineage_record: LineageRecord | None,
    context: OpenLineageDatasetContext,
) -> list[dict[str, Any]]:
    if lineage_record is None:
        return []

    outputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for output in lineage_record.materialized_outputs:
        dataset = _materialized_output_dataset(output, context)
        identity = (dataset["namespace"], dataset["name"])
        if identity not in seen:
            outputs.append(dataset)
            seen.add(identity)

    materialized_by_path = {output.path: output for output in lineage_record.materialized_outputs}
    for artifact in lineage_record.artifacts:
        namespace, name = _storage_dataset_identity(artifact.path)
        identity = (namespace, name)
        if identity in seen:
            continue
        artifact_dataset: dict[str, Any] = {"namespace": namespace, "name": name}
        if (
            len(lineage_record.artifacts) == 1
            and lineage_record.records_extracted is not None
            and artifact.path not in materialized_by_path
        ):
            artifact_dataset["outputFacets"] = {
                "outputStatistics": _output_statistics(lineage_record.records_extracted)
            }
        outputs.append(artifact_dataset)
        seen.add(identity)
    return outputs


def _materialized_output_dataset(
    output: MaterializedOutput,
    context: OpenLineageDatasetContext,
) -> dict[str, Any]:
    if output.zone == "bronze" and output.format.lower() == "iceberg":
        namespace, name = context.iceberg_namespace, output.path
    else:
        namespace, name = _storage_dataset_identity(output.path)
    dataset: dict[str, Any] = {"namespace": namespace, "name": name}
    if output.records_written is not None:
        dataset["outputFacets"] = {
            "outputStatistics": _output_statistics(output.records_written)
        }
    return dataset


def _output_statistics(row_count: int) -> dict[str, Any]:
    return {
        **facet_base(OUTPUT_STATISTICS_FACET_SCHEMA_URL),
        "rowCount": row_count,
    }


def _storage_dataset_identity(path: str) -> tuple[str, str]:
    parsed = urlsplit(path)
    if parsed.scheme and len(parsed.scheme) > 1:
        namespace = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else parsed.scheme
        name = parsed.path.lstrip("/") or path
        return namespace, name
    return "file", path
