"""Shared bronze-materialization pipeline used by the live executor and the raw-replay loader.

Both `janus.runtime.executor` and `janus.scripts.raw_to_bronze` drive the same
batch -> Spark read -> normalize -> write-to-bronze pipeline; everything downstream
of the normalization handoff is defined here exactly once. This module is a leaf:
it must not import from either caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, WriteResult
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport
from janus.readers import SparkDatasetReader
from janus.schema_contracts import resolve_spark_schema_for_plan
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_FILE_HANDOFF_ARTIFACTS_PER_BATCH = 5


@dataclass(slots=True)
class BronzeMaterializer:
    """Owns the batch -> read -> normalize -> write-to-bronze loop for both entry points."""

    reader: SparkDatasetReader
    normalizer: BaseNormalizer
    writer_factory: Callable[[StorageLayout], SparkDatasetWriter]

    def materialize(
        self,
        planned_run: PlannedRun,
        plan: ExecutionPlan,
        spark: SparkSession,
        handoff: ExtractionResult,
        storage_layout: StorageLayout,
        logger: StructuredLogger | None,
        *,
        bronze_target_identifier: str | None = None,
    ) -> tuple[tuple[WriteResult, ...], Any | None]:
        batches = _normalization_handoff_batches(planned_run, handoff)
        if len(batches) > 1:
            _log_info(
                logger,
                "bronze_write_batches_started",
                batch_count=len(batches),
                handoff_artifact_count=len(handoff.artifacts),
                batch_artifact_limit=_FILE_HANDOFF_ARTIFACTS_PER_BATCH,
            )

        writer = self.writer_factory(storage_layout)
        bronze_results: list[WriteResult] = []
        normalized_dataframe = None
        for batch_index, batch_artifacts in enumerate(batches, start=1):
            batch_handoff = replace(handoff, artifacts=batch_artifacts).with_metadata(
                "normalization_artifact_count",
                str(len(batch_artifacts)),
            )
            batch_metadata = _batch_log_metadata(
                batch_index,
                len(batches),
                batch_artifacts,
            )

            _log_info(
                logger,
                "spark_read_started",
                artifact_count=len(batch_artifacts),
                **batch_metadata,
            )
            handoff_format = batch_handoff.single_artifact_format()
            spark_schema = (
                resolve_spark_schema_for_plan(plan)
                if handoff_format == plan.source_config.spark.input_format
                else None
            )
            read_options = (
                plan.source_config.spark.read_options
                if handoff_format == plan.source_config.spark.input_format
                else None
            )
            raw_dataframe = self.reader.read_extraction_result(
                spark,
                batch_handoff,
                format_name=handoff_format,
                schema=spark_schema,
                options=read_options,
            )
            _log_info(logger, "spark_read_finished", **batch_metadata)

            _log_info(logger, "normalization_started", **batch_metadata)
            normalized_dataframe = self.normalizer.normalize(raw_dataframe, plan)
            _log_info(logger, "normalization_finished", **batch_metadata)

            write_mode = _write_mode_for_batch(plan, batch_index)
            started_fields: dict[str, Any] = {
                "bronze_output_path": plan.bronze_output.path,
                "write_mode": write_mode,
                **batch_metadata,
            }
            if bronze_target_identifier is not None:
                started_fields["target_table"] = bronze_target_identifier
            _log_info(logger, "bronze_write_started", **started_fields)
            bronze_result = writer.write(
                normalized_dataframe,
                plan,
                "bronze",
                mode=write_mode,
                count_records=_should_count_records_for_handoff(planned_run),
            )
            bronze_results.append(bronze_result)
            _log_info(
                logger,
                "bronze_write_finished",
                path=bronze_result.path,
                format=bronze_result.format,
                mode=bronze_result.mode,
                records_written=bronze_result.records_written,
                partition_by=list(bronze_result.partition_by),
                **batch_metadata,
            )

        if len(batches) > 1:
            _log_info(
                logger,
                "bronze_write_batches_finished",
                batch_count=len(batches),
                materialized_output_count=len(bronze_results),
            )
        return tuple(bronze_results), normalized_dataframe


def _normalization_handoff_batches(
    planned_run: PlannedRun,
    handoff: ExtractionResult,
) -> tuple[tuple[ExtractedArtifact, ...], ...]:
    artifacts = handoff.artifacts
    if not artifacts:
        return ()

    if getattr(planned_run.strategy, "strategy_family", None) != "file":
        return (artifacts,)

    limit = _FILE_HANDOFF_ARTIFACTS_PER_BATCH
    return tuple(artifacts[index : index + limit] for index in range(0, len(artifacts), limit))


def _write_mode_for_batch(plan: ExecutionPlan, batch_index: int) -> str:
    write_mode = plan.source_config.spark.write_mode
    if batch_index > 1 and write_mode == "overwrite":
        return "append"
    return write_mode


def _should_count_records_for_handoff(planned_run: PlannedRun) -> bool:
    return getattr(planned_run.strategy, "strategy_family", None) != "file"


def _batch_log_metadata(
    batch_index: int,
    batch_count: int,
    batch_artifacts: tuple[ExtractedArtifact, ...],
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "batch_count": batch_count,
        "batch_artifact_count": len(batch_artifacts),
        "batch_first_artifact": batch_artifacts[0].path if batch_artifacts else None,
    }


def _bind_execution_logger(
    logger: StructuredLogger | None,
    plan: ExecutionPlan,
) -> StructuredLogger | None:
    if logger is None:
        return None
    return logger.bind(
        run_id=plan.run_context.run_id,
        source_id=plan.source.source_id,
        source_name=plan.source.name,
        environment=plan.run_context.environment,
        strategy_family=plan.source.strategy,
        strategy_variant=plan.source.strategy_variant,
    )


def _log_info(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.info(event, **fields)


def _log_error(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.error(event, **fields)


def _log_exception(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.exception(event, **fields)


def _default_storage_layout(
    plan: ExecutionPlan,
    environment_config: Mapping[str, Any],
) -> StorageLayout:
    return StorageLayout.from_environment_config(
        environment_config,
        plan.run_context.project_root,
    )


def _quality_failure_message(report: PersistedValidationReport) -> str:
    failed_checks = ", ".join(
        f"{check.phase}.{check.name}" for check in report.report.failed_checks
    )
    if not failed_checks:
        return "Quality validation failed"
    return f"Quality validation failed: {failed_checks}"


def _raw_write_results(
    plan: ExecutionPlan,
    extraction_result: ExtractionResult,
) -> tuple[WriteResult, ...]:
    return tuple(
        WriteResult.from_plan(
            plan,
            "raw",
            path=artifact.path,
            format_name=artifact.format,
            mode="overwrite",
            records_written=1,
            partition_by=(),
            metadata={"checksum": artifact.checksum} if artifact.checksum else None,
        )
        for artifact in extraction_result.artifacts
    )
