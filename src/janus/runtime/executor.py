from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, WriteResult
from janus.models.source_config import OutputTarget
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, QualityGate, ValidationReportStore
from janus.readers import SparkDatasetReader
from janus.schema_contracts import resolve_spark_schema_for_plan
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_FILE_HANDOFF_ARTIFACTS_PER_BATCH = 5


@dataclass(frozen=True, slots=True)
class ExecutedRun:
    planned_run: PlannedRun
    status: str
    extraction_result: ExtractionResult | None = None
    write_results: tuple[WriteResult, ...] = ()
    validation_report: PersistedValidationReport | None = None
    strategy_metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata_path: Path | None = None
    lineage_path: Path | None = None
    checkpoint_state_path: Path | None = None
    checkpoint_history_path: Path | None = None
    failure_reason: str | None = None
    error_type: str | None = None

    @property
    def is_successful(self) -> bool:
        return self.status == "succeeded"

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": self.status,
            "strategy_metadata": self.strategy_metadata,
            "materialized_outputs": [
                {
                    "zone": write_result.zone,
                    "path": write_result.path,
                    "format": write_result.format,
                    "mode": write_result.mode,
                    "records_written": write_result.records_written,
                    "partition_by": list(write_result.partition_by),
                    "metadata": write_result.metadata_as_dict(),
                }
                for write_result in self.write_results
            ],
            "metadata_outputs": {
                "run_metadata_path": (
                    str(self.run_metadata_path) if self.run_metadata_path is not None else None
                ),
                "lineage_path": str(self.lineage_path) if self.lineage_path is not None else None,
                "checkpoint_state_path": (
                    str(self.checkpoint_state_path)
                    if self.checkpoint_state_path is not None
                    else None
                ),
                "checkpoint_history_path": (
                    str(self.checkpoint_history_path)
                    if self.checkpoint_history_path is not None
                    else None
                ),
                "validation_report_path": (
                    str(self.validation_report.path)
                    if self.validation_report is not None
                    else None
                ),
            },
        }

        if self.extraction_result is not None:
            summary["records_extracted"] = self.extraction_result.records_extracted
            summary["checkpoint_value"] = self.extraction_result.checkpoint_value
            summary["artifact_count"] = len(self.extraction_result.artifacts)
            summary["artifacts"] = [
                {
                    "path": artifact.path,
                    "format": artifact.format,
                    "checksum": artifact.checksum,
                }
                for artifact in self.extraction_result.artifacts
            ]
            summary["extraction_metadata"] = self.extraction_result.metadata_as_dict()

        if self.validation_report is not None:
            summary["validation"] = {
                "is_successful": self.validation_report.report.is_successful,
                "summary": self.validation_report.report.summary(),
                "failed_checks": [
                    f"{check.phase}.{check.name}"
                    for check in self.validation_report.report.failed_checks
                ],
            }

        if self.failure_reason is not None:
            summary["failure_reason"] = self.failure_reason
        if self.error_type is not None:
            summary["error_type"] = self.error_type

        return summary


@dataclass(slots=True)
class SourceExecutor:
    logger: StructuredLogger | None = None
    reader: SparkDatasetReader = field(default_factory=SparkDatasetReader)
    normalizer: BaseNormalizer = field(default_factory=BaseNormalizer)
    quality_gate: QualityGate = field(
        default_factory=lambda: QualityGate(ValidationReportStore())
    )
    observer: RunObserver = field(default_factory=RunObserver)
    writer_factory: Callable[[StorageLayout], SparkDatasetWriter] = SparkDatasetWriter
    storage_layout_resolver: Callable[[ExecutionPlan, Mapping[str, Any]], StorageLayout] = field(
        default_factory=lambda: _default_storage_layout
    )

    def execute(
        self,
        planned_run: PlannedRun,
        spark: SparkSession,
        environment_config: Mapping[str, Any],
    ) -> ExecutedRun:
        storage_layout = self.storage_layout_resolver(planned_run.plan, environment_config)
        plan = _plan_with_storage_layout_outputs(planned_run.plan, storage_layout)
        runtime_planned_run = replace(planned_run, plan=plan)
        logger = _bind_execution_logger(self.logger, plan)
        _attach_strategy_logger(planned_run.strategy, logger)

        extraction_result: ExtractionResult | None = None
        write_results: tuple[WriteResult, ...] = ()
        validation_report: PersistedValidationReport | None = None
        strategy_metadata: dict[str, Any] = {}

        try:
            _log_info(
                logger,
                "source_execution_started",
                extraction_mode=plan.extraction_mode,
                checkpoint_strategy=plan.checkpoint_strategy,
                raw_output_path=plan.raw_output.path,
                bronze_output_path=plan.bronze_output.path,
                metadata_output_path=plan.metadata_output.path,
            )

            self.observer.start_run(plan)
            _log_info(logger, "run_observation_started")

            _log_info(
                logger,
                "source_extraction_started",
                pagination_type=plan.source_config.access.pagination.type,
                auth_type=plan.source_config.access.auth.type,
            )
            extraction_result = planned_run.strategy.extract(
                plan,
                hook=planned_run.hook,
                spark=spark,
            )
            _log_info(
                logger,
                "source_extraction_finished",
                records_extracted=extraction_result.records_extracted or 0,
                artifact_count=len(extraction_result.artifacts),
                checkpoint_value=extraction_result.checkpoint_value,
            )

            raw_write_results = _raw_write_results(plan, extraction_result)
            write_results = raw_write_results
            _log_info(
                logger,
                "raw_outputs_materialized",
                artifact_count=len(raw_write_results),
            )

            handoff = planned_run.strategy.build_normalization_handoff(
                plan,
                extraction_result,
                hook=planned_run.hook,
            )
            _log_info(
                logger,
                "normalization_handoff_prepared",
                artifact_count=len(handoff.artifacts),
                is_empty=handoff.is_empty,
            )

            normalized_dataframe = None
            if not handoff.is_empty:
                bronze_results, normalized_dataframe = self._write_handoff_to_bronze(
                    runtime_planned_run,
                    plan,
                    spark,
                    handoff,
                    storage_layout,
                    logger,
                )
                write_results = raw_write_results + bronze_results

            strategy_metadata = dict(
                planned_run.strategy.emit_metadata(
                    plan,
                    extraction_result,
                    write_results,
                    hook=planned_run.hook,
                )
            )
            _log_info(
                logger,
                "strategy_metadata_emitted",
                metadata_keys=sorted(strategy_metadata),
            )

            _log_info(logger, "quality_validation_started")
            validation_report = self.quality_gate.validate_and_store(
                plan,
                dataframe=normalized_dataframe,
                write_results=write_results,
                raise_on_failure=False,
            )
            _log_info(
                logger,
                "quality_validation_finished",
                is_successful=validation_report.report.is_successful,
                summary=validation_report.report.summary(),
                validation_report_path=str(validation_report.path),
            )
            if not validation_report.report.is_successful:
                failure = RuntimeError(_quality_failure_message(validation_report))
                persisted = self.observer.record_failure(
                    plan,
                    failure,
                    extraction_result,
                    write_results,
                )
                _log_error(
                    logger,
                    "source_execution_failed",
                    failure_reason=str(failure),
                    error_type=type(failure).__name__,
                )
                return _build_executed_run(
                    runtime_planned_run,
                    status="failed",
                    persisted=persisted,
                    extraction_result=extraction_result,
                    write_results=write_results,
                    validation_report=validation_report,
                    strategy_metadata=strategy_metadata,
                    failure_reason=str(failure),
                    error_type=type(failure).__name__,
                )

            persisted = self.observer.record_success(plan, extraction_result, write_results)
            _log_info(
                logger,
                "source_execution_succeeded",
                records_extracted=extraction_result.records_extracted or 0,
                materialized_output_count=len(write_results),
                run_metadata_path=str(getattr(persisted, "run_metadata_path", "")),
                lineage_path=str(getattr(persisted, "lineage_path", "")),
            )
            return _build_executed_run(
                runtime_planned_run,
                status="succeeded",
                persisted=persisted,
                extraction_result=extraction_result,
                write_results=write_results,
                validation_report=validation_report,
                strategy_metadata=strategy_metadata,
            )
        except Exception as exc:
            _log_exception(
                logger,
                "source_execution_failed",
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )
            persisted = self.observer.record_failure(plan, exc, extraction_result, write_results)
            return _build_executed_run(
                runtime_planned_run,
                status="failed",
                persisted=persisted,
                extraction_result=extraction_result,
                write_results=write_results,
                validation_report=validation_report,
                strategy_metadata=strategy_metadata,
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )

    def _write_handoff_to_bronze(
        self,
        planned_run: PlannedRun,
        plan: ExecutionPlan,
        spark: SparkSession,
        handoff: ExtractionResult,
        storage_layout: StorageLayout,
        logger: StructuredLogger | None,
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
            _log_info(
                logger,
                "bronze_write_started",
                bronze_output_path=plan.bronze_output.path,
                write_mode=write_mode,
                **batch_metadata,
            )
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


def _attach_strategy_logger(strategy: Any, logger: StructuredLogger | None) -> None:
    if logger is None or not hasattr(strategy, "logger"):
        return
    if strategy.logger is None:
        strategy.logger = logger


def _log_info(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.info(event, **fields)


def _log_error(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.error(event, **fields)


def _log_exception(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.exception(event, **fields)


def _plan_with_storage_layout_outputs(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
) -> ExecutionPlan:
    raw_target = storage_layout.resolve_output(plan, "raw")
    bronze_target = storage_layout.resolve_output(plan, "bronze")
    metadata_target = storage_layout.resolve_output(plan, "metadata")
    return replace(
        plan,
        raw_output=OutputTarget(
            path=str(raw_target.resolved_path),
            format=raw_target.format,
            namespace=plan.raw_output.namespace,
            table_name=plan.raw_output.table_name,
        ),
        bronze_output=OutputTarget(
            path=str(bronze_target.resolved_path),
            format=bronze_target.format,
            namespace=plan.bronze_output.namespace,
            table_name=plan.bronze_output.table_name,
        ),
        metadata_output=OutputTarget(
            path=str(metadata_target.resolved_path),
            format=metadata_target.format,
            namespace=plan.metadata_output.namespace,
            table_name=plan.metadata_output.table_name,
        ),
    )


def _build_executed_run(
    planned_run: PlannedRun,
    *,
    status: str,
    persisted,
    extraction_result: ExtractionResult | None,
    write_results: tuple[WriteResult, ...],
    validation_report: PersistedValidationReport | None,
    strategy_metadata: dict[str, Any],
    failure_reason: str | None = None,
    error_type: str | None = None,
) -> ExecutedRun:
    checkpoint_result = getattr(persisted, "checkpoint_result", None)
    checkpoint_state_path = None
    checkpoint_history_path = None
    if checkpoint_result is not None:
        checkpoint_state_path = checkpoint_result.current_path
        checkpoint_history_path = checkpoint_result.history_path

    return ExecutedRun(
        planned_run=planned_run,
        status=status,
        extraction_result=extraction_result,
        write_results=write_results,
        validation_report=validation_report,
        strategy_metadata=strategy_metadata,
        run_metadata_path=getattr(persisted, "run_metadata_path", None),
        lineage_path=getattr(persisted, "lineage_path", None),
        checkpoint_state_path=checkpoint_state_path,
        checkpoint_history_path=checkpoint_history_path,
        failure_reason=failure_reason,
        error_type=error_type,
    )


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
