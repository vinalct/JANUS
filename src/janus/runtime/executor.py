from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractionResult, WriteResult
from janus.models.source_config import OutputTarget
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, QualityGate, ValidationReportStore
from janus.readers import SparkDatasetReader
from janus.runtime.materialize import (
    BronzeMaterializer,
    _bind_execution_logger,
    _default_storage_layout,
    _log_error,
    _log_exception,
    _log_info,
    _quality_failure_message,
    _raw_write_results,
)
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter


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
        spark_provider: SparkSessionProvider,
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
                # The provider — never a live session — crosses into extract(). Only the
                # api/catalog request-input lookup materializes one from it, scoped to
                # that lookup; the download itself always runs session-free.
                extraction_result = planned_run.strategy.extract(
                    plan,
                    hook=planned_run.hook,
                    spark=spark_provider,
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
                    materializer = BronzeMaterializer(
                        reader=self.reader,
                        normalizer=self.normalizer,
                        writer_factory=self.writer_factory,
                    )
                    bronze_results, normalized_dataframe = materializer.materialize(
                        runtime_planned_run,
                        plan,
                        spark_provider.get(),
                        handoff,
                        storage_layout,
                        logger,
                    )
                    write_results = raw_write_results + bronze_results
                else:
                    _log_info(logger, "spark_session_skipped")

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
                # Quality validation was the last consumer of the session-bound
                # DataFrame; everything below — observation, checkpointing, metadata
                # persistence — runs session-free. The outer `finally` is the backstop
                # for paths that never reach this point.
                spark_provider.stop()

                if not validation_report.report.is_successful:
                    failure = RuntimeError(_quality_failure_message(validation_report))
                    persisted = self.observer.record_failure(
                        plan,
                        failure,
                        extraction_result,
                        write_results,
                        strategy_metadata=strategy_metadata,
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

                persisted = self.observer.record_success(
                    plan,
                    extraction_result,
                    write_results,
                    strategy_metadata=strategy_metadata,
                )
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
            finally:
                # Idempotent, so the release above is not repeated; this guarantees one
                # on the paths that bypassed it — extraction, materialization
                spark_provider.stop()
        except Exception as exc:
            _log_exception(
                logger,
                "source_execution_failed",
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )
            persisted = self.observer.record_failure(
                plan,
                exc,
                extraction_result,
                write_results,
                strategy_metadata=strategy_metadata,
            )
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


def _attach_strategy_logger(strategy: Any, logger: StructuredLogger | None) -> None:
    if logger is None or not hasattr(strategy, "logger"):
        return
    if strategy.logger is None:
        strategy.logger = logger


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
