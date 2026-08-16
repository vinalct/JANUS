"""Replay entry point: re-materialize bronze from an existing raw zone.

`RawToBronzeLoader` is the loader half of `--ingest-raw-to-bronze`: it drives the run
lifecycle (observation, Spark session, quality gate, metadata) and owns nothing about
the raw zone's shape. Rebuilding the `ExtractionResult` a previous run left on disk is
`janus.scripts.rehydrate`'s job; the write path itself (batch -> read -> normalize ->
write) is delegated to `janus.runtime.materialize.BronzeMaterializer`, the same
component the live executor uses, so replaying raw produces the same bronze as
`--execute`.

`_artifact_format_for_path`, `_rediscover_raw_artifacts` and `_sha256` are re-exported
below for callers that imported them from here before the split (FR-3).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractionResult, WriteResult
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, QualityGate, ValidationReportStore
from janus.readers import SparkDatasetReader
from janus.runtime.executor import _plan_with_storage_layout_outputs
from janus.runtime.materialize import (
    BronzeMaterializer,
    _bind_execution_logger,
    _default_storage_layout,
    _log_error,
    _log_exception,
    _log_info,
    _quality_failure_message,
    _raw_write_results,
    read_committed_bronze,
)
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.scripts.checksums import _artifact_format_for_path, _sha256
from janus.scripts.rehydrate import (
    _build_extraction_result_from_raw,
    _rediscover_raw_artifacts,
)
from janus.scripts.replay_plan import _bronze_target_identifier, _override_bronze_output
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

__all__ = [
    "RawToBronzeLoader",
    "RawToBronzeRun",
    "_artifact_format_for_path",
    "_rediscover_raw_artifacts",
    "_sha256",
    "ingest_raw_to_bronze",
]


@dataclass(frozen=True, slots=True)
class RawToBronzeRun:
    planned_run: PlannedRun
    status: str
    extraction_result: ExtractionResult
    handoff: ExtractionResult
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
        bronze_target = _bronze_target_identifier(self.planned_run.plan)
        actual_bronze_path = next(
            (result.path for result in self.write_results if result.zone == "bronze"),
            bronze_target,
        )
        summary: dict[str, Any] = {
            "status": self.status,
            "raw_artifact_count": len(self.extraction_result.artifacts),
            "handoff_artifact_count": len(self.handoff.artifacts),
            "target_table": actual_bronze_path,
            "strategy_metadata": self.strategy_metadata,
            "materialized_outputs": [
                {
                    "zone": result.zone,
                    "path": result.path,
                    "format": result.format,
                    "mode": result.mode,
                    "records_written": result.records_written,
                    "partition_by": list(result.partition_by),
                    "metadata": result.metadata_as_dict(),
                }
                for result in self.write_results
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
class RawToBronzeLoader:
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

    def ingest(
        self,
        planned_run: PlannedRun,
        spark: SparkSessionProvider | SparkSession,
        environment_config: Mapping[str, Any],
        *,
        bronze_table: str,
    ) -> RawToBronzeRun:
        # Replay converges on the live path's lifecycle: rediscovery and rehydration
        # are pure Python, so Spark starts only where BronzeMaterializer needs it.
        # A caller that hands over a live session keeps owning it.
        spark_provider = (
            spark
            if isinstance(spark, SparkSessionProvider)
            else SparkSessionProvider.wrapping(spark)
        )
        storage_layout = self.storage_layout_resolver(planned_run.plan, environment_config)
        plan = _plan_with_storage_layout_outputs(planned_run.plan, storage_layout)
        plan = _override_bronze_output(plan, bronze_table)
        runtime_planned_run = replace(planned_run, plan=plan)
        logger = _bind_execution_logger(self.logger, plan)

        extraction_result = ExtractionResult.from_plan(plan, ())
        handoff = extraction_result
        write_results: tuple[WriteResult, ...] = ()
        validation_report: PersistedValidationReport | None = None
        strategy_metadata: dict[str, Any] = {}

        try:
            try:
                _log_info(
                    logger,
                    "raw_to_bronze_started",
                    raw_output_path=plan.raw_output.path,
                    bronze_output_path=plan.bronze_output.path,
                    target_table=_bronze_target_identifier(plan),
                )

                self.observer.start_run(plan)
                _log_info(logger, "run_observation_started")

                extraction_result = _build_extraction_result_from_raw(
                    runtime_planned_run,
                    plan,
                    spark_provider,
                    storage_layout,
                )
                write_results = _raw_write_results(plan, extraction_result)
                _log_info(
                    logger,
                    "raw_artifacts_rediscovered",
                    artifact_count=len(extraction_result.artifacts),
                    raw_output_path=plan.raw_output.path,
                )

                handoff = runtime_planned_run.strategy.build_normalization_handoff(
                    plan,
                    extraction_result,
                    hook=runtime_planned_run.hook,
                )
                _log_info(
                    logger,
                    "normalization_handoff_prepared",
                    artifact_count=len(handoff.artifacts),
                    is_empty=handoff.is_empty,
                )

                normalized_dataframe = None
                bronze_dataframe = None
                run_keys = None
                if not handoff.is_empty:
                    materializer = BronzeMaterializer(
                        reader=self.reader,
                        normalizer=self.normalizer,
                        writer_factory=self.writer_factory,
                    )
                    bronze_results, normalized_dataframe, run_keys = materializer.materialize(
                        runtime_planned_run,
                        plan,
                        spark_provider.get(),
                        handoff,
                        storage_layout,
                        logger,
                        bronze_target_identifier=_bronze_target_identifier(plan),
                    )
                    write_results = write_results + bronze_results
                    # Symmetric with the executor: read the committed table for the bronze
                    # uniqueness oracle while the replay's session is still live.
                    bronze_dataframe = read_committed_bronze(
                        spark_provider.get(),
                        bronze_results,
                    )
                else:
                    _log_info(logger, "spark_session_skipped")

                strategy_metadata = dict(
                    runtime_planned_run.strategy.emit_metadata(
                        plan,
                        extraction_result,
                        write_results,
                        hook=runtime_planned_run.hook,
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
                    bronze_dataframe=bronze_dataframe,
                    run_keys=run_keys,
                    raise_on_failure=False,
                )
                _log_info(
                    logger,
                    "quality_validation_finished",
                    is_successful=validation_report.report.is_successful,
                    summary=validation_report.report.summary(),
                    validation_report_path=str(validation_report.path),
                )
                # Mirrors the executor boundary: quality validation was the last
                # consumer of the session-bound DataFrame, so observation and
                # metadata persistence below run session-free.
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
                        "raw_to_bronze_failed",
                        failure_reason=str(failure),
                        error_type=type(failure).__name__,
                    )
                    return _build_result(
                        runtime_planned_run,
                        status="failed",
                        extraction_result=extraction_result,
                        handoff=handoff,
                        write_results=write_results,
                        validation_report=validation_report,
                        strategy_metadata=strategy_metadata,
                        persisted=persisted,
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
                    "raw_to_bronze_succeeded",
                    raw_artifact_count=len(extraction_result.artifacts),
                    handoff_artifact_count=len(handoff.artifacts),
                    materialized_output_count=len(write_results),
                    target_table=_bronze_target_identifier(plan),
                )
                return _build_result(
                    runtime_planned_run,
                    status="succeeded",
                    extraction_result=extraction_result,
                    handoff=handoff,
                    write_results=write_results,
                    validation_report=validation_report,
                    strategy_metadata=strategy_metadata,
                    persisted=persisted,
                )
            finally:
                # Idempotent, so the release above is not repeated; this guarantees
                # one on the paths that bypassed it.
                spark_provider.stop()
        except Exception as exc:
            _log_exception(
                logger,
                "raw_to_bronze_failed",
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
            return _build_result(
                runtime_planned_run,
                status="failed",
                extraction_result=extraction_result,
                handoff=handoff,
                write_results=write_results,
                validation_report=validation_report,
                strategy_metadata=strategy_metadata,
                persisted=persisted,
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )


def ingest_raw_to_bronze(
    planned_run: PlannedRun,
    spark: SparkSessionProvider | SparkSession,
    environment_config: Mapping[str, Any],
    *,
    bronze_table: str,
    logger: StructuredLogger | None = None,
) -> RawToBronzeRun:
    return RawToBronzeLoader(logger=logger).ingest(
        planned_run,
        spark,
        environment_config,
        bronze_table=bronze_table,
    )


def _build_result(
    planned_run: PlannedRun,
    *,
    status: str,
    extraction_result: ExtractionResult,
    handoff: ExtractionResult,
    write_results: tuple[WriteResult, ...],
    validation_report: PersistedValidationReport | None,
    strategy_metadata: dict[str, Any],
    persisted,
    failure_reason: str | None = None,
    error_type: str | None = None,
) -> RawToBronzeRun:
    checkpoint_result = getattr(persisted, "checkpoint_result", None)
    checkpoint_state_path = None
    checkpoint_history_path = None
    if checkpoint_result is not None:
        checkpoint_state_path = checkpoint_result.current_path
        checkpoint_history_path = checkpoint_result.history_path

    return RawToBronzeRun(
        planned_run=planned_run,
        status=status,
        extraction_result=extraction_result,
        handoff=handoff,
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
