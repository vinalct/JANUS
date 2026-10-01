"""Shared bronze-materialization pipeline used by the live executor and the raw-replay loader.

Both `janus.runtime.executor` and `janus.scripts.raw_to_bronze` drive the same
batch -> Spark read -> contract check -> normalize -> write-to-bronze pipeline; everything
downstream of the normalization handoff is defined here exactly once. This module is a leaf:
it must not import from either caller.

No batch reaches ``writer.write`` without passing the pre-write contract check. A refusal is
raised, never handled here: the entry points decide what a failed run records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from janus.models import (
    BronzeWriteIntent,
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    WriteResult,
    resolve_bronze_write_intent,
)
from janus.models.data_contracts import DataContract
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.quality import (
    CORRUPT_RECORD_COLUMN,
    ContractEnforcementError,
    MissingContractError,
    PersistedValidationReport,
    PreWriteEvidence,
    run_pre_write_pass,
)
from janus.quality.pre_write import pre_write_aggregates
from janus.readers import SparkDatasetReader
from janus.readers.spark import CORRUPT_RECORD_READ_FORMATS
from janus.schema_contracts import spark_schema_from_contract
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from janus.runtime.spark_lifecycle import SparkSessionProvider

_FILE_HANDOFF_ARTIFACTS_PER_BATCH = 5


@dataclass(frozen=True, slots=True)
class MaterializedHandoff:
    """What materializing one handoff hands the quality gate; empty when the handoff was."""

    bronze_results: tuple[WriteResult, ...] = ()
    normalized_dataframe: Any | None = None
    bronze_dataframe: Any | None = None
    run_keys: Any | None = None
    pre_write_evidence: tuple[PreWriteEvidence, ...] | None = None


@dataclass(slots=True)
class BronzeMaterializer:
    """Owns the batch -> read -> check -> normalize -> write-to-bronze loop of both entry points."""

    reader: SparkDatasetReader
    normalizer: BaseNormalizer
    writer_factory: Callable[[StorageLayout], SparkDatasetWriter]

    def materialize_handoff(
        self,
        planned_run: PlannedRun,
        plan: ExecutionPlan,
        spark_provider: SparkSessionProvider,
        handoff: ExtractionResult,
        storage_layout: StorageLayout,
        logger: StructuredLogger | None,
        *,
        bronze_target_identifier: str | None = None,
    ) -> MaterializedHandoff:
        """Materialize a non-empty handoff in the provider's session; start none for an empty."""
        if handoff.is_empty:
            _log_info(logger, "spark_session_skipped")
            return MaterializedHandoff()
        bronze_results, normalized_dataframe, run_keys, evidence = self.materialize(
            planned_run,
            plan,
            spark_provider.get(),
            handoff,
            storage_layout,
            logger,
            bronze_target_identifier=bronze_target_identifier,
        )
        return MaterializedHandoff(
            bronze_results=bronze_results,
            normalized_dataframe=normalized_dataframe,
            # Read the committed table for the bronze uniqueness oracle while the session is
            # still live — this is not a new lifetime.
            bronze_dataframe=read_committed_bronze(spark_provider.get(), bronze_results),
            run_keys=run_keys,
            pre_write_evidence=evidence,
        )

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
    ) -> tuple[tuple[WriteResult, ...], Any | None, Any | None, tuple[PreWriteEvidence, ...]]:
        """Write every batch that passes its contract check; return the evidence of each one.

        Raises ``MissingContractError`` for a plan without a contract, and the check's
        ``ContractEnforcementError`` for the first batch that fails it, carrying the evidence of
        every batch checked so far and the bronze results of the batches already committed.
        """
        contract = plan.data_contract
        if contract is None:
            raise MissingContractError(plan.source.source_id)

        batches = _normalization_handoff_batches(planned_run, handoff)
        if len(batches) > 1:
            _log_info(
                logger,
                "bronze_write_batches_started",
                batch_count=len(batches),
                handoff_artifact_count=len(handoff.artifacts),
                batch_artifact_limit=_FILE_HANDOFF_ARTIFACTS_PER_BATCH,
            )

        # Resolve the write intent exactly once per run
        run_intent = resolve_bronze_write_intent(plan)
        # Keys the run wrote, accumulated across every batch so the bronze uniqueness
        # oracle covers rows from batches 1..n-1, not only the last one it validates.
        primary_key = contract.primary_key
        writer = self.writer_factory(storage_layout)
        bronze_results: list[WriteResult] = []
        evidence: list[PreWriteEvidence] = []
        normalized_dataframe = None
        run_keys = None
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

            tracks_corrupt = _tracks_corrupt_records(
                self.reader, batch_handoff.single_artifact_format(), plan
            )
            raw_dataframe = _read_batch(
                self.reader, spark, batch_handoff, plan, contract, logger, batch_metadata
            )

            # A strict pass aggregates before the write reads the same rows again: persist so
            # the two share one scan, and release it only after the write (or its failure).
            persisted = pre_write_aggregates(
                contract, enforcement=contract.janus.enforcement, tracks_corrupt=tracks_corrupt
            )
            if persisted:
                raw_dataframe = _persist(raw_dataframe)
            try:
                try:
                    checked_dataframe, batch_evidence = run_pre_write_pass(
                        raw_dataframe,
                        contract,
                        enforcement=contract.janus.enforcement,
                        batch_index=batch_metadata["batch_index"],
                        batch_count=batch_metadata["batch_count"],
                        max_malformed_rows=contract.janus.max_malformed_rows,
                        tracks_corrupt=tracks_corrupt,
                    )
                except ContractEnforcementError as exc:
                    _annotate_contract_failure(
                        exc, evidence, bronze_results, logger, contract.id, batch_metadata
                    )
                    raise
                evidence.append(batch_evidence)
                _log_contract_check_passed(logger, batch_evidence, batch_metadata)

                _log_info(logger, "normalization_started", **batch_metadata)
                normalized_dataframe = self.normalizer.normalize(checked_dataframe, plan)
                _log_info(logger, "normalization_finished", **batch_metadata)

                run_keys = _accumulate_run_keys(run_keys, normalized_dataframe, primary_key)

                batch_intent = run_intent.for_batch(batch_index)
                intent_fields = _intent_log_fields(batch_intent)
                started_fields: dict[str, Any] = {
                    "bronze_output_path": plan.bronze_output.path,
                    **intent_fields,
                    **batch_metadata,
                }
                if bronze_target_identifier is not None:
                    started_fields["target_table"] = bronze_target_identifier
                _log_info(logger, "bronze_write_started", **started_fields)
                bronze_result = writer.write(
                    normalized_dataframe,
                    plan,
                    "bronze",
                    intent=batch_intent,
                    batch_index=batch_index,
                    count_records=_should_count_records_for_handoff(planned_run),
                )
            finally:
                if persisted:
                    raw_dataframe.unpersist()
            bronze_results.append(bronze_result)
            _log_info(
                logger,
                "bronze_write_finished",
                path=bronze_result.path,
                format=bronze_result.format,
                mode=bronze_result.mode,
                records_written=bronze_result.records_written,
                partition_by=list(bronze_result.partition_by),
                **intent_fields,
                **batch_metadata,
            )

        if len(batches) > 1:
            _log_info(
                logger,
                "bronze_write_batches_finished",
                batch_count=len(batches),
                materialized_output_count=len(bronze_results),
            )
        if run_keys is not None:
            run_keys = run_keys.distinct()
        return tuple(bronze_results), normalized_dataframe, run_keys, tuple(evidence)


def _tracks_corrupt_records(
    reader: SparkDatasetReader, handoff_format: str, plan: ExecutionPlan
) -> bool:
    """Only readers that can expose corrupt records opt into the pre-write count."""
    return (
        getattr(reader, "supports_corrupt_record_read", False)
        and handoff_format in CORRUPT_RECORD_READ_FORMATS
        and handoff_format == plan.source_config.spark.input_format
    )


def _read_batch(
    reader: SparkDatasetReader,
    spark: SparkSession,
    handoff: ExtractionResult,
    plan: ExecutionPlan,
    contract: DataContract,
    logger: StructuredLogger | None,
    batch_metadata: Mapping[str, Any],
) -> Any:
    """Read one batch with the source's schema and options when its format matches."""
    _log_info(
        logger,
        "spark_read_started",
        artifact_count=len(handoff.artifacts),
        contract_id=contract.id,
        schema_version=contract.schema_version,
        **batch_metadata,
    )
    handoff_format = handoff.single_artifact_format()
    tracks_corrupt = _tracks_corrupt_records(reader, handoff_format, plan)
    spark_schema = (
        spark_schema_from_contract(contract, with_corrupt_record=tracks_corrupt)
        if handoff_format == plan.source_config.spark.input_format
        else None
    )
    read_options = (
        plan.source_config.spark.read_options
        if handoff_format == plan.source_config.spark.input_format
        else None
    )
    read_kwargs: dict[str, Any] = {}
    if tracks_corrupt:
        read_kwargs["corrupt_record_column"] = CORRUPT_RECORD_COLUMN
    dataframe = reader.read_extraction_result(
        spark,
        handoff,
        format_name=handoff_format,
        schema=spark_schema,
        options=read_options,
        **read_kwargs,
    )
    _log_info(logger, "spark_read_finished", **batch_metadata)
    return dataframe


def _annotate_contract_failure(
    exc: ContractEnforcementError,
    evidence: Sequence[PreWriteEvidence],
    committed: Sequence[WriteResult],
    logger: StructuredLogger | None,
    contract_id: str,
    batch_metadata: Mapping[str, Any],
) -> None:
    """Keep checked-batch evidence and earlier commits on a refused batch."""
    exc.evidence = (*evidence, *exc.evidence)
    exc.committed_results = tuple(committed)
    _log_error(
        logger,
        "contract_check_failed",
        failure_stage=exc.failure_stage,
        error_type=type(exc).__name__,
        contract_id=contract_id,
        **batch_metadata,
    )


def _log_contract_check_passed(
    logger: StructuredLogger | None,
    evidence: PreWriteEvidence,
    batch_metadata: Mapping[str, Any],
) -> None:
    _log_info(
        logger,
        "contract_check_passed",
        mismatches=0,
        nullability_relaxed=list(evidence.contract_check.nullability_relaxed),
        **batch_metadata,
    )


def _persist(dataframe: Any) -> Any:
    """Keep one batch for the pre-write aggregation and the write; spills rather than failing."""
    from pyspark import StorageLevel

    return dataframe.persist(StorageLevel.MEMORY_AND_DISK)


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


def _accumulate_run_keys(
    run_keys: Any | None,
    normalized_dataframe: Any,
    primary_key: tuple[str, ...],
) -> Any | None:
    """Union this batch's distinct primary-key frame into the run-level key set."""

    if not primary_key:
        return run_keys
    columns = getattr(normalized_dataframe, "columns", ())
    if any(field not in columns for field in primary_key):
        return run_keys
    batch_keys = normalized_dataframe.select(*primary_key).distinct()
    if run_keys is None:
        return batch_keys
    return run_keys.unionByName(batch_keys)


def read_committed_bronze(
    spark: SparkSession,
    bronze_results: Sequence[WriteResult],
) -> Any | None:
    """Return the committed bronze Iceberg table frame, or ``None`` when there is none.

    Both entry points call this between materialization and quality validation, inside the
    live session window, so ``spark`` is the already-open session and no new lifetime is
    created. The identifier is the bronze write result's own ``path`` — the writer sets it
    to the resolved table identifier — so there is one derivation, not a re-computation.
    Path-based (non-iceberg) bronze and runs that wrote no bronze return ``None``, leaving
    the bronze uniqueness oracle to skip.
    """
    bronze_result = next(
        (
            result
            for result in bronze_results
            if result.zone == "bronze" and result.format.strip().lower() == "iceberg"
        ),
        None,
    )
    if bronze_result is None:
        return None
    return spark.table(bronze_result.path)


def _intent_log_fields(intent: BronzeWriteIntent) -> dict[str, Any]:
    """Fields the two bronze write events carry so every run's log records the intent."""
    fields: dict[str, Any] = {
        "write_strategy": intent.strategy,
        "write_mode": intent.reported_mode,
        "write_intent_reason": intent.reason,
    }
    if intent.merge_keys:
        fields["merge_keys"] = list(intent.merge_keys)
    return fields


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
