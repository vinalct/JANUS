from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, WriteResult
from janus.models.source_config import OutputTarget
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, QualityGate, ValidationReportStore
from janus.readers import SparkDatasetReader
from janus.runtime.executor import _plan_with_storage_layout_outputs
from janus.schema_contracts import resolve_spark_schema_for_plan
from janus.strategies.api import ApiRequest, build_paginator
from janus.strategies.api.request_inputs import load_request_inputs
from janus.strategies.catalog.core import (
    ENTITY_TYPE_ORDER,
    CatalogStrategy,
    _apply_per_input_params,
    _rediscover_catalog_input_artifacts,
    _replay_catalog_entities_from_dir,
    _resolve_url,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import RawArtifactWriter, SparkDatasetWriter

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_READABLE_ARTIFACT_FORMATS = frozenset(
    {"binary", "csv", "json", "jsonl", "parquet", "text"}
)
_FILE_HANDOFF_ARTIFACTS_PER_BATCH = 1


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
        spark: SparkSession,
        environment_config: Mapping[str, Any],
        *,
        bronze_table: str,
    ) -> RawToBronzeRun:
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
                spark,
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
            if not handoff.is_empty:
                bronze_results, normalized_dataframe = self._write_handoff_to_bronze(
                    runtime_planned_run,
                    plan,
                    spark,
                    handoff,
                    storage_layout,
                    logger,
                )
                write_results = write_results + bronze_results

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

            persisted = self.observer.record_success(plan, extraction_result, write_results)
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
        except Exception as exc:
            _log_exception(
                logger,
                "raw_to_bronze_failed",
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )
            persisted = self.observer.record_failure(plan, exc, extraction_result, write_results)
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
                target_table=_bronze_target_identifier(plan),
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


def ingest_raw_to_bronze(
    planned_run: PlannedRun,
    spark: SparkSession,
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


def _default_storage_layout(
    plan: ExecutionPlan,
    environment_config: Mapping[str, Any],
) -> StorageLayout:
    return StorageLayout.from_environment_config(
        environment_config,
        plan.run_context.project_root,
    )


def _override_bronze_output(plan: ExecutionPlan, bronze_table: str) -> ExecutionPlan:
    normalized_target = bronze_table.strip()
    if not normalized_target:
        raise ValueError("bronze_table must not be empty")

    namespace = plan.bronze_output.namespace
    table_name = normalized_target

    if "." in normalized_target:
        if normalized_target.count(".") != 1:
            raise ValueError(
                "bronze_table must be a table name or one namespace.table identifier"
            )
        namespace, table_name = normalized_target.split(".", 1)
        namespace = namespace.strip() or None
        table_name = table_name.strip()
        if not table_name:
            raise ValueError("bronze_table table name must not be empty")

    return replace(
        plan,
        bronze_output=OutputTarget(
            path=plan.bronze_output.path,
            format=plan.bronze_output.format,
            namespace=namespace,
            table_name=table_name,
        ),
    )


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


def _build_extraction_result_from_raw(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    spark: SparkSession,
    storage_layout: StorageLayout,
) -> ExtractionResult:
    if isinstance(planned_run.strategy, CatalogStrategy):
        return _build_catalog_extraction_result_from_raw(
            planned_run,
            plan,
            spark,
            storage_layout,
        )

    raw_plan = _plan_with_active_raw_root(plan)
    raw_artifacts = _rediscover_raw_artifacts(raw_plan)
    raw_artifacts = _rehydrate_file_raw_artifacts(
        planned_run,
        raw_plan,
        raw_artifacts,
        storage_layout,
    )
    return ExtractionResult.from_plan(
        plan,
        raw_artifacts,
        metadata={
            "raw_to_bronze": "true",
            "rediscovered_raw_artifact_count": str(len(raw_artifacts)),
            "raw_artifact_root": raw_plan.raw_output.path,
        },
    )


def _rediscover_raw_artifacts(plan: ExecutionPlan) -> tuple[ExtractedArtifact, ...]:
    raw_root = Path(plan.raw_output.path)
    if not raw_root.exists():
        raise FileNotFoundError(f"Configured raw output path does not exist: {raw_root}")

    artifacts = tuple(
        ExtractedArtifact(
            path=str(path),
            format=_artifact_format_for_path(path, fallback=plan.source_config.spark.input_format),
            checksum=_sha256(path),
        )
        for path in sorted(candidate for candidate in raw_root.rglob("*") if candidate.is_file())
    )
    if not artifacts:
        raise FileNotFoundError(f"No raw artifacts were found under {raw_root}")
    return artifacts


def _plan_with_active_raw_root(plan: ExecutionPlan) -> ExecutionPlan:
    raw_root = Path(plan.raw_output.path)
    latest_run_root = _latest_raw_run_root(raw_root)
    if latest_run_root is None:
        return plan
    return replace(plan, raw_output=replace(plan.raw_output, path=str(latest_run_root)))


def _latest_raw_run_root(raw_root: Path) -> Path | None:
    runs_root = raw_root / "runs"
    if not runs_root.exists():
        return None

    candidates = sorted(
        path
        for path in runs_root.glob("ingestion_date=*/run_id=*")
        if path.is_dir()
    )
    if not candidates:
        return None
    return candidates[-1]


def _rehydrate_file_raw_artifacts(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    raw_artifacts: tuple[ExtractedArtifact, ...],
    storage_layout: StorageLayout,
) -> tuple[ExtractedArtifact, ...]:
    if getattr(planned_run.strategy, "strategy_family", None) != "file":
        return raw_artifacts

    from janus.strategies.files.core import (
        DiscoveredFile,
        FileHook,
        _archive_member_payloads,
        _filter_members,
        _infer_handoff_format,
        _raw_extracted_relative_path,
    )

    artifacts_by_path = {artifact.path: artifact for artifact in raw_artifacts}
    raw_writer = RawArtifactWriter(storage_layout)
    file_hook = planned_run.hook if isinstance(planned_run.hook, FileHook) else None
    raw_root = Path(plan.raw_output.path)

    for artifact in raw_artifacts:
        artifact_path = Path(artifact.path)
        if not _is_archive_download_path(raw_root, artifact_path):
            continue

        archive_file = DiscoveredFile(
            source_kind="local",
            location=str(artifact_path),
            filename=artifact_path.name,
            format="binary",
            version=_download_version(raw_root, artifact_path),
        )
        member_payloads = _archive_member_payloads(
            artifact_path.read_bytes(),
            archive_file.filename,
        )
        members = tuple(
            DiscoveredFile(
                source_kind="archive",
                location=member_name,
                filename=PurePosixPath(member_name).name,
                format=_artifact_format_for_path(
                    Path(member_name),
                    fallback=plan.source_config.spark.input_format,
                ),
                version=archive_file.version,
            )
            for member_name in member_payloads
        )
        members = _filter_members(members, plan.source_config.access.file_pattern)
        if file_hook is not None:
            members = tuple(file_hook.archive_members(plan, archive_file, members))

        for member in members:
            member_payload = member_payloads.get(member.location)
            if member_payload is None:
                continue
            persisted = raw_writer.write_bytes(
                plan,
                _raw_extracted_relative_path(
                    archive_file.version or "current",
                    archive_file.filename,
                    member.location,
                ),
                member_payload,
                mode="ignore",
                metadata={
                    "archive_filename": archive_file.filename,
                    "archive_member": member.location,
                    "resolved_version": archive_file.version or "current",
                },
            )
            artifacts_by_path[str(persisted.artifact.path)] = ExtractedArtifact(
                path=str(persisted.artifact.path),
                format=_infer_handoff_format(
                    member,
                    fallback=plan.source_config.spark.input_format,
                ),
                checksum=persisted.artifact.checksum,
            )

    return tuple(sorted(artifacts_by_path.values(), key=lambda artifact: artifact.path))


def _is_archive_download_path(raw_root: Path, artifact_path: Path) -> bool:
    try:
        relative_path = artifact_path.relative_to(raw_root)
    except ValueError:
        return False
    if len(relative_path.parts) < 3 or relative_path.parts[0] != "downloads":
        return False
    lower_name = artifact_path.name.lower()
    return lower_name.endswith((".zip", ".tar.gz", ".tgz"))


def _download_version(raw_root: Path, artifact_path: Path) -> str:
    try:
        relative_path = artifact_path.relative_to(raw_root)
    except ValueError:
        return "current"
    if len(relative_path.parts) >= 3 and relative_path.parts[0] == "downloads":
        return relative_path.parts[1]
    return "current"


def _build_catalog_extraction_result_from_raw(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    spark: SparkSession,
    storage_layout: StorageLayout,
) -> ExtractionResult:
    strategy = planned_run.strategy
    assert isinstance(strategy, CatalogStrategy)

    request_inputs = tuple(
        load_request_inputs(plan.source_config.access.request_inputs, spark=spark)
    )
    if not request_inputs:
        request_inputs = (None,)

    raw_plan = _plan_with_active_raw_root(plan)

    raw_artifacts = _rediscover_catalog_raw_artifacts(
        raw_plan,
        storage_layout,
        request_input_count=len(request_inputs),
    )
    if not raw_artifacts:
        raise FileNotFoundError(f"No raw artifacts were found under {raw_plan.raw_output.path}")

    normalized_records: dict[str, list[dict[str, Any]]] = {
        entity_type: [] for entity_type in ENTITY_TYPE_ORDER
    }
    entity_indexes: dict[tuple[str, str], int] = {}
    paginator = build_paginator(plan.source_config.access.pagination)
    base_request = _catalog_base_request(plan)
    checkpoint_value: str | None = None

    for request_input_index, request_input in enumerate(request_inputs, start=1):
        per_input_request = _apply_per_input_params(
            base_request,
            plan.source_config.access.parameter_bindings,
            request_input,
        )
        checkpoint_value = _replay_catalog_entities_from_dir(
            strategy,
            raw_plan,
            storage_layout,
            per_input_request,
            paginator,
            request_input_index,
            len(request_inputs),
            checkpoint_state=None,
            normalized_records=normalized_records,
            entity_indexes=entity_indexes,
            current_checkpoint_value=checkpoint_value,
        )

    raw_writer = RawArtifactWriter(storage_layout)
    normalized_artifacts = strategy._persist_normalized_records(
        raw_plan,
        raw_writer,
        normalized_records,
    )
    all_artifacts = tuple(raw_artifacts + tuple(normalized_artifacts))
    records_extracted = sum(len(records) for records in normalized_records.values())
    normalized_artifact_count = len(normalized_artifacts)

    return ExtractionResult.from_plan(
        plan,
        all_artifacts,
        records_extracted=records_extracted,
        checkpoint_value=checkpoint_value,
        metadata={
            "raw_to_bronze": "true",
            "rediscovered_raw_artifact_count": str(len(raw_artifacts)),
            "normalized_artifact_count": str(normalized_artifact_count),
            "raw_artifact_root": raw_plan.raw_output.path,
            "organizations_extracted": str(len(normalized_records["organization"])),
            "groups_extracted": str(len(normalized_records["group"])),
            "datasets_extracted": str(len(normalized_records["dataset"])),
            "resources_extracted": str(len(normalized_records["resource"])),
        },
    )


def _rediscover_catalog_raw_artifacts(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    *,
    request_input_count: int,
) -> tuple[ExtractedArtifact, ...]:
    artifacts: list[ExtractedArtifact] = []
    for request_input_index in range(1, request_input_count + 1):
        artifacts.extend(
            _rediscover_catalog_input_artifacts(
                plan,
                storage_layout,
                request_input_index,
                request_input_count,
            )
        )

    if artifacts:
        return tuple(artifacts)

    return _rediscover_raw_artifacts(plan)


def _catalog_base_request(plan: ExecutionPlan) -> ApiRequest:
    source_access = plan.source_config.access
    return ApiRequest(
        method=source_access.method,
        url=_resolve_url(plan.source_config),
        timeout_seconds=source_access.timeout_seconds,
        headers=_freeze_string_mapping(source_access.headers or {}),
        params=_freeze_string_mapping(source_access.params or {}),
    )


def _artifact_format_for_path(path: Path, *, fallback: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".txt", ".tsv"}:
        return "text"
    if suffix in {".zip", ".xlsx", ".bin"}:
        return "binary"

    normalized_fallback = fallback.strip().lower()
    if normalized_fallback in _READABLE_ARTIFACT_FORMATS:
        return normalized_fallback
    return "binary"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _bronze_target_identifier(plan: ExecutionPlan) -> str:
    return bronze_table_identifier(
        plan.bronze_output.path,
        fallback_name=plan.source.source_id,
        namespace=plan.bronze_output.namespace,
        table_name=plan.bronze_output.table_name,
    )


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


def _quality_failure_message(report: PersistedValidationReport) -> str:
    failed_checks = ", ".join(
        f"{check.phase}.{check.name}" for check in report.report.failed_checks
    )
    if not failed_checks:
        return "Quality validation failed"
    return f"Quality validation failed: {failed_checks}"


def _freeze_string_mapping(values: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()

    frozen_items: list[tuple[str, str]] = []
    for key, value in values.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if not normalized_key or not normalized_value:
            continue
        frozen_items.append((normalized_key, normalized_value))
    return tuple(sorted(frozen_items))
