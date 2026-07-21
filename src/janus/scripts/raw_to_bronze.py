"""Replay entry point: re-materialize bronze from an existing raw zone.

`RawToBronzeLoader` owns only raw rehydration — rediscovering raw artifacts on
disk and rebuilding the `ExtractionResult` the live run would have produced.
The write path itself (batch -> read -> normalize -> write) is delegated to
`janus.runtime.materialize.BronzeMaterializer`, the same component the live
executor uses, so replaying raw produces the same bronze as `--execute`.
"""

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
from janus.runtime.spark_lifecycle import SparkSessionProvider, scoped_request_input_session
from janus.strategies.api import ApiRequest, build_paginator
from janus.strategies.api.request_inputs import load_request_inputs
from janus.strategies.catalog.core import (
    ENTITY_TYPE_ORDER,
    CatalogStrategy,
    _apply_per_input_params,
    _rediscover_catalog_input_artifacts,
    _replay_catalog_entities_from_dir,
)
from janus.strategies.http import resolve_url
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SIDECAR_SUFFIX, RawArtifactWriter, SparkDatasetWriter

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_READABLE_ARTIFACT_FORMATS = frozenset(
    {"binary", "csv", "json", "jsonl", "parquet", "text"}
)


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
                        bronze_target_identifier=_bronze_target_identifier(plan),
                    )
                    write_results = write_results + bronze_results
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


def _build_extraction_result_from_raw(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    spark_provider: SparkSessionProvider,
    storage_layout: StorageLayout,
) -> ExtractionResult:
    if isinstance(planned_run.strategy, CatalogStrategy):
        return _build_catalog_extraction_result_from_raw(
            planned_run,
            plan,
            spark_provider,
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


def _rediscover_raw_artifacts(
    plan: ExecutionPlan, *, verify_checksums: bool = False
) -> tuple[ExtractedArtifact, ...]:
    raw_root = Path(plan.raw_output.path)
    if not raw_root.exists():
        raise FileNotFoundError(f"Configured raw output path does not exist: {raw_root}")

    artifacts = tuple(
        ExtractedArtifact(
            path=str(path),
            format=_artifact_format_for_path(path, fallback=plan.source_config.spark.input_format),
            checksum=_resolve_raw_checksum(path, verify=verify_checksums),
        )
        for path in sorted(
            candidate
            for candidate in raw_root.rglob("*")
            if candidate.is_file() and candidate.suffix != SIDECAR_SUFFIX
        )
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
    spark_provider: SparkSessionProvider,
    storage_layout: StorageLayout,
) -> ExtractionResult:
    strategy = planned_run.strategy
    assert isinstance(strategy, CatalogStrategy)

    # Same scoped seam the live catalog strategy uses: only `iceberg_rows` inputs
    # open a session here, and it is stopped before the replay walk begins.
    request_inputs_config = plan.source_config.access.request_inputs
    with scoped_request_input_session(spark_provider, request_inputs_config) as session:
        request_inputs = tuple(load_request_inputs(request_inputs_config, spark=session))
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
        url=resolve_url(plan.source_config, family_label="Catalog"),
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


def _resolve_raw_checksum(path: Path, *, verify: bool = False) -> str:
    """Return the artifact's SHA-256, reading the ``.sha256`` sidecar when present."""

    cached = _read_sidecar_checksum(path.with_name(path.name + SIDECAR_SUFFIX))
    if cached is None:
        return _sha256(path)
    if verify:
        recomputed = _sha256(path)
        if recomputed != cached:
            raise ValueError(
                f"Checksum sidecar mismatch for {path}: "
                f"sidecar={cached}, recomputed={recomputed}"
            )
    return cached


def _read_sidecar_checksum(sidecar: Path) -> str | None:
    """Read a bare-digest ``.sha256`` sidecar, or ``None`` if absent/empty."""

    if not sidecar.is_file():
        return None
    content = sidecar.read_text(encoding="utf-8").strip()
    if not content:
        return None
    return content.split()[0]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bronze_target_identifier(plan: ExecutionPlan) -> str:
    return bronze_table_identifier(
        plan.bronze_output.path,
        fallback_name=plan.source.source_id,
        namespace=plan.bronze_output.namespace,
        table_name=plan.bronze_output.table_name,
    )


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
