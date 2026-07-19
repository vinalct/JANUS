from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from janus.checkpoints.store import CheckpointStore, CheckpointWriteResult
from janus.lineage.models import LineageRecord, RunMetadata
from janus.lineage.persistence import MetadataZonePaths, write_json_atomic
from janus.models import ExecutionPlan, ExtractionResult, WriteResult
from janus.utils.logging import StructuredLogger

STRATEGY_METADATA_PREFIX = "strategy."


@dataclass(frozen=True, slots=True)
class PersistedArtifacts:
    """Paths and records produced by one observability persistence call."""

    run_metadata: RunMetadata
    run_metadata_path: Path
    lineage_record: LineageRecord | None = None
    lineage_path: Path | None = None
    checkpoint_result: CheckpointWriteResult | None = None


@dataclass(slots=True)
class RunMetadataStore:
    """Persistence helper for run lifecycle records in the metadata zone."""

    def write(self, plan: ExecutionPlan, record: RunMetadata) -> Path:
        metadata_paths = MetadataZonePaths.from_plan(plan)
        return write_json_atomic(metadata_paths.run_metadata_path(record.run_id), record.to_dict())


@dataclass(slots=True)
class LineageStore:
    """Persistence helper for lineage artifacts in the metadata zone."""

    def write(self, plan: ExecutionPlan, record: LineageRecord) -> Path:
        metadata_paths = MetadataZonePaths.from_plan(plan)
        return write_json_atomic(metadata_paths.lineage_path(record.run_id), record.to_dict())


@dataclass(slots=True)
class RunObserver:
    """Shared observability service that future strategies can call around execution."""

    run_metadata_store: RunMetadataStore = field(default_factory=RunMetadataStore)
    lineage_store: LineageStore = field(default_factory=LineageStore)
    checkpoint_store: CheckpointStore = field(default_factory=CheckpointStore)
    logger: StructuredLogger | None = None

    def start_run(self, plan: ExecutionPlan) -> PersistedArtifacts:
        run_metadata = RunMetadata.started(plan)
        run_metadata_path = self.run_metadata_store.write(plan, run_metadata)
        return PersistedArtifacts(
            run_metadata=run_metadata,
            run_metadata_path=run_metadata_path,
        )

    def record_success(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        *,
        finished_at: datetime | None = None,
        strategy_metadata: Mapping[str, Any] | None = None,
    ) -> PersistedArtifacts:
        prepared_metadata = _prepare_strategy_metadata(strategy_metadata, logger=self.logger)

        run_metadata = RunMetadata.succeeded(
            plan,
            extraction_result,
            write_results,
            finished_at=finished_at,
            metadata=prepared_metadata,
        )
        run_metadata_path = self.run_metadata_store.write(plan, run_metadata)

        lineage_record = LineageRecord.from_runtime(
            plan,
            status="succeeded",
            extraction_result=extraction_result,
            write_results=write_results,
            emitted_at=finished_at,
            metadata=prepared_metadata,
        )
        lineage_path = self.lineage_store.write(plan, lineage_record)

        checkpoint_result = self.checkpoint_store.save(
            plan,
            extraction_result.checkpoint_value,
            run_id=plan.run_context.run_id,
            updated_at=finished_at,
            metadata=_checkpoint_metadata(extraction_result),
        )
        return PersistedArtifacts(
            run_metadata=run_metadata,
            run_metadata_path=run_metadata_path,
            lineage_record=lineage_record,
            lineage_path=lineage_path,
            checkpoint_result=checkpoint_result,
        )

    def record_failure(
        self,
        plan: ExecutionPlan,
        error: Exception | str,
        extraction_result: ExtractionResult | None = None,
        write_results: tuple[WriteResult, ...] = (),
        *,
        finished_at: datetime | None = None,
        strategy_metadata: Mapping[str, Any] | None = None,
    ) -> PersistedArtifacts:
        prepared_metadata = _prepare_strategy_metadata(strategy_metadata, logger=self.logger)

        run_metadata = RunMetadata.failed(
            plan,
            error,
            extraction_result,
            write_results,
            finished_at=finished_at,
            metadata=prepared_metadata,
        )
        run_metadata_path = self.run_metadata_store.write(plan, run_metadata)

        lineage_record = LineageRecord.from_runtime(
            plan,
            status="failed",
            extraction_result=extraction_result,
            write_results=write_results,
            failure_reason=run_metadata.failure_reason,
            error_type=run_metadata.error_type,
            emitted_at=finished_at,
            metadata=prepared_metadata,
        )
        lineage_path = self.lineage_store.write(plan, lineage_record)
        return PersistedArtifacts(
            run_metadata=run_metadata,
            run_metadata_path=run_metadata_path,
            lineage_record=lineage_record,
            lineage_path=lineage_path,
        )


def _checkpoint_metadata(extraction_result: ExtractionResult) -> dict[str, str]:
    metadata = extraction_result.metadata_as_dict()
    if extraction_result.records_extracted is not None:
        metadata["records_extracted"] = str(extraction_result.records_extracted)
    return metadata


def _prepare_strategy_metadata(
    raw: Mapping[str, Any] | None,
    *,
    prefix: str = STRATEGY_METADATA_PREFIX,
    logger: StructuredLogger | None = None,
) -> dict[str, str]:
    """Coerce, drop, and namespace strategy/hook metadata into a safe string mapping.

    Turns the flat ``emit_metadata()`` dict — which may carry non-string scalars from
    strategies and hooks — into entries that always satisfy the frozen models'
    ``_freeze_string_mapping`` invariant. Every surviving key is namespaced under a
    single uniform ``prefix`` so strategy/hook keys can never clobber a reserved
    run-metadata field. Invalid entries are dropped with a logged warning rather than
    raised, so a malformed hook payload never aborts a live run.
    """
    if not raw:
        return {}

    prepared: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            _log_warning(
                logger,
                "strategy_metadata_entry_dropped",
                key=repr(key),
                reason="non_string_or_empty_key",
            )
            continue

        coerced, reason = _coerce_metadata_value(value)
        if coerced is None:
            _log_warning(
                logger,
                "strategy_metadata_entry_dropped",
                key=key.strip(),
                reason=reason,
            )
            continue

        prepared[f"{prefix}{key.strip()}"] = coerced
    return prepared


def _coerce_metadata_value(value: Any) -> tuple[str | None, str]:
    """Stringify a scalar metadata value, or report why it must be dropped."""
    if value is None:
        return None, "none_value"
    if isinstance(value, bool):
        return ("true" if value else "false"), ""
    if isinstance(value, (int, float)):
        return str(value), ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, "empty_value"
        return stripped, ""
    return None, "non_scalar_value"


def _log_warning(logger: StructuredLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.warning(event, **fields)
