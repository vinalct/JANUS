"""One run, projected into one flat row — the definition of "queryable"."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from janus.checkpoints.store import SUPPORTED_CHECKPOINT_DECISIONS, CheckpointWriteResult
from janus.lineage.models import LineageRecord, MaterializedOutput, RunMetadata
from janus.observability.vocabulary import (
    BRONZE_ZONE,
    MAX_FAILURE_REASON_LENGTH,
    PIPELINE_ATTEMPT_ATTRIBUTE,
    PIPELINE_RUN_ID_ATTRIBUTE,
    QUALITY_NOT_RUN,
    RUN_RECORD_SCHEMA_VERSION,
    RUN_RECORD_STATUSES,
    SUPPORTED_QUALITY_OUTCOMES,
    TRIGGER_ATTRIBUTE,
)
from janus.quality.models import ValidationReport


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One terminal run flattened into the row ``metadata.runs`` stores."""

    # Identity and correlation
    run_id: str
    source_id: str
    source_name: str
    environment: str
    strategy_family: str
    strategy_variant: str
    extraction_mode: str
    # Status and timing
    status: str
    started_at: datetime
    emitted_at: datetime
    # Provenance
    config_version: str
    source_config_path: str
    # Volume
    artifact_count: int
    # Quality
    quality_outcome: str

    source_hook: str | None = None
    pipeline_run_id: str | None = None
    pipeline_attempt: int | None = None
    trigger: str | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    records_extracted: int | None = None
    records_written: int | None = None
    bronze_table_identifier: str | None = None
    bronze_write_mode: str | None = None
    # Checkpoint
    checkpoint_field: str | None = None
    checkpoint_strategy: str | None = None
    checkpoint_value: str | None = None
    checkpoint_decision: str | None = None
    checkpoint_advanced: bool | None = None
    # Quality detail
    quality_checks_passed: int | None = None
    quality_checks_failed: int | None = None
    quality_checks_skipped: int | None = None
    quality_failed_checks: tuple[str, ...] | None = None
    # Failure
    failure_reason: str | None = None
    failure_reason_truncated: bool | None = None
    failure_reason_length: int | None = None
    error_type: str | None = None
    # Evidence links
    run_metadata_path: str | None = None
    lineage_path: str | None = None
    checkpoint_history_path: str | None = None
    validation_report_path: str | None = None
    # Schema versioning
    record_schema_version: int = RUN_RECORD_SCHEMA_VERSION
    schema_version: str | None = None
    contract_id: str | None = None
    contract_version: str | None = None

    def __post_init__(self) -> None:
        if self.status not in RUN_RECORD_STATUSES:
            allowed = ", ".join(sorted(RUN_RECORD_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        if self.quality_outcome not in SUPPORTED_QUALITY_OUTCOMES:
            allowed = ", ".join(sorted(SUPPORTED_QUALITY_OUTCOMES))
            raise ValueError(f"quality_outcome must be one of: {allowed}")
        if (
            self.checkpoint_decision is not None
            and self.checkpoint_decision not in SUPPORTED_CHECKPOINT_DECISIONS
        ):
            allowed = ", ".join(sorted(SUPPORTED_CHECKPOINT_DECISIONS))
            raise ValueError(f"checkpoint_decision must be one of: {allowed}")
        for name in (
            "run_id",
            "source_id",
            "source_name",
            "environment",
            "strategy_family",
            "strategy_variant",
            "extraction_mode",
            "config_version",
            "source_config_path",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.record_schema_version < 1:
            raise ValueError("record_schema_version must be positive")
        if self.artifact_count < 0:
            raise ValueError("artifact_count must not be negative")
        _validate_timezone_aware("started_at", self.started_at)
        _validate_timezone_aware("emitted_at", self.emitted_at)
        if self.ended_at is not None:
            _validate_timezone_aware("ended_at", self.ended_at)
        self._validate_three_valued_columns()

    def _validate_three_valued_columns(self) -> None:
        """Pin the NULL / zero / empty-list distinctions a later refactor would collapse."""
        if (self.status == "failed") != (self.failure_reason is not None):
            raise ValueError("a failed run must carry a failure_reason, and a success must not")
        reason_companions = (self.failure_reason_truncated, self.failure_reason_length)
        if (self.failure_reason is None) != all(value is None for value in reason_companions):
            raise ValueError(
                "failure_reason_truncated and failure_reason_length are present exactly "
                "when failure_reason is"
            )
        if (self.checkpoint_decision is None) != (self.checkpoint_advanced is None):
            raise ValueError(
                "checkpoint_advanced is present exactly when checkpoint_decision is; NULL "
                "means no checkpoint write was attempted"
            )
        quality_detail = (
            self.quality_checks_passed,
            self.quality_checks_failed,
            self.quality_checks_skipped,
            self.quality_failed_checks,
        )
        validation_ran = self.quality_outcome != QUALITY_NOT_RUN
        if validation_ran == all(value is None for value in quality_detail):
            raise ValueError(
                f"quality detail columns are NULL exactly when quality_outcome is "
                f"{QUALITY_NOT_RUN!r}"
            )
        if self.quality_outcome == "failed" and not self.quality_failed_checks:
            raise ValueError("a failed quality outcome must name at least one failed check")
        if self.quality_outcome == "passed" and self.quality_failed_checks:
            raise ValueError("a passed quality outcome must name no failed checks")

    @classmethod
    def from_run(
        cls,
        run_metadata: RunMetadata,
        lineage_record: LineageRecord,
        *,
        emitted_at: datetime,
        checkpoint_result: CheckpointWriteResult | None = None,
        validation_report: ValidationReport | None = None,
        evidence: RunEvidencePaths | None = None,
    ) -> Self:
        """Flatten the four records describing one terminal run into one row."""
        if run_metadata.run_id != lineage_record.run_id:
            raise ValueError(
                "run_metadata and lineage_record must describe the same run "
                f"({run_metadata.run_id!r} != {lineage_record.run_id!r})"
            )
        if run_metadata.status != lineage_record.status:
            raise ValueError(
                "run_metadata and lineage_record must agree on status "
                f"({run_metadata.status!r} != {lineage_record.status!r})"
            )

        attributes = run_metadata.run_attributes_as_dict()
        bronze_outputs = _bronze_outputs(lineage_record)
        reason, reason_truncated, reason_length = _bounded_failure_reason(
            run_metadata.failure_reason
        )
        outcome, passed, failed, skipped, failed_checks = _project_quality(validation_report)
        paths = evidence or RunEvidencePaths()

        return cls(
            run_id=run_metadata.run_id,
            source_id=run_metadata.source_id,
            source_name=run_metadata.source_name,
            environment=run_metadata.environment,
            strategy_family=run_metadata.strategy_family,
            strategy_variant=run_metadata.strategy_variant,
            extraction_mode=run_metadata.extraction_mode,
            source_hook=lineage_record.source_hook,
            pipeline_run_id=_lift(attributes, PIPELINE_RUN_ID_ATTRIBUTE),
            pipeline_attempt=_optional_int(_lift(attributes, PIPELINE_ATTEMPT_ATTRIBUTE)),
            trigger=_lift(attributes, TRIGGER_ATTRIBUTE),
            status=run_metadata.status,
            started_at=run_metadata.started_at,
            ended_at=run_metadata.ended_at,
            emitted_at=emitted_at,
            duration_seconds=run_metadata.duration_seconds,
            config_version=lineage_record.config_version,
            schema_version=lineage_record.schema_version,
            contract_id=lineage_record.contract_id,
            contract_version=lineage_record.contract_version,
            source_config_path=run_metadata.source_config_path,
            records_extracted=run_metadata.records_extracted,
            artifact_count=len(lineage_record.artifacts),
            records_written=_sum_records_written(bronze_outputs),
            bronze_table_identifier=bronze_outputs[0].path if bronze_outputs else None,
            bronze_write_mode=bronze_outputs[0].mode if bronze_outputs else None,
            checkpoint_field=run_metadata.checkpoint_field,
            checkpoint_strategy=run_metadata.checkpoint_strategy,
            checkpoint_value=run_metadata.checkpoint_value,
            checkpoint_decision=(
                checkpoint_result.decision if checkpoint_result is not None else None
            ),
            checkpoint_advanced=(
                checkpoint_result.advanced if checkpoint_result is not None else None
            ),
            quality_outcome=outcome,
            quality_checks_passed=passed,
            quality_checks_failed=failed,
            quality_checks_skipped=skipped,
            quality_failed_checks=failed_checks,
            failure_reason=reason,
            failure_reason_truncated=reason_truncated,
            failure_reason_length=reason_length,
            error_type=run_metadata.error_type,
            run_metadata_path=_as_optional_string(paths.run_metadata_path),
            lineage_path=_as_optional_string(paths.lineage_path),
            checkpoint_history_path=_as_optional_string(
                checkpoint_result.history_path if checkpoint_result is not None else None
            ),
            validation_report_path=_as_optional_string(paths.validation_report_path),
        )

    def to_dict(self) -> dict[str, Any]:
        """Every column, always present."""
        return {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "environment": self.environment,
            "strategy_family": self.strategy_family,
            "strategy_variant": self.strategy_variant,
            "extraction_mode": self.extraction_mode,
            "source_hook": self.source_hook,
            "pipeline_run_id": self.pipeline_run_id,
            "pipeline_attempt": self.pipeline_attempt,
            "trigger": self.trigger,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "emitted_at": self.emitted_at,
            "duration_seconds": self.duration_seconds,
            "config_version": self.config_version,
            "source_config_path": self.source_config_path,
            "records_extracted": self.records_extracted,
            "artifact_count": self.artifact_count,
            "records_written": self.records_written,
            "bronze_table_identifier": self.bronze_table_identifier,
            "bronze_write_mode": self.bronze_write_mode,
            "checkpoint_field": self.checkpoint_field,
            "checkpoint_strategy": self.checkpoint_strategy,
            "checkpoint_value": self.checkpoint_value,
            "checkpoint_decision": self.checkpoint_decision,
            "checkpoint_advanced": self.checkpoint_advanced,
            "quality_outcome": self.quality_outcome,
            "quality_checks_passed": self.quality_checks_passed,
            "quality_checks_failed": self.quality_checks_failed,
            "quality_checks_skipped": self.quality_checks_skipped,
            "quality_failed_checks": (
                None if self.quality_failed_checks is None else list(self.quality_failed_checks)
            ),
            "failure_reason": self.failure_reason,
            "failure_reason_truncated": self.failure_reason_truncated,
            "failure_reason_length": self.failure_reason_length,
            "error_type": self.error_type,
            "run_metadata_path": self.run_metadata_path,
            "lineage_path": self.lineage_path,
            "checkpoint_history_path": self.checkpoint_history_path,
            "validation_report_path": self.validation_report_path,
            "record_schema_version": self.record_schema_version,
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True, slots=True)
class RunEvidencePaths:
    """Where the authoritative JSON for one run lives (NFR-1's link back)."""

    run_metadata_path: Path | str | None = None
    lineage_path: Path | str | None = None
    validation_report_path: Path | str | None = None


def _bronze_outputs(lineage_record: LineageRecord) -> tuple[MaterializedOutput, ...]:
    """The bronze half of ``materialized_outputs``; empty for an empty handoff."""
    return tuple(
        output for output in lineage_record.materialized_outputs if output.zone == BRONZE_ZONE
    )


def _sum_records_written(outputs: tuple[MaterializedOutput, ...]) -> int | None:
    """Total bronze rows written, or ``NULL`` when no bronze output reported a count.

    A run whose normalization handoff was empty has no bronze output at all, and a writer
    may skip counting; neither is "wrote zero rows", so neither may render as ``0``.
    """
    counts = [output.records_written for output in outputs if output.records_written is not None]
    return sum(counts) if counts else None


def _project_quality(
    report: ValidationReport | None,
) -> tuple[str, int | None, int | None, int | None, tuple[str, ...] | None]:
    """Collapse the validation report into one outcome plus its counts and failed checks."""
    if report is None:
        return QUALITY_NOT_RUN, None, None, None, None

    summary = report.summary()
    failed_checks = tuple(
        f"{check.phase}.{check.name}" for check in report.failed_checks
    )
    outcome = "passed" if report.is_successful else "failed"
    return (
        outcome,
        summary["passed"],
        summary["failed"],
        summary["skipped"],
        failed_checks,
    )


def _bounded_failure_reason(reason: str | None) -> tuple[str | None, bool | None, int | None]:
    """Cut the reason to the bound, and say whether it was cut and how long it was."""
    if reason is None:
        return None, None, None

    original_length = len(reason)
    if original_length <= MAX_FAILURE_REASON_LENGTH:
        return reason, False, original_length
    return reason[:MAX_FAILURE_REASON_LENGTH], True, original_length


def _lift(attributes: Mapping[str, str], key: str) -> str | None:
    """Read one run attribute into its own column; absent means ``NULL``, never ``""``."""
    value = attributes.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_int(raw: str | None) -> int | None:
    """Parse a numeric run attribute, or ``NULL`` when it is absent or not a number."""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _as_optional_string(value: Path | str | None) -> str | None:
    """Stringify a path the caller handed over. No resolution, no filesystem access."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_timezone_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
