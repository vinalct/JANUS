from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Self

from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, WriteResult

SUPPORTED_RUN_STATUSES = frozenset({"failed", "running", "succeeded"})


@dataclass(frozen=True, slots=True)
class ConfiguredOutput:
    """Configured output target kept in metadata and lineage artifacts."""

    zone: str
    path: str
    format: str

    def __post_init__(self) -> None:
        if not self.zone.strip():
            raise ValueError("zone must not be empty")
        if not self.path.strip():
            raise ValueError("path must not be empty")
        if not self.format.strip():
            raise ValueError("format must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "path": self.path,
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class MaterializedOutput:
    """One output that was actually written during a run."""

    zone: str
    path: str
    format: str
    mode: str
    records_written: int | None = None
    partition_by: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.zone.strip():
            raise ValueError("zone must not be empty")
        if not self.path.strip():
            raise ValueError("path must not be empty")
        if not self.format.strip():
            raise ValueError("format must not be empty")
        if not self.mode.strip():
            raise ValueError("mode must not be empty")

    @classmethod
    def from_write_result(cls, write_result: WriteResult) -> Self:
        return cls(
            zone=write_result.zone,
            path=write_result.path,
            format=write_result.format,
            mode=write_result.mode,
            records_written=write_result.records_written,
            partition_by=write_result.partition_by,
            metadata=write_result.metadata,
        )

    def metadata_as_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "zone": self.zone,
            "path": self.path,
            "format": self.format,
            "mode": self.mode,
            "partition_by": list(self.partition_by),
            "metadata": self.metadata_as_dict(),
        }
        if self.records_written is not None:
            payload["records_written"] = self.records_written
        return payload


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Raw extraction artifact preserved in lineage records."""

    path: str
    format: str
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("path must not be empty")
        if not self.format.strip():
            raise ValueError("format must not be empty")

    @classmethod
    def from_extracted_artifact(cls, artifact: ExtractedArtifact) -> Self:
        return cls(
            path=artifact.path,
            format=artifact.format,
            checksum=artifact.checksum,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "format": self.format,
        }
        if self.checksum is not None:
            payload["checksum"] = self.checksum
        return payload


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Traceable run record persisted to the metadata zone."""

    run_id: str
    source_id: str
    source_name: str
    environment: str
    strategy_family: str
    strategy_variant: str
    extraction_mode: str
    checkpoint_strategy: str
    status: str
    started_at: datetime
    source_config_path: str
    configured_outputs: tuple[ConfiguredOutput, ...]
    materialized_outputs: tuple[MaterializedOutput, ...] = ()
    checkpoint_field: str | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    records_extracted: int | None = None
    checkpoint_value: str | None = None
    failure_reason: str | None = None
    error_type: str | None = None
    run_attributes: tuple[tuple[str, str], ...] = ()
    plan_notes: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in SUPPORTED_RUN_STATUSES:
            allowed_statuses = ", ".join(sorted(SUPPORTED_RUN_STATUSES))
            raise ValueError(f"status must be one of: {allowed_statuses}")
        _validate_timezone_aware("started_at", self.started_at)
        if self.ended_at is not None:
            _validate_timezone_aware("ended_at", self.ended_at)
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.environment.strip():
            raise ValueError("environment must not be empty")
        if not self.strategy_family.strip():
            raise ValueError("strategy_family must not be empty")
        if not self.strategy_variant.strip():
            raise ValueError("strategy_variant must not be empty")
        if not self.extraction_mode.strip():
            raise ValueError("extraction_mode must not be empty")
        if not self.checkpoint_strategy.strip():
            raise ValueError("checkpoint_strategy must not be empty")
        if not self.source_config_path.strip():
            raise ValueError("source_config_path must not be empty")

    @classmethod
    def started(
        cls,
        plan: ExecutionPlan,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        return cls(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            source_name=plan.source.name,
            environment=plan.run_context.environment,
            strategy_family=plan.source.strategy,
            strategy_variant=plan.source.strategy_variant,
            extraction_mode=plan.extraction_mode,
            checkpoint_strategy=plan.checkpoint_strategy,
            checkpoint_field=plan.checkpoint_field,
            status="running",
            started_at=plan.run_context.started_at,
            source_config_path=str(plan.source_config.config_path),
            configured_outputs=configured_outputs_from_plan(plan),
            run_attributes=plan.run_context.attributes,
            plan_notes=plan.notes,
            metadata=_freeze_string_mapping(metadata),
        )

    @classmethod
    def succeeded(
        cls,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        *,
        finished_at: datetime | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        completed_at = finished_at or datetime.now(tz=UTC)
        return cls(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            source_name=plan.source.name,
            environment=plan.run_context.environment,
            strategy_family=plan.source.strategy,
            strategy_variant=plan.source.strategy_variant,
            extraction_mode=plan.extraction_mode,
            checkpoint_strategy=plan.checkpoint_strategy,
            checkpoint_field=plan.checkpoint_field,
            status="succeeded",
            started_at=plan.run_context.started_at,
            ended_at=completed_at,
            duration_seconds=_duration_seconds(plan.run_context.started_at, completed_at),
            source_config_path=str(plan.source_config.config_path),
            configured_outputs=configured_outputs_from_plan(plan),
            materialized_outputs=tuple(
                MaterializedOutput.from_write_result(write_result)
                for write_result in write_results
            ),
            records_extracted=extraction_result.records_extracted,
            checkpoint_value=extraction_result.checkpoint_value,
            run_attributes=plan.run_context.attributes,
            plan_notes=plan.notes,
            metadata=_freeze_string_mapping(metadata),
        )

    @classmethod
    def failed(
        cls,
        plan: ExecutionPlan,
        error: Exception | str,
        extraction_result: ExtractionResult | None = None,
        write_results: tuple[WriteResult, ...] = (),
        *,
        finished_at: datetime | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        failed_at = finished_at or datetime.now(tz=UTC)
        failure_reason = str(error).strip()
        if not failure_reason:
            raise ValueError("failure_reason must not be empty")

        error_type = type(error).__name__ if isinstance(error, Exception) else "RuntimeError"
        return cls(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            source_name=plan.source.name,
            environment=plan.run_context.environment,
            strategy_family=plan.source.strategy,
            strategy_variant=plan.source.strategy_variant,
            extraction_mode=plan.extraction_mode,
            checkpoint_strategy=plan.checkpoint_strategy,
            checkpoint_field=plan.checkpoint_field,
            status="failed",
            started_at=plan.run_context.started_at,
            ended_at=failed_at,
            duration_seconds=_duration_seconds(plan.run_context.started_at, failed_at),
            source_config_path=str(plan.source_config.config_path),
            configured_outputs=configured_outputs_from_plan(plan),
            materialized_outputs=tuple(
                MaterializedOutput.from_write_result(write_result)
                for write_result in write_results
            ),
            records_extracted=(
                extraction_result.records_extracted if extraction_result is not None else None
            ),
            checkpoint_value=(
                extraction_result.checkpoint_value if extraction_result is not None else None
            ),
            failure_reason=failure_reason,
            error_type=error_type,
            run_attributes=plan.run_context.attributes,
            plan_notes=plan.notes,
            metadata=_freeze_string_mapping(metadata),
        )

    def metadata_as_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def run_attributes_as_dict(self) -> dict[str, str]:
        return dict(self.run_attributes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "environment": self.environment,
            "strategy_family": self.strategy_family,
            "strategy_variant": self.strategy_variant,
            "extraction_mode": self.extraction_mode,
            "checkpoint_strategy": self.checkpoint_strategy,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "source_config_path": self.source_config_path,
            "configured_outputs": [output.to_dict() for output in self.configured_outputs],
            "materialized_outputs": [
                output.to_dict() for output in self.materialized_outputs
            ],
            "run_attributes": self.run_attributes_as_dict(),
            "plan_notes": list(self.plan_notes),
            "metadata": self.metadata_as_dict(),
        }
        if self.checkpoint_field is not None:
            payload["checkpoint_field"] = self.checkpoint_field
        if self.ended_at is not None:
            payload["ended_at"] = self.ended_at.isoformat()
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        if self.records_extracted is not None:
            payload["records_extracted"] = self.records_extracted
        if self.checkpoint_value is not None:
            payload["checkpoint_value"] = self.checkpoint_value
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """Lineage artifact that links one run to config version and produced outputs."""

    run_id: str
    source_id: str
    source_name: str
    environment: str
    strategy_family: str
    strategy_variant: str
    extraction_mode: str
    checkpoint_strategy: str
    status: str
    emitted_at: datetime
    source_config_path: str
    config_version: str
    configured_outputs: tuple[ConfiguredOutput, ...]
    materialized_outputs: tuple[MaterializedOutput, ...] = ()
    artifacts: tuple[ArtifactSnapshot, ...] = ()
    checkpoint_field: str | None = None
    source_hook: str | None = None
    records_extracted: int | None = None
    checkpoint_value: str | None = None
    failure_reason: str | None = None
    error_type: str | None = None
    run_attributes: tuple[tuple[str, str], ...] = ()
    plan_notes: tuple[str, ...] = ()
    extraction_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in SUPPORTED_RUN_STATUSES:
            allowed_statuses = ", ".join(sorted(SUPPORTED_RUN_STATUSES))
            raise ValueError(f"status must be one of: {allowed_statuses}")
        _validate_timezone_aware("emitted_at", self.emitted_at)
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.environment.strip():
            raise ValueError("environment must not be empty")
        if not self.strategy_family.strip():
            raise ValueError("strategy_family must not be empty")
        if not self.strategy_variant.strip():
            raise ValueError("strategy_variant must not be empty")
        if not self.extraction_mode.strip():
            raise ValueError("extraction_mode must not be empty")
        if not self.checkpoint_strategy.strip():
            raise ValueError("checkpoint_strategy must not be empty")
        if not self.source_config_path.strip():
            raise ValueError("source_config_path must not be empty")
        if not self.config_version.strip():
            raise ValueError("config_version must not be empty")

    @classmethod
    def from_runtime(
        cls,
        plan: ExecutionPlan,
        *,
        status: str,
        extraction_result: ExtractionResult | None = None,
        write_results: tuple[WriteResult, ...] = (),
        failure_reason: str | None = None,
        error_type: str | None = None,
        emitted_at: datetime | None = None,
    ) -> Self:
        resolved_emitted_at = emitted_at or datetime.now(tz=UTC)
        return cls(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            source_name=plan.source.name,
            environment=plan.run_context.environment,
            strategy_family=plan.source.strategy,
            strategy_variant=plan.source.strategy_variant,
            extraction_mode=plan.extraction_mode,
            checkpoint_strategy=plan.checkpoint_strategy,
            checkpoint_field=plan.checkpoint_field,
            source_hook=plan.source.source_hook,
            status=status,
            emitted_at=resolved_emitted_at,
            source_config_path=str(plan.source_config.config_path),
            config_version=compute_config_version(plan.source_config.config_path),
            configured_outputs=configured_outputs_from_plan(plan),
            materialized_outputs=tuple(
                MaterializedOutput.from_write_result(write_result)
                for write_result in write_results
            ),
            artifacts=tuple(
                ArtifactSnapshot.from_extracted_artifact(artifact)
                for artifact in extraction_result.artifacts
            )
            if extraction_result is not None
            else (),
            records_extracted=(
                extraction_result.records_extracted if extraction_result is not None else None
            ),
            checkpoint_value=(
                extraction_result.checkpoint_value if extraction_result is not None else None
            ),
            failure_reason=failure_reason,
            error_type=error_type,
            run_attributes=plan.run_context.attributes,
            plan_notes=plan.notes,
            extraction_metadata=(
                extraction_result.metadata if extraction_result is not None else ()
            ),
        )

    def extraction_metadata_as_dict(self) -> dict[str, str]:
        return dict(self.extraction_metadata)

    def run_attributes_as_dict(self) -> dict[str, str]:
        return dict(self.run_attributes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "environment": self.environment,
            "strategy_family": self.strategy_family,
            "strategy_variant": self.strategy_variant,
            "extraction_mode": self.extraction_mode,
            "checkpoint_strategy": self.checkpoint_strategy,
            "status": self.status,
            "emitted_at": self.emitted_at.isoformat(),
            "source_config_path": self.source_config_path,
            "config_version": self.config_version,
            "configured_outputs": [output.to_dict() for output in self.configured_outputs],
            "materialized_outputs": [
                output.to_dict() for output in self.materialized_outputs
            ],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "run_attributes": self.run_attributes_as_dict(),
            "plan_notes": list(self.plan_notes),
            "extraction_metadata": self.extraction_metadata_as_dict(),
        }
        if self.checkpoint_field is not None:
            payload["checkpoint_field"] = self.checkpoint_field
        if self.source_hook is not None:
            payload["source_hook"] = self.source_hook
        if self.records_extracted is not None:
            payload["records_extracted"] = self.records_extracted
        if self.checkpoint_value is not None:
            payload["checkpoint_value"] = self.checkpoint_value
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload


def configured_outputs_from_plan(plan: ExecutionPlan) -> tuple[ConfiguredOutput, ...]:
    return (
        ConfiguredOutput(zone="raw", path=plan.raw_output.path, format=plan.raw_output.format),
        ConfiguredOutput(
            zone="bronze",
            path=plan.bronze_output.path,
            format=plan.bronze_output.format,
        ),
        ConfiguredOutput(
            zone="metadata",
            path=plan.metadata_output.path,
            format=plan.metadata_output.format,
        ),
    )


def compute_config_version(config_path: Path) -> str:
    """Hash the checked-in source config so lineage can pin one exact config version."""
    return sha256(config_path.read_bytes()).hexdigest()


def _duration_seconds(started_at: datetime, ended_at: datetime) -> float:
    return round((ended_at - started_at).total_seconds(), 6)


def _freeze_string_mapping(values: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()

    frozen_items: list[tuple[str, str]] = []
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("mapping keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"mapping value for {key!r} must be a non-empty string")
        frozen_items.append((key.strip(), value.strip()))
    return tuple(sorted(frozen_items))


def _validate_timezone_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
