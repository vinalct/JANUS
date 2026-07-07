from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from janus.models.source_config import OutputTarget, SourceConfig

SUPPORTED_OUTPUT_ZONES = frozenset({"bronze", "metadata", "raw"})


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Small source identity used across planning, extraction, and metadata flows."""

    source_id: str
    name: str
    owner: str
    domain: str
    federation_level: str
    source_type: str
    strategy: str
    strategy_variant: str
    source_hook: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_source_config(cls, source_config: SourceConfig) -> Self:
        """Project the full source config into the smaller reference shared by runtime flows."""
        return cls(
            source_id=source_config.source_id,
            name=source_config.name,
            owner=source_config.owner,
            domain=source_config.domain,
            federation_level=source_config.federation_level,
            source_type=source_config.source_type,
            strategy=source_config.strategy,
            strategy_variant=source_config.strategy_variant,
            source_hook=source_config.source_hook,
            tags=source_config.tags,
        )


@dataclass(frozen=True, slots=True)
class RunContext:
    """Execution-scoped details that keep runs deterministic and traceable."""

    run_id: str
    environment: str
    project_root: Path
    started_at: datetime
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.environment.strip():
            raise ValueError("environment must not be empty")
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")

    @classmethod
    def create(
        cls,
        run_id: str,
        environment: str,
        project_root: Path,
        *,
        started_at: datetime | None = None,
        attributes: Mapping[str, str] | None = None,
    ) -> Self:
        """Build a normalized run context with stable path and attribute handling."""
        return cls(
            run_id=run_id,
            environment=environment,
            project_root=project_root.resolve(),
            started_at=started_at or datetime.now(tz=UTC),
            attributes=_freeze_string_mapping(attributes),
        )

    def attributes_as_dict(self) -> dict[str, str]:
        """Return the run attributes in a mapping form useful to logging or metadata code."""
        return dict(self.attributes)

    def with_attribute(self, key: str, value: str) -> Self:
        """Return a new run context with one additional string attribute."""
        return replace(self, attributes=_merge_string_mapping(self.attributes, key, value))


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Planner output consumed by strategy implementations and later runtime layers."""

    run_context: RunContext
    source: SourceReference
    source_config: SourceConfig
    extraction_mode: str
    checkpoint_strategy: str
    raw_output: OutputTarget
    bronze_output: OutputTarget
    metadata_output: OutputTarget
    checkpoint_field: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source.source_id != self.source_config.source_id:
            raise ValueError("source.source_id must match source_config.source_id")

    @classmethod
    def from_source_config(cls, source_config: SourceConfig, run_context: RunContext) -> Self:
        """Build the strategy-agnostic execution plan shape expected by planner code."""
        return cls(
            run_context=run_context,
            source=SourceReference.from_source_config(source_config),
            source_config=source_config,
            extraction_mode=source_config.extraction.mode,
            checkpoint_strategy=source_config.extraction.checkpoint_strategy,
            checkpoint_field=source_config.extraction.checkpoint_field,
            raw_output=source_config.outputs.raw,
            bronze_output=source_config.outputs.bronze,
            metadata_output=source_config.outputs.metadata,
        )

    def with_note(self, note: str) -> Self:
        """Return a new execution plan with an appended planner or hook note."""
        normalized_note = note.strip()
        if not normalized_note:
            raise ValueError("note must not be empty")
        return replace(self, notes=self.notes + (normalized_note,))


@dataclass(frozen=True, slots=True)
class ExtractedArtifact:
    """One extracted raw artifact that can later be handed to normalization code."""

    path: str
    format: str
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("artifact path must not be empty")
        if not self.format.strip():
            raise ValueError("artifact format must not be empty")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Strategy extraction output and the handoff object for later normalization work."""

    source: SourceReference
    run_id: str
    artifacts: tuple[ExtractedArtifact, ...]
    records_extracted: int | None = None
    checkpoint_value: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_plan(
        cls,
        plan: ExecutionPlan,
        artifacts: tuple[ExtractedArtifact, ...],
        *,
        records_extracted: int | None = None,
        checkpoint_value: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        """Build an extraction result that stays linked to one source and one run."""
        return cls(
            source=plan.source,
            run_id=plan.run_context.run_id,
            artifacts=artifacts,
            records_extracted=records_extracted,
            checkpoint_value=checkpoint_value,
            metadata=_freeze_string_mapping(metadata),
        )

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        """Expose raw artifact paths in the order produced by the extraction phase."""
        return tuple(artifact.path for artifact in self.artifacts)

    @property
    def is_empty(self) -> bool:
        """Tell downstream code whether extraction produced any raw artifacts."""
        return not self.artifacts

    def metadata_as_dict(self) -> dict[str, str]:
        """Return extraction metadata as a plain mapping for simpler assertions or logging."""
        return dict(self.metadata)

    def single_artifact_format(self) -> str:
        """Return the single handoff format carried by this extraction result."""
        formats = {artifact.format for artifact in self.artifacts}
        if len(formats) != 1:
            raise ValueError(
                "Artifacts contain multiple formats; the caller must resolve the handoff "
                "before reading it."
            )
        return next(iter(formats))

    def with_metadata(self, key: str, value: str) -> Self:
        """Return a new extraction result with one additional metadata entry."""
        return replace(self, metadata=_merge_string_mapping(self.metadata, key, value))


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Result of writing one explicit JANUS output zone."""

    source: SourceReference
    run_id: str
    zone: str
    path: str
    format: str
    mode: str
    records_written: int | None = None
    partition_by: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.zone not in SUPPORTED_OUTPUT_ZONES:
            allowed_zones = ", ".join(sorted(SUPPORTED_OUTPUT_ZONES))
            raise ValueError(f"zone must be one of: {allowed_zones}")
        if not self.path.strip():
            raise ValueError("path must not be empty")
        if not self.format.strip():
            raise ValueError("format must not be empty")
        if not self.mode.strip():
            raise ValueError("mode must not be empty")

    @classmethod
    def from_plan(
        cls,
        plan: ExecutionPlan,
        zone: str,
        *,
        path: str | None = None,
        format_name: str | None = None,
        mode: str | None = None,
        records_written: int | None = None,
        partition_by: tuple[str, ...] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        """Build a write result from the configured output target for one zone."""
        target = _output_target_for_zone(plan, zone)
        return cls(
            source=plan.source,
            run_id=plan.run_context.run_id,
            zone=zone,
            path=path or target.path,
            format=format_name or target.format,
            mode=mode or plan.source_config.spark.write_mode,
            records_written=records_written,
            partition_by=partition_by or plan.source_config.spark.partition_by,
            metadata=_freeze_string_mapping(metadata),
        )

    def metadata_as_dict(self) -> dict[str, str]:
        """Return write metadata as a plain mapping for metadata-emission code."""
        return dict(self.metadata)

    def with_metadata(self, key: str, value: str) -> Self:
        """Return a new write result with one additional metadata entry."""
        return replace(self, metadata=_merge_string_mapping(self.metadata, key, value))


def _output_target_for_zone(plan: ExecutionPlan, zone: str) -> OutputTarget:
    if zone == "raw":
        return plan.raw_output
    if zone == "bronze":
        return plan.bronze_output
    if zone == "metadata":
        return plan.metadata_output

    allowed_zones = ", ".join(sorted(SUPPORTED_OUTPUT_ZONES))
    raise ValueError(f"zone must be one of: {allowed_zones}")


def _freeze_string_mapping(mapping: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not mapping:
        return ()

    pairs: list[tuple[str, str]] = []
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError("mapping keys must be strings")
        if not isinstance(value, str):
            raise TypeError("mapping values must be strings")
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            raise ValueError("mapping keys must not be empty")
        if not normalized_value:
            raise ValueError("mapping values must not be empty")
        pairs.append((normalized_key, normalized_value))

    return tuple(sorted(pairs))


def _merge_string_mapping(
    pairs: tuple[tuple[str, str], ...], key: str, value: str
) -> tuple[tuple[str, str], ...]:
    merged = dict(pairs)
    normalized_key = key.strip()
    normalized_value = value.strip()
    if not normalized_key:
        raise ValueError("mapping keys must not be empty")
    if not normalized_value:
        raise ValueError("mapping values must not be empty")
    merged[normalized_key] = normalized_value
    return tuple(sorted(merged.items()))
