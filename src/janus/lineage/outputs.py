"""Configured and materialized outputs carried by run and lineage records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from janus.models import ExtractedArtifact, WriteResult


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
