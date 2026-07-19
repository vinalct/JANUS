from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan, ExtractedArtifact, WriteResult
from janus.utils.storage import StorageLayout

SUPPORTED_RAW_ARTIFACT_FORMATS = frozenset({"binary", "json", "jsonl", "text"})
SUPPORTED_FILE_OUTPUT_ZONES = frozenset({"metadata", "raw"})
SUPPORTED_FILE_WRITE_MODES = frozenset({"append", "ignore", "overwrite"})

# Suffix of the per-artifact checksum sidecar written next to every raw payload.
# Replay discovery (`scripts/raw_to_bronze.py`) must skip files ending with this
# so a sidecar is never mistaken for a data artifact.
SIDECAR_SUFFIX = ".sha256"


@dataclass(frozen=True, slots=True)
class PersistedArtifact:
    """Artifact plus write metadata returned by the raw file writer."""

    artifact: ExtractedArtifact
    write_result: WriteResult


class RawArtifactWriter:
    """Persist exact payloads into raw-like JANUS zones with deterministic paths."""

    def __init__(
        self,
        storage_layout: StorageLayout,
        *,
        raw_path_prefix: str | Path | None = None,
    ) -> None:
        self.storage_layout = storage_layout
        self._raw_path_prefix = Path(raw_path_prefix) if raw_path_prefix is not None else None

    @property
    def raw_path_prefix(self) -> Path | None:
        return self._raw_path_prefix

    def with_raw_path_prefix(self, raw_path_prefix: str | Path | None) -> RawArtifactWriter:
        return RawArtifactWriter(self.storage_layout, raw_path_prefix=raw_path_prefix)

    def write_bytes(
        self,
        plan: ExecutionPlan,
        relative_path: str | Path,
        payload: bytes,
        *,
        zone: str = "raw",
        mode: str = "overwrite",
        metadata: Mapping[str, str] | None = None,
    ) -> PersistedArtifact:
        return self._write_payload(
            plan,
            relative_path,
            payload,
            zone=zone,
            format_name="binary",
            mode=mode,
            records_written=1,
            metadata=metadata,
        )

    def write_text(
        self,
        plan: ExecutionPlan,
        relative_path: str | Path,
        payload: str,
        *,
        zone: str = "raw",
        mode: str = "overwrite",
        metadata: Mapping[str, str] | None = None,
    ) -> PersistedArtifact:
        return self._write_payload(
            plan,
            relative_path,
            payload.encode("utf-8"),
            zone=zone,
            format_name="text",
            mode=mode,
            records_written=1,
            metadata=metadata,
        )

    def write_json(
        self,
        plan: ExecutionPlan,
        relative_path: str | Path,
        payload: Any,
        *,
        zone: str = "raw",
        mode: str = "overwrite",
        metadata: Mapping[str, str] | None = None,
    ) -> PersistedArtifact:
        if mode == "append":
            raise ValueError("append mode is not supported for json artifacts")
        body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        return self._write_payload(
            plan,
            relative_path,
            body.encode("utf-8"),
            zone=zone,
            format_name="json",
            mode=mode,
            records_written=1,
            metadata=metadata,
        )

    def write_json_lines(
        self,
        plan: ExecutionPlan,
        relative_path: str | Path,
        records: Iterable[Any],
        *,
        zone: str = "raw",
        mode: str = "overwrite",
        metadata: Mapping[str, str] | None = None,
    ) -> PersistedArtifact:
        serialized_records = [
            json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records
        ]
        payload = ("\n".join(serialized_records) + ("\n" if serialized_records else "")).encode(
            "utf-8"
        )
        return self._write_payload(
            plan,
            relative_path,
            payload,
            zone=zone,
            format_name="jsonl",
            mode=mode,
            records_written=len(serialized_records),
            metadata=metadata,
        )

    def _write_payload(
        self,
        plan: ExecutionPlan,
        relative_path: str | Path,
        payload: bytes,
        *,
        zone: str,
        format_name: str,
        mode: str,
        records_written: int,
        metadata: Mapping[str, str] | None,
    ) -> PersistedArtifact:
        _validate_output_zone(zone)
        _validate_format_name(format_name)
        _validate_write_mode(mode)

        target = self.storage_layout.resolve_output(plan, zone)
        path = target.child(self._relative_path_for_zone(zone, relative_path))
        checksum, persisted_path = _write_bytes(path, payload, mode)
        _write_checksum_sidecar(persisted_path, checksum)

        resolved_metadata = dict(metadata or {})
        resolved_metadata.setdefault("checksum", checksum)
        write_result = WriteResult.from_plan(
            plan,
            zone,
            path=str(persisted_path),
            format_name=format_name,
            mode=mode,
            records_written=records_written,
            metadata=resolved_metadata,
        )
        artifact = ExtractedArtifact(
            path=str(persisted_path),
            format=format_name,
            checksum=checksum,
        )
        return PersistedArtifact(artifact=artifact, write_result=write_result)

    def _relative_path_for_zone(self, zone: str, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if zone == "raw" and self._raw_path_prefix is not None:
            return self._raw_path_prefix / path
        return path


def _write_bytes(path: Path, payload: bytes, mode: str) -> tuple[str, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "ignore" and path.exists():
        existing_payload = path.read_bytes()
        return sha256(existing_payload).hexdigest(), path

    if mode == "append":
        with path.open("ab") as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)
    return sha256(path.read_bytes()).hexdigest(), path


def _write_checksum_sidecar(path: Path, checksum: str) -> None:
    """Persist ``checksum`` next to ``path`` as ``<path>.sha256`` (bare digest).

    The digest is the identical string returned as ``ExtractedArtifact.checksum``
    for the same write; replay reads it instead of re-hashing the payload. The
    format matches the file family's integrity sidecars, so
    ``_read_checksum_sidecar`` can parse it.
    """

    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    sidecar.write_text(f"{checksum}\n", encoding="utf-8")


def _validate_output_zone(zone: str) -> None:
    if zone not in SUPPORTED_FILE_OUTPUT_ZONES:
        allowed = ", ".join(sorted(SUPPORTED_FILE_OUTPUT_ZONES))
        raise ValueError(f"zone must be one of: {allowed}")


def _validate_format_name(format_name: str) -> None:
    if format_name not in SUPPORTED_RAW_ARTIFACT_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_RAW_ARTIFACT_FORMATS))
        raise ValueError(f"format_name must be one of: {allowed}")


def _validate_write_mode(mode: str) -> None:
    if mode not in SUPPORTED_FILE_WRITE_MODES:
        allowed = ", ".join(sorted(SUPPORTED_FILE_WRITE_MODES))
        raise ValueError(f"mode must be one of: {allowed}")
