from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from janus.models import ExecutionPlan, ExtractedArtifact, WriteResult
from janus.utils.storage import StorageLayout

SUPPORTED_RAW_ARTIFACT_FORMATS = frozenset({"binary", "json", "jsonl", "text"})
SUPPORTED_FILE_OUTPUT_ZONES = frozenset({"metadata", "raw"})
SUPPORTED_FILE_WRITE_MODES = frozenset({"append", "ignore", "overwrite"})

WRITE_CHUNK_BYTES = 1024 * 1024
SPOOL_THRESHOLD_BYTES = 64 * 1024 * 1024
STAGING_DIRNAME = ".staging"
PARTIAL_SUFFIX_MARKER = ".partial-"

# Suffix of the per-artifact checksum sidecar written next to every raw payload.
# Replay discovery (`scripts/raw_to_bronze.py`) must skip files ending with this
# so a sidecar is never mistaken for a data artifact.
SIDECAR_SUFFIX = ".sha256"


class RawWriteLimitError(ValueError):
    """Raised when a streamed artifact grows beyond its configured byte cap."""

    def __init__(self, *, max_bytes: int, bytes_seen: int) -> None:
        self.max_bytes = max_bytes
        self.bytes_seen = bytes_seen
        super().__init__(f"stream exceeded max_bytes={max_bytes}: {bytes_seen} bytes seen")


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

    def write_stream(
        self,
        plan: ExecutionPlan,
        relative_path: str | Path,
        stream: BinaryIO,
        *,
        zone: str = "raw",
        mode: str = "overwrite",
        metadata: Mapping[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> PersistedArtifact:
        """Persist a binary stream atomically while computing its checksum."""
        _validate_output_zone(zone)
        _validate_write_mode(mode)
        _validate_max_bytes(max_bytes)
        if mode == "append":
            raise ValueError("append mode is not supported for streamed artifacts")

        target = self.storage_layout.resolve_output(plan, zone)
        path = target.child(self._relative_path_for_zone(zone, relative_path))
        if mode == "ignore" and path.exists():
            checksum = _hash_file(path)
            return self._finalize_persisted_artifact(
                plan,
                path,
                checksum=checksum,
                zone=zone,
                format_name="binary",
                mode=mode,
                records_written=1,
                metadata=metadata,
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.parent / (
            f".{path.name}{PARTIAL_SUFFIX_MARKER}{uuid4().hex}"
        )
        try:
            with temporary_path.open("xb") as destination:
                _bytes_written, checksum = _copy_hashing(
                    stream,
                    destination,
                    max_bytes=max_bytes,
                )
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        return self._finalize_persisted_artifact(
            plan,
            path,
            checksum=checksum,
            zone=zone,
            format_name="binary",
            mode=mode,
            records_written=1,
            metadata=metadata,
        )

    def begin_staged_write(
        self,
        plan: ExecutionPlan,
        *,
        zone: str = "raw",
        max_bytes: int | None = None,
    ) -> StagedWrite:
        """Start a two-phase streamed write inside the selected output zone."""
        _validate_output_zone(zone)
        _validate_max_bytes(max_bytes)
        target = self.storage_layout.resolve_output(plan, zone)
        staging_dir = target.child(self._relative_path_for_zone(zone, STAGING_DIRNAME))
        staging_dir.mkdir(parents=True, exist_ok=True)
        return StagedWrite(
            writer=self,
            plan=plan,
            zone=zone,
            path=staging_dir / f"{uuid4().hex}.partial",
            max_bytes=max_bytes,
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
        return self._finalize_persisted_artifact(
            plan,
            persisted_path,
            checksum=checksum,
            zone=zone,
            format_name=format_name,
            mode=mode,
            records_written=records_written,
            metadata=metadata,
        )

    def _finalize_persisted_artifact(
        self,
        plan: ExecutionPlan,
        path: Path,
        *,
        checksum: str,
        zone: str,
        format_name: str,
        mode: str,
        records_written: int,
        metadata: Mapping[str, str] | None,
    ) -> PersistedArtifact:
        _write_checksum_sidecar(path, checksum)
        resolved_metadata = dict(metadata or {})
        resolved_metadata.setdefault("checksum", checksum)
        write_result = WriteResult.from_plan(
            plan,
            zone,
            path=str(path),
            format_name=format_name,
            mode=mode,
            records_written=records_written,
            metadata=resolved_metadata,
        )
        artifact = ExtractedArtifact(
            path=str(path),
            format=format_name,
            checksum=checksum,
        )
        return PersistedArtifact(artifact=artifact, write_result=write_result)

    def _relative_path_for_zone(self, zone: str, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if zone == "raw" and self._raw_path_prefix is not None:
            return self._raw_path_prefix / path
        return path


class StagedWrite:
    """A streamed payload held inside its output zone until its final path is known."""

    __slots__ = (
        "_committed",
        "_discarded",
        "_max_bytes",
        "_plan",
        "_sha256_hex",
        "_writer",
        "_written",
        "bytes_written",
        "path",
        "zone",
    )

    def __init__(
        self,
        *,
        writer: RawArtifactWriter,
        plan: ExecutionPlan,
        zone: str,
        path: Path,
        max_bytes: int | None,
    ) -> None:
        self._writer = writer
        self._plan = plan
        self.zone = zone
        self.path = path
        self._max_bytes = max_bytes
        self.bytes_written = 0
        self._sha256_hex: str | None = None
        self._written = False
        self._committed = False
        self._discarded = False

    @property
    def sha256_hex(self) -> str:
        if self._sha256_hex is None:
            raise RuntimeError("staged write has not finished copying")
        return self._sha256_hex

    def write_from(self, stream: BinaryIO) -> None:
        """Copy one stream into staging while computing its size and checksum."""
        self._ensure_active()
        if self._written:
            raise RuntimeError("staged write already contains a payload")

        try:
            with self.path.open("xb") as destination:
                bytes_written, checksum = _copy_hashing(
                    stream,
                    destination,
                    max_bytes=self._max_bytes,
                )
        except BaseException:
            self.discard()
            raise

        self.bytes_written = bytes_written
        self._sha256_hex = checksum
        self._written = True

    def open(self) -> BinaryIO:
        """Open the completed staged payload for archive sniffing or materialization."""
        self._ensure_active()
        if not self._written:
            raise RuntimeError("staged write has not finished copying")
        return self.path.open("rb")

    def commit(
        self,
        relative_path: str | Path,
        *,
        format_name: str = "binary",
        mode: str = "overwrite",
        metadata: Mapping[str, str] | None = None,
    ) -> PersistedArtifact:
        """Atomically move the staged payload to its final artifact path."""
        self._ensure_active()
        if not self._written:
            raise RuntimeError("staged write has not finished copying")
        _validate_format_name(format_name)
        _validate_write_mode(mode)
        if mode == "append":
            raise ValueError("append mode is not supported for streamed artifacts")

        target = self._writer.storage_layout.resolve_output(self._plan, self.zone)
        path = target.child(self._writer._relative_path_for_zone(self.zone, relative_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "ignore" and path.exists():
                checksum = _hash_file(path)
                self._remove_staged_path()
            else:
                checksum = self.sha256_hex
                os.replace(self.path, path)
                _remove_empty_staging_dir(self.path.parent)
            persisted = self._writer._finalize_persisted_artifact(
                self._plan,
                path,
                checksum=checksum,
                zone=self.zone,
                format_name=format_name,
                mode=mode,
                records_written=1,
                metadata=metadata,
            )
        except BaseException:
            self.discard()
            raise

        self._committed = True
        return persisted

    def discard(self) -> None:
        """Remove an uncommitted staged payload; safe to call more than once."""
        if self._committed or self._discarded:
            return
        self._remove_staged_path()
        self._discarded = True

    def __enter__(self) -> StagedWrite:
        self._ensure_active()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        if not self._committed:
            self.discard()

    def _ensure_active(self) -> None:
        if self._committed:
            raise RuntimeError("staged write has already been committed")
        if self._discarded:
            raise RuntimeError("staged write has already been discarded")

    def _remove_staged_path(self) -> None:
        self.path.unlink(missing_ok=True)
        _remove_empty_staging_dir(self.path.parent)


def _write_bytes(path: Path, payload: bytes, mode: str) -> tuple[str, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "ignore" and path.exists():
        return _hash_file(path), path

    if mode == "append":
        with path.open("ab") as stream:
            stream.write(payload)
        return _hash_file(path), path

    checksum = sha256(payload).hexdigest()
    path.write_bytes(payload)
    return checksum, path


def _copy_hashing(
    stream: BinaryIO,
    destination: BinaryIO,
    *,
    max_bytes: int | None,
) -> tuple[int, str]:
    digest = sha256()
    bytes_seen = 0
    while True:
        chunk = stream.read(WRITE_CHUNK_BYTES)
        if not chunk:
            break
        bytes_seen += len(chunk)
        if max_bytes is not None and bytes_seen > max_bytes:
            raise RawWriteLimitError(max_bytes=max_bytes, bytes_seen=bytes_seen)
        destination.write(chunk)
        digest.update(chunk)
    return bytes_seen, digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(WRITE_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_staging_dir(path: Path) -> None:
    if path.name != STAGING_DIRNAME:
        return
    with suppress(OSError):
        path.rmdir()


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


def _validate_max_bytes(max_bytes: int | None) -> None:
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be greater than or equal to 0")
