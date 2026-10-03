"""What a raw file *is*, inferred from its path and from what extraction left beside it.

Replay meets raw artifacts it did not write in this run and has to answer two questions
about each one — what format to hand the reader, and what its SHA-256 is. Both answers
come from the on-disk layout, never from the run that produced the file, which is why
these helpers sit next to ``rehydrate`` rather than inside it.

The SHA-256 half lives in ``janus.writers.sidecar``, beside the writer that produces the
sidecar, because the catalog strategy resolves digests the same way and must not import
``janus.scripts``. It is re-exported here so every existing import path keeps working.
"""

from __future__ import annotations

from pathlib import Path

from janus.writers.sidecar import (
    RawArtifactIntegrityError,
    _read_sidecar_checksum,
    _resolve_raw_checksum,
    _sha256,
)

__all__ = [
    "RawArtifactIntegrityError",
    "_artifact_format_for_path",
    "_read_sidecar_checksum",
    "_resolve_raw_checksum",
    "_sha256",
]

_READABLE_ARTIFACT_FORMATS = frozenset(
    {"binary", "csv", "json", "jsonl", "parquet", "text"}
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
