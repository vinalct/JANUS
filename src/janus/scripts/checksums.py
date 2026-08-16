"""What a raw file *is*, inferred from its path and from what extraction left beside it.

Replay meets raw artifacts it did not write in this run and has to answer two questions
about each one — what format to hand the reader, and what its SHA-256 is. Both answers
come from the on-disk layout, never from the run that produced the file, which is why
these helpers sit next to ``rehydrate`` rather than inside it.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from janus.writers import SIDECAR_SUFFIX

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


def _resolve_raw_checksum(path: Path, *, verify: bool = False) -> str:
    """Return the artifact's SHA-256, reading the ``.sha256`` sidecar when present.

    Both branches are load-bearing; neither is dead code. Extraction writes a bare-hex
    sidecar next to every raw artifact precisely so replay can read a digest instead of
    re-hashing the body — on the multi-GB CNPJ files that is the difference between a
    cheap rediscovery and re-reading the whole raw zone. The full-read fallback is what
    keeps ``--ingest-raw-to-bronze`` working against zones written before sidecars
    existed; delete it and replay of a legacy zone stops resolving a checksum at all.

    ``verify`` re-hashes even on a sidecar hit and rejects a sidecar that disagrees with
    the body. It stays opt-in because paying that read on every artifact is exactly the
    cost the sidecar was introduced to avoid.
    """

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
