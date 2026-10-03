"""The read side of the ``.sha256`` sidecar that ``RawArtifactWriter`` writes beside raw artifacts.

``raw.py`` writes the sidecar from the bytes it has just persisted. This module answers the
question every later reader asks: which digest did extraction record, and, when the caller
asks, does the file on disk still agree with it?

It lives in ``janus.writers`` rather than ``janus.scripts`` so that the catalog strategy's
rediscovery and the replay scripts read a digest through the same function, without
``janus.strategies`` importing ``janus.scripts``. ``janus.scripts.checksums`` re-exports
every name here, so the old import paths still work.
"""

from __future__ import annotations

from pathlib import Path

from janus.writers.raw import SIDECAR_SUFFIX, _hash_file


class RawArtifactIntegrityError(ValueError):
    """A raw artifact does not match the digest that extraction recorded beside it.

    It subclasses ``ValueError`` because ``_resolve_raw_checksum`` already raised that on a
    mismatch. The replay loader's existing failure handling therefore records it as a failed
    run, not as a crash of a different shape.
    """

    def __init__(self, path: Path, *, sidecar_digest: str, computed_digest: str) -> None:
        self.path = path
        self.sidecar_digest = sidecar_digest
        self.computed_digest = computed_digest
        super().__init__(
            f"Raw artifact integrity check failed for {path}: sidecar sha256 {sidecar_digest}, "
            f"computed {computed_digest}. The raw zone has changed since extraction; "
            "bronze was not written."
        )


def _resolve_raw_checksum(path: Path, *, verify: bool = False) -> str:
    """Return the artifact's SHA-256, reading the ``.sha256`` sidecar when one exists."""

    cached = _read_sidecar_checksum(path.with_name(path.name + SIDECAR_SUFFIX))
    if cached is None:
        return _sha256(path)
    if verify:
        recomputed = _sha256(path)
        if recomputed != cached:
            raise RawArtifactIntegrityError(path, sidecar_digest=cached, computed_digest=recomputed)
    return cached


def _read_sidecar_checksum(sidecar: Path) -> str | None:
    """Read a bare-digest ``.sha256`` sidecar. Return ``None`` if it is missing or empty."""

    if not sidecar.is_file():
        return None
    content = sidecar.read_text(encoding="utf-8").strip()
    if not content:
        return None
    return content.split()[0]

_sha256 = _hash_file
