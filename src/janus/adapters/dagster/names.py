"""Stable Dagster identifiers derived from arbitrary JANUS source ids."""

from __future__ import annotations

import hashlib
import re

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_]+")
_REPEATED_UNDERSCORE = re.compile(r"_+")
_MAX_STEM_LENGTH = 48


def source_op_name(source_id: str) -> str:
    """Return a valid, stable, collision-resistant Dagster op name.

    The readable stem is deliberately not treated as identity: punctuation, spaces,
    case, and truncation can all collapse distinct source ids. A digest of the original
    UTF-8 value preserves that identity while the fixed prefix handles leading digits.
    """
    stem = _UNSAFE_NAME.sub("_", source_id.strip()).strip("_").lower()
    stem = _REPEATED_UNDERSCORE.sub("_", stem) or "source"
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return f"janus_source_{stem[:_MAX_STEM_LENGTH]}_{digest}"
