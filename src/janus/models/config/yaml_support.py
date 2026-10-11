"""Safe YAML parsing with LibYAML when available, without caching mutable documents."""

from __future__ import annotations

from typing import IO, Any

import yaml

_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def safe_load_yaml(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Use the same safe constructors as ``safe_load``, with the faster C parser."""
    return yaml.load(stream, Loader=_SAFE_LOADER)
