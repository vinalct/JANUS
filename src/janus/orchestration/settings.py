"""Strict, engine-free orchestration settings from an environment profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ORCHESTRATION_BLOCK_KEY = "orchestration"
SUPPORTED_ORCHESTRATION_KEYS = frozenset({"max_parallel_default", "lock_wait_seconds"})


@dataclass(frozen=True, slots=True)
class OrchestrationSettings:
    """Batch capacity and the bounded wait for external source-lock contention."""

    max_parallel_default: int = 1
    lock_wait_seconds: int = 0


DEFAULT_ORCHESTRATION_SETTINGS = OrchestrationSettings()


def resolve_orchestration_settings(config: Mapping[str, Any]) -> OrchestrationSettings:
    """Absent block uses defaults; a present block must declare supported integers."""
    if ORCHESTRATION_BLOCK_KEY not in config:
        return DEFAULT_ORCHESTRATION_SETTINGS
    block = config[ORCHESTRATION_BLOCK_KEY]
    if not isinstance(block, Mapping):
        raise ValueError(f"{ORCHESTRATION_BLOCK_KEY} must be a mapping")
    unsupported = sorted(str(key) for key in block if key not in SUPPORTED_ORCHESTRATION_KEYS)
    if unsupported:
        raise ValueError(
            f"{ORCHESTRATION_BLOCK_KEY} has unsupported key(s): {', '.join(unsupported)}"
        )
    return OrchestrationSettings(
        max_parallel_default=_integer(block, "max_parallel_default", minimum=1, default=1),
        lock_wait_seconds=_integer(block, "lock_wait_seconds", minimum=0, default=0),
    )


def _integer(block: Mapping[str, Any], key: str, *, minimum: int, default: int) -> int:
    value = block.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{ORCHESTRATION_BLOCK_KEY}.{key} must be an integer >= {minimum}, "
            f"got {type(value).__name__}"
        )
    return value
