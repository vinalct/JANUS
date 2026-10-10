"""Structural validation of a source's declared operational expectations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from janus.models.config.coercion import _optional_int, _require_mapping
from janus.models.config.constants import FRESHNESS_UNIT_SECONDS, MIN_MAX_DURATION_SECONDS
from janus.models.config.issues import ValidationIssue
from janus.models.config.types import FreshnessConfig, OperationsConfig


def _build_operations_config(
    raw_value: Any, issues: list[ValidationIssue]
) -> OperationsConfig:
    """Absent means undeclared; malformed declarations collect issues without raising."""
    data = {} if raw_value is None else _require_mapping(raw_value, "operations", issues)
    max_duration_seconds = _optional_int(
        data, "max_duration_seconds", issues, "operations", minimum=MIN_MAX_DURATION_SECONDS
    )
    freshness = None
    if "freshness" in data:
        raw_freshness = data["freshness"]
        if not isinstance(raw_freshness, Mapping):
            issues.append(ValidationIssue("operations.freshness", "must be a mapping"))
        else:
            max_age_seconds = _parse_max_age(raw_freshness.get("max_age"))
            if max_age_seconds is None:
                issues.append(
                    ValidationIssue(
                        "operations.freshness.max_age",
                        "must be a positive integer followed by one of: "
                        f"{', '.join(sorted(FRESHNESS_UNIT_SECONDS))} (for example '24h')",
                    )
                )
            else:
                freshness = FreshnessConfig(max_age_seconds=max_age_seconds)
    return OperationsConfig(max_duration_seconds=max_duration_seconds, freshness=freshness)


def _parse_max_age(raw: object) -> int | None:
    """Convert a positive integer and a unit from ``FRESHNESS_UNIT_SECONDS`` to seconds."""
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"([0-9]+)([a-z]+)", raw.strip())
    if match is None:
        return None
    unit_seconds = FRESHNESS_UNIT_SECONDS.get(match[2])
    if unit_seconds is None:
        return None
    try:
        count = int(match[1])
    except ValueError:
        # An integer exceeding Python's digit limit is still a collected parse issue.
        return None
    return count * unit_seconds if count > 0 else None
