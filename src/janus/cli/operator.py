"""Who changed operator state, and why: shared by every verb that mutates it.

The stores record the identity and reason they are handed and never read the environment.
Both are resolved here, at the edge, so `dead-letters` and `checkpoint` cannot answer "who
did this" in two different ways. The identity is deliberately narrow: an environment
variable or `unknown`, never an email address, a hostname or a system account lookup.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

from janus.planner import normalize_run_id_segment

OPERATOR_ENVIRONMENT_VARIABLES = ("JANUS_OPERATOR", "USER")
UNKNOWN_OPERATOR = "unknown"


def resolve_operator() -> str:
    """$JANUS_OPERATOR, else $USER, else "unknown" — nothing else.

    Deliberately not `getpass.getuser()`: it falls back to reading system account
    databases, and an operator record is an artifact that gets shared. A variable that is
    set but blank counts as unset.
    """
    for variable in OPERATOR_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    return UNKNOWN_OPERATOR


def manual_run_id(operator: str, *, now: datetime | None = None) -> str:
    """`manual-<YYYYMMDDTHHMMSSZ>-<operator slug>`, path-safe."""
    resolved_now = now or datetime.now(tz=UTC)
    if resolved_now.tzinfo is None or resolved_now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    timestamp = resolved_now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"manual-{timestamp}-{normalize_run_id_segment(operator)}"


def require_reason(parser: argparse.ArgumentParser, reason: str | None) -> str:
    """Every state mutation carries a reason; an empty one is an argument error."""
    normalized = (reason or "").strip()
    if not normalized:
        parser.error("--reason is required and must not be empty: every state change is recorded")
    return normalized
