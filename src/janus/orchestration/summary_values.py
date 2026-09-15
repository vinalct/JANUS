"""Safe immutable values used inside pipeline result records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from janus.orchestration.plans import MAX_FAILURE_REASON_LENGTH
from janus.utils.logging import REDACTED_VALUE, is_sensitive_field, sanitize_log_payload


@dataclass(frozen=True, slots=True)
class FailureDetails:
    """A bounded source or operational failure with no live exception attached."""

    phase: str
    error_type: str
    reason: str

    def __post_init__(self) -> None:
        if not self.phase.strip():
            raise ValueError("phase must not be empty")
        if not self.error_type.strip():
            raise ValueError("error_type must not be empty")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        object.__setattr__(self, "phase", self.phase.strip())
        object.__setattr__(self, "error_type", self.error_type.strip())
        object.__setattr__(self, "reason", _bounded_reason(self.reason))

    @classmethod
    def from_exception(cls, exc: BaseException, *, phase: str) -> Self:
        reason = str(exc).strip() or type(exc).__name__
        return cls(phase=phase, error_type=type(exc).__name__, reason=reason)

    def to_summary(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "error_type": self.error_type,
            "reason": self.reason,
        }


def freeze_evidence(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep-freeze and sanitize one established per-source result summary."""
    frozen = _freeze_json(evidence)
    if not isinstance(frozen, Mapping):
        raise TypeError("source evidence must be a JSON object")
    return frozen


def thaw_json(value: Any) -> Any:
    """Return ordinary dict/list values suitable for json.dumps."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _freeze_json(value: Any, *, field_name: str | None = None) -> Any:
    normalized_field = field_name.strip().lower().replace("_", "-") if field_name else None
    if normalized_field is not None and (
        is_sensitive_field(normalized_field) or _excluded_evidence_field(normalized_field)
    ):
        return REDACTED_VALUE
    if normalized_field == "failure-reason" and isinstance(value, str):
        return _bounded_reason(value)

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key in sorted(value, key=str):
            text_key = str(key)
            if text_key in frozen:
                raise ValueError(
                    f"JSON evidence keys collide after string conversion: {text_key!r}"
                )
            frozen[text_key] = _freeze_json(value[key], field_name=text_key)
        result: Any = MappingProxyType(frozen)
    elif isinstance(value, list | tuple):
        result = tuple(_freeze_json(item, field_name=field_name) for item in value)
    elif value is None or isinstance(value, str | bool | int):
        result = sanitize_log_payload(value, field_name=field_name)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON evidence numbers must be finite")
        result = value
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, datetime):
        _validate_aware(field_name or "datetime", value)
        result = value.isoformat()
    else:
        raise TypeError(f"source evidence contains non-JSON value {type(value).__name__}")
    return result


def _excluded_evidence_field(field_name: str) -> bool:
    if "credential" in field_name:
        return True
    if "traceback" in field_name or field_name in {"stack-trace", "stacktrace"}:
        return True
    return field_name == "body" or (
        field_name.endswith("-body")
        and any(marker in field_name for marker in ("http", "raw", "response"))
    )


def _bounded_reason(reason: str) -> str:
    sanitized = sanitize_log_payload(reason.strip())
    assert isinstance(sanitized, str)
    if len(sanitized) <= MAX_FAILURE_REASON_LENGTH:
        return sanitized
    return f"{sanitized[:MAX_FAILURE_REASON_LENGTH]}… (truncated)"


def _validate_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
