"""Clock injection and elapsed-time records shared by batch entry points."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True, slots=True)
class ClockSample:
    """The two clocks sampled at the beginning of an operation."""

    wall_time: datetime
    monotonic_time: float

    def __post_init__(self) -> None:
        _validate_aware("wall_time", self.wall_time)
        if not math.isfinite(self.monotonic_time):
            raise ValueError("monotonic_time must be finite")


@dataclass(frozen=True, slots=True)
class ExecutionTiming:
    """Wall-clock evidence plus a monotonic, non-negative elapsed duration."""

    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float

    def __post_init__(self) -> None:
        if (self.started_at is None) != (self.ended_at is None):
            raise ValueError("started_at and ended_at must either both be set or both be absent")
        if self.started_at is not None:
            _validate_aware("started_at", self.started_at)
            assert self.ended_at is not None
            _validate_aware("ended_at", self.ended_at)
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.started_at is None and self.duration_seconds != 0:
            raise ValueError("an operation that did not execute must have zero duration")

    @classmethod
    def not_executed(cls) -> ExecutionTiming:
        """Return the only valid timing for a skipped or planning-failed source."""
        return cls(started_at=None, ended_at=None, duration_seconds=0.0)

    @property
    def executed(self) -> bool:
        return self.started_at is not None

    def to_summary(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at is not None else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class PipelineClock:
    """Inject wall time for evidence and monotonic time for elapsed durations."""

    wall_clock: Callable[[], datetime] = _utc_now
    monotonic_clock: Callable[[], float] = time.monotonic

    def start(self) -> ClockSample:
        return ClockSample(
            wall_time=self.wall_clock(),
            monotonic_time=self.monotonic_clock(),
        )

    def finish(self, started: ClockSample) -> ExecutionTiming:
        """Finish a sample without deriving elapsed time from a shifting wall clock."""
        ended_at = self.wall_clock()
        ended_monotonic = self.monotonic_clock()
        if not math.isfinite(ended_monotonic):
            raise ValueError("monotonic clock returned a non-finite value")
        elapsed = max(0.0, ended_monotonic - started.monotonic_time)
        return ExecutionTiming(
            started_at=started.wall_time,
            ended_at=ended_at,
            duration_seconds=round(elapsed, 6),
        )


def _validate_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
