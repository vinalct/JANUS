"""Thread-safe request pacing shared by the api, catalog, and file strategies.

One throttle for every HTTP-shaped family. It carries the ``Lock`` that the API
strategy relies on for concurrent pagination; the catalog and file paths gain the
same thread-safety for free, closing the latent rate-limit trap that their
lock-less copies used to hide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class HttpRequestThrottle:
    """Thread-safe request pacing aligned with JANUS safe defaults."""

    requests_per_minute: int | None
    clock: Callable[[], float]
    sleeper: Callable[[float], None]
    _next_allowed_at: float | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def wait_for_turn(self) -> None:
        if self.requests_per_minute is None:
            return

        interval_seconds = 60 / self.requests_per_minute
        with self._lock:
            now = self.clock()
            scheduled_at = now
            if self._next_allowed_at is not None and now < self._next_allowed_at:
                scheduled_at = self._next_allowed_at
            self._next_allowed_at = scheduled_at + interval_seconds

        delay = scheduled_at - now
        if delay > 0:
            self.sleeper(delay)
