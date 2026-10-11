"""One daemon-worker deadline shared by run and pipeline telemetry."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

DEFAULT_EMISSION_TIMEOUT_SECONDS = 5.0


class EmissionGuardError(RuntimeError):
    """Safe stage and exception type for a guard failure, without exception text."""

    def __init__(self, stage: str, exception_type: str) -> None:
        self.stage = stage
        self.exception_type = exception_type
        super().__init__(exception_type)


def bounded_emission[T](
    worker: Callable[[], T],
    *,
    timeout_seconds: float = DEFAULT_EMISSION_TIMEOUT_SECONDS,
    thread_name: str = "janus-run-event-emission",
) -> T:
    """Bound one operation while bridging KeyboardInterrupt and SystemExit."""
    if not _valid_timeout(timeout_seconds):
        raise EmissionGuardError("budget", "InvalidEmissionTimeout")
    results: list[T] = []
    fatal_errors: list[BaseException] = []

    def bridge() -> None:
        try:
            results.append(worker())
        except BaseException as exc:
            fatal_errors.append(exc)

    try:
        thread = threading.Thread(target=bridge, name=thread_name, daemon=True)
        thread.start()
        thread.join(timeout_seconds)
    except Exception as exc:
        raise EmissionGuardError("worker", type(exc).__name__) from exc
    if thread.is_alive():
        raise EmissionGuardError("budget", "EmissionTimeoutError")
    if fatal_errors:
        raise fatal_errors[0]
    if results:
        return results[0]
    raise EmissionGuardError("worker", "WorkerExitedWithoutResult")


def _valid_timeout(value: float) -> bool:
    try:
        return math.isfinite(value) and value > 0
    except Exception:
        return False
