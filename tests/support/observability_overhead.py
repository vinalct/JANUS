"""NFR-3: measure one run and its observer share before or after guarded emission."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

from janus.observability import (
    IcebergAppendOutcome,
    IcebergAppendResult,
    build_run_event_emitter,
)
from tests.support import observability_baseline as baseline

REPEATS = 30
MEASURED_CASES = ("api_success", "quality_failure", "empty_handoff")
GUARDED_EMISSION = "--guarded-emission" in sys.argv[1:]


def _successful_sink(*args: object, **kwargs: object) -> IcebergAppendResult:
    del args, kwargs
    return IcebergAppendResult(
        IcebergAppendOutcome.EMITTED,
        "metadata.runs",
    )


class TimedObserver(baseline.FixedObserver):
    """The captured observer, timed in place: it does exactly what it did before."""

    calls: ClassVar[list[tuple[str, int]]] = []

    def __init__(self) -> None:
        if GUARDED_EMISSION:
            super().__init__(
                emitter=build_run_event_emitter({}, {}, runs_table_sink=_successful_sink)
            )
        else:
            super().__init__()

    def start_run(self, *args, **kwargs):
        started = time.perf_counter_ns()
        result = super().start_run(*args, **kwargs)
        TimedObserver.calls.append(("start_run", time.perf_counter_ns() - started))
        return result

    def record_success(self, *args, **kwargs):
        started = time.perf_counter_ns()
        result = super().record_success(*args, **kwargs)
        TimedObserver.calls.append(("record_success", time.perf_counter_ns() - started))
        return result

    def record_failure(self, *args, **kwargs):
        started = time.perf_counter_ns()
        result = super().record_failure(*args, **kwargs)
        TimedObserver.calls.append(("record_failure", time.perf_counter_ns() - started))
        return result


def _milliseconds(values: list[int]) -> list[float]:
    return [value / 1_000_000 for value in values]


def _report(label: str, values: list[float]) -> None:
    ordered = sorted(values)
    print(
        f"  {label:<34} n={len(ordered):<3} min={min(ordered):8.3f} "
        f"median={statistics.median(ordered):8.3f} mean={statistics.fmean(ordered):8.3f} "
        f"p95={ordered[int(len(ordered) * 0.95) - 1]:8.3f} max={max(ordered):8.3f} "
        f"stdev={statistics.stdev(ordered):7.3f}  (ms)"
    )


def measure(case: str, repeats: int = REPEATS) -> None:
    end_to_end: list[int] = []
    per_method: dict[str, list[int]] = {}
    for _ in range(repeats):
        TimedObserver.calls = []
        with TemporaryDirectory(prefix=f"overhead-{case}-") as temporary:
            root = Path(temporary).resolve()
            original = baseline.FixedObserver
            baseline.FixedObserver = TimedObserver
            try:
                started = time.perf_counter_ns()
                baseline.capture_case(root, case)
                end_to_end.append(time.perf_counter_ns() - started)
            finally:
                baseline.FixedObserver = original
        for name, duration in TimedObserver.calls:
            per_method.setdefault(name, []).append(duration)

    print(f"\ncase: {case}")
    _report("run end to end", _milliseconds(end_to_end))
    observer_total = [0] * len(end_to_end)
    for name, durations in sorted(per_method.items()):
        _report(f"observer.{name}", _milliseconds(durations))
        for index, duration in enumerate(durations[: len(observer_total)]):
            observer_total[index] += duration
    _report("observer total (all calls)", _milliseconds(observer_total))
    share = [total / run * 100 for total, run in zip(observer_total, end_to_end, strict=True)]
    print(
        f"  observer share of end-to-end       median={statistics.median(share):.2f}%  "
        f"max={max(share):.2f}%"
    )


def main() -> None:
    print(f"python {sys.version.split()[0]}  repeats={REPEATS}  timer=time.perf_counter_ns")
    for case in MEASURED_CASES:
        measure(case)


if __name__ == "__main__":
    main()
