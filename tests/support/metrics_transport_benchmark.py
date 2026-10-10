"""Measure the same realistic batch through each real metrics transport, using loopback only."""

from __future__ import annotations

import json
import socket
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from janus.observability.metrics.sink import MetricsRunSink
from janus.observability.metrics.transport import (
    MetricSample,
    MetricsEmissionOutcome,
    MetricsEmissionResult,
)
from janus.observability.records import RunRecord
from tests.support.metrics_order23 import record, sink

REPEATS = 30


@dataclass(frozen=True)
class TransportBenchmark:
    timings_ms: dict[str, dict[str, float]]
    sinks: dict[str, MetricsRunSink]
    sample_count: int
    received_datagrams: int

    @property
    def slowest(self) -> str:
        return max(self.timings_ms, key=lambda kind: self.timings_ms[kind]["median"])


class _CaptureTransport:
    kind = "file"

    def __init__(self) -> None:
        self.samples: tuple[MetricSample, ...] = ()

    def send(self, samples, *, budget_seconds):
        assert budget_seconds > 0
        self.samples = tuple(samples)
        return MetricsEmissionResult(MetricsEmissionOutcome.EMITTED, self.kind)


def _samples(run: RunRecord) -> tuple[MetricSample, ...]:
    capture = _CaptureTransport()
    assert MetricsRunSink(capture).emit(run, budget_seconds=5).emitted
    assert capture.samples, "the transport benchmark cannot measure an empty batch"
    return capture.samples


def _statistics(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": sorted(values)[int(len(values) * 0.95) - 1],
        "max": max(values),
        "stdev": statistics.stdev(values),
    }


@contextmanager
def measured_transports(root: Path, *, repeats: int = REPEATS) -> Iterator[TransportBenchmark]:
    """Seed textfile state, measure delivery, and keep the slowest sink usable by fan-out.

    A successful seed followed by failed batches makes textfile read and parse its
    existing last-success state. DNS and loopback receive time are outside the timer.
    Every batch uses identical catalogue samples; no result or outlier is discarded.
    """
    assert repeats >= 2
    run = record(
        root,
        source_id="baseline_api",
        trigger="baseline",
        status="failed",
        failure_reason="quality_gate_failed",
        failure_reason_truncated=False,
        failure_reason_length=len("quality_gate_failed"),
        quality_outcome="failed",
        quality_checks_passed=2,
        quality_checks_skipped=0,
        quality_failed_checks=("required_fields",),
        request_count=24,
        retry_count=2,
        attempt_count=26,
        dead_letter_count=1,
        quality_checks_failed=1,
        duration_budget_seconds=4,
        duration_budget_outcome="exceeded",
        freshness_max_age_seconds=86400,
    )
    values = _samples(run)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        destinations = {
            "file": sink(root / "file", "file"),
            "textfile": sink(root / "textfile", "textfile"),
            "statsd": sink(root, "statsd", host="127.0.0.1", port=receiver.getsockname()[1]),
        }
        seed = _samples(
            replace(
                run,
                status="succeeded",
                failure_reason=None,
                failure_reason_truncated=None,
                failure_reason_length=None,
                quality_outcome="passed",
                quality_checks_failed=0,
                quality_failed_checks=(),
            )
        )
        assert destinations["textfile"].transport.send(seed, budget_seconds=5).emitted
        received = 0
        timings: dict[str, list[float]] = {kind: [] for kind in destinations}
        # Interleave destinations across repeats to reduce ordering/load bias.
        for _ in range(repeats):
            for kind, destination in destinations.items():
                started = time.perf_counter_ns()
                result = destination.transport.send(values, budget_seconds=5)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                assert result.emitted, result.to_summary()
                assert result.sample_count >= len(values)
                timings[kind].append(elapsed_ms)
                if kind == "statsd":
                    datagrams = int(result.reason.split(":")[1]) if result.reason else 1
                    lines = []
                    for _ in range(datagrams):
                        packet, _address = receiver.recvfrom(65535)
                        assert packet, "the benchmark must deliver actual datagrams"
                        lines.extend(packet.splitlines())
                        received += 1
                    assert len(lines) == len(values)
        yield TransportBenchmark(
            {kind: _statistics(values) for kind, values in timings.items()},
            destinations,
            len(values),
            received,
        )


def main() -> None:
    with (
        TemporaryDirectory(prefix="metrics-transports-") as temporary,
        measured_transports(Path(temporary)) as measured,
    ):
        print(
            json.dumps(
                {
                    "repeats": REPEATS,
                    "samples_per_batch": measured.sample_count,
                    "received_datagrams": measured.received_datagrams,
                    "timings_ms": measured.timings_ms,
                    "slowest_by_median": measured.slowest,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
