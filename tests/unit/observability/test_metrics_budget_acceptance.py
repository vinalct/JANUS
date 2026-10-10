"""AC-5: the measured slowest real destination and a slow transport share one deadline."""

from __future__ import annotations

import shutil
import threading
import time

from janus.observability import IcebergAppendOutcome, IcebergAppendResult, build_run_event_emitter
from janus.observability.emission import DEFAULT_EMISSION_TIMEOUT_SECONDS, RunEmissionOutcome
from janus.observability.metrics.sink import MetricsRunSink
from janus.observability.metrics.transport import MetricsEmissionOutcome, MetricsEmissionResult
from tests.support import observability_baseline as baseline
from tests.support.metrics_transport_benchmark import measured_transports
from tests.unit.observability.test_acceptance_evidence import _authoritative_files, _emitter_config

SCHEDULING_TOLERANCE_SECONDS = 0.5


def _append(*args, **kwargs):
    return IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")


def _capture(root, emitter, monkeypatch, case="quality_failure"):
    fixed = baseline.FixedObserver

    class EmittingObserver(fixed):
        def __init__(self):
            super().__init__(emitter=emitter)

    monkeypatch.setattr(baseline, "FixedObserver", EmittingObserver)
    return baseline.capture_case(root, case)


def test_fanout_with_the_measured_slowest_transport_finishes_within_five_seconds(
    tmp_path, monkeypatch, record_property
):
    root = tmp_path / "project"
    with measured_transports(tmp_path / "benchmark") as measured:
        record_property("transport_timings_ms", str(measured.timings_ms))
        record_property("slowest_transport", measured.slowest)
        emitter = build_run_event_emitter(
            _emitter_config(),
            {
                "metadata_dir": root / "data/metadata",
                "iceberg_warehouse_dir": root / "data/bronze/iceberg",
            },
            runs_table_sink=_append,
            metrics_sink=measured.sinks[measured.slowest],
        )
        assert emitter.timeout_seconds == DEFAULT_EMISSION_TIMEOUT_SECONDS == 5.0
        started = time.monotonic()
        manifest = _capture(root, emitter, monkeypatch)
        elapsed = time.monotonic() - started
        result = emitter.last_result
        record_property("fanout_duration_seconds", result.duration_seconds)
        assert result.outcome is RunEmissionOutcome.EMITTED
        assert result.openlineage.emitted
        assert result.metrics.emitted
        assert result.metrics.transport == measured.slowest
        assert result.metrics.sample_count > 0
        assert result.duration_seconds < DEFAULT_EMISSION_TIMEOUT_SECONDS
        assert elapsed < DEFAULT_EMISSION_TIMEOUT_SECONDS
        assert manifest["status"] == "failed"
        assert _authoritative_files(root / "data/metadata")


def test_a_4_9_second_transport_cannot_extend_the_shared_five_second_deadline(
    tmp_path, monkeypatch, record_property
):
    root = tmp_path / "project"
    baseline.capture_case(root, "api_success")
    expected_files = _authoritative_files(root / "data/metadata")
    shutil.rmtree(root)
    completed = threading.Event()
    remaining_budgets = []

    class SleepingTransport:
        kind = "file"

        def send(self, samples, *, budget_seconds):
            assert samples
            remaining_budgets.append(budget_seconds)
            try:
                time.sleep(4.9)
                return MetricsEmissionResult(MetricsEmissionOutcome.EMITTED, self.kind)
            finally:
                completed.set()

    def delayed_append(*args, **kwargs):
        time.sleep(0.25)
        return _append(*args, **kwargs)

    emitter = build_run_event_emitter(
        _emitter_config(),
        {
            "metadata_dir": root / "data/metadata",
            "iceberg_warehouse_dir": root / "data/bronze/iceberg",
        },
        runs_table_sink=delayed_append,
        metrics_sink=MetricsRunSink(SleepingTransport()),
    )
    try:
        manifest = _capture(root, emitter, monkeypatch, "api_success")
        result = emitter.last_result
        record_property("sleeping_fanout_duration_seconds", result.duration_seconds)
        assert result.outcome is RunEmissionOutcome.FAILED
        assert result.stage == "budget"
        assert result.metrics.outcome is MetricsEmissionOutcome.FAILED
        assert result.metrics.step == "budget"
        assert result.metrics.exception_type == "EmissionTimeoutError"
        assert remaining_budgets and 0 < remaining_budgets[0] < 4.9
        assert not completed.is_set(), "the run must return before the slow transport finishes"
        assert result.duration_seconds < (
            DEFAULT_EMISSION_TIMEOUT_SECONDS + SCHEDULING_TOLERANCE_SECONDS
        )
        assert manifest["status"] == "succeeded"
        assert _authoritative_files(root / "data/metadata") == expected_files
    finally:
        assert completed.wait(2), "the bounded worker must finish after its deliberate sleep"
