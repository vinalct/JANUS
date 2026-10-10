"""Catalogue projection and per-run metrics destination selection."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest

import janus.observability.metrics.sink as sink_module
from janus.observability import RunEmissionOutcome, RunEmissionResult
from janus.observability.metrics import (
    DisabledMetricsTransport,
    FileMetricsTransport,
    MetricsEmissionOutcome,
    MetricsEmissionResult,
    MetricsRunSink,
    StatsdMetricsTransport,
    TextfileMetricsTransport,
    build_metrics_sink,
    disabled_metrics_sink,
)
from janus.observability.metrics.catalogue import ALLOWED_LABELS, run_metrics
from janus.observability.openlineage.transport import (
    OpenLineageEmissionOutcome,
    OpenLineageEmissionResult,
)
from tests.support.metrics_order23 import NOW, WarningSpy, record


def _capture(projected, *, emission_result=None):
    transport = Mock(kind="file")
    transport.send.return_value = MetricsEmissionResult(MetricsEmissionOutcome.EMITTED, "file")
    destination = MetricsRunSink(transport)
    result = destination.emit(projected, budget_seconds=0.125, emission_result=emission_result)
    transport.send.assert_called_once()
    assert transport.send.call_args.kwargs == {"budget_seconds": 0.125}
    samples = transport.send.call_args.args[0]
    assert isinstance(samples, tuple) and samples
    assert result.emitted
    return samples


def test_sink_maps_all_run_declarations_and_reads_labels_once(tmp_path, monkeypatch):
    projected = record(
        tmp_path,
        duration_budget_outcome="exceeded",
        duration_budget_seconds=4,
        request_count=4,
        retry_count=2,
        dead_letter_count=1,
        quality_checks_failed=3,
        quality_outcome="failed",
        quality_failed_checks=("schema", "nulls", "duplicates"),
        trigger="scheduled",
    )
    labels = Mock(wraps=sink_module.label_values)
    monkeypatch.setattr(sink_module, "label_values", labels)
    context = RunEmissionResult("succeeded", RunEmissionOutcome.EMITTED, "metadata.runs")
    samples = _capture(projected, emission_result=context)
    labels.assert_called_once_with(projected)
    assert [sample.declaration for sample in samples] == list(run_metrics())
    assert {sample.name: sample.value for sample in samples} == {
        "janus_run_duration_seconds": 5.0,
        "janus_run_total": 1,
        "janus_records_extracted": 3,
        "janus_records_written": 3,
        "janus_http_requests_total": 4,
        "janus_http_retries_total": 2,
        "janus_dead_letters_total": 1,
        "janus_quality_checks_failed": 3,
        "janus_duration_budget_exceeded_total": 1,
        "janus_last_run_timestamp_seconds": NOW.timestamp(),
        "janus_last_success_timestamp_seconds": NOW.timestamp(),
        "janus_emission_failures_total": 0,
    }
    assert all(sample.emitted_at == projected.emitted_at for sample in samples)
    expected_labels = {label: getattr(projected, label) for label in ALLOWED_LABELS}
    assert all(dict(sample.labels) == expected_labels for sample in samples)
    assert all(set(dict(sample.labels)) == ALLOWED_LABELS for sample in samples)
    assert all(dict(sample.labels)["source_id"] == projected.source_id for sample in samples)


@pytest.mark.parametrize("status", ["succeeded", "failed"])
@pytest.mark.parametrize("budget_outcome", [None, "undeclared", "within", "exceeded"])
def test_optional_and_conditional_measurements_remain_distinct_from_zero(
    status, budget_outcome, tmp_path
):
    samples = _capture(
        record(
            tmp_path,
            status=status,
            failure_reason="run failed" if status == "failed" else None,
            failure_reason_truncated=False if status == "failed" else None,
            failure_reason_length=10 if status == "failed" else None,
            records_extracted=None,
            records_written=0,
            request_count=None,
            retry_count=0,
            duration_budget_outcome=budget_outcome,
            duration_budget_seconds=5 if budget_outcome in ("within", "exceeded") else None,
            trigger="  ",
        )
    )
    values = {sample.name: sample.value for sample in samples}
    assert "janus_records_extracted" not in values
    assert "janus_http_requests_total" not in values
    assert "janus_dead_letters_total" not in values
    assert "janus_emission_failures_total" not in values, "no destination outcome was measured"
    assert values["janus_records_written"] == values["janus_http_retries_total"] == 0
    assert ("janus_last_success_timestamp_seconds" in values) == (status == "succeeded")
    if budget_outcome in (None, "undeclared"):
        assert "janus_duration_budget_exceeded_total" not in values
    else:
        assert values["janus_duration_budget_exceeded_total"] == (budget_outcome == "exceeded")
    assert all("trigger" not in dict(sample.labels) for sample in samples)
    assert all(sample.value is not None for sample in samples)


@pytest.mark.parametrize("row_outcome", list(RunEmissionOutcome))
@pytest.mark.parametrize("lineage_outcome", [None, *OpenLineageEmissionOutcome])
def test_emission_failure_counter_uses_only_preceding_destinations(
    row_outcome, lineage_outcome, tmp_path
):
    context = RunEmissionResult(
        "failed",
        row_outcome,
        "metadata.runs",
        openlineage=(
            OpenLineageEmissionResult(lineage_outcome, "file") if lineage_outcome else None
        ),
        metrics=MetricsEmissionResult(MetricsEmissionOutcome.FAILED, "statsd"),
    )
    samples = _capture(record(tmp_path), emission_result=context)
    failures = next(sample for sample in samples if sample.name == "janus_emission_failures_total")
    assert failures.value == (row_outcome == "failed") + (lineage_outcome == "failed")


@pytest.mark.parametrize(
    ("kind", "transport_type"),
    [
        ("disabled", DisabledMetricsTransport),
        ("file", FileMetricsTransport),
        ("textfile", TextfileMetricsTransport),
        ("statsd", StatsdMetricsTransport),
    ],
)
def test_builder_selects_each_transport_without_creating_files(kind, transport_type, tmp_path):
    settings = {"transport": kind}
    if kind == "statsd":
        settings["host"] = "127.0.0.1"
    destination = build_metrics_sink(
        {"observability": {"metrics": settings}}, {"metadata_dir": tmp_path}
    )
    assert isinstance(destination.transport, transport_type)
    assert destination.transport.kind == kind
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FrozenInstanceError):
        destination.transport = DisabledMetricsTransport()


@pytest.mark.parametrize(
    ("settings", "paths"),
    [
        ({"transport": "typo"}, {}),
        ({"transport": "file"}, {}),
        ({"transport": "file", "path": "../outside"}, None),
        ({"transport": "textfile", "path": "/tmp/outside"}, None),
    ],
)
def test_invalid_profile_degrades_once_and_disabled_emits_without_mapping(
    settings, paths, tmp_path
):
    warnings = WarningSpy()
    destination = build_metrics_sink(
        {"observability": {"metrics": settings}},
        {"metadata_dir": tmp_path} if paths is None else paths,
        logger=warnings,
    )
    assert isinstance(destination.transport, DisabledMetricsTransport)
    for _ in range(2):
        result = destination.emit(None, budget_seconds=1, logger=warnings)
        assert result.outcome is MetricsEmissionOutcome.SKIPPED
        assert result.reason == "profile_invalid"
    assert len(warnings.warnings) == 1
    assert warnings.warnings[0][0] == "metrics_profile_invalid"
    assert list(tmp_path.iterdir()) == []


def test_disabled_sink_is_a_real_silent_default():
    destination = disabled_metrics_sink()
    assert isinstance(destination.transport, DisabledMetricsTransport)
    result = destination.emit(None, budget_seconds=0.125)
    assert result.outcome is MetricsEmissionOutcome.SKIPPED
    assert result.reason == "transport_not_configured"


def test_sink_threads_budget_and_leaves_defective_transport_guarding_to_caller(tmp_path):
    transport = Mock(kind="file")
    transport.send.side_effect = OSError("receiver failed")
    with pytest.raises(OSError):
        MetricsRunSink(transport).emit(record(tmp_path), budget_seconds=0.125)
    assert transport.send.call_args.kwargs == {"budget_seconds": 0.125}


def test_statsd_failure_is_reported_once_by_transport(tmp_path, monkeypatch):
    import socket

    warnings = WarningSpy()
    destination = build_metrics_sink(
        {"observability": {"metrics": {"transport": "statsd", "host": "127.0.0.1"}}},
        {},
        logger=warnings,
    )
    monkeypatch.setattr(socket, "socket", Mock(side_effect=OSError("secret-token")))
    result = destination.emit(record(tmp_path), budget_seconds=1, logger=warnings)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert len(warnings.warnings) == 1
    assert "secret-token" not in repr(warnings.warnings)


def test_unresolvable_statsd_address_warns_once_at_selection(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", Mock(side_effect=socket.gaierror("secret-host")))
    warnings = WarningSpy()
    destination = build_metrics_sink(
        {"observability": {"metrics": {"transport": "statsd", "host": "private.invalid"}}},
        {},
        logger=warnings,
    )
    assert isinstance(destination.transport, DisabledMetricsTransport)
    destination.emit(record(tmp_path), budget_seconds=1, logger=warnings)
    assert len(warnings.warnings) == 1
    assert "secret-host" not in repr(warnings.warnings)
