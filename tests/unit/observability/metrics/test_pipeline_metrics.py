"""Batch totals produce four bounded aggregate series through every shared transport."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from jsonschema import ValidationError

from janus.observability.metrics.catalogue import MetricType
from janus.observability.metrics.exposition import parse_exposition
from janus.observability.metrics.pipeline import emit_pipeline_metrics, pipeline_samples
from janus.observability.metrics.transport import MetricsEmissionOutcome
from janus.orchestration import SourceOutcome
from tests.support.metrics_order23 import WarningSpy, sink
from tests.unit.observability.metrics.test_transports import _udp_receiver, _validator
from tests.unit.orchestration.test_pipeline_results_and_summary import (
    ENDED_AT,
    _attempt,
    _outcome,
    _plan,
    _successful_outcomes,
)


def _mixed_outcome(tmp_path):
    plan = _plan(tmp_path, "chain_and_peer")
    sources = []
    for source in plan.sources:
        if source.source_id in {"B", "D"}:
            sources.append(
                SourceOutcome.skipped(
                    source,
                    direct_blocking_upstream_ids=source.upstream_ids,
                    root_failed_source_ids=("A",),
                )
            )
        else:
            sources.append(
                SourceOutcome.from_attempts(
                    source,
                    (
                        _attempt(
                            source, status="failed" if source.source_id == "A" else "succeeded"
                        ),
                    ),
                )
            )
    return _outcome(plan, tuple(sources))


def test_mixed_batch_projects_exactly_four_gauges_from_the_existing_totals(tmp_path, monkeypatch):
    outcome = _mixed_outcome(tmp_path)
    monkeypatch.setattr(
        "janus.observability.metrics.pipeline.time.monotonic",
        lambda: pytest.fail("the projection read a clock"),
    )
    samples = pipeline_samples(outcome)
    assert len(samples) == 4
    assert [(sample.name, sample.value) for sample in samples] == [
        ("janus_pipeline_duration_seconds", 5.0),
        ("janus_pipeline_sources_total", 1),
        ("janus_pipeline_sources_total", 1),
        ("janus_pipeline_sources_total", 2),
    ]
    assert all(sample.metric_type is MetricType.GAUGE for sample in samples)
    assert all(sample.scope == "pipeline" and sample.emitted_at == ENDED_AT for sample in samples)
    assert dict(samples[0].labels) == {"environment": "local", "trigger": "run-all"}
    assert [dict(sample.labels) for sample in samples[1:]] == [
        {"environment": "local", "trigger": "run-all", "status": status}
        for status in ("succeeded", "failed", "skipped")
    ]


def test_a_finished_success_emits_measured_zero_for_both_failed_and_skipped(tmp_path):
    plan = _plan(tmp_path, "chain")
    outcome = _outcome(plan, _successful_outcomes(plan))
    samples = pipeline_samples(outcome)
    assert len(samples) == 4
    assert [sample.value for sample in samples[1:]] == [2, 0, 0]


def test_a_batch_where_every_source_failed_still_emits_four_samples(tmp_path):
    plan = _plan(tmp_path, "independent")
    outcomes = tuple(
        SourceOutcome.from_attempts(source, (_attempt(source, status="failed"),))
        for source in plan.sources
    )
    outcome = _outcome(plan, outcomes)
    samples = pipeline_samples(outcome)
    assert outcome.status == "failed"
    assert len(samples) == 4
    assert [sample.value for sample in samples[1:]] == [0, 2, 0]


def test_skipped_nodes_have_no_per_source_metric_or_fabricated_attempt(tmp_path):
    outcome = _mixed_outcome(tmp_path)
    skipped = [source for source in outcome.sources if source.status == "skipped"]
    assert skipped
    assert all(not source.attempts and not source.timing.executed for source in skipped)
    samples = pipeline_samples(outcome)
    assert len(samples) == 4
    assert all(sample.name.startswith("janus_pipeline_") for sample in samples)
    assert all(
        set(dict(sample.labels)) <= {"environment", "trigger", "status"} for sample in samples
    )
    assert not any(
        {"source_id", "pipeline_run_id", "run_id"} & set(dict(sample.labels)) for sample in samples
    )


def test_file_transport_writes_one_schema_valid_pipeline_batch_with_status_labels(tmp_path):
    samples = pipeline_samples(_mixed_outcome(tmp_path))
    destination = sink(tmp_path, "file")
    result = destination.emit_samples(samples, budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.EMITTED
    assert result.sample_count == 4
    lines = destination.transport.path_for(samples).read_text().splitlines()
    assert len(lines) == 1
    batch = json.loads(lines[0])
    _validator().validate(batch)
    assert batch["scope"] == "pipeline"
    assert batch["emitted_at"] == ENDED_AT.isoformat()
    assert batch["labels"] == {"environment": "local", "trigger": "run-all"}
    assert "labels" not in batch["samples"][0]
    assert [(sample["labels"], sample["value"]) for sample in batch["samples"][1:]] == [
        ({"status": "succeeded"}, 1),
        ({"status": "failed"}, 1),
        ({"status": "skipped"}, 2),
    ]


@pytest.mark.parametrize("violation", ["source_id", "pipeline_run_id", "status", "run-label"])
def test_pipeline_ndjson_schema_rejects_unbounded_or_inconsistent_labels(tmp_path, violation):
    destination = sink(tmp_path, "file")
    samples = pipeline_samples(_mixed_outcome(tmp_path))
    destination.emit_samples(samples, budget_seconds=1)
    batch = json.loads(destination.transport.path_for(samples).read_text())
    if violation in {"source_id", "pipeline_run_id"}:
        batch["labels"][violation] = "unbounded"
    elif violation == "status":
        batch["samples"][1]["labels"]["status"] = "pending"
    else:
        batch["scope"] = "run"
    with pytest.raises(ValidationError):
        _validator().validate(batch)


def test_pipeline_textfile_replaces_the_same_aggregate_and_keeps_zero_series(tmp_path):
    mixed = _mixed_outcome(tmp_path)
    destination = sink(tmp_path, "textfile")
    result = destination.emit_samples(pipeline_samples(mixed), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.EMITTED
    assert result.sample_count == 4
    path = destination.transport.directory / "pipeline-local-run_all.prom"
    assert len(parse_exposition(path.read_text())) == 4
    success_plan = _plan(tmp_path, "chain")
    success = _outcome(success_plan, _successful_outcomes(success_plan))
    result = destination.emit_samples(pipeline_samples(success), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.EMITTED
    rendered = parse_exposition(path.read_text())
    assert len(rendered) == 4
    assert {
        dict(sample.labels)["status"]: sample.value
        for sample in rendered
        if sample.name == "janus_pipeline_sources_total"
    } == {"succeeded": 2, "failed": 0, "skipped": 0}
    assert list(destination.transport.directory.iterdir()) == [path]


def test_pipeline_statsd_delivers_the_same_four_samples_in_one_datagram(tmp_path):
    samples = pipeline_samples(_mixed_outcome(tmp_path))
    with _udp_receiver() as (address, received):
        destination = sink(tmp_path, "statsd", host=address[0], port=address[1])
        result = destination.emit_samples(samples, budget_seconds=1)
        payload = received.get(timeout=1).decode()
        assert received.empty()
    assert result.outcome is MetricsEmissionOutcome.EMITTED
    assert result.sample_count == 4
    assert payload.splitlines() == [
        "janus.pipeline_duration_seconds:5.0|g|#environment:local,trigger:run-all",
        *[
            f"janus.pipeline_sources_total:{value}|g|#environment:local,status:{status},trigger:run-all"
            for status, value in (("succeeded", 1), ("failed", 1), ("skipped", 2))
        ],
    ]


@pytest.mark.parametrize("kind", ["file", "textfile"])
@pytest.mark.parametrize("violation", ["source_id", "environment", "status", "scope"])
def test_pipeline_transports_reject_mixed_aggregates_and_invalid_labels(tmp_path, kind, violation):
    samples = pipeline_samples(_mixed_outcome(tmp_path))
    first = samples[1]
    if violation == "scope":
        first = replace(first, declaration=replace(first.declaration, scope="run"))
    else:
        labels = dict(first.labels)
        labels[violation] = "invalid"
        first = replace(first, labels=tuple(labels.items()))
    result = sink(tmp_path, kind).emit_samples((samples[0], first, *samples[2:]), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED


def test_disabled_pipeline_metrics_are_silent_and_create_no_files(tmp_path):
    outcome = _mixed_outcome(tmp_path)
    before = set(tmp_path.rglob("*"))
    logger = WarningSpy()
    emit_pipeline_metrics(outcome, {}, {"metadata_dir": tmp_path}, logger=logger)
    assert set(tmp_path.rglob("*")) == before
    assert logger.warnings == []


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(130)])
def test_pipeline_emission_bridges_process_interruptions_from_the_shared_worker(
    tmp_path, monkeypatch, error
):
    outcome = _mixed_outcome(tmp_path)

    def interrupted_builder(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(
        "janus.observability.metrics.pipeline.build_metrics_sink", interrupted_builder
    )
    logger = WarningSpy()
    with pytest.raises(type(error)) as caught:
        emit_pipeline_metrics(outcome, {}, {"metadata_dir": tmp_path}, logger=logger)
    assert caught.value is error
    assert logger.warnings == []
