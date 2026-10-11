import ast
import inspect
import re
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from janus.observability import RunRecord
from janus.observability.metrics import catalogue
from janus.observability.metrics.catalogue import (
    ALLOWED_LABELS,
    DELIBERATELY_UNMAPPED,
    MetricType,
    label_values,
    pipeline_metrics,
    run_metrics,
)
from tests.support.metrics_order23 import record
from tests.unit.runtime.test_batch_executor import _run

LABELS = {"source_id", "strategy_family", "strategy_variant", "environment", "status", "trigger"}


def test_every_declaration_resolves_to_a_real_record_or_pipeline_summary_field(tmp_path):
    run_fields = {field.name for field in fields(RunRecord)}
    pipeline, _, _ = _run(tmp_path)
    summary_fields = set(pipeline.totals())
    run_entries, pipeline_entries = run_metrics(), pipeline_metrics()
    assert run_fields and summary_fields
    assert run_entries and pipeline_entries

    derived = {"janus_run_total", "janus_emission_failures_total"}
    mapped_entries = [entry for entry in run_entries if entry.name not in derived]
    assert mapped_entries
    assert {entry.source_field for entry in mapped_entries} <= run_fields
    assert {entry.name for entry in run_entries} >= derived
    assert {entry.source_field for entry in pipeline_entries} <= summary_fields
    assert all(entry.scope == "run" for entry in run_entries)
    assert all(entry.scope == "pipeline" for entry in pipeline_entries)
    assert all(set(entry.source_fields) <= summary_fields for entry in pipeline_entries)
    assert {entry.name: entry.source_field for entry in run_entries if entry.name in derived} == {
        "janus_run_total": "record",
        "janus_emission_failures_total": "emission_result",
    }


def test_every_numeric_or_timestamp_field_is_mapped_or_has_a_written_exclusion(tmp_path):
    hints = get_type_hints(RunRecord)
    numeric = {
        field.name
        for field in fields(RunRecord)
        if set(get_args(hints[field.name]) or (hints[field.name],)) & {int, float, datetime}
    }
    pipeline, _, _ = _run(tmp_path)
    summary_numeric = {
        name for name, value in pipeline.totals().items() if isinstance(value, int | float)
    }
    numeric |= summary_numeric
    mapped = {entry.source_field for entry in (*run_metrics(), *pipeline_metrics())}
    mapped |= {name for entry in pipeline_metrics() for name in entry.source_fields}
    assert summary_numeric, "numeric summary field discovery must not be empty"
    excluded = set(DELIBERATELY_UNMAPPED)
    assert numeric, "numeric field discovery must not be empty"
    assert mapped, "a catalogue mapping nothing must not satisfy drift coverage"
    assert excluded, "intentional exclusions need their own non-empty register"
    assert numeric <= mapped | excluded
    assert excluded <= numeric, "stale exclusions must fail in the reverse direction"
    assert not mapped & excluded
    assert all(
        isinstance(reason, str) and reason.strip() for reason in DELIBERATELY_UNMAPPED.values()
    )


def test_label_union_is_exactly_the_six_bounded_labels():
    entries = (*run_metrics(), *pipeline_metrics())
    assert entries
    declared = {label for entry in entries for label in entry.labels}
    assert declared
    assert declared == set(ALLOWED_LABELS) == LABELS
    assert all(set(entry.labels) <= ALLOWED_LABELS for entry in entries)


def test_catalogue_detector_rejects_a_deliberate_run_id_label():
    entries = run_metrics()
    assert entries
    with pytest.raises(ValueError, match="run_id"):
        replace(entries[0], labels=("source_id", "run_id"))


def test_catalogue_detector_accepts_a_clean_declaration():
    entries = run_metrics()
    assert entries
    clean = replace(entries[0], labels=("source_id", "status"))
    assert clean.labels == ("source_id", "status")
    assert clean.name == entries[0].name


def test_catalogue_contains_all_fourteen_unique_named_metrics():
    entries = (*run_metrics(), *pipeline_metrics())
    assert entries
    assert len(entries) == len({entry.name for entry in entries}) == 14
    assert {entry.name for entry in entries} == {
        "janus_run_duration_seconds",
        "janus_run_total",
        "janus_records_extracted",
        "janus_records_written",
        "janus_http_requests_total",
        "janus_http_retries_total",
        "janus_dead_letters_total",
        "janus_quality_checks_failed",
        "janus_duration_budget_exceeded_total",
        "janus_last_run_timestamp_seconds",
        "janus_last_success_timestamp_seconds",
        "janus_emission_failures_total",
        "janus_pipeline_duration_seconds",
        "janus_pipeline_sources_total",
    }
    assert len(run_metrics()) == 12
    assert len(pipeline_metrics()) == 2
    assert all(re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", entry.name) for entry in entries)
    assert all(entry.help_text.strip() for entry in entries)


@pytest.mark.parametrize(
    ("metric_name", "source_field", "metric_type"),
    [
        ("janus_run_duration_seconds", "duration_seconds", MetricType.GAUGE),
        ("janus_run_total", "record", MetricType.COUNTER),
        ("janus_records_extracted", "records_extracted", MetricType.GAUGE),
        ("janus_records_written", "records_written", MetricType.GAUGE),
        ("janus_http_requests_total", "request_count", MetricType.COUNTER),
        ("janus_http_retries_total", "retry_count", MetricType.COUNTER),
        ("janus_dead_letters_total", "dead_letter_count", MetricType.COUNTER),
        ("janus_quality_checks_failed", "quality_checks_failed", MetricType.GAUGE),
        ("janus_duration_budget_exceeded_total", "duration_budget_outcome", MetricType.COUNTER),
        ("janus_last_run_timestamp_seconds", "ended_at", MetricType.GAUGE),
        ("janus_last_success_timestamp_seconds", "ended_at", MetricType.GAUGE),
        ("janus_emission_failures_total", "emission_result", MetricType.COUNTER),
        ("janus_pipeline_duration_seconds", "duration_seconds", MetricType.GAUGE),
        ("janus_pipeline_sources_total", "succeeded", MetricType.GAUGE),
    ],
)
def test_named_metrics_keep_their_required_source_and_type(metric_name, source_field, metric_type):
    entry = next(
        entry for entry in (*run_metrics(), *pipeline_metrics()) if entry.name == metric_name
    )
    assert entry.source_field == source_field
    assert entry.metric_type is metric_type


@pytest.mark.parametrize("name", ["", " ", "9metric", "janus.metric", "janus-metric", "a\n", "é"])
def test_invalid_prometheus_names_are_rejected(name):
    with pytest.raises(ValueError, match="metric name"):
        replace(run_metrics()[0], name=name)


@pytest.mark.parametrize("name", ["janus_metric", "_metric", ":metric", "a:metric_9"])
def test_valid_prometheus_names_are_accepted(name):
    assert replace(run_metrics()[0], name=name).name == name


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"help_text": " \n"}, "help_text"),
        ({"source_field": ""}, "source_field"),
        ({"scope": "unknown"}, "scope"),
        ({"metric_type": "histogram"}, "metric_type"),
        ({"labels": ("source_id", "source_id")}, "duplicates"),
        ({"labels": ["source_id"]}, "immutable tuple"),
        ({"source_fields": ["duration_seconds"]}, "immutable tuples"),
        ({"outcome_values": {"within": 0}}, "immutable tuples"),
    ],
)
def test_invalid_declarations_are_rejected_before_rendering(overrides, message):
    with pytest.raises(ValueError, match=message):
        replace(run_metrics()[0], **overrides)


@pytest.mark.parametrize("label", ["pipeline_run_id", "config_version", "bronze_table_identifier"])
def test_other_forbidden_labels_are_rejected(label):
    with pytest.raises(ValueError, match=label):
        replace(run_metrics()[0], labels=(label,))


def test_declarations_are_frozen_and_have_no_mutable_attribute_dictionary():
    entry = run_metrics()[0]
    with pytest.raises(FrozenInstanceError):
        entry.labels = ("run_id",)
    assert not hasattr(entry, "__dict__")


def test_last_success_has_a_declarative_status_filter_and_documents_it():
    success = next(
        entry for entry in run_metrics() if entry.name == "janus_last_success_timestamp_seconds"
    )
    assert success.required_status == "succeeded"
    assert "only" in success.help_text and "succeeded" in success.help_text
    assert all(entry.required_status is None for entry in run_metrics() if entry is not success)


@pytest.mark.parametrize(
    ("outcome", "value"), [("exceeded", 1), ("within", 0), ("undeclared", None)]
)
def test_budget_outcome_rules_omit_undeclared_rather_than_reporting_zero(outcome, value):
    budget = next(
        entry for entry in run_metrics() if entry.name == "janus_duration_budget_exceeded_total"
    )
    assert dict(budget.outcome_values)[outcome] == value
    assert "absent when undeclared" in budget.help_text


@pytest.mark.parametrize(
    ("runs_table", "openlineage", "metrics", "count"),
    [
        ("emitted", "emitted", "failed", 0),
        ("failed", "emitted", "failed", 1),
        ("failed", "failed", "emitted", 2),
        ("skipped", "skipped", "failed", 0),
    ],
)
def test_emission_failure_sources_count_other_destinations_only(
    runs_table, openlineage, metrics, count
):
    entry = next(entry for entry in run_metrics() if entry.name == "janus_emission_failures_total")
    assert entry.source_fields == ("runs_table", "openlineage")
    assert "never counts a metrics failure" in entry.help_text
    outcomes = {"runs_table": runs_table, "openlineage": openlineage, "metrics": metrics}
    assert sum(outcomes[destination] == "failed" for destination in entry.source_fields) == count
    with pytest.raises(ValueError, match="only runs_table and openlineage"):
        replace(entry, source_fields=(*entry.source_fields, "metrics"))


def test_pipeline_status_counts_use_the_bounded_status_label():
    duration, sources = pipeline_metrics()
    assert duration.labels == ("environment", "trigger")
    assert sources.labels == ("environment", "trigger", "status")
    assert sources.source_fields == ("succeeded", "failed", "skipped")
    assert "status" in catalogue.__doc__ and "outcome" in catalogue.__doc__


@pytest.mark.parametrize("trigger", [None, "", " \t "])
def test_absent_labels_are_omitted_instead_of_rendered_as_empty_strings(tmp_path, trigger):
    values = label_values(record(tmp_path, trigger=trigger))
    assert set(values) == ALLOWED_LABELS - {"trigger"}
    assert all(values.values())


def test_label_values_read_only_allowed_record_fields(tmp_path):
    run = record(tmp_path, trigger=" cron ", pipeline_run_id="pipeline", pipeline_attempt=3)
    values = label_values(run)
    assert values == {
        "source_id": run.source_id,
        "strategy_family": run.strategy_family,
        "strategy_variant": run.strategy_variant,
        "environment": run.environment,
        "status": run.status,
        "trigger": "cron",
    }
    assert set(values) == ALLOWED_LABELS


def test_catalogue_is_pure_data_without_engine_or_transport_imports():
    path = Path(inspect.getfile(catalogue))
    source = path.read_text(encoding="utf-8")
    imports = {
        name
        for node in ast.walk(ast.parse(source))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert imports
    assert not any(
        name.startswith(("pyspark", "pyiceberg", "pyarrow", "janus.runtime"))
        or "transport" in name.split(".")
        for name in imports
    )
    assert len(source.splitlines()) < 300
