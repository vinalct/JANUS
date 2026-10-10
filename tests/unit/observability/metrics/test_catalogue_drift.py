from dataclasses import fields, replace
from datetime import datetime
from typing import get_args, get_type_hints

import pytest

from janus.observability import RunRecord
from tests.unit.runtime.test_batch_executor import _run

pytestmark = pytest.mark.xfail(
    strict=True, reason="metrics catalogue and label validation are absent"
)
LABELS = {"source_id", "strategy_family", "strategy_variant", "environment", "status", "trigger"}


def test_every_declaration_resolves_to_a_real_record_or_pipeline_summary_field(tmp_path):
    from janus.observability.metrics.catalogue import pipeline_metrics, run_metrics

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


def test_every_numeric_or_timestamp_field_is_mapped_or_has_a_written_exclusion():
    from janus.observability.metrics.catalogue import DELIBERATELY_UNMAPPED, run_metrics

    hints = get_type_hints(RunRecord)
    numeric = {
        field.name
        for field in fields(RunRecord)
        if set(get_args(hints[field.name]) or (hints[field.name],)) & {int, float, datetime}
    }
    mapped = {entry.source_field for entry in run_metrics()}
    excluded = set(DELIBERATELY_UNMAPPED)
    assert numeric and mapped and excluded
    assert numeric <= mapped | excluded
    assert excluded <= numeric, "stale exclusions must fail in the reverse direction"
    assert not mapped & excluded
    assert all(
        isinstance(reason, str) and reason.strip() for reason in DELIBERATELY_UNMAPPED.values()
    )


def test_label_union_is_exactly_the_six_bounded_labels():
    from janus.observability.metrics.catalogue import ALLOWED_LABELS, pipeline_metrics, run_metrics

    entries = (*run_metrics(), *pipeline_metrics())
    assert entries
    declared = {label for entry in entries for label in entry.labels}
    assert declared
    assert declared == set(ALLOWED_LABELS) == LABELS


def test_catalogue_detector_rejects_a_deliberate_run_id_label():
    from janus.observability.metrics.catalogue import run_metrics

    entries = run_metrics()
    assert entries
    with pytest.raises(ValueError, match="run_id"):
        replace(entries[0], labels=("source_id", "run_id"))


def test_catalogue_detector_accepts_a_clean_declaration():
    from janus.observability.metrics.catalogue import run_metrics

    entries = run_metrics()
    assert entries
    clean = replace(entries[0], labels=("source_id", "status"))
    assert clean.labels == ("source_id", "status")
    assert clean.name == entries[0].name


def test_catalogue_contains_all_fourteen_unique_named_metrics():
    from janus.observability.metrics.catalogue import pipeline_metrics, run_metrics

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
