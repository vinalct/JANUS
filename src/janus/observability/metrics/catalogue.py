"""One engine-free declaration per JANUS metric, shared by renderers and drift tests.

Counters are per-run totals, not an in-process monotonic registry: JANUS exits
after each invocation. Collectors receive the latest run's value, including when
the textfile collector exposes a counter for a downstream ``rate()`` rule.

Pipeline source counts use the label ``status``, rather than FR-2's ``outcome``,
to stay within the same six-label allow-list. ``record`` and ``emission_result``
are the two synthetic sources: record existence and the preceding destinations'
results. Conditional emission and multi-field sources are immutable declaration
data; the sink resolves them before any transport renders a sample.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from janus.observability.records import RunRecord

_METRIC_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_EMISSION_DESTINATIONS = frozenset({"runs_table", "openlineage"})


class MetricType(StrEnum):
    """Prometheus types used by the catalogue, without a collector dependency."""

    GAUGE = "gauge"
    COUNTER = "counter"


# Registry-bounded identity and outcome labels. A run_id creates unbounded series.
ALLOWED_LABELS = frozenset(
    {"source_id", "strategy_family", "strategy_variant", "environment", "status", "trigger"}
)
_RUN_LABELS = (
    "source_id",
    "strategy_family",
    "strategy_variant",
    "environment",
    "status",
    "trigger",
)
_PIPELINE_LABELS = ("environment", "trigger")


@dataclass(frozen=True, slots=True)
class MetricDeclaration:
    """One metric and its projection rules, validated before reaching a renderer.

    ``source_fields`` enumerates multi-field projections (pipeline status counts
    or emission destinations). ``outcome_values`` maps outcomes to numbers; a
    ``None`` value means omit the sample, rather than emit a measured zero.
    """

    name: str
    metric_type: MetricType
    help_text: str
    source_field: str
    scope: Literal["run", "pipeline"]
    labels: tuple[str, ...]
    required_status: str | None = None
    outcome_values: tuple[tuple[str, int | None], ...] = ()
    source_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _METRIC_NAME.fullmatch(self.name) is None:
            raise ValueError("metric name must match [a-zA-Z_:][a-zA-Z0-9_:]*")
        if not isinstance(self.metric_type, MetricType):
            raise ValueError("metric_type must be a MetricType gauge or counter")
        if not isinstance(self.help_text, str) or not self.help_text.strip():
            raise ValueError("metric help_text must not be empty")
        if not isinstance(self.source_field, str) or not self.source_field.strip():
            raise ValueError("metric source_field must not be empty")
        if self.scope not in {"run", "pipeline"}:
            raise ValueError("metric scope must be 'run' or 'pipeline'")
        if not isinstance(self.labels, tuple):
            raise ValueError("metric labels must be an immutable tuple")
        unknown = set(self.labels) - ALLOWED_LABELS
        if unknown:
            raise ValueError(f"metric labels are not allowed: {', '.join(sorted(unknown))}")
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("metric labels must not contain duplicates")
        if not isinstance(self.source_fields, tuple) or not isinstance(self.outcome_values, tuple):
            raise ValueError("metric projection rules must be immutable tuples")
        if self.source_field == "emission_result" and (
            not self.source_fields or set(self.source_fields) - _EMISSION_DESTINATIONS
        ):
            raise ValueError("emission failures may count only runs_table and openlineage")


_RUN_METRICS = (
    MetricDeclaration(
        "janus_run_duration_seconds",
        MetricType.GAUGE,
        "Duration of the terminal run in seconds.",
        "duration_seconds",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_run_total",
        MetricType.COUNTER,
        "One terminal run, counted by status from the record's existence.",
        "record",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_records_extracted",
        MetricType.GAUGE,
        "Records extracted by the run.",
        "records_extracted",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_records_written",
        MetricType.GAUGE,
        "Records written to bronze by the run.",
        "records_written",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_http_requests_total",
        MetricType.COUNTER,
        "Successful HTTP requests in this run; absent when the family does not report them.",
        "request_count",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_http_retries_total",
        MetricType.COUNTER,
        "HTTP retries in this run; absent when the family does not report them.",
        "retry_count",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_dead_letters_total",
        MetricType.COUNTER,
        "Dead letters reported by the run across all strategy families.",
        "dead_letter_count",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_quality_checks_failed",
        MetricType.GAUGE,
        "Failed quality checks in the run.",
        "quality_checks_failed",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_duration_budget_exceeded_total",
        MetricType.COUNTER,
        "Duration budget exceeded: 1 when exceeded, 0 when within, absent when undeclared.",
        "duration_budget_outcome",
        "run",
        _RUN_LABELS,
        outcome_values=(("exceeded", 1), ("within", 0), ("undeclared", None)),
    ),
    MetricDeclaration(
        "janus_last_run_timestamp_seconds",
        MetricType.GAUGE,
        "Terminal run end time in Unix seconds.",
        "ended_at",
        "run",
        _RUN_LABELS,
    ),
    MetricDeclaration(
        "janus_last_success_timestamp_seconds",
        MetricType.GAUGE,
        "Run end time in Unix seconds, emitted only when status is succeeded.",
        "ended_at",
        "run",
        _RUN_LABELS,
        required_status="succeeded",
    ),
    MetricDeclaration(
        "janus_emission_failures_total",
        MetricType.COUNTER,
        "Failed runs_table and openlineage destinations in RunEmissionResult; "
        "never counts a metrics failure or its own send result.",
        "emission_result",
        "run",
        _RUN_LABELS,
        source_fields=("runs_table", "openlineage"),
    ),
)

_PIPELINE_METRICS = (
    MetricDeclaration(
        "janus_pipeline_duration_seconds",
        MetricType.GAUGE,
        "Duration of the finished pipeline in seconds.",
        "duration_seconds",
        "pipeline",
        _PIPELINE_LABELS,
    ),
    MetricDeclaration(
        "janus_pipeline_sources_total",
        MetricType.GAUGE,
        "Pipeline sources by status: succeeded, failed and skipped, including measured zeros.",
        "succeeded",
        "pipeline",
        (*_PIPELINE_LABELS, "status"),
        source_fields=("succeeded", "failed", "skipped"),
    ),
)

DELIBERATELY_UNMAPPED: dict[str, str] = {
    "attempt_count": (
        "Total HTTP attempts including retries. janus_http_requests_total (successful "
        "requests) and janus_http_retries_total (retries) already span it; a third "
        "series that is the sum of two others invites dashboards that double-count."
    ),
    "artifact_count": (
        "Raw file count is a storage concern, answered by the runs table, not a scrape."
    ),
    "pipeline_attempt": (
        "Correlation, not a measurement. It is a label candidate at most, "
        "and it is unbounded over time."
    ),
    "failure_reason_length": "Diagnostic detail about one message; meaningless as a series.",
    "record_schema_version": "A constant per release.",
    "quality_checks_passed": (
        "Derivable from total minus failed; only the failure count drives an alert."
    ),
    "quality_checks_skipped": "As above; only the failure count drives an alert.",
    "duration_budget_seconds": (
        "A declared threshold, not a measurement. It travels in the runs table for queries; "
        "a Prometheus rule compares against its own literal."
    ),
    "freshness_max_age_seconds": (
        "A declared threshold, not a measurement; janus status and stale-sources.sql use it."
    ),
    "started_at": "Covered by janus_last_run_timestamp_seconds (ended_at) plus duration.",
    "emitted_at": "An emission-time artifact, not a run measurement.",
    "malformed_rows": (
        "Contract-enforcement diagnostic available in the runs table; "
        "the quality failure count already signals a failed check."
    ),
}


def run_metrics() -> tuple[MetricDeclaration, ...]:
    """The twelve terminal-run declarations, in stable emission order."""
    return _RUN_METRICS


def pipeline_metrics() -> tuple[MetricDeclaration, ...]:
    """The two batch-summary declarations; skipped nodes have no per-run metric."""
    return _PIPELINE_METRICS


def label_values(record: RunRecord) -> dict[str, str]:
    """Read the six bounded labels once; omit absent or whitespace-only values."""
    values = {label: getattr(record, label) for label in _RUN_LABELS}
    return {
        label: value.strip()
        for label, value in values.items()
        if value is not None and value.strip()
    }
