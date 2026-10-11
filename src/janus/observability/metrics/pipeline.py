"""Pure batch-summary projection and bounded delivery through the profile's sink."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from janus.observability.bounded import (
    DEFAULT_EMISSION_TIMEOUT_SECONDS,
    EmissionGuardError,
    bounded_emission,
)
from janus.observability.metrics.catalogue import pipeline_metrics
from janus.observability.metrics.sink import _warning, build_metrics_sink
from janus.observability.metrics.transport import (
    MetricSample,
    MetricsEmissionOutcome,
    MetricsEmissionResult,
    WarningLogger,
)
from janus.orchestration.results import PipelineOutcome
from janus.utils.environment import RuntimeLocation


def pipeline_samples(outcome: PipelineOutcome) -> tuple[MetricSample, ...]:
    """Project a finished batch's existing totals; no I/O, clock or per-node evidence."""
    totals = outcome.totals()
    labels = {"environment": outcome.environment, "trigger": outcome.trigger}
    samples = []
    for declaration in pipeline_metrics():
        for field in declaration.source_fields or (declaration.source_field,):
            values = {**labels, "status": field}
            # These zeros are measured: the batch finished with no failed/skipped sources.
            # Keep each series at zero so an alert fired by a previous batch can resolve.
            samples.append(
                MetricSample(
                    declaration,
                    tuple((label, values[label]) for label in declaration.labels),
                    totals[field],
                    outcome.timing.ended_at,
                )
            )
    return tuple(samples)


def emit_pipeline_metrics(
    outcome: PipelineOutcome,
    environment_config: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: WarningLogger | None = None,
    timeout_seconds: float = DEFAULT_EMISSION_TIMEOUT_SECONDS,
) -> None:
    """Emit last, including sink construction in the shared guard; retain failures as logs."""
    started_at = time.monotonic()

    def operation() -> MetricsEmissionResult:
        config = environment_config() if callable(environment_config) else environment_config
        sink = build_metrics_sink(config, resolved_paths, logger=logger)
        remaining = timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            raise EmissionGuardError("budget", "EmissionTimeoutError")
        return sink.emit_samples(pipeline_samples(outcome), budget_seconds=remaining, logger=logger)

    try:
        bounded_emission(
            operation,
            timeout_seconds=timeout_seconds,
            thread_name="janus-pipeline-metrics-emission",
        )
    except Exception as exc:
        fields = {"outcome": MetricsEmissionOutcome.FAILED.value}
        fields["step"] = exc.stage if isinstance(exc, EmissionGuardError) else "metrics"
        fields["exception_type"] = (
            exc.exception_type if isinstance(exc, EmissionGuardError) else type(exc).__name__
        )
        _warning(logger, "metrics_emission_degraded", fields)
