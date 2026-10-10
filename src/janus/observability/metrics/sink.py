"""Project the catalogue once and deliver metrics inside the shared run deadline."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from janus.observability.metrics.catalogue import MetricDeclaration, label_values, run_metrics
from janus.observability.metrics.settings import (
    MetricsProfileError,
    MetricsTransportKind,
    resolve_metrics_directory,
    resolve_metrics_settings,
)
from janus.observability.metrics.transport import (
    NOT_CONFIGURED_REASON,
    DisabledMetricsTransport,
    FileMetricsTransport,
    MetricSample,
    MetricsEmissionOutcome,
    MetricsEmissionResult,
    MetricsTransport,
    TextfileMetricsTransport,
    WarningLogger,
    build_statsd_transport,
)
from janus.observability.records import RunRecord
from janus.utils.environment import RuntimeLocation

if TYPE_CHECKING:
    from janus.observability.emission import RunEmissionResult

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetricsRunSink:
    """Map one RunRecord to samples and hand one batch to one transport."""

    transport: MetricsTransport

    def emit(
        self,
        run_record: RunRecord,
        *,
        budget_seconds: float,
        logger: WarningLogger | None = None,
        emission_result: RunEmissionResult | None = None,
    ) -> MetricsEmissionResult:
        if self.transport.kind == MetricsTransportKind.DISABLED.value:
            return self.transport.send((), budget_seconds=budget_seconds)
        labels = label_values(run_record)
        samples = []
        for declaration in run_metrics():
            if declaration.required_status not in (None, run_record.status):
                continue
            value = _sample_value(declaration, run_record, emission_result)
            if value is not None:
                samples.append(
                    MetricSample(
                        declaration,
                        tuple(
                            (label, labels[label])
                            for label in declaration.labels
                            if label in labels
                        ),
                        value,
                        run_record.emitted_at,
                    )
                )
        result = self.transport.send(tuple(samples), budget_seconds=budget_seconds)
        # StatsD reports delivery failures itself; file transports leave that to the sink.
        if result.outcome is MetricsEmissionOutcome.FAILED and self.transport.kind != "statsd":
            _warning(logger, "metrics_emission_degraded", result.to_summary())
        return result


def build_metrics_sink(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: WarningLogger | None = None,
) -> MetricsRunSink:
    """Select once per run; invalid settings or containment degrade with one warning."""
    try:
        settings = resolve_metrics_settings(config)
        if settings.file is not None:
            transport: MetricsTransport = FileMetricsTransport(
                resolve_metrics_directory(settings.file.directory, resolved_paths)
            )
        elif settings.textfile is not None:
            transport = TextfileMetricsTransport(
                resolve_metrics_directory(settings.textfile.directory, resolved_paths), logger
            )
        elif settings.statsd is not None:
            transport = build_statsd_transport(settings.statsd, logger=logger)
        else:
            return disabled_metrics_sink()
    except MetricsProfileError as exc:
        _warning(logger, "metrics_profile_invalid", {"reason": str(exc)})
        transport = DisabledMetricsTransport(reason="profile_invalid", detail=str(exc))
    return MetricsRunSink(transport)


def disabled_metrics_sink() -> MetricsRunSink:
    """A real, silent default for directly constructed emitters."""
    return MetricsRunSink(DisabledMetricsTransport())


def emit_guarded_metrics(
    sink: MetricsRunSink,
    run_record: RunRecord,
    *,
    timeout_seconds: float,
    started_at: float,
    emission_result: RunEmissionResult | None,
    logger: WarningLogger | None,
) -> MetricsEmissionResult | None:
    """Attempt the final destination using only the deadline's remaining time."""
    remaining = timeout_seconds - (time.monotonic() - started_at)
    if remaining <= 0:
        return _failure(sink, "budget", "EmissionTimeoutError", logger)
    context = {"emission_result": emission_result} if isinstance(sink, MetricsRunSink) else {}
    try:
        result = sink.emit(
            run_record,
            budget_seconds=remaining,
            logger=logger,
            **context,
        )
    except Exception as exc:
        return _failure(sink, "metrics", type(exc).__name__, logger)
    return None if result.reason == NOT_CONFIGURED_REASON else result


def complete_metrics_result(
    sink: MetricsRunSink,
    result: RunEmissionResult,
    *,
    logger: WarningLogger | None,
) -> MetricsEmissionResult | None:
    """Describe a destination prevented by an early failure or the worker's join."""
    if result.metrics is not None:
        return result.metrics
    if result.stage == "budget":
        return _failure(sink, "budget", result.exception_type or "EmissionTimeoutError", logger)
    if result.stage is None or (
        isinstance(sink.transport, DisabledMetricsTransport)
        and sink.transport.reason == NOT_CONFIGURED_REASON
    ):
        return None
    return _failure(sink, result.stage, result.exception_type or "EmissionError", logger)


def _sample_value(
    declaration: MetricDeclaration,
    record: RunRecord,
    emission_result: RunEmissionResult | None,
) -> int | float | None:
    if declaration.source_field == "record":
        return 1
    if declaration.source_field == "emission_result":
        if emission_result is None:
            return None
        outcomes = {
            "runs_table": emission_result.outcome,
            "openlineage": (
                emission_result.openlineage.outcome if emission_result.openlineage else None
            ),
        }
        return sum(outcomes[destination] == "failed" for destination in declaration.source_fields)
    value = getattr(record, declaration.source_field)
    if value is None:
        return None
    if declaration.outcome_values:
        value = dict(declaration.outcome_values)[value]
    if isinstance(value, datetime):
        return value.timestamp()
    if value is None or isinstance(value, int | float):
        return value
    raise TypeError("a resolved metric value must be numeric")


def _failure(
    sink: MetricsRunSink, step: str, exception_type: str, logger: WarningLogger | None
) -> MetricsEmissionResult:
    result = MetricsEmissionResult(
        MetricsEmissionOutcome.FAILED,
        sink.transport.kind,
        reason=f"{step}_failed",
        step=step,
        exception_type=exception_type,
    )
    _warning(logger, "metrics_emission_degraded", result.to_summary())
    return result


def _warning(logger: WarningLogger | None, event: str, fields: Mapping[str, Any]) -> None:
    try:
        if logger is None:
            _LOG.warning(event, extra={"event_fields": dict(fields)})
        else:
            logger.warning(event, **fields)
    except Exception:
        pass
