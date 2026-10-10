"""Engine-free, best-effort delivery of a rendered metrics batch."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from janus.observability.metrics.catalogue import (
    ALLOWED_LABELS,
    MetricDeclaration,
    MetricType,
)
from janus.observability.metrics.settings import MetricsTransportKind
from janus.observability.openlineage.transport import PartialWriteError, _append_line

NOT_CONFIGURED_REASON = "transport_not_configured"
PROFILE_ERROR_REASON = "transport_profile_error"
METRICS_FILE_PREFIX = "metrics-"
METRICS_FILE_SUFFIX = ".ndjson"


class MetricsEmissionOutcome(StrEnum):
    """One delivery's outcome, without changing the run's own status."""

    EMITTED = "emitted"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MetricsEmissionResult:
    """Safe delivery evidence; exception messages never enter the summary."""

    outcome: MetricsEmissionOutcome
    transport: str
    target: str | None = None
    reason: str | None = None
    step: str | None = None
    exception_type: str | None = None
    sample_count: int = 0

    @property
    def emitted(self) -> bool:
        return self.outcome is MetricsEmissionOutcome.EMITTED

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "outcome": self.outcome.value,
            "transport": self.transport,
            "sample_count": self.sample_count,
        }
        for name in ("target", "reason", "step", "exception_type"):
            value = getattr(self, name)
            if value is not None:
                summary[name] = value
        return summary


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One catalogue declaration, resolved labels and value, with batch metadata.

    The sink supplies the record's same ``emitted_at`` to every sample. It is
    optional for samples parsed from exposition, which carries no batch time.
    NDJSON writes that timestamp once, and refuses inconsistent batch metadata.
    """

    declaration: MetricDeclaration
    labels: tuple[tuple[str, str], ...]
    value: int | float
    emitted_at: datetime | None = None

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def metric_type(self) -> MetricType:
        return self.declaration.metric_type

    @property
    def help_text(self) -> str:
        return self.declaration.help_text

    @property
    def scope(self) -> str:
        return self.declaration.scope


class MetricsTransport(Protocol):
    """One batch out, one result back. Implementations never raise or retry."""

    @property
    def kind(self) -> str: ...

    def send(
        self, samples: Sequence[MetricSample], *, budget_seconds: float
    ) -> MetricsEmissionResult: ...


@dataclass(frozen=True, slots=True)
class DisabledMetricsTransport:
    """A real disabled destination, including in-run profile degradation."""

    reason: str = NOT_CONFIGURED_REASON
    detail: str | None = None

    @property
    def kind(self) -> str:
        return MetricsTransportKind.DISABLED.value

    @property
    def reportable(self) -> bool:
        return self.reason != NOT_CONFIGURED_REASON or self.detail is not None

    def send(
        self, samples: Sequence[MetricSample], *, budget_seconds: float
    ) -> MetricsEmissionResult:
        del samples, budget_seconds
        return MetricsEmissionResult(
            outcome=MetricsEmissionOutcome.SKIPPED,
            transport=self.kind,
            reason=self.reason,
            step="transport_selection",
        )


@dataclass(frozen=True, slots=True)
class FileMetricsTransport:
    """One schema-valid NDJSON batch per run, appended to its own UTC day's file."""

    directory: Path

    @property
    def kind(self) -> str:
        return MetricsTransportKind.FILE.value

    def path_for(self, samples: Sequence[MetricSample]) -> Path:
        day = _emitted_at(samples[0]).date().isoformat()
        return self.directory / f"{METRICS_FILE_PREFIX}{day}{METRICS_FILE_SUFFIX}"

    def send(
        self, samples: Sequence[MetricSample], *, budget_seconds: float
    ) -> MetricsEmissionResult:
        del budget_seconds  # A local append is bounded by the caller's own deadline.
        path = self.directory
        try:
            if not samples:
                return MetricsEmissionResult(
                    MetricsEmissionOutcome.SKIPPED, self.kind, reason="no_samples"
                )
            path = self.path_for(samples)
            payload = _serialize(samples)
        except Exception as exc:
            return self._failed(path, step="serialization", exception=exc)

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return self._failed(path, step="directory_create", exception=exc)

        try:
            _append_line(path, payload)
        except Exception as exc:
            return self._failed(path, step="append", exception=exc)

        return MetricsEmissionResult(
            outcome=MetricsEmissionOutcome.EMITTED,
            transport=self.kind,
            target=str(path),
            sample_count=len(samples),
        )

    def _failed(self, path: Path, *, step: str, exception: Exception) -> MetricsEmissionResult:
        return MetricsEmissionResult(
            outcome=MetricsEmissionOutcome.FAILED,
            transport=self.kind,
            target=str(path),
            reason=(
                "partial_write" if isinstance(exception, PartialWriteError) else f"{step}_failed"
            ),
            step=step,
            exception_type=type(exception).__name__,
        )


def _emitted_at(sample: MetricSample) -> datetime:
    value = sample.emitted_at
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a metrics batch requires an aware emitted_at from its record")
    return value.astimezone(UTC)


def _labels(sample: MetricSample) -> dict[str, str]:
    labels: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in sample.labels:
        if name not in ALLOWED_LABELS or name not in sample.declaration.labels:
            raise ValueError("metric label is not declared or allowed")
        if name in seen:
            raise ValueError("metric labels must not contain duplicates")
        seen.add(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError("metric label values must be text")
        if value.strip():
            labels[name] = value.strip()
    return labels


def _serialize(samples: Sequence[MetricSample]) -> bytes:
    """One timestamp and one shared label set, never a timestamp per measurement."""
    first = samples[0]
    emitted_at = _emitted_at(first)
    labels = _labels(first)
    rendered = []
    for sample in samples:
        if (
            _emitted_at(sample) != emitted_at
            or sample.scope != first.scope
            or _labels(sample) != labels
        ):
            raise ValueError("metrics samples must share their batch time, scope and labels")
        if isinstance(sample.value, bool) or not isinstance(sample.value, int | float):
            raise TypeError("metric values must be numeric")
        rendered.append(
            {"name": sample.name, "type": sample.metric_type.value, "value": sample.value}
        )
    batch = {
        "emitted_at": emitted_at.isoformat(),
        "scope": first.scope,
        "labels": labels,
        "samples": rendered,
    }
    return json.dumps(batch, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
