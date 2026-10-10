"""Engine-free, best-effort delivery of a rendered metrics batch.

Textfiles replace last-run state atomically. Failed runs preserve the previous
last-success gauge: deleting it would stop freshness alerts when a source fails.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from janus.observability.metrics.catalogue import (
    ALLOWED_LABELS,
    MetricDeclaration,
    MetricType,
)
from janus.observability.metrics.exposition import ParsedSample, parse_exposition, render_exposition
from janus.observability.metrics.settings import MetricsTransportKind
from janus.observability.openlineage.transport import PartialWriteError, WarningLogger, _append_line
from janus.utils.storage import sanitize_identifier_segment

NOT_CONFIGURED_REASON = "transport_not_configured"
PROFILE_ERROR_REASON = "transport_profile_error"
METRICS_FILE_PREFIX = "metrics-"
METRICS_FILE_SUFFIX = ".ndjson"
_LAST_SUCCESS = "janus_last_success_timestamp_seconds"
_LOG = logging.getLogger(__name__)


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
    value: int | float | None
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


@dataclass(frozen=True, slots=True)
class TextfileMetricsTransport:
    """One atomic last-run file per source, retaining success across failures."""

    directory: Path
    logger: WarningLogger | None = None

    @property
    def kind(self) -> str:
        return MetricsTransportKind.TEXTFILE.value

    def path_for(self, labels: Mapping[str, str]) -> Path:
        source = sanitize_identifier_segment(labels["source_id"])
        if not source:
            raise ValueError("textfile metrics require a non-empty source_id")
        return self.directory / f"{source}.prom"

    def send(
        self, samples: Sequence[MetricSample], *, budget_seconds: float
    ) -> MetricsEmissionResult:
        del budget_seconds  # Local I/O shares the caller's guarded deadline.
        path = self.directory
        try:
            current = tuple(sample for sample in samples if sample.value is not None)
            if not current:
                return MetricsEmissionResult(
                    MetricsEmissionOutcome.SKIPPED, self.kind, reason="no_samples"
                )
            labels = _labels(current[0])
            path = self.path_for(labels)
            current = _textfile_samples(current, labels)
            payload = render_exposition(current)
        except Exception as exc:
            return self._failed(path, step="render", exception=exc)

        try:
            retained = self._read_success(path, labels, current)
        except Exception as exc:
            return self._failed(path, step="read_existing", exception=exc)
        try:
            if retained:
                payload = render_exposition((*current, *retained))
        except Exception as exc:
            return self._failed(path, step="render", exception=exc)
        return self._rewrite(path, payload, sample_count=len(current) + len(retained))

    def _rewrite(self, path: Path, payload: str, *, sample_count: int) -> MetricsEmissionResult:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return self._failed(path, step="directory_create", exception=exc)

        temporary_path = None
        try:
            temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
            if temporary_path.write_text(payload, encoding="utf-8") != len(payload):
                raise PartialWriteError("the textfile write did not write a complete exposition")
            temporary_path.chmod(0o644)
        except Exception as exc:
            _remove_temporary(temporary_path)
            return self._failed(path, step="write_temp", exception=exc)
        try:
            os.replace(temporary_path, path)
        except Exception as exc:
            _remove_temporary(temporary_path)
            return self._failed(path, step="replace", exception=exc)
        return MetricsEmissionResult(
            MetricsEmissionOutcome.EMITTED,
            self.kind,
            target=str(path),
            sample_count=sample_count,
        )

    def _read_success(
        self, path: Path, labels: Mapping[str, str], samples: Sequence[MetricSample]
    ) -> tuple[ParsedSample, ...]:
        if labels.get("status") == "succeeded" or any(
            sample.name == _LAST_SUCCESS for sample in samples
        ):
            return ()
        try:
            previous = parse_exposition(path.read_text(encoding="utf-8"))
            successful = tuple(sample for sample in previous if sample.name == _LAST_SUCCESS)
            if len(successful) > 1 or any(
                sample.metric_type is not MetricType.GAUGE
                or dict(sample.labels).get("source_id") != labels["source_id"]
                or set(dict(sample.labels)) - ALLOWED_LABELS
                for sample in successful
            ):
                raise ValueError("invalid retained success gauge")
            return successful
        except FileNotFoundError:
            return ()
        except (ValueError, UnicodeError) as exc:
            self._warn_discarded(exc)
            return ()

    def _warn_discarded(self, exception: Exception) -> None:
        fields = {"step": "read_existing", "exception_type": type(exception).__name__}
        try:
            if self.logger is None:
                _LOG.warning("metrics_textfile_existing_discarded", extra={"event_fields": fields})
            else:
                self.logger.warning("metrics_textfile_existing_discarded", **fields)
        except Exception:
            pass

    def _failed(self, path: Path, *, step: str, exception: Exception) -> MetricsEmissionResult:
        return MetricsEmissionResult(
            MetricsEmissionOutcome.FAILED,
            self.kind,
            target=str(path),
            reason=f"{step}_failed",
            step=step,
            exception_type=type(exception).__name__,
        )


def _textfile_samples(
    samples: Sequence[MetricSample], labels: Mapping[str, str]
) -> tuple[MetricSample, ...]:
    normalized = []
    for sample in samples:
        if sample.scope != "run" or _labels(sample) != labels:
            raise ValueError("textfile samples must belong to one run and source")
        if sample.name == _LAST_SUCCESS and labels.get("status") != "succeeded":
            raise ValueError("a last-success measurement requires a succeeded run")
        normalized.append(
            MetricSample(
                sample.declaration, tuple(sorted(labels.items())), sample.value, sample.emitted_at
            )
        )
    return tuple(normalized)


def _remove_temporary(path: Path | None) -> None:
    if path is not None:
        with suppress(OSError):
            path.unlink(missing_ok=True)


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
