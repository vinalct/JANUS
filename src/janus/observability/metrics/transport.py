"""Engine-free, best-effort delivery of a rendered metrics batch.

Textfiles replace last-run state atomically. Failed runs preserve the previous
last-success gauge: deleting it would stop freshness alerts when a source fails.

StatsD uses the stdlib socket directly: JANUS has no UDP primitive to compose.
It is fire-and-forget; a successful send cannot confirm that a receiver got it.
DNS is resolved once when building the transport; sends reuse its address.
"""

from __future__ import annotations

import logging
import math
import os
import re
import socket
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from janus.observability.metrics.catalogue import (
    ALLOWED_LABELS,
    MetricDeclaration,
    MetricType,
)
from janus.observability.metrics.exposition import ParsedSample, parse_exposition, render_exposition
from janus.observability.metrics.serialization import (
    emitted_at as _emitted_at,
)
from janus.observability.metrics.serialization import (
    label_values as _labels,
)
from janus.observability.metrics.serialization import (
    pipeline_labels,
)
from janus.observability.metrics.serialization import (
    serialize_batch as _serialize,
)
from janus.observability.metrics.settings import MetricsTransportKind, StatsdMetricsSettings
from janus.observability.openlineage.transport import PartialWriteError, WarningLogger, _append_line
from janus.utils.storage import sanitize_identifier_segment

NOT_CONFIGURED_REASON = "transport_not_configured"
PROFILE_ERROR_REASON = "transport_profile_error"
METRICS_FILE_PREFIX = "metrics-"
METRICS_FILE_SUFFIX = ".ndjson"
SAFE_DATAGRAM_BYTES = 1432  # 1500-byte MTU - 20-byte IPv4 - 8-byte UDP - 40-byte margin.
SEND_TIMEOUT_SECONDS = 0.5
_STATSD_NAME = re.compile(r"[A-Za-z0-9_]+")
_STATSD_TAG_SEPARATORS = str.maketrans({char: "_" for char in ",|:\n\r"})
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
            if current[0].scope == "pipeline":
                labels = pipeline_labels(current)
                environment = sanitize_identifier_segment(labels["environment"])
                trigger = sanitize_identifier_segment(labels["trigger"])
                if not environment or not trigger:
                    raise ValueError("pipeline textfile identity must not be empty")
                path = self.directory / f"pipeline-{environment}-{trigger}.prom"
                return self._rewrite(path, render_exposition(current), sample_count=len(current))
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


@dataclass(frozen=True, slots=True)
class StatsdMetricsTransport:
    """One UDP batch per run, using a pre-resolved address. No response or retry."""

    address: tuple[str, int]
    prefix: str
    logger: WarningLogger | None = None

    @property
    def kind(self) -> str:
        return MetricsTransportKind.STATSD.value

    def send(
        self, samples: Sequence[MetricSample], *, budget_seconds: float
    ) -> MetricsEmissionResult:
        started_at = time.monotonic()
        step = "render"
        connection = None
        failure = None
        current: tuple[MetricSample, ...] = ()
        datagrams: tuple[bytes, ...] = ()
        try:
            current = tuple(sample for sample in samples if sample.value is not None)
            if not current:
                return MetricsEmissionResult(
                    MetricsEmissionOutcome.SKIPPED, self.kind, reason="no_samples"
                )
            datagrams = _statsd_datagrams(current, self.prefix)
            step = "budget"
            if not math.isfinite(budget_seconds) or budget_seconds <= 0:
                raise ValueError("a StatsD send requires a positive finite budget")
            deadline = started_at + budget_seconds
            if time.monotonic() >= deadline:
                raise TimeoutError("the metrics emission budget is exhausted")
            step = "socket_create"
            connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for payload in datagrams:
                step = "budget"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("the metrics emission budget is exhausted")
                step = "settimeout"
                connection.settimeout(min(remaining, SEND_TIMEOUT_SECONDS))
                step = "send"
                if connection.sendto(payload, self.address) != len(payload):
                    raise PartialWriteError("the UDP send did not write a complete datagram")
        except Exception as exc:
            failure = (step, exc)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception as exc:
                    if failure is None:
                        failure = ("close", exc)
        if failure is not None:
            step, exception = failure
            result = MetricsEmissionResult(
                MetricsEmissionOutcome.FAILED,
                self.kind,
                reason="partial_write"
                if isinstance(exception, PartialWriteError)
                else f"{step}_failed",
                step=step,
                exception_type=type(exception).__name__,
            )
            _statsd_warning(self.logger, {**result.to_summary(), "port": self.address[1]})
            return result
        return MetricsEmissionResult(
            MetricsEmissionOutcome.EMITTED,
            self.kind,
            reason=f"datagrams_sent:{len(datagrams)}" if len(datagrams) > 1 else None,
            sample_count=len(current),
        )


def build_statsd_transport(
    settings: StatsdMetricsSettings, *, logger: WarningLogger | None = None
) -> MetricsTransport:
    """Resolve once before emission; report only the port and resolver failure type."""
    try:
        resolved = socket.getaddrinfo(
            settings.host, settings.port, family=socket.AF_INET, type=socket.SOCK_DGRAM
        )[0][4]
    except OSError as exc:
        reason = "statsd_address_unresolvable"
        _statsd_warning(
            logger,
            {
                "transport": MetricsTransportKind.STATSD.value,
                "outcome": MetricsEmissionOutcome.SKIPPED.value,
                "reason": reason,
                "step": "address_resolution",
                "port": settings.port,
                "exception_type": type(exc).__name__,
            },
        )
        return DisabledMetricsTransport(reason=reason, detail=type(exc).__name__)
    # AF_INET restricts the resolver's union-typed address to an IPv4 pair.
    return StatsdMetricsTransport(cast(tuple[str, int], resolved), settings.prefix, logger)


def _statsd_datagrams(samples: Sequence[MetricSample], prefix: str) -> tuple[bytes, ...]:
    """Pack whole lines greedily; an oversized single line cannot be split safely."""
    if _STATSD_NAME.fullmatch(prefix) is None:
        raise ValueError("invalid StatsD prefix")
    datagrams: list[bytes] = []
    batch = b""
    for sample in samples:
        name = sample.name.removeprefix("janus_")
        if _STATSD_NAME.fullmatch(name) is None:
            raise ValueError("invalid StatsD metric name")
        if isinstance(sample.value, bool) or not isinstance(sample.value, int | float):
            raise TypeError("metric values must be numeric")
        if not math.isfinite(sample.value):
            raise ValueError("metric values must be finite")
        tags = ",".join(
            f"{key}:{value.translate(_STATSD_TAG_SEPARATORS)}"
            for key, value in sorted(_labels(sample).items())
        )
        metric_type = "g" if sample.metric_type is MetricType.GAUGE else "c"
        line = f"{prefix}.{name}:{sample.value}|{metric_type}"
        payload = (line + (f"|#{tags}" if tags else "")).encode("utf-8")
        if len(payload) > SAFE_DATAGRAM_BYTES:
            raise ValueError("one StatsD sample exceeds the safe datagram size")
        if batch and len(batch) + 1 + len(payload) > SAFE_DATAGRAM_BYTES:
            datagrams.append(batch)
            batch = b""
        batch = batch + b"\n" + payload if batch else payload
    if batch:
        datagrams.append(batch)
    return tuple(datagrams)


def _statsd_warning(logger: WarningLogger | None, fields: Mapping[str, Any]) -> None:
    try:
        if logger is None:
            _LOG.warning("metrics_statsd_degraded", extra={"event_fields": dict(fields)})
        else:
            logger.warning("metrics_statsd_degraded", **fields)
    except Exception:
        pass


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
