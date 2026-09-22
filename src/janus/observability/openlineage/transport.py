"""The three transports an OpenLineage event can take out of JANUS."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from janus.observability.openlineage.settings import (
    HttpTransportSettings,
    OpenLineageProfileError,
    OpenLineageSettings,
    OpenLineageTransportKind,
    resolve_openlineage_settings,
)
from janus.utils.environment import RuntimeLocation
from janus.utils.logging import StructuredLogger, redact_url
from janus.utils.storage import ResolvedOutputTarget

METADATA_ROOT_KEY = "metadata_dir"
EVENTS_FILE_PREFIX = "events-"
EVENTS_FILE_SUFFIX = ".ndjson"
UNDATED_EVENTS_DAY = "undated"
EVENT_TIME_KEY = "eventTime"
JSON_CONTENT_TYPE = "application/json"
HTTP_SUCCESS_MIN = 200
HTTP_SUCCESS_MAX = 299
OPENLINEAGE_RESPONSE_LIMIT_BYTES = 1024**2
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}")
_LOG = logging.getLogger(__name__)

NOT_CONFIGURED_REASON = "transport_not_configured"
PROFILE_ERROR_REASON = "transport_profile_error"


class WarningLogger(Protocol):
    """The one method every degradation path needs. Shared so the seam has one spelling."""

    def warning(self, event: str, **fields: Any) -> None: ...


class OpenLineageEmissionOutcome(StrEnum):
    """The three outcomes a caller can report without inspecting the transport."""

    EMITTED = "emitted"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OpenLineageEmissionResult:
    """Observable outcome of one best-effort event delivery."""

    outcome: OpenLineageEmissionOutcome
    transport: str
    target: str | None = None
    reason: str | None = None
    step: str | None = None
    exception_type: str | None = None
    status_code: int | None = None

    @property
    def emitted(self) -> bool:
        return self.outcome is OpenLineageEmissionOutcome.EMITTED

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "outcome": self.outcome.value,
            "transport": self.transport,
        }
        for name in ("target", "reason", "step", "exception_type", "status_code"):
            value = getattr(self, name)
            if value is not None:
                summary[name] = value
        return summary


class OpenLineageTransport(Protocol):
    """One event out, one result back. Implementations never raise and never retry."""

    @property
    def kind(self) -> str: ...

    def send(
        self,
        event: Mapping[str, Any],
        *,
        budget_seconds: float,
    ) -> OpenLineageEmissionResult: ...


@dataclass(frozen=True, slots=True)
class DisabledOpenLineageTransport:
    """The real object ``disabled`` resolves to, so no call site branches on ``None``."""

    reason: str = NOT_CONFIGURED_REASON
    detail: str | None = None

    @property
    def kind(self) -> str:
        return OpenLineageTransportKind.DISABLED.value

    @property
    def reportable(self) -> bool:
        """Whether this transport has something to say beyond "nobody asked for events"."""
        return self.reason != NOT_CONFIGURED_REASON

    def send(
        self,
        event: Mapping[str, Any],
        *,
        budget_seconds: float,
    ) -> OpenLineageEmissionResult:
        del event, budget_seconds
        return OpenLineageEmissionResult(
            outcome=OpenLineageEmissionOutcome.SKIPPED,
            transport=self.kind,
            reason=self.reason,
            step="transport_selection",
            exception_type=self.detail,
        )


@dataclass(frozen=True, slots=True)
class FileOpenLineageTransport:
    """Newline-delimited JSON in the metadata zone, one event per line, appended."""

    directory: Path

    @property
    def kind(self) -> str:
        return OpenLineageTransportKind.FILE.value

    def path_for(self, event: Mapping[str, Any]) -> Path:
        return self.directory / f"{EVENTS_FILE_PREFIX}{_event_day(event)}{EVENTS_FILE_SUFFIX}"

    def send(
        self,
        event: Mapping[str, Any],
        *,
        budget_seconds: float,
    ) -> OpenLineageEmissionResult:
        del budget_seconds  # A local append is bounded by the caller's own deadline.
        path = self.path_for(event)
        try:
            payload = _serialize(event)
        except Exception as exc:
            return self._failed(path, step="serialization", exception=exc)

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return self._failed(path, step="directory_create", exception=exc)

        try:
            writes = _append_line(path, payload)
        except Exception as exc:
            return self._failed(path, step="append", exception=exc)

        return OpenLineageEmissionResult(
            outcome=OpenLineageEmissionOutcome.EMITTED,
            transport=self.kind,
            target=str(path),
            reason="partial_write" if writes > 1 else None,
        )

    def _failed(self, path: Path, *, step: str, exception: Exception) -> OpenLineageEmissionResult:
        return OpenLineageEmissionResult(
            outcome=OpenLineageEmissionOutcome.FAILED,
            transport=self.kind,
            target=str(path),
            reason=f"{step}_failed",
            step=step,
            exception_type=type(exception).__name__,
        )


@dataclass(frozen=True, slots=True)
class HttpOpenLineageTransport:
    """One POST to one receiver: no retry, no backoff, one short timeout.

    Nothing is taken from the response but its status — a lineage receiver's error body is
    not this run's business, and a status plus a redacted URL is the whole warning.
    """

    settings: HttpTransportSettings

    @property
    def kind(self) -> str:
        return OpenLineageTransportKind.HTTP.value

    @property
    def target(self) -> str:
        return redact_url(self.settings.target_url)

    def send(
        self,
        event: Mapping[str, Any],
        *,
        budget_seconds: float,
    ) -> OpenLineageEmissionResult:
        try:
            payload = _serialize(event)
        except Exception as exc:
            return self._failed(step="serialization", exception=exc)

        timeout = _request_timeout(self.settings.timeout_seconds, budget_seconds)
        try:
            # Lazy by design: the hermetic transports never import the strategy package.
            from janus.strategies.http.transport import ApiRequest, UrllibApiTransport
        except Exception as exc:
            return self._failed(step="transport_import", exception=exc)

        request = ApiRequest(
            method="POST",
            url=self.settings.target_url,
            timeout_seconds=timeout,
            headers=self._headers(),
            body=payload,
            max_payload_bytes=OPENLINEAGE_RESPONSE_LIMIT_BYTES,
        )
        transport = UrllibApiTransport()
        try:
            response = transport.send(request)
        except Exception as exc:
            return self._failed(step="request", exception=exc)
        finally:
            transport.close()

        if HTTP_SUCCESS_MIN <= response.status_code <= HTTP_SUCCESS_MAX:
            return OpenLineageEmissionResult(
                outcome=OpenLineageEmissionOutcome.EMITTED,
                transport=self.kind,
                target=self.target,
                status_code=response.status_code,
            )
        return OpenLineageEmissionResult(
            outcome=OpenLineageEmissionOutcome.FAILED,
            transport=self.kind,
            target=self.target,
            reason="unexpected_status",
            step="response",
            status_code=response.status_code,
        )

    def _headers(self) -> tuple[tuple[str, str], ...]:
        """An unset token yields no header at all, never an empty or ``Bearer None`` one."""
        headers = {"Content-Type": JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE}
        if self.settings.token is not None:
            headers["Authorization"] = f"Bearer {self.settings.token}"
        return tuple(sorted(headers.items()))

    def _failed(self, *, step: str, exception: Exception) -> OpenLineageEmissionResult:
        return OpenLineageEmissionResult(
            outcome=OpenLineageEmissionOutcome.FAILED,
            transport=self.kind,
            target=self.target,
            reason=f"{step}_failed",
            step=step,
            exception_type=type(exception).__name__,
        )


def build_openlineage_transport(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: StructuredLogger | WarningLogger | None = None,
) -> OpenLineageTransport:
    """Select one transport from one profile, degrading instead of raising inside a run.

    The profile error itself is raised by :func:`resolve_openlineage_settings` at profile read
    time, where a wrong profile is a configuration error. Here — inside a run — it becomes a
    disabled transport carrying the reason, and one warning.
    """
    try:
        settings = resolve_openlineage_settings(config)
        return _transport_for(settings, resolved_paths)
    except Exception as exc:
        return _disabled_by_profile_error(exc, logger)


def _transport_for(
    settings: OpenLineageSettings,
    resolved_paths: Mapping[str, RuntimeLocation],
) -> OpenLineageTransport:
    if settings.file is not None:
        return FileOpenLineageTransport(
            directory=_resolve_events_directory(settings.file.directory, resolved_paths)
        )
    if settings.http is not None:
        return HttpOpenLineageTransport(settings=settings.http)
    return DisabledOpenLineageTransport()


def _resolve_events_directory(
    directory: str,
    resolved_paths: Mapping[str, RuntimeLocation],
) -> Path:
    """Resolve the events directory inside the metadata zone, refusing any way out of it."""
    metadata_root = resolved_paths.get(METADATA_ROOT_KEY)
    if metadata_root is None:
        raise OpenLineageProfileError(
            "The file transport needs a resolved metadata zone; this profile resolved none"
        )
    root = Path(str(metadata_root))
    if not root.is_absolute():
        raise OpenLineageProfileError(
            "The resolved metadata zone must be absolute before events can be written under it"
        )
    zone = ResolvedOutputTarget(
        zone="metadata",
        configured_path=directory,
        resolved_path=root,
        format="ndjson",
    )
    try:
        return zone.child(directory)
    except ValueError as exc:
        raise OpenLineageProfileError(
            f"The OpenLineage events path must stay inside the metadata zone: {exc}"
        ) from exc


def _disabled_by_profile_error(
    exception: Exception,
    logger: StructuredLogger | WarningLogger | None,
) -> DisabledOpenLineageTransport:
    fields = {
        "outcome": OpenLineageEmissionOutcome.SKIPPED.value,
        "reason": PROFILE_ERROR_REASON,
        "exception_type": type(exception).__name__,
        "message": str(exception),
    }
    try:
        if logger is None:
            _LOG.warning("openlineage_transport_unavailable", extra={"event_fields": fields})
        else:
            logger.warning("openlineage_transport_unavailable", **fields)
    except Exception:
        pass
    return DisabledOpenLineageTransport(
        reason=PROFILE_ERROR_REASON,
        detail=type(exception).__name__,
    )


def _serialize(event: Mapping[str, Any]) -> bytes:
    """One event, one line: deterministic key order and no embedded newline."""
    return json.dumps(dict(event), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _append_line(path: Path, payload: bytes) -> int:
    """Append ``payload`` plus a newline, reporting how many ``write(2)`` calls it took."""
    line = payload + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    writes = 0
    try:
        written = 0
        while written < len(line):
            written += os.write(descriptor, line[written:])
            writes += 1
    finally:
        os.close(descriptor)
    return writes


def _event_day(event: Mapping[str, Any]) -> str:
    """The UTC day the event names, taken from the event itself so the file stays pure."""
    event_time = event.get(EVENT_TIME_KEY)
    if isinstance(event_time, str) and _ISO_DAY.match(event_time):
        return event_time[:10]
    if isinstance(event_time, datetime):
        return event_time.astimezone(UTC).date().isoformat()
    return UNDATED_EVENTS_DAY


def _request_timeout(configured: float, budget_seconds: float) -> int:
    """The per-request timeout: the configured one, never more than what is left to spend."""
    bounded = min(configured, budget_seconds) if budget_seconds > 0 else configured
    return max(1, int(bounded))
