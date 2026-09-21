from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Self
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

REDACTED_VALUE = "***REDACTED***"
_ENCODED_REDACTED_VALUE = quote_plus(REDACTED_VALUE)
SENSITIVE_FIELD_MARKERS = (
    "api-key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "session",
    "set-cookie",
    "token",
)
DEFAULT_SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "cookie",
        "key",
        "password",
        "secret",
        "session",
        "sig",
        "signature",
        "token",
    }
)
_RESERVED_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


@dataclass(slots=True)
class StructuredLogger:
    """Small wrapper that emits JSON-friendly structured log events."""

    logger: logging.Logger
    context: dict[str, Any] = field(default_factory=dict)

    def bind(self, **fields: Any) -> Self:
        merged_context = dict(self.context)
        merged_context.update(fields)
        return type(self)(logger=self.logger, context=merged_context)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        payload = dict(self.context)
        payload.update(fields)
        self.logger.exception(event, extra={"event_fields": payload})

    def _log(self, level: int, event: str, **fields: Any) -> None:
        payload = dict(self.context)
        payload.update(fields)
        self.logger.log(level, event, extra={"event_fields": payload})


class StructuredJsonFormatter(logging.Formatter):
    """Format log records as one JSON document per line with secret redaction."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        event_fields = getattr(record, "event_fields", None)
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_FIELDS and key != "event_fields"
        }
        if event_fields is not None and not isinstance(event_fields, Mapping):
            raise TypeError("event_fields must be a mapping when provided")

        merged_fields = dict(extra_fields)
        if event_fields:
            merged_fields.update(event_fields)
        if merged_fields:
            payload["fields"] = sanitize_log_payload(merged_fields)

        if record.exc_info:
            payload["exception"] = sanitize_log_payload(self.formatException(record.exc_info))
        return json.dumps(payload, sort_keys=True)


def build_structured_logger(
    name: str,
    *,
    level: int | str = logging.INFO,
    stream: Any | None = None,
    propagate: bool = False,
) -> StructuredLogger:
    """Create a logger configured for JANUS structured logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredJsonFormatter())
    logger.addHandler(handler)
    logger.propagate = propagate
    return StructuredLogger(logger=logger)


def sanitize_log_payload(value: Any, *, field_name: str | None = None) -> Any:
    """Recursively redact sensitive headers, cookies, tokens, and query parameters."""
    if field_name is not None and is_sensitive_field(field_name):
        return REDACTED_VALUE

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_log_payload(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_log_payload(item, field_name=field_name) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_log_payload(item, field_name=field_name) for item in value)
    if isinstance(value, str):
        return _sanitize_string(value, field_name=field_name)
    return value


def is_sensitive_field(field_name: str) -> bool:
    normalized = field_name.strip().lower().replace("_", "-")
    return any(marker in normalized for marker in SENSITIVE_FIELD_MARKERS)


def _sanitize_string(value: str, *, field_name: str | None = None) -> str:
    if field_name is not None and is_sensitive_field(field_name):
        return REDACTED_VALUE
    return redact_url(value)


def redact_url(url: str) -> str:
    """Redact sensitive query parameters from URLs while leaving the rest readable."""
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    redacted_path = _redact_sensitive_path(parsed.path)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True) if parsed.query else []

    redacted_pairs = []
    changed = False
    for key, value in query_pairs:
        if is_sensitive_field(key) or key.strip().lower() in DEFAULT_SENSITIVE_QUERY_PARAMS:
            redacted_pairs.append((key, REDACTED_VALUE))
            changed = True
        else:
            redacted_pairs.append((key, value))

    if not changed and redacted_path == parsed.path:
        return url
    redacted_query = urlencode(redacted_pairs).replace(
        _ENCODED_REDACTED_VALUE,
        REDACTED_VALUE,
    )
    return urlunsplit(parsed._replace(path=redacted_path, query=redacted_query))


def _redact_sensitive_path(path: str) -> str:
    parts = path.split("/")
    for index, part in enumerate(parts[:-1]):
        if part == "s" and index > 0 and parts[index - 1] == "index.php":
            parts[index + 1] = REDACTED_VALUE
        if (
            part == "public.php"
            and index + 3 < len(parts)
            and parts[index + 1 : index + 3] == ["dav", "files"]
        ):
            parts[index + 3] = REDACTED_VALUE
    return "/".join(parts)
