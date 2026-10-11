"""The ``observability.metrics`` block of an environment profile."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from janus.utils.environment import RuntimeLocation, non_empty_text
from janus.utils.storage import resolve_metadata_directory

OBSERVABILITY_BLOCK_KEY = "observability"
METRICS_BLOCK_KEY = "metrics"
TRANSPORT_KEY = "transport"
PATH_KEY = "path"
HOST_KEY = "host"
PORT_KEY = "port"
PREFIX_KEY = "prefix"

SUPPORTED_METRICS_KEYS = frozenset({TRANSPORT_KEY, PATH_KEY, HOST_KEY, PORT_KEY, PREFIX_KEY})
DEFAULT_METRICS_DIRECTORY = "metrics"
DEFAULT_TEXTFILE_DIRECTORY = "metrics/textfile"
DEFAULT_STATSD_PORT = 8125
MAX_STATSD_PORT = 65535
DEFAULT_METRIC_PREFIX = "janus"
_METRIC_PREFIX = re.compile(r"[A-Za-z0-9_]+")
_QUALIFIED_BLOCK = f"{OBSERVABILITY_BLOCK_KEY}.{METRICS_BLOCK_KEY}"


class MetricsTransportKind(StrEnum):
    """The only accepted ways metrics can leave JANUS."""

    DISABLED = "disabled"
    FILE = "file"
    TEXTFILE = "textfile"
    STATSD = "statsd"


SUPPORTED_TRANSPORTS = frozenset(kind.value for kind in MetricsTransportKind)


class MetricsProfileError(ValueError):
    """An unusable metrics profile, mapped to exit code 2 by existing CLI handlers."""


@dataclass(frozen=True, slots=True)
class FileMetricsSettings:
    """NDJSON directory, relative to the metadata zone."""

    directory: str = DEFAULT_METRICS_DIRECTORY


@dataclass(frozen=True, slots=True)
class TextfileMetricsSettings:
    """Prometheus textfile directory, relative to the metadata zone."""

    directory: str = DEFAULT_TEXTFILE_DIRECTORY


@dataclass(frozen=True, slots=True)
class StatsdMetricsSettings:
    """One UDP receiver and a bounded metric-name prefix."""

    host: str
    port: int = DEFAULT_STATSD_PORT
    prefix: str = DEFAULT_METRIC_PREFIX


@dataclass(frozen=True, slots=True)
class MetricsSettings:
    """One profile's transport selection and exactly its matching settings block."""

    kind: MetricsTransportKind
    file: FileMetricsSettings | None = None
    textfile: TextfileMetricsSettings | None = None
    statsd: StatsdMetricsSettings | None = None

    def __post_init__(self) -> None:
        carried = {
            MetricsTransportKind.FILE: self.file,
            MetricsTransportKind.TEXTFILE: self.textfile,
            MetricsTransportKind.STATSD: self.statsd,
        }
        present = {kind for kind, block in carried.items() if block is not None}
        expected = set() if self.kind is MetricsTransportKind.DISABLED else {self.kind}
        if present != expected:
            raise MetricsProfileError(
                f"{self.kind.value!r} settings must carry exactly that transport's block"
            )

    @property
    def enabled(self) -> bool:
        return self.kind is not MetricsTransportKind.DISABLED


DISABLED_METRICS_SETTINGS = MetricsSettings(kind=MetricsTransportKind.DISABLED)


def resolve_metrics_settings(config: Mapping[str, Any]) -> MetricsSettings:
    """Resolve the transport one profile declares, or raise naming what it accepts."""
    block = _metrics_block(config)
    if block is None:
        return DISABLED_METRICS_SETTINGS
    _reject_unsupported_keys(block)
    kind = _resolve_kind(block)
    if kind is MetricsTransportKind.DISABLED:
        return DISABLED_METRICS_SETTINGS
    directory = non_empty_text(block.get(PATH_KEY))
    if kind is MetricsTransportKind.FILE:
        return MetricsSettings(
            kind=kind, file=FileMetricsSettings(directory or DEFAULT_METRICS_DIRECTORY)
        )
    if kind is MetricsTransportKind.TEXTFILE:
        return MetricsSettings(
            kind=kind, textfile=TextfileMetricsSettings(directory or DEFAULT_TEXTFILE_DIRECTORY)
        )
    return MetricsSettings(kind=kind, statsd=_statsd_settings(block))


def resolve_metrics_directory(
    settings_directory: str,
    resolved_paths: Mapping[str, RuntimeLocation],
) -> Path:
    """Resolve a directory inside the metadata zone through the zone's containment rule."""
    try:
        return resolve_metadata_directory(settings_directory, resolved_paths, format="ndjson")
    except ValueError as exc:
        raise MetricsProfileError(
            f"The {_QUALIFIED_BLOCK}.{PATH_KEY} must stay inside the metadata zone: {exc}"
        ) from exc


def _metrics_block(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    observability = config.get(OBSERVABILITY_BLOCK_KEY)
    if observability is None:
        return None
    if not isinstance(observability, Mapping):
        raise MetricsProfileError(f"{OBSERVABILITY_BLOCK_KEY} must be a mapping")
    block = observability.get(METRICS_BLOCK_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise MetricsProfileError(f"{_QUALIFIED_BLOCK} must be a mapping")
    return block


def _reject_unsupported_keys(block: Mapping[str, Any]) -> None:
    unsupported = sorted(str(key) for key in block if key not in SUPPORTED_METRICS_KEYS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_METRICS_KEYS))
        raise MetricsProfileError(
            f"Environment config has unsupported {_QUALIFIED_BLOCK} key(s): "
            f"{', '.join(unsupported)}; supported keys: {supported}"
        )


def _resolve_kind(block: Mapping[str, Any]) -> MetricsTransportKind:
    declared = non_empty_text(block.get(TRANSPORT_KEY))
    if declared is None:
        return MetricsTransportKind.DISABLED
    if declared not in SUPPORTED_TRANSPORTS:
        supported = ", ".join(sorted(SUPPORTED_TRANSPORTS))
        raise MetricsProfileError(
            f"Environment config has an unsupported {_QUALIFIED_BLOCK}.{TRANSPORT_KEY}: "
            f"{declared!r}; supported values: {supported}"
        )
    return MetricsTransportKind(declared)


def _statsd_settings(block: Mapping[str, Any]) -> StatsdMetricsSettings:
    host = non_empty_text(block.get(HOST_KEY))
    if host is None:
        raise MetricsProfileError(
            f"Environment config must set a non-empty {_QUALIFIED_BLOCK}.{HOST_KEY} "
            f"for {_QUALIFIED_BLOCK}.{TRANSPORT_KEY} {MetricsTransportKind.STATSD.value!r}"
        )
    prefix = non_empty_text(block.get(PREFIX_KEY)) or DEFAULT_METRIC_PREFIX
    if _METRIC_PREFIX.fullmatch(prefix) is None:
        raise MetricsProfileError(
            f"Environment config has an invalid {_QUALIFIED_BLOCK}.{PREFIX_KEY}; "
            "it must contain only A-Z, a-z, 0-9 or _"
        )
    return StatsdMetricsSettings(host=host, port=_port(block), prefix=prefix)


def _port(block: Mapping[str, Any]) -> int:
    declared = non_empty_text(block.get(PORT_KEY))
    if declared is None:
        return DEFAULT_STATSD_PORT
    qualified = f"{_QUALIFIED_BLOCK}.{PORT_KEY}"
    try:
        port = int(declared)
    except ValueError as exc:
        raise MetricsProfileError(
            f"Environment config has a non-numeric {qualified}: {declared!r}; "
            "it must be an integer"
        ) from exc
    if not 0 < port <= MAX_STATSD_PORT:
        raise MetricsProfileError(
            f"Environment config has an out-of-range {qualified}: {declared!r}; "
            f"it must be greater than 0 and at most {MAX_STATSD_PORT}"
        )
    return port
