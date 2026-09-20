"""The ``observability.openlineage`` block of an environment profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from janus.utils.environment import non_empty_text

OBSERVABILITY_BLOCK_KEY = "observability"
OPENLINEAGE_BLOCK_KEY = "openlineage"
TRANSPORT_KEY = "transport"
PATH_KEY = "path"
URL_KEY = "url"
ENDPOINT_KEY = "endpoint"
TIMEOUT_KEY = "timeout_seconds"
AUTH_KEY = "auth"
TOKEN_KEY = "token"

SUPPORTED_OPENLINEAGE_KEYS = frozenset(
    {TRANSPORT_KEY, PATH_KEY, URL_KEY, ENDPOINT_KEY, TIMEOUT_KEY, AUTH_KEY}
)
SUPPORTED_OPENLINEAGE_AUTH_KEYS = frozenset({TOKEN_KEY})

DEFAULT_EVENTS_DIRECTORY = "lineage/openlineage"
DEFAULT_HTTP_ENDPOINT = "api/v1/lineage"

DEFAULT_HTTP_TIMEOUT_SECONDS = 2.0
MAX_HTTP_TIMEOUT_SECONDS = 30.0

SUPPORTED_URL_SCHEMES = frozenset({"http", "https"})


class OpenLineageTransportKind(StrEnum):
    """The three ways an event can leave JANUS, and the only accepted ``transport`` values."""

    FILE = "file"
    HTTP = "http"
    DISABLED = "disabled"


SUPPORTED_TRANSPORTS = frozenset(kind.value for kind in OpenLineageTransportKind)


class OpenLineageProfileError(ValueError):
    """The environment profile cannot name a usable OpenLineage transport.

    A ``ValueError`` so both CLI entry points already map it to their configuration exit
    code without learning a new exception.
    """


@dataclass(frozen=True, slots=True)
class FileTransportSettings:
    """Where newline-delimited events are appended, relative to the metadata zone."""

    directory: str = DEFAULT_EVENTS_DIRECTORY


@dataclass(frozen=True, slots=True)
class HttpTransportSettings:
    """One receiver, one endpoint, one timeout, and an optional bearer token."""

    url: str
    endpoint: str = DEFAULT_HTTP_ENDPOINT
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS
    token: str | None = None

    @property
    def target_url(self) -> str:
        """The URL a single POST is sent to: the base with the endpoint appended once."""
        return f"{self.url.rstrip('/')}/{self.endpoint.strip('/')}"


@dataclass(frozen=True, slots=True)
class OpenLineageSettings:
    """One profile's resolved transport selection."""

    kind: OpenLineageTransportKind
    file: FileTransportSettings | None = None
    http: HttpTransportSettings | None = None

    def __post_init__(self) -> None:
        """The kind and the settings it names travel together, so no caller re-derives it."""
        carried = {
            OpenLineageTransportKind.FILE: self.file is not None,
            OpenLineageTransportKind.HTTP: self.http is not None,
            OpenLineageTransportKind.DISABLED: self.file is None and self.http is None,
        }
        if not carried[self.kind]:
            raise OpenLineageProfileError(
                f"{self.kind.value!r} settings must carry exactly that transport's block"
            )

    @property
    def enabled(self) -> bool:
        return self.kind is not OpenLineageTransportKind.DISABLED


DISABLED_OPENLINEAGE_SETTINGS = OpenLineageSettings(kind=OpenLineageTransportKind.DISABLED)


def resolve_openlineage_settings(config: Mapping[str, Any]) -> OpenLineageSettings:
    """Resolve the transport one profile declares, or raise naming what it accepts."""
    block = _openlineage_block(config)
    if block is None:
        return DISABLED_OPENLINEAGE_SETTINGS

    _reject_unsupported_keys(block, SUPPORTED_OPENLINEAGE_KEYS, "")
    kind = _resolve_kind(block)
    if kind is OpenLineageTransportKind.DISABLED:
        return DISABLED_OPENLINEAGE_SETTINGS
    if kind is OpenLineageTransportKind.FILE:
        return OpenLineageSettings(kind=kind, file=_file_settings(block))
    return OpenLineageSettings(kind=kind, http=_http_settings(block))


def _openlineage_block(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The block itself, or ``None`` when the profile declares no lineage emission."""
    observability = config.get(OBSERVABILITY_BLOCK_KEY)
    if observability is None:
        return None
    if not isinstance(observability, Mapping):
        raise OpenLineageProfileError(f"{OBSERVABILITY_BLOCK_KEY} must be a mapping")

    block = observability.get(OPENLINEAGE_BLOCK_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise OpenLineageProfileError(
            f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY} must be a mapping"
        )
    return block


def _resolve_kind(block: Mapping[str, Any]) -> OpenLineageTransportKind:
    """The declared transport. Unset means disabled; unrecognised is a profile error."""
    declared = non_empty_text(block.get(TRANSPORT_KEY))
    if declared is None:
        return OpenLineageTransportKind.DISABLED
    if declared not in SUPPORTED_TRANSPORTS:
        supported = ", ".join(sorted(SUPPORTED_TRANSPORTS))
        raise OpenLineageProfileError(
            f"Environment config has an unsupported "
            f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY}.{TRANSPORT_KEY}: "
            f"{declared!r}; supported values: {supported}"
        )
    return OpenLineageTransportKind(declared)


def _file_settings(block: Mapping[str, Any]) -> FileTransportSettings:
    directory = non_empty_text(block.get(PATH_KEY))
    return FileTransportSettings(directory=directory or DEFAULT_EVENTS_DIRECTORY)


def _http_settings(block: Mapping[str, Any]) -> HttpTransportSettings:
    return HttpTransportSettings(
        url=_required_url(block),
        endpoint=non_empty_text(block.get(ENDPOINT_KEY)) or DEFAULT_HTTP_ENDPOINT,
        timeout_seconds=_timeout_seconds(block),
        token=_token(block),
    )


def _required_url(block: Mapping[str, Any]) -> str:
    """The receiver's base URL, which an HTTP transport cannot be inferred without."""
    qualified = f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY}.{URL_KEY}"
    url = non_empty_text(block.get(URL_KEY))
    if url is None:
        raise OpenLineageProfileError(
            f"Environment config must set a non-empty {qualified} for "
            f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY}.{TRANSPORT_KEY} "
            f"{OpenLineageTransportKind.HTTP.value!r}"
        )

    scheme = urlsplit(url).scheme.lower()
    if scheme not in SUPPORTED_URL_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_URL_SCHEMES))
        raise OpenLineageProfileError(
            f"Environment config has a {qualified} with an unsupported scheme: "
            f"{scheme or '<none>'!r}; supported schemes: {supported}"
        )
    return url


def _timeout_seconds(block: Mapping[str, Any]) -> float:
    qualified = f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY}.{TIMEOUT_KEY}"
    declared = non_empty_text(block.get(TIMEOUT_KEY))
    if declared is None:
        return DEFAULT_HTTP_TIMEOUT_SECONDS
    try:
        timeout = float(declared)
    except ValueError as exc:
        raise OpenLineageProfileError(
            f"Environment config has a non-numeric {qualified}: {declared!r}"
        ) from exc
    if not 0 < timeout <= MAX_HTTP_TIMEOUT_SECONDS:
        raise OpenLineageProfileError(
            f"Environment config has an out-of-range {qualified}: {declared!r}; "
            f"it must be greater than 0 and at most {MAX_HTTP_TIMEOUT_SECONDS}"
        )
    return timeout


def _token(block: Mapping[str, Any]) -> str | None:
    """The bearer token, or ``None`` — an unexported ``${VAR:-}`` is not a token."""
    auth = block.get(AUTH_KEY)
    if auth is None:
        return None
    if not isinstance(auth, Mapping):
        raise OpenLineageProfileError(
            f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY}.{AUTH_KEY} must be a mapping"
        )
    _reject_unsupported_keys(auth, SUPPORTED_OPENLINEAGE_AUTH_KEYS, f".{AUTH_KEY}")
    return non_empty_text(auth.get(TOKEN_KEY))


def _reject_unsupported_keys(
    block: Mapping[str, Any],
    supported_keys: frozenset[str],
    suffix: str,
) -> None:
    """Fail closed on a key the block does not define, naming the key but never its value."""
    unsupported = sorted(set(block) - supported_keys)
    if not unsupported:
        return
    supported = ", ".join(sorted(supported_keys))
    raise OpenLineageProfileError(
        f"Environment config has unsupported "
        f"{OBSERVABILITY_BLOCK_KEY}.{OPENLINEAGE_BLOCK_KEY}{suffix} key(s): "
        f"{', '.join(unsupported)}; supported keys: {supported}"
    )
