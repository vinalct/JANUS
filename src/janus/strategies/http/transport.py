"""Reusable HTTP transport shared by the api, catalog, and file strategies.

Owns the request/response value objects (ApiRequest, ApiResponse), the stdlib
urllib transport (UrllibApiTransport) and its context-managed client (ApiClient),
SSL/CA-bundle resolution, and auth injection (inject_auth). Moved verbatim from
janus.strategies.api.http; the shared behavioral layer built on top of it lives
alongside in janus.strategies.http.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPSHandler, OpenerDirector, Request, build_opener

from janus.models import AuthConfig
from janus.strategies.common import _freeze_string_mapping, _stringify_mapping
from janus.utils.logging import redact_url


@dataclass(frozen=True, slots=True)
class ApiRequest:
    """Normalized HTTP request used by the reusable API strategy."""

    method: str
    url: str
    timeout_seconds: int
    headers: tuple[tuple[str, str], ...] = ()
    params: tuple[tuple[str, str], ...] = ()
    body: bytes | None = None

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("method must not be empty")
        if not self.url.strip():
            raise ValueError("url must not be empty")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be greater than zero")

    def headers_as_dict(self) -> dict[str, str]:
        return dict(self.headers)

    def params_as_dict(self) -> dict[str, str]:
        return dict(self.params)

    def with_header(self, name: str, value: str) -> ApiRequest:
        headers = self.headers_as_dict()
        headers[name] = value
        return replace(self, headers=_freeze_string_mapping(headers))

    def with_url(self, url: str) -> ApiRequest:
        return replace(self, url=url)

    def with_params(self, params: Mapping[str, Any]) -> ApiRequest:
        merged_params = self.params_as_dict()
        merged_params.update(_stringify_mapping(params))
        return replace(self, params=_freeze_string_mapping(merged_params))

    def full_url(self) -> str:
        parsed = urlsplit(self.url)
        existing_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        existing_params.update(self.params_as_dict())
        if not existing_params:
            return self.url
        return urlunsplit(parsed._replace(query=urlencode(existing_params)))


@dataclass(frozen=True, slots=True)
class ApiResponse:
    """HTTP response returned by the transport layer."""

    request: ApiRequest
    status_code: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()
    received_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def __post_init__(self) -> None:
        if self.status_code < 100:
            raise ValueError("status_code must be a valid HTTP status")
        if self.received_at.tzinfo is None or self.received_at.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")

    def headers_as_dict(self) -> dict[str, str]:
        return dict(self.headers)

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding)

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.text())


class ApiTransportError(RuntimeError):
    """Raised when the transport could not reach the remote API."""


class AuthResolutionError(RuntimeError):
    """Raised when API auth cannot be resolved from the configured environment."""


class ApiTransport(Protocol):
    """Small transport contract so tests can inject deterministic fake clients."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def send(self, request: ApiRequest) -> ApiResponse: ...


@dataclass(slots=True)
class UrllibApiTransport:
    """Stdlib-backed HTTP transport with an explicit open/close lifecycle."""

    opener: OpenerDirector | None = None
    ca_bundle_path: str | None = None

    def open(self) -> None:
        if self.opener is None:
            self.opener = build_opener(
                HTTPSHandler(context=_build_ssl_context(self.ca_bundle_path))
            )

    def close(self) -> None:
        self.opener = None

    def send(self, request: ApiRequest) -> ApiResponse:
        self.open()
        if self.opener is None:
            raise ApiTransportError("API transport failed to initialize urllib opener")

        urllib_request = Request(
            request.full_url(),
            data=request.body,
            method=request.method,
            headers=request.headers_as_dict(),
        )

        try:
            with self.opener.open(urllib_request, timeout=request.timeout_seconds) as stream:
                return ApiResponse(
                    request=request,
                    status_code=stream.getcode(),
                    body=stream.read(),
                    headers=_freeze_string_mapping(dict(stream.headers.items())),
                )
        except HTTPError as exc:
            headers = dict(exc.headers.items()) if exc.headers is not None else {}
            return ApiResponse(
                request=request,
                status_code=exc.code,
                body=exc.read(),
                headers=_freeze_string_mapping(headers),
            )
        except (URLError, OSError) as exc:
            raise ApiTransportError(
                f"Request failed for {redact_url(request.full_url())!r}: "
                f"{type(exc).__name__}"
            ) from exc


@dataclass(slots=True)
class ApiClient:
    """Context-managed client wrapper around a transport implementation."""

    transport: ApiTransport

    def __enter__(self) -> ApiClient:
        self.transport.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb
        self.transport.close()

    def send(self, request: ApiRequest) -> ApiResponse:
        return self.transport.send(request)


def _build_ssl_context(ca_bundle_path: str | None = None) -> ssl.SSLContext:
    context = ssl.create_default_context()
    for candidate in _ca_bundle_candidates(ca_bundle_path):
        if _is_optional_ca_bundle(candidate) and not Path(candidate).is_file():
            continue
        context.load_verify_locations(cafile=candidate)
    return context


def _resolve_ca_bundle(ca_bundle_path: str | None = None) -> str | None:
    candidates = _ca_bundle_candidates(ca_bundle_path)
    if not candidates:
        return None
    return candidates[0]


def _ca_bundle_candidates(ca_bundle_path: str | None = None) -> tuple[str, ...]:
    candidates: list[str] = []
    _append_ca_candidate(candidates, ca_bundle_path)

    for env_var in ("JANUS_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        _append_ca_candidate(candidates, os.getenv(env_var))

    try:
        import certifi
    except ImportError:
        pass
    else:
        _append_ca_candidate(candidates, certifi.where())

    _append_ca_candidate(candidates, os.getenv("JANUS_SYSTEM_CA_BUNDLE"))
    _append_ca_candidate(candidates, "/etc/ssl/certs/ca-certificates.crt")
    return tuple(candidates)


def _append_ca_candidate(candidates: list[str], value: str | None) -> None:
    if value is None or not value.strip():
        return
    candidate = value.strip()
    if candidate not in candidates:
        candidates.append(candidate)


def _is_optional_ca_bundle(path: str) -> bool:
    configured_paths = {
        value.strip()
        for value in (
            os.getenv("JANUS_CA_BUNDLE"),
            os.getenv("SSL_CERT_FILE"),
            os.getenv("REQUESTS_CA_BUNDLE"),
        )
        if value is not None and value.strip()
    }
    return path not in configured_paths


def inject_auth(
    request: ApiRequest,
    auth: AuthConfig,
    *,
    env_reader: Callable[[str], str | None] = os.getenv,
) -> ApiRequest:
    """Inject auth headers or query params without leaking secret handling into strategy code."""

    if auth.type == "none":
        return request

    if auth.type == "basic":
        username = _require_secret(auth.username_env_var, env_reader)
        password = _require_secret(auth.password_env_var, env_reader)
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        return request.with_header("Authorization", f"Basic {token}")

    token = _require_secret(auth.env_var, env_reader)
    rendered_token = _render_token(auth.token_prefix, token)

    if auth.type == "bearer_token":
        header_name = auth.header_name or "Authorization"
        token_prefix = auth.token_prefix or "Bearer"
        return request.with_header(header_name, _render_token(token_prefix, token))

    if auth.type == "header_token":
        header_name = auth.header_name or "Authorization"
        return request.with_header(header_name, rendered_token)

    if auth.type == "query_token":
        query_param = auth.query_param or "token"
        return request.with_params({query_param: rendered_token})

    raise ValueError(f"Unsupported auth type: {auth.type}")


def _require_secret(
    env_var: str | None,
    env_reader: Callable[[str], str | None],
) -> str:
    if env_var is None or not env_var.strip():
        raise AuthResolutionError("Configured auth env var must not be empty")

    value = env_reader(env_var)
    if value is None or not value.strip():
        raise AuthResolutionError(
            f"Required auth secret {env_var!r} is not available in the environment"
        )
    return value.strip()


def _render_token(prefix: str | None, token: str) -> str:
    normalized_prefix = (prefix or "").strip()
    if not normalized_prefix:
        return token
    return f"{normalized_prefix} {token}"

