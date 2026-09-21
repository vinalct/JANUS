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
from urllib.request import (
    HTTPDefaultErrorHandler,
    HTTPErrorProcessor,
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    OpenerDirector,
    ProxyHandler,
    Request,
)

from janus.models import AuthConfig
from janus.models.config.constants import DEFAULT_MAX_REDIRECTS, MAX_REDIRECTS_CEILING
from janus.strategies.common import _freeze_string_mapping, _stringify_mapping
from janus.utils.logging import redact_url

HTTP_STATUS_MIN = 100
HTTP_STATUS_SUCCESS = 200
HTTP_STATUS_REDIRECT = 300
HTTP_STATUS_CLIENT_ERROR = 400
SUPPORTED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})
HTTP_DEFAULT_PORT = 80
HTTPS_DEFAULT_PORT = 443


@dataclass(frozen=True, slots=True)
class ApiRequest:
    """Normalized HTTP request used by the reusable API strategy."""

    method: str
    url: str
    timeout_seconds: int
    headers: tuple[tuple[str, str], ...] = ()
    params: tuple[tuple[str, str], ...] = ()
    body: bytes | None = None
    sensitive_headers: tuple[str, ...] = ()
    sensitive_params: tuple[str, ...] = ()
    max_redirects: int = DEFAULT_MAX_REDIRECTS

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

    def with_sensitive_header(self, name: str, value: str) -> ApiRequest:
        request = self.with_header(name, value)
        sensitive_headers = tuple(
            existing
            for existing in request.sensitive_headers
            if existing.lower() != name.lower()
        )
        return replace(request, sensitive_headers=(*sensitive_headers, name))

    def with_url(self, url: str) -> ApiRequest:
        return replace(self, url=url)

    def with_params(self, params: Mapping[str, Any]) -> ApiRequest:
        merged_params = self.params_as_dict()
        merged_params.update(_stringify_mapping(params))
        return replace(self, params=_freeze_string_mapping(merged_params))

    def with_sensitive_param(self, name: str, value: str) -> ApiRequest:
        request = self.with_params({name: value})
        sensitive_params = tuple(
            existing for existing in request.sensitive_params if existing != name
        )
        return replace(request, sensitive_params=(*sensitive_params, name))

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
        if self.status_code < HTTP_STATUS_MIN:
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


class RedirectRefused(URLError):
    """Raised when a redirect crosses a forbidden transport boundary."""


class RedirectLimitExceeded(URLError):
    """Raised when a request exceeds its configured redirect-hop limit."""


@dataclass(frozen=True, slots=True)
class RedirectPolicy:
    """Credential and hop policy carried by each urllib request."""

    origin: tuple[str, str, int]
    sensitive_headers: frozenset[str]
    sensitive_params: frozenset[str]
    max_redirects: int


class _JanusRequest(Request):
    janus_redirect_policy: RedirectPolicy
    janus_redirect_hops: int


def _origin_of(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    default_port = HTTPS_DEFAULT_PORT if scheme == "https" else HTTP_DEFAULT_PORT
    return scheme, (parsed.hostname or "").lower(), parsed.port or default_port


def _credentials_may_travel(
    origin: tuple[str, str, int], target: tuple[str, str, int]
) -> bool:
    """Return whether credentials may travel from the original request to a target."""
    if origin == target:
        return True
    return (
        origin[0] == "http"
        and origin[2] == HTTP_DEFAULT_PORT
        and target[0] == "https"
        and target[2] == HTTPS_DEFAULT_PORT
        and origin[1] == target[1]
    )


def _strip_sensitive(request: Request, policy: RedirectPolicy) -> None:
    for headers in (request.headers, request.unredirected_hdrs):
        for name in tuple(headers):
            if (
                name.lower() == "authorization"
                or name.lower() in policy.sensitive_headers
            ):
                del headers[name]

    parsed = urlsplit(request.full_url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(name, value) for name, value in query if name not in policy.sensitive_params]
    if filtered != query:
        request.full_url = urlunsplit(parsed._replace(query=urlencode(filtered)))


def _carry_sensitive_params(
    source: Request, target: Request, policy: RedirectPolicy
) -> None:
    source_query = parse_qsl(urlsplit(source.full_url).query, keep_blank_values=True)
    target_url = urlsplit(target.full_url)
    target_query = parse_qsl(target_url.query, keep_blank_values=True)
    target_names = {name for name, _value in target_query}
    carried = [
        (name, value)
        for name, value in source_query
        if name in policy.sensitive_params and name not in target_names
    ]
    if carried:
        target.full_url = urlunsplit(
            target_url._replace(query=urlencode([*target_query, *carried]))
        )


class JanusRedirectHandler(HTTPRedirectHandler):
    """Apply JANUS's per-request redirect and credential policy."""

    max_redirections = MAX_REDIRECTS_CEILING
    max_repeats = MAX_REDIRECTS_CEILING

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        policy = getattr(req, "janus_redirect_policy", None)
        if policy is None:
            raise RedirectRefused("redirect without a JANUS policy")

        hops = getattr(req, "janus_redirect_hops", 0) + 1
        if hops > policy.max_redirects:
            raise RedirectLimitExceeded(
                "redirect exceeded "
                f"access.limits.max_redirects={policy.max_redirects}"
            )

        target_scheme = urlsplit(newurl).scheme.lower()
        if target_scheme not in SUPPORTED_URL_SCHEMES:
            raise RedirectRefused(f"redirect to scheme {target_scheme!r} refused")
        if urlsplit(req.full_url).scheme.lower() == "https" and target_scheme == "http":
            raise RedirectRefused("https → http downgrade refused")

        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        if _credentials_may_travel(policy.origin, _origin_of(newurl)):
            _carry_sensitive_params(req, redirected, policy)
        else:
            _strip_sensitive(redirected, policy)

        redirected.janus_redirect_policy = policy
        redirected.janus_redirect_hops = hops
        return redirected


class ApiTransport(Protocol):
    """Small transport contract so tests can inject deterministic fake clients."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def send(self, request: ApiRequest) -> ApiResponse: ...


def _require_supported_scheme(url: str) -> None:
    scheme = urlsplit(url).scheme.lower()
    if scheme not in SUPPORTED_URL_SCHEMES:
        raise ApiTransportError(
            f"Refusing to open {redact_url(url)!r}: scheme "
            f"{scheme or '<none>'!r} is not http or https"
        )


def _build_opener(
    context: ssl.SSLContext,
    *,
    redirect_handler: HTTPRedirectHandler,
) -> OpenerDirector:
    """Build an opener with exactly the handlers JANUS needs."""
    opener = OpenerDirector()
    for handler in (
        ProxyHandler(),
        HTTPHandler(),
        HTTPSHandler(context=context),
        HTTPDefaultErrorHandler(),
        redirect_handler,
        HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


@dataclass(slots=True)
class UrllibApiTransport:
    """Stdlib-backed HTTP transport with an explicit open/close lifecycle."""

    opener: OpenerDirector | None = None
    ca_bundle_path: str | None = None

    def open(self) -> None:
        if self.opener is None:
            self.opener = _build_opener(
                _build_ssl_context(self.ca_bundle_path),
                redirect_handler=JanusRedirectHandler(),
            )

    def close(self) -> None:
        self.opener = None

    def send(self, request: ApiRequest) -> ApiResponse:
        self.open()
        full_url = request.full_url()
        _require_supported_scheme(full_url)
        if self.opener is None:
            raise ApiTransportError("API transport failed to initialize urllib opener")

        urllib_request = _JanusRequest(
            full_url,
            data=request.body,
            method=request.method,
            headers=request.headers_as_dict(),
        )
        urllib_request.janus_redirect_policy = RedirectPolicy(
            origin=_origin_of(full_url),
            sensitive_headers=frozenset(
                {"authorization", *(name.lower() for name in request.sensitive_headers)}
            ),
            sensitive_params=frozenset(request.sensitive_params),
            max_redirects=request.max_redirects,
        )
        urllib_request.janus_redirect_hops = 0

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
            message = (
                f"Request failed for {redact_url(request.full_url())!r}: "
                f"{type(exc).__name__}"
            )
            if isinstance(exc, RedirectRefused | RedirectLimitExceeded):
                message = f"{message}: {exc.reason}"
            raise ApiTransportError(message) from exc


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
        return request.with_sensitive_header("Authorization", f"Basic {token}")

    token = _require_secret(auth.env_var, env_reader)
    rendered_token = _render_token(auth.token_prefix, token)

    if auth.type == "bearer_token":
        header_name = auth.header_name or "Authorization"
        token_prefix = auth.token_prefix or "Bearer"
        return request.with_sensitive_header(
            header_name, _render_token(token_prefix, token)
        )

    if auth.type == "header_token":
        header_name = auth.header_name or "Authorization"
        return request.with_sensitive_header(header_name, rendered_token)

    if auth.type == "query_token":
        query_param = auth.query_param or "token"
        return request.with_sensitive_param(query_param, rendered_token)

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

