"""The trust boundary the shared transport draws: scheme, redirect and size."""

from __future__ import annotations

import email.message
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import Request

import pytest

from janus.models import AuthConfig
from janus.strategies.http import (
    ApiClient,
    ApiRequest,
    ApiTransportError,
    JanusRedirectHandler,
    RedirectLimitExceeded,
    RedirectPolicy,
    RedirectRefused,
    UrllibApiTransport,
    inject_auth,
    send_with_retries,
)

from .conftest import CountingThrottle, FakeTransport, build_plan, make_client, recording_logger

REDIRECT_POLICY_ATTR = "janus_redirect_policy"
REDIRECT_HOPS_ATTR = "janus_redirect_hops"

# Reasons, spelled once so the -ra summary reads as a task list.
RED_UNTIL_09 = "red until: transport byte cap and stream()"

UNSUPPORTED_URLS = (
    "file:///etc/hostname",
    "data:text/plain;base64,aGk=",
    "ftp://example.invalid/x",
    "gopher://example.invalid/",
    "example.invalid/no-scheme",
)


# ---------------------------------------------------------------------------
# A fake OpenerDirector, injected through the transport's existing ``opener`` field.


class FakeUrllibResponse:
    """What ``OpenerDirector.open`` hands back: ``getcode``/``headers``/``read``."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self._status_code = status_code
        self._chunks = list(chunks or [])
        self.read_calls: list[int | None] = []
        self.closed = False
        message = email.message.Message()
        for name, value in (headers or {}).items():
            message[name] = value
        self.headers = message

    def getcode(self) -> int:
        return self._status_code

    def read(self, amount: int | None = -1) -> bytes:
        self.read_calls.append(amount)
        if not self._chunks:
            return b""
        if amount is None or amount < 0:
            body = b"".join(self._chunks)
            self._chunks = []
            return body
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeUrllibResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SpyOpener:
    """Records every ``open`` call so a test can assert the opener was never reached."""

    def __init__(self, response: FakeUrllibResponse | None = None) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self._response = response

    def open(self, request: Any, timeout: Any = None) -> FakeUrllibResponse:
        self.calls.append((request, timeout))
        if self._response is None:
            raise AssertionError("the opener was reached for a URL that must be refused")
        return self._response


def _handler_class_names(transport: UrllibApiTransport) -> list[str]:
    assert transport.opener is not None
    return [type(handler).__name__ for handler in transport.opener.handlers]


# ---------------------------------------------------------------------------
# FR-1 — scheme allow-list at the transport


@pytest.mark.parametrize("url", UNSUPPORTED_URLS)
def test_transport_refuses_each_unsupported_scheme(url):
    """A non-``http(s)`` URL is refused *before* the opener is touched."""
    spy = SpyOpener()
    transport = UrllibApiTransport(opener=spy)

    with pytest.raises(ApiTransportError):
        transport.send(ApiRequest(method="GET", url=url, timeout_seconds=5))

    assert spy.calls == [], f"the opener was reached for {url!r} before the scheme was checked"


def test_opener_installs_no_file_ftp_data_or_unknown_handler():
    """The opener carries exactly the handlers JANUS needs — never ``build_opener``'s nine."""
    transport = UrllibApiTransport()
    transport.open()

    names = _handler_class_names(transport)

    forbidden = {"FileHandler", "FTPHandler", "DataHandler", "UnknownHandler"}
    assert forbidden.isdisjoint(names), (
        f"the opener installs handlers for schemes JANUS does not speak: "
        f"{sorted(forbidden.intersection(names))}"
    )
    for required in (
        "HTTPHandler",
        "HTTPSHandler",
        "HTTPDefaultErrorHandler",
        "HTTPErrorProcessor",
    ):
        assert required in names, f"{required} is missing from {names}"


def test_the_one_redirect_handler_is_the_janus_one():
    """Exactly one redirect handler, and it is JANUS's — not CPython's header-copying default."""
    from urllib.request import HTTPRedirectHandler

    transport = UrllibApiTransport()
    transport.open()
    assert transport.opener is not None

    redirect_handlers = [
        handler
        for handler in transport.opener.handlers
        if isinstance(handler, HTTPRedirectHandler)
    ]

    assert len(redirect_handlers) == 1, (
        f"expected one redirect handler, found {[type(h).__name__ for h in redirect_handlers]}"
    )
    assert isinstance(redirect_handlers[0], JanusRedirectHandler)


def test_https_handler_still_carries_the_janus_ssl_context():
    """Green on arrival: a pin, not a red test.

    TLS verification is the one thing the audit found already correct; FR-1 rebuilds the
    opener by hand, and this is what makes a silently weakened context fail the suite.
    """
    transport = UrllibApiTransport()
    transport.open()
    assert transport.opener is not None

    https_handlers = [
        handler
        for handler in transport.opener.handlers
        if type(handler).__name__ == "HTTPSHandler"
    ]
    assert len(https_handlers) == 1

    context = https_handlers[0]._context
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_openlineage_http_transport_inherits_the_scheme_rule(tmp_path):
    """The lineage POST composes the same transport, so it inherits the boundary."""

    from janus.observability.openlineage.settings import HttpTransportSettings
    from janus.observability.openlineage.transport import (
        HttpOpenLineageTransport,
        OpenLineageEmissionOutcome,
    )

    target = tmp_path / "lineage-receiver.json"
    target.write_text('{"local":"file"}', encoding="utf-8")

    transport = HttpOpenLineageTransport(
        settings=HttpTransportSettings(url=f"file://{tmp_path}", endpoint=target.name)
    )

    result = transport.send({"eventType": "START"}, budget_seconds=5.0)

    assert result.outcome is OpenLineageEmissionOutcome.FAILED
    assert result.step == "request"
    assert result.exception_type == "ApiTransportError", (
        "the lineage transport read a local file and failed on the response shape, not on "
        "the scheme"
    )


# ---------------------------------------------------------------------------
# FR-3 — redirect policy 


ORIGIN_URL = "https://api.example.gov.br/v1/a?chave=Q&pagina=1"
SENSITIVE_HEADER = "chave-api-dados"
SENSITIVE_PARAM = "chave"

#: ``(label, from_url, to_url, credentials_survive)`` for every followed hop.
FOLLOWED_REDIRECTS = (
    ("relative_same_origin", ORIGIN_URL, urljoin(ORIGIN_URL, "/v1/b"), True),
    ("absolute_same_origin", ORIGIN_URL, "https://api.example.gov.br/v1/b", True),
    ("different_host", ORIGIN_URL, "https://other.example.gov.br/v1/b", False),
    ("different_port", ORIGIN_URL, "https://api.example.gov.br:8443/v1/b", False),
    (
        "canonical_upgrade",
        "http://api.example.gov.br/v1/a?chave=Q&pagina=1",
        "https://api.example.gov.br/v1/b",
        True,
    ),
)

#: ``(label, from_url, to_url)`` for every hop the handler must refuse outright.
REFUSED_REDIRECTS = (
    ("downgrade", ORIGIN_URL, "http://api.example.gov.br/v1/b"),
    ("foreign_scheme", ORIGIN_URL, "ftp://api.example.gov.br/v1/b"),
)


def _redirect_policy(from_url: str, *, max_redirects: int = 5):
    """Build the per-request policy attaches, from the URL ``send()`` was given."""
    parsed = urlsplit(from_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return RedirectPolicy(
        origin=(parsed.scheme, (parsed.hostname or "").lower(), port),
        sensitive_headers=frozenset({"authorization", SENSITIVE_HEADER}),
        sensitive_params=frozenset({SENSITIVE_PARAM}),
        max_redirects=max_redirects,
    )


def _redirected_request(from_url: str, *, max_redirects: int = 5, hops: int = 0) -> Request:
    """A urllib request shaped the way ``send()`` shapes one, policy attached."""
    request = Request(
        from_url,
        method="GET",
        headers={
            "Authorization": "Bearer T",
            SENSITIVE_HEADER: "K",
            "Accept": "application/json",
        },
    )
    setattr(request, REDIRECT_POLICY_ATTR, _redirect_policy(from_url, max_redirects=max_redirects))
    setattr(request, REDIRECT_HOPS_ATTR, hops)
    return request


def _redirect(from_url: str, to_url: str, *, max_redirects: int = 5, hops: int = 0):
    headers = email.message.Message()
    headers["Location"] = to_url
    return JanusRedirectHandler().redirect_request(
        _redirected_request(from_url, max_redirects=max_redirects, hops=hops),
        None,
        302,
        "Found",
        headers,
        to_url,
    )


@pytest.mark.parametrize(
    ("label", "from_url", "to_url", "credentials_survive"),
    FOLLOWED_REDIRECTS,
    ids=[row[0] for row in FOLLOWED_REDIRECTS],
)
def test_redirect_carries_credentials_only_within_the_origin(
    label, from_url, to_url, credentials_survive
):
    """Origin is scheme + host + port; ``http → https`` on one host is the single exception."""
    del label
    redirected = _redirect(from_url, to_url)

    assert redirected is not None
    assert redirected.get_header("Accept") == "application/json", (
        "a non-sensitive header must travel on every followed hop"
    )

    header_names = {name.lower() for name in redirected.headers}
    query = parse_qs(urlsplit(redirected.full_url).query)

    if credentials_survive:
        assert "authorization" in header_names
        assert SENSITIVE_HEADER in header_names
        assert query.get(SENSITIVE_PARAM) == ["Q"]
    else:
        assert "authorization" not in header_names
        assert SENSITIVE_HEADER not in header_names
        assert SENSITIVE_PARAM not in query


@pytest.mark.parametrize(
    ("label", "from_url", "to_url"),
    REFUSED_REDIRECTS,
    ids=[row[0] for row in REFUSED_REDIRECTS],
)
def test_redirect_to_a_downgrade_or_a_foreign_scheme_is_refused(label, from_url, to_url):
    """§8 Q2: a silent ``https → http`` downgrade is not worth a public-data mirror.

    The stdlib would have followed the ``ftp:`` target — ``http_error_302`` permits
    ``('http', 'https', 'ftp', '')`` — so FR-3(b) is JANUS's own check, not an assertion
    about CPython.
    """
    del label
    with pytest.raises(RedirectRefused):
        _redirect(from_url, to_url)


def test_the_hop_after_the_cap_is_refused():
    """The per-request cap must bite before CPython's own ``max_redirections`` of 10."""
    with pytest.raises(RedirectLimitExceeded):
        _redirect(ORIGIN_URL, "https://api.example.gov.br/v1/b", max_redirects=5, hops=5)


def test_the_last_hop_within_the_cap_is_still_followed():
    """The cap counts hops, not attempts: the fifth of five must go through."""
    redirected = _redirect(ORIGIN_URL, "https://api.example.gov.br/v1/b", max_redirects=5, hops=4)

    assert redirected is not None
    assert getattr(redirected, REDIRECT_HOPS_ATTR) == 5


# ---------------------------------------------------------------------------
# FR-3 end to end: two loopback servers, one cross-origin hop (TASK-04)


class _RedirectingHandler(BaseHTTPRequestHandler):
    """Answers ``/cross`` and ``/same`` with a 302 and records what reaches ``/land``."""

    protocol_version = "HTTP/1.1"
    cross_origin_target = ""
    received: ClassVar[list[dict[str, str]]] = []

    def do_GET(self) -> None:  
        if self.path == "/cross":
            self._redirect(self.cross_origin_target)
        elif self.path == "/same":
            self._redirect("/land")
        else:
            type(self).received.append({k.lower(): v for k, v in self.headers.items()})
            body = json.dumps({"path": self.path}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr access log."""


def _serve() -> tuple[ThreadingHTTPServer, threading.Thread, type[_RedirectingHandler]]:
    handler = type("_ScopedRedirectingHandler", (_RedirectingHandler,), {"received": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, handler


def test_a_real_cross_origin_redirect_drops_the_credential_and_a_same_origin_one_keeps_it():
    """Both servers bound to 127.0.0.1 and torn down in ``finally`` — hermetic, per CONTRIBUTING."""
    server_a, thread_a, handler_a = _serve()
    server_b, thread_b, handler_b = _serve()
    try:
        port_b = server_b.server_address[1]
        handler_a.cross_origin_target = f"http://127.0.0.1:{port_b}/land"
        port_a = server_a.server_address[1]

        transport = UrllibApiTransport()
        try:
            for path in ("/cross", "/same"):
                request = ApiRequest(
                    method="GET",
                    url=f"http://127.0.0.1:{port_a}{path}",
                    timeout_seconds=10,
                    headers=(("Authorization", "Bearer T"),),
                )
                assert transport.send(request).status_code == 200
        finally:
            transport.close()

        assert len(handler_b.received) == 1, "the cross-origin hop never landed on server B"
        assert "authorization" not in handler_b.received[0], (
            "the API key travelled to a different origin — SEC-02, verbatim"
        )
        assert len(handler_a.received) == 1, "the same-origin hop never landed on server A"
        assert handler_a.received[0].get("authorization") == "Bearer T"
    finally:
        for server, thread in ((server_a, thread_a), (server_b, thread_b)):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


# ---------------------------------------------------------------------------
# FR-3 — inject_auth records *names*, never values (TASK-04)

AUTH_CASES = (
    (
        "basic",
        AuthConfig(type="basic", username_env_var="JANUS_U", password_env_var="JANUS_P"),
        {"JANUS_U": "user", "JANUS_P": "pass"},
        ("authorization",),
        (),
    ),
    (
        "bearer_token",
        AuthConfig(type="bearer_token", env_var="JANUS_T"),
        {"JANUS_T": "SECRETVALUE"},
        ("authorization",),
        (),
    ),
    (
        "header_token",
        AuthConfig(type="header_token", env_var="JANUS_T", header_name=SENSITIVE_HEADER),
        {"JANUS_T": "SECRETVALUE"},
        (SENSITIVE_HEADER,),
        (),
    ),
    (
        "query_token",
        AuthConfig(type="query_token", env_var="JANUS_T", query_param=SENSITIVE_PARAM),
        {"JANUS_T": "SECRETVALUE"},
        (),
        (SENSITIVE_PARAM,),
    ),
)


@pytest.mark.parametrize(
    ("label", "auth", "env", "expected_headers", "expected_params"),
    AUTH_CASES,
    ids=[row[0] for row in AUTH_CASES],
)
def test_inject_auth_records_what_it_injected_by_name_only(
    label, auth, env, expected_headers, expected_params
):
    """The transport must know *which* header to strip without ever knowing the value."""
    del label
    request = inject_auth(
        ApiRequest(method="GET", url="https://api.example.gov.br/v1/a", timeout_seconds=30),
        auth,
        env_reader=env.get,
    )

    recorded_headers = {name.lower() for name in request.sensitive_headers}
    recorded_params = {name.lower() for name in request.sensitive_params}

    assert recorded_headers == set(expected_headers)
    assert recorded_params == set(expected_params)

    secrets = {value for value in env.values()}
    recorded = recorded_headers | recorded_params
    assert secrets.isdisjoint(recorded), "inject_auth recorded a secret value, not a name"


# ---------------------------------------------------------------------------
# FR-4 — the byte cap


@pytest.mark.xfail(strict=True, reason=RED_UNTIL_09)
def test_a_declared_content_length_over_the_cap_is_refused_before_the_body_is_read():
    """The preflight is the whole point: an 8 GiB declaration must cost zero bytes of RAM."""
    from janus.strategies.http import ApiResponseTooLargeError

    response = FakeUrllibResponse(headers={"Content-Length": "10"}, chunks=[b"0123456789"])
    transport = UrllibApiTransport(opener=SpyOpener(response))

    with pytest.raises(ApiResponseTooLargeError) as excinfo:
        transport.send(
            ApiRequest(
                method="GET",
                url="https://api.example.gov.br/v1/a",
                timeout_seconds=5,
                max_payload_bytes=5,
            )
        )

    assert response.read_calls == [], "the body was read despite an over-cap Content-Length"
    message = str(excinfo.value)
    assert "max_payload_bytes=5" in message
    assert "10" in message


@pytest.mark.xfail(strict=True, reason=RED_UNTIL_09)
def test_a_chunked_body_is_aborted_the_moment_it_passes_the_cap():
    """No ``Content-Length`` is not a licence to read forever; the counter is the backstop."""
    from janus.strategies.http import ApiResponseTooLargeError

    response = FakeUrllibResponse(chunks=[b"aaaa", b"bbbb", b"cccc"])
    transport = UrllibApiTransport(opener=SpyOpener(response))

    with pytest.raises(ApiResponseTooLargeError) as excinfo:
        transport.send(
            ApiRequest(
                method="GET",
                url="https://api.example.gov.br/v1/a",
                timeout_seconds=5,
                max_payload_bytes=10,
            )
        )

    message = str(excinfo.value)
    assert "max_payload_bytes=10" in message
    assert "12" in message, f"the message must name the bytes seen; got {message!r}"


@pytest.mark.xfail(strict=True, reason=RED_UNTIL_09)
def test_a_body_exactly_at_the_cap_is_allowed():
    """The cap is a ceiling, not a strict bound — an off-by-one here dead-letters a valid run."""
    response = FakeUrllibResponse(
        headers={"Content-Length": "8"}, chunks=[b"aaaa", b"bbbb"]
    )
    transport = UrllibApiTransport(opener=SpyOpener(response))

    result = transport.send(
        ApiRequest(
            method="GET",
            url="https://api.example.gov.br/v1/a",
            timeout_seconds=5,
            max_payload_bytes=8,
        )
    )

    assert result.body == b"aaaabbbb"


@pytest.mark.xfail(strict=True, reason=RED_UNTIL_09)
def test_an_http_error_body_is_capped_too():
    """``exc.read()`` is the second unbounded read in ``send`` — a 404 can carry 2 MiB."""
    from urllib.error import HTTPError

    from janus.strategies.http import ApiResponseTooLargeError

    headers = email.message.Message()
    headers["Content-Type"] = "text/html"

    class _RaisingOpener:
        def open(self, request: Any, timeout: Any = None) -> Any:
            raise HTTPError(
                "https://api.example.gov.br/v1/a", 404, "Not Found", headers, _BigBody()
            )

    class _BigBody:
        def read(self, amount: int | None = -1) -> bytes:
            size = 2 * 1024 * 1024 if amount is None or amount < 0 else amount
            return b"x" * size

        def close(self) -> None:
            return None

    with pytest.raises(ApiResponseTooLargeError):
        UrllibApiTransport(opener=_RaisingOpener()).send(
            ApiRequest(
                method="GET",
                url="https://api.example.gov.br/v1/a",
                timeout_seconds=5,
                max_payload_bytes=64 * 1024,
            )
        )


@pytest.mark.xfail(strict=True, reason=RED_UNTIL_09)
def test_stream_hands_back_a_reader_that_refuses_to_pass_the_cap():
    """``stream()`` is what lets the file family never hold a payload in memory (FR-4)."""
    from janus.strategies.http import ApiResponseTooLargeError

    response = FakeUrllibResponse(chunks=[b"aaaa", b"bbbb", b"cccc"])
    transport = UrllibApiTransport(opener=SpyOpener(response))

    streamed = transport.stream(
        ApiRequest(
            method="GET",
            url="https://api.example.gov.br/v1/a",
            timeout_seconds=5,
            max_payload_bytes=6,
        )
    )
    try:
        with pytest.raises(ApiResponseTooLargeError):
            streamed.body.read()
    finally:
        streamed.close()


@pytest.mark.xfail(strict=True, reason=RED_UNTIL_09)
def test_an_over_cap_response_is_not_retried(tmp_path):
    """Re-downloading cannot shrink a payload: one attempt, no sleep."""
    import janus.strategies.api.requests as api_requests
    from janus.strategies.http import ApiResponseTooLargeError

    plan = build_plan("api", tmp_path, source_id="cap_not_retried", retry_max_attempts=3)
    client, transport = make_client(
        [ApiResponseTooLargeError("https://example.invalid/records", limit_bytes=5)] * 3
    )
    slept: list[float] = []
    logger, _stream = recording_logger()

    with pytest.raises(api_requests.ApiStrategyError):
        send_with_retries(
            plan,
            client,
            ApiRequest(method="GET", url="https://example.invalid/records", timeout_seconds=5),
            CountingThrottle(),
            logger,
            policy=api_requests._RETRY_POLICY,
            sleeper=slept.append,
        )

    assert len(transport.requests) == 1, "a too-large response must not be re-sent"
    assert slept == [], "a non-retryable transport error must not consume a backoff"


def test_a_plain_transport_error_is_still_retried_to_max_attempts(tmp_path):
    """Green on arrival: a pin, not a red test."""
    import janus.strategies.api.requests as api_requests

    plan = build_plan("api", tmp_path, source_id="plain_error_retried", retry_max_attempts=3)
    transport = FakeTransport([ApiTransportError("boom")] * 3)
    client = ApiClient(transport)
    slept: list[float] = []
    logger, _stream = recording_logger()

    with pytest.raises(api_requests.ApiStrategyError):
        send_with_retries(
            plan,
            client,
            ApiRequest(method="GET", url="https://example.invalid/records", timeout_seconds=5),
            CountingThrottle(),
            logger,
            policy=api_requests._RETRY_POLICY,
            sleeper=slept.append,
        )

    assert len(transport.requests) == 3
    assert len(slept) == 2


def test_a_successful_send_through_a_fake_opener_is_unchanged():
    """Green on arrival: the shape every case above perturbs, pinned once.

    Status, body and headers come back exactly as they do today, so a failure in this test
    localizes a regression to the transport rather than to any one boundary rule.
    """
    response = FakeUrllibResponse(
        status_code=200,
        headers={"Content-Type": "application/json", "Content-Length": "9"},
        chunks=[b'{"ok":1}\n'],
    )
    transport = UrllibApiTransport(opener=SpyOpener(response))

    result = transport.send(
        ApiRequest(method="GET", url="https://api.example.gov.br/v1/a", timeout_seconds=5)
    )

    assert result.status_code == 200
    assert result.body == b'{"ok":1}\n'
    assert result.headers_as_dict()["Content-Type"] == "application/json"
