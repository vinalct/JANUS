"""Focused contract tests for value-based response-body secret scrubbing."""

from __future__ import annotations

from urllib.parse import quote, quote_plus

import pytest

import janus.strategies.files.download as file_download
from janus.models import AuthConfig
from janus.strategies.files import FileDownloadError
from janus.strategies.http import (
    ApiClient,
    ApiRequest,
    HttpRequestThrottle,
    SecretScrubber,
    inject_auth,
    stream_with_retries,
)
from janus.utils.logging import REDACTED_VALUE

from .conftest import FakeTransport, ResponseSpec, build_plan

SECRET = "portal/key with space"


def test_scrubs_raw_and_url_encoded_forms_in_bytes_and_text():
    scrubber = SecretScrubber()
    scrubber.register(SECRET)
    variants = (SECRET, quote(SECRET, safe=""), quote_plus(SECRET, safe=""))
    text = "|".join(variants)

    scrubbed_text = scrubber.scrub_text(text)
    scrubbed_bytes = scrubber.scrub_bytes(text.encode())

    for value in variants:
        assert value not in scrubbed_text
        assert value.encode() not in scrubbed_bytes
    assert REDACTED_VALUE in scrubbed_text
    assert REDACTED_VALUE.encode() in scrubbed_bytes


def test_inject_auth_registers_raw_and_rendered_bearer_forms():
    request = inject_auth(
        ApiRequest(
            method="GET",
            url="https://api.example.gov.br/v1/records",
            timeout_seconds=30,
        ),
        AuthConfig(type="bearer_token", env_var="TOKEN"),
        env_reader=lambda _name: SECRET,
    )

    assert request.scrubber is not None
    body = f'{{"token":"{SECRET}","header":"Bearer {SECRET}"}}'.encode()
    scrubbed = request.scrubber.scrub_bytes(body)

    assert SECRET.encode() not in scrubbed
    assert f"Bearer {SECRET}".encode() not in scrubbed
    assert REDACTED_VALUE.encode() in scrubbed
    assert "scrubber=" not in repr(request)
    assert "SecretScrubber" not in repr(request)


def test_no_match_returns_the_original_objects():
    scrubber = SecretScrubber()
    scrubber.register(SECRET)
    body = b'{"error":"not found"}'
    text = body.decode()

    assert scrubber.scrub_bytes(body) is body
    assert scrubber.scrub_text(text) is text


def test_streamed_file_error_scrubs_an_echoed_custom_header_value(tmp_path):
    plan = build_plan(
        "file",
        tmp_path,
        source_id="file_secret_scrubber",
        retry_max_attempts=1,
    )
    request = inject_auth(
        ApiRequest(
            method="GET",
            url="https://files.example.gov.br/data.csv",
            timeout_seconds=30,
        ),
        AuthConfig(
            type="header_token",
            env_var="TOKEN",
            header_name="chave-api-dados",
        ),
        env_reader=lambda _name: SECRET,
    )
    client = ApiClient(
        FakeTransport(
            [ResponseSpec(status_code=500, body=f'{{"echo":"{SECRET}"}}'.encode())]
        )
    )
    throttle = HttpRequestThrottle(
        requests_per_minute=None,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(FileDownloadError) as excinfo:
        stream_with_retries(
            plan,
            client,
            request,
            throttle,
            None,
            policy=file_download._RETRY_POLICY,
            sleeper=lambda _seconds: None,
        )

    message = str(excinfo.value)
    assert SECRET not in message
    assert REDACTED_VALUE in message
