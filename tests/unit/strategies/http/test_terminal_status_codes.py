"""The shared retry loop's opt-in terminal-status return path.

A caller may declare statuses that are a normal terminal outcome (a speculative
page past the end of a stream) rather than a failure. These tests pin the
mechanism *and* its precedence against retry, plus the fact that the default
empty set leaves every existing caller untouched.
"""

from __future__ import annotations

import pytest

import janus.strategies.api.core as api_core
import janus.strategies.api.requests as api_requests
from janus.strategies.http import send_with_retries

from .conftest import (
    CountingThrottle,
    ResponseSpec,
    build_plan,
    json_response,
    logged_events,
    make_client,
    make_request,
    recording_logger,
)

REQUEST_URL = "https://example.invalid/records"


def _send(plan, script, *, terminal=None, logger=None, decode=None):
    """Call the shared loop the way the api family does, returning the observables."""
    sleeps: list[float] = []
    client, transport = make_client(script)
    throttle = CountingThrottle()
    executor = api_requests.ApiRequestExecutor(sleeper=sleeps.append)
    kwargs = {} if terminal is None else {"terminal_status_codes": terminal}
    result = send_with_retries(
        plan,
        client,
        make_request(REQUEST_URL),
        throttle,
        logger,
        policy=api_requests._RETRY_POLICY,
        sleeper=sleeps.append,
        decode=decode or (lambda response: executor.decode_payload(plan, response)),
        payload_error_types=(api_core.ApiPayloadError,),
        **kwargs,
    )
    return result, transport, sleeps, throttle


def test_terminal_status_returns_response_without_raising(tmp_path):
    plan = build_plan("api", tmp_path, source_id="terminal_404_returns")

    result, transport, sleeps, throttle = _send(
        plan,
        [ResponseSpec(status_code=404)],
        terminal=frozenset({404}),
    )

    response, payload, attempts = result
    assert response.status_code == 404
    assert payload is None
    assert attempts == 1
    assert len(transport.requests) == 1
    assert throttle.calls == 1
    assert sleeps == []


def test_terminal_status_is_not_retried_even_when_retryable(tmp_path):
    """Terminal wins over the retryable set: a definitive answer is not re-asked."""
    plan = build_plan("api", tmp_path, source_id="terminal_beats_retryable")

    result, transport, sleeps, _throttle = _send(
        plan,
        [ResponseSpec(status_code=429), json_response({"records": []})],
        terminal=frozenset({429}),
    )

    response, payload, attempts = result
    assert response.status_code == 429
    assert payload is None
    assert attempts == 1
    assert len(transport.requests) == 1
    assert sleeps == []


def test_terminal_status_does_not_decode_the_body(tmp_path):
    plan = build_plan("api", tmp_path, source_id="terminal_no_decode")
    decode_calls: list[object] = []

    def spy_decode(response):
        decode_calls.append(response)
        raise AssertionError("decode must not run for a terminal response")

    result, _transport, _sleeps, _throttle = _send(
        plan,
        [ResponseSpec(status_code=416, body=b"<html>Range Not Satisfiable</html>")],
        terminal=frozenset({416}),
        decode=spy_decode,
    )

    _response, payload, _attempts = result
    assert payload is None
    assert decode_calls == []


def test_non_terminal_error_still_raises_response_error(tmp_path):
    plan = build_plan("api", tmp_path, source_id="terminal_absent_404")

    with pytest.raises(api_core.ApiResponseError) as excinfo:
        _send(plan, [ResponseSpec(status_code=404)], terminal=frozenset({416}))

    assert excinfo.value.response.status_code == 404
    assert str(excinfo.value) == f"API request failed with status 404 for {REQUEST_URL}"


def test_retryable_status_outside_the_terminal_set_still_retries(tmp_path):
    plan = build_plan("api", tmp_path, source_id="terminal_503_still_retries")

    result, transport, sleeps, _throttle = _send(
        plan,
        [ResponseSpec(status_code=503), json_response({})],
        terminal=frozenset({404, 416}),
    )

    assert result[-1] == 2
    assert len(transport.requests) == 2
    assert sleeps == [2.0]


def test_default_call_shape_is_unchanged(tmp_path):
    """Without the keyword, 404 raises and 503 retries exactly as characterized."""
    plan = build_plan("api", tmp_path, source_id="terminal_default_shape")

    with pytest.raises(api_core.ApiResponseError):
        _send(plan, [ResponseSpec(status_code=404)])

    result, transport, sleeps, throttle = _send(
        plan,
        [ResponseSpec(status_code=503), json_response({"ok": 1})],
    )

    response, payload, attempts = result
    assert response.status_code == 200
    assert payload == {"ok": 1}
    assert attempts == 2
    assert len(transport.requests) == 2
    assert throttle.calls == 2
    assert sleeps == [2.0]


def test_terminal_status_emits_an_info_event(tmp_path):
    plan = build_plan("api", tmp_path, source_id="terminal_info_event")
    logger, stream = recording_logger()

    _send(plan, [ResponseSpec(status_code=404)], terminal=frozenset({404}), logger=logger)

    events = logged_events(stream)
    assert [event["event"] for event in events] == ["http_terminal_status_returned"]
    assert events[0]["level"] == "INFO"
    assert events[0]["fields"] == {"status_code": 404, "attempt": 1}
