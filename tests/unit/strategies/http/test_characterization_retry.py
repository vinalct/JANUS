"""Characterization: the per-family `_send_with_retries` loops.
"""

from __future__ import annotations

import pytest

from janus.strategies.api.http import ApiTransportError, AuthResolutionError

from .conftest import (
    CountingThrottle,
    ResponseSpec,
    build_plan,
    build_strategy,
    json_response,
    make_client,
    make_request,
    recording_logger,
    warning_events,
)

REQUEST_URL = "https://example.invalid/records"


def _send(family, plan, script, *, request=None, logger=None):
    """Run one `_send_with_retries` call and return (result, transport, sleeps, throttle)."""
    sleeps: list[float] = []
    strategy = build_strategy(family, sleeper=sleeps.append)
    client, transport = make_client(script)
    throttle = CountingThrottle()
    result = family.send_with_retries(
        strategy,
        plan,
        client,
        request or make_request(REQUEST_URL),
        throttle,
        logger,
    )
    return result, transport, sleeps, throttle


def test_success_first_try_returns_attempt_one_with_family_return_shape(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_first_try")
    result, transport, sleeps, throttle = _send(family, plan, [json_response({"records": []})])

    if family.returns_payload:
        assert len(result) == 3
        response, payload, attempts = result
        assert payload == {"records": []}
    else:
        assert len(result) == 2
        response, attempts = result

    assert response.status_code == 200
    assert attempts == 1
    assert len(transport.requests) == 1
    assert throttle.calls == 1
    assert sleeps == []


def test_transport_error_then_success_retries_once(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_transport_retry")
    result, transport, sleeps, throttle = _send(
        family,
        plan,
        [ApiTransportError("connection reset"), json_response({})],
    )

    attempts = result[-1]
    assert attempts == 2
    assert len(transport.requests) == 2
    assert throttle.calls == 2  # throttle consulted once per attempt
    assert sleeps == [2.0]  # fixed backoff_seconds=2 via common._retry_delay_seconds


def test_transport_error_exhaustion_raises_family_error_with_last_message(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_transport_exhausted")
    last_error = ApiTransportError("still unreachable")

    with pytest.raises(family.transport_exhaustion_error) as excinfo:
        _send(
            family,
            plan,
            [ApiTransportError("boom-1"), ApiTransportError("boom-2"), last_error],
        )

    assert type(excinfo.value) is family.transport_exhaustion_error
    assert str(excinfo.value) == "still unreachable"
    assert excinfo.value.__cause__ is last_error


def test_auth_resolution_error_is_retried_like_a_transport_error(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_auth_retry")
    result, transport, sleeps, _throttle = _send(
        family,
        plan,
        [AuthResolutionError("secret not available"), json_response({})],
    )

    assert result[-1] == 2
    assert len(transport.requests) == 2
    assert sleeps == [2.0]


def test_auth_resolution_error_exhaustion_raises_family_error(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_auth_exhausted")
    last_error = AuthResolutionError("secret never resolved")

    with pytest.raises(family.transport_exhaustion_error) as excinfo:
        _send(
            family,
            plan,
            [
                AuthResolutionError("nope-1"),
                AuthResolutionError("nope-2"),
                last_error,
            ],
        )

    assert str(excinfo.value) == "secret never resolved"
    assert excinfo.value.__cause__ is last_error


def test_retryable_status_code_set_is_pinned(family):
    assert family.retryable_status_codes == frozenset({408, 429, 500, 502, 503, 504})


def test_retryable_status_then_success_retries(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_503_retry")
    result, transport, sleeps, _throttle = _send(
        family,
        plan,
        [ResponseSpec(status_code=503), json_response({})],
    )

    assert result[-1] == 2
    assert len(transport.requests) == 2
    assert sleeps == [2.0]


def test_non_retryable_status_raises_immediately_with_family_shape(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_404")

    with pytest.raises(family.response_error) as excinfo:
        _send(family, plan, [ResponseSpec(status_code=404)])

    assert type(excinfo.value) is family.response_error
    assert str(excinfo.value) == family.status_error_message(404, REQUEST_URL)
    if family.response_error_has_response:
        assert excinfo.value.response.status_code == 404


def test_non_retryable_status_only_sends_one_request_and_never_sleeps(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_404_once")
    sleeps: list[float] = []
    strategy = build_strategy(family, sleeper=sleeps.append)
    client, transport = make_client([ResponseSpec(status_code=404)])

    with pytest.raises(family.response_error):
        family.send_with_retries(
            strategy, plan, client, make_request(REQUEST_URL), CountingThrottle(), None
        )

    assert len(transport.requests) == 1
    assert sleeps == []


def test_non_retryable_status_message_redacts_sensitive_query_params(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_404_redacted")
    request = make_request(REQUEST_URL, params={"token": "sekret-value"})

    with pytest.raises(family.response_error) as excinfo:
        _send(family, plan, [ResponseSpec(status_code=404)], request=request)

    assert "sekret-value" not in str(excinfo.value)
    assert "REDACTED" in str(excinfo.value)


def test_retryable_status_exhaustion_raises_family_response_error(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_503_exhausted")

    with pytest.raises(family.response_error) as excinfo:
        _send(
            family,
            plan,
            [
                ResponseSpec(status_code=503),
                ResponseSpec(status_code=503),
                ResponseSpec(status_code=503),
            ],
        )

    assert type(excinfo.value) is family.response_error
    assert str(excinfo.value) == family.status_error_message(503, REQUEST_URL)


def test_retryable_status_exhaustion_sleeps_between_all_attempts(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_503_sleeps")
    sleeps: list[float] = []
    strategy = build_strategy(family, sleeper=sleeps.append)
    client, transport = make_client([ResponseSpec(status_code=503)] * 3)

    with pytest.raises(family.response_error):
        family.send_with_retries(
            strategy, plan, client, make_request(REQUEST_URL), CountingThrottle(), None
        )

    assert len(transport.requests) == 3
    assert sleeps == [2.0, 2.0]


def test_decode_failure_is_retried_then_succeeds(payload_family, tmp_path):
    plan = build_plan(
        payload_family.name, tmp_path, source_id=f"{payload_family.name}_decode_retry"
    )
    result, transport, sleeps, _throttle = _send(
        payload_family,
        plan,
        [ResponseSpec(status_code=200, body=b"not json"), json_response({"ok": 1})],
    )

    response, payload, attempts = result
    assert attempts == 2
    assert payload == {"ok": 1}
    assert len(transport.requests) == 2
    assert sleeps == [2.0]


def test_decode_failure_exhaustion_reraises_the_payload_error_itself(payload_family, tmp_path):
    plan = build_plan(
        payload_family.name, tmp_path, source_id=f"{payload_family.name}_decode_exhausted"
    )

    with pytest.raises(payload_family.payload_error) as excinfo:
        _send(
            payload_family,
            plan,
            [ResponseSpec(status_code=200, body=b"not json")] * 3,
        )

    # The payload error is re-raised as-is on exhaustion, not wrapped in the
    # family strategy error.
    assert type(excinfo.value) is payload_family.payload_error
    assert str(excinfo.value).startswith("Failed to decode")


def test_retry_after_header_is_honored_when_larger_than_backoff(family, tmp_path):
    plan = build_plan(
        family.name,
        tmp_path,
        source_id=f"{family.name}_retry_after",
        rate_limit_backoff_seconds=60,
    )
    result, _transport, sleeps, _throttle = _send(
        family,
        plan,
        [
            ResponseSpec(status_code=429, headers=(("Retry-After", "30"),)),
            json_response({}),
        ],
    )

    assert result[-1] == 2
    assert sleeps == [30.0]


def test_retry_after_is_capped_by_the_rate_limit_backoff(family, tmp_path):
    plan = build_plan(
        family.name,
        tmp_path,
        source_id=f"{family.name}_retry_after_capped",
        rate_limit_backoff_seconds=5,
    )
    _result, _transport, sleeps, _throttle = _send(
        family,
        plan,
        [
            ResponseSpec(status_code=429, headers=(("Retry-After", "30"),)),
            json_response({}),
        ],
    )

    assert sleeps == [5.0]


def test_unparseable_retry_after_is_ignored(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_retry_after_bad")
    _result, _transport, sleeps, _throttle = _send(
        family,
        plan,
        [
            ResponseSpec(status_code=429, headers=(("Retry-After", "soon"),)),
            json_response({}),
        ],
    )

    assert sleeps == [2.0]


def test_exponential_backoff_progression(family, tmp_path):
    plan = build_plan(
        family.name,
        tmp_path,
        source_id=f"{family.name}_exponential",
        retry_backoff_strategy="exponential",
    )
    result, _transport, sleeps, _throttle = _send(
        family,
        plan,
        [ApiTransportError("boom-1"), ApiTransportError("boom-2"), json_response({})],
    )

    assert result[-1] == 3
    assert sleeps == [2.0, 4.0]


def test_fixed_backoff_progression(family, tmp_path):
    plan = build_plan(
        family.name,
        tmp_path,
        source_id=f"{family.name}_fixed",
        retry_backoff_strategy="fixed",
    )
    result, _transport, sleeps, _throttle = _send(
        family,
        plan,
        [ApiTransportError("boom-1"), ApiTransportError("boom-2"), json_response({})],
    )

    assert result[-1] == 3
    assert sleeps == [2.0, 2.0]


def test_retry_emits_exactly_one_family_warning_event_with_status_code(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_log_status")
    logger, stream = recording_logger()

    _send(family, plan, [ResponseSpec(status_code=503), json_response({})], logger=logger)

    warnings = warning_events(stream)
    assert len(warnings) == 1
    assert warnings[0]["event"] == family.retry_log_event
    assert warnings[0]["fields"] == {
        "attempt": 1,
        "delay_seconds": 2.0,
        "status_code": 503,
    }


def test_retry_warning_event_has_null_status_code_for_transport_errors(family, tmp_path):
    plan = build_plan(family.name, tmp_path, source_id=f"{family.name}_log_transport")
    logger, stream = recording_logger()

    _send(family, plan, [ApiTransportError("boom"), json_response({})], logger=logger)

    warnings = warning_events(stream)
    assert len(warnings) == 1
    assert warnings[0]["event"] == family.retry_log_event
    assert warnings[0]["fields"] == {
        "attempt": 1,
        "delay_seconds": 2.0,
        "status_code": None,
    }
