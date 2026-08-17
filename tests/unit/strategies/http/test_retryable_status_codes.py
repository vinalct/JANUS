"""Which failures are worth re-sending is the source's declaration, not a constant."""

from __future__ import annotations

import pytest

import janus.strategies.api.core as api_core
import janus.strategies.api.requests as api_requests
from janus.strategies.http import RETRYABLE_STATUS_CODES, send_with_retries

from .conftest import (
    CountingThrottle,
    ResponseSpec,
    build_plan,
    json_response,
    make_client,
    make_request,
)

REQUEST_URL = "https://example.invalid/records"


def _send(plan, script, *, terminal=None):
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
        None,
        policy=api_requests._RETRY_POLICY,
        sleeper=sleeps.append,
        decode=lambda response: executor.decode_payload(plan, response),
        payload_error_types=(api_core.ApiPayloadError,),
        **kwargs,
    )
    return result, transport, sleeps


def test_status_outside_the_default_set_is_raised_on_the_first_attempt(tmp_path):
    """The pre-fix behaviour, pinned: an undeclared 400 never reaches attempt 2."""
    plan = build_plan("api", tmp_path, source_id="undeclared_400", retry_max_attempts=4)

    with pytest.raises(api_core.ApiResponseError):
        _send(plan, [ResponseSpec(status_code=400)])


def test_a_declared_status_is_retried_and_the_recovered_page_is_returned(tmp_path):
    """The fix: a transient 400 costs one retry instead of the whole run."""
    plan = build_plan(
        "api",
        tmp_path,
        source_id="declared_400_recovers",
        retry_max_attempts=4,
        retry_retryable_status_codes=[400, 408, 429, 500, 502, 503, 504],
    )

    (response, payload, attempts), transport, sleeps = _send(
        plan,
        [
            ResponseSpec(status_code=400, body=b'{"message": "transient"}'),
            json_response({"records": [{"id": 1}]}),
        ],
    )

    assert response.status_code == 200
    assert payload == {"records": [{"id": 1}]}
    assert attempts == 2
    assert len(transport.requests) == 2
    assert len(sleeps) == 1


def test_a_declared_status_still_fails_once_max_attempts_is_spent(tmp_path):
    """Declaring a status retryable buys attempts, not immunity."""
    plan = build_plan(
        "api",
        tmp_path,
        source_id="declared_400_exhausts",
        retry_max_attempts=3,
        retry_retryable_status_codes=[400],
    )

    with pytest.raises(api_core.ApiResponseError):
        _send(plan, [ResponseSpec(status_code=400)] * 3)


def test_the_default_declaration_leaves_the_previous_classification_intact(tmp_path):
    """A source that declares nothing behaves exactly as it did before the option existed."""
    plan = build_plan("api", tmp_path, source_id="default_retryable", retry_max_attempts=2)

    assert frozenset(plan.source_config.extraction.retry.retryable_status_codes) == (
        RETRYABLE_STATUS_CODES
    )

    (response, _payload, attempts), transport, _sleeps = _send(
        plan,
        [ResponseSpec(status_code=503), json_response({"records": []})],
    )

    assert response.status_code == 200
    assert attempts == 2
    assert len(transport.requests) == 2


def test_an_empty_declaration_disables_status_retries_entirely(tmp_path):
    """``retryable_status_codes: []`` is a policy — every non-2xx fails on attempt one."""
    plan = build_plan(
        "api",
        tmp_path,
        source_id="no_status_retries",
        retry_max_attempts=5,
        retry_retryable_status_codes=[],
    )

    with pytest.raises(api_core.ApiResponseError):
        _send(plan, [ResponseSpec(status_code=503)])


def test_a_terminal_status_still_wins_over_a_retryable_declaration(tmp_path):
    """The documented branch order holds even when a caller declares the same status twice.

    The config layer rejects a *declared* overlap (see ``test_retry_status_contract``), so
    this uses a status the config cannot name as past-end — ``terminal_status_codes`` is a
    caller argument, and the precedence must not depend on validation having run.
    """
    plan = build_plan(
        "api",
        tmp_path,
        source_id="terminal_beats_retryable",
        retry_max_attempts=4,
        retry_retryable_status_codes=[503],
    )

    (response, payload, attempts), transport, sleeps = _send(
        plan,
        [ResponseSpec(status_code=503)],
        terminal=frozenset({503}),
    )

    assert response.status_code == 503
    assert payload is None
    assert attempts == 1
    assert len(transport.requests) == 1
    assert sleeps == []
