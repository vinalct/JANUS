"""One retry loop for every HTTP-shaped strategy family.

Collapses the three copied ``_send_with_retries`` / ``_sleep_for_retry`` pairs
(api, catalog, file) into a single body. The only verified differences between
the copies become explicit parameters, never a silent fork.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from janus.models import ExecutionPlan
from janus.models.config.constants import DEFAULT_RETRYABLE_STATUS_CODES
from janus.strategies.common import _retry_delay_seconds
from janus.strategies.http.scrubber import SecretScrubber
from janus.strategies.http.throttle import HttpRequestThrottle
from janus.strategies.http.transport import (
    HTTP_STATUS_REDIRECT,
    HTTP_STATUS_SUCCESS,
    ApiClient,
    ApiNonRetryableTransportError,
    ApiRequest,
    ApiResponse,
    ApiStreamedResponse,
    ApiTransportError,
    AuthResolutionError,
)
from janus.utils.logging import StructuredLogger

#: The statuses re-sent when a source declares no ``extraction.retry.retryable_status_codes``.
#: The tuple is owned by the config layer so the loop and the validator cannot drift apart.
RETRYABLE_STATUS_CODES = frozenset(DEFAULT_RETRYABLE_STATUS_CODES)
ERROR_BODY_READ_LIMIT = 64 * 1024

_ResponseT = TypeVar("_ResponseT", ApiResponse, ApiStreamedResponse)


@dataclass(frozen=True, slots=True)
class RetryErrorPolicy:
    """Family-specific error/log wiring for the shared retry loop.

    ``transport_error_factory`` builds the strategy error raised when transport /
    auth failures exhaust the retries (called with ``str(exc)``); it is also used
    for the post-loop safety re-raise. ``response_error_factory`` builds the error
    raised for a non-retryable or exhausted non-2xx response (called with the
    :class:`ApiResponse`). ``retry_log_event`` is the warn-log event name emitted
    before each backoff sleep.
    """

    transport_error_factory: Callable[[str], Exception]
    response_error_factory: Callable[[ApiResponse], Exception]
    retry_log_event: str


def send_with_retries(
    plan: ExecutionPlan,
    client: ApiClient,
    request: ApiRequest,
    throttle: HttpRequestThrottle,
    logger: StructuredLogger | None,
    *,
    policy: RetryErrorPolicy,
    sleeper: Callable[[float], None],
    decode: Callable[[ApiResponse], Any] | None = None,
    payload_error_types: tuple[type[Exception], ...] = (),
    terminal_status_codes: frozenset[int] = frozenset(),
) -> tuple[ApiResponse, Any | None, int]:
    """Send ``request`` with the shared retry/throttle/backoff policy.

    Always returns ``(response, payload, attempts)``; ``payload`` is ``None`` when
    ``decode`` is ``None`` (the file strategy). The throttle is consulted once per
    attempt, including the first, matching every family's current pacing.

    ``terminal_status_codes`` lets a caller declare statuses that are a normal
    terminal outcome rather than a failure — a speculative page past the end of a
    paginated stream, for instance. Such a response is *returned*, not raised, so
    the knowledge stays in this one loop instead of being re-derived by callers
    sniffing ``exc.response.status_code``. The ordering is part of the contract:

    * ``2xx`` is handled first, so a success is never terminal-by-status.
    * Terminal is checked **before** the retryable set, so a status in both wins as
      terminal and is sent exactly once — retrying a definitive "there is nothing
      here" only burns rate-limit budget.
    * ``payload`` is ``None`` for a terminal return; the body is an error document,
      not a payload, and is never decoded. Callers must therefore branch on
      ``response.status_code``, never on ``payload is None``.

    The default empty set keeps every existing caller bit-for-bit unchanged.

    Which *failures* are worth re-sending is read from the plan's
    ``extraction.retry.retryable_status_codes`` rather than from a constant here: "transient"
    is a property of the upstream API, not of HTTP. A status outside that set is raised on the
    first attempt without consuming ``max_attempts``, so an API that answers a valid request
    with an unusual client error can end a whole run on one response unless the source
    declares it.
    """

    def _success(response: ApiResponse) -> Any | None:
        if decode is None:
            return None
        return decode(response)

    response, payload, attempts = _attempt_loop(
        plan,
        request,
        throttle,
        logger,
        policy=policy,
        sleeper=sleeper,
        sender=client.send,
        success_handler=_success,
        payload_error_types=payload_error_types,
        terminal_status_codes=terminal_status_codes,
        response_error_handler=lambda response: response,
        close_before_retry=lambda _response: None,
    )
    return response, payload, attempts


def stream_with_retries(
    plan: ExecutionPlan,
    client: ApiClient,
    request: ApiRequest,
    throttle: HttpRequestThrottle,
    logger: StructuredLogger | None,
    *,
    policy: RetryErrorPolicy,
    sleeper: Callable[[float], None],
    terminal_status_codes: frozenset[int] = frozenset(),
) -> tuple[ApiStreamedResponse, int]:
    """Open a streamed response with the shared retry/throttle/backoff policy."""
    response, _payload, attempts = _attempt_loop(
        plan,
        request,
        throttle,
        logger,
        policy=policy,
        sleeper=sleeper,
        sender=client.stream,
        success_handler=lambda _response: None,
        payload_error_types=(),
        terminal_status_codes=terminal_status_codes,
        response_error_handler=lambda response: response.materialize(limit=ERROR_BODY_READ_LIMIT),
        close_before_retry=ApiStreamedResponse.close,
    )
    return response, attempts


def _attempt_loop(
    plan: ExecutionPlan,
    request: ApiRequest,
    throttle: HttpRequestThrottle,
    logger: StructuredLogger | None,
    *,
    policy: RetryErrorPolicy,
    sleeper: Callable[[float], None],
    sender: Callable[[ApiRequest], _ResponseT],
    success_handler: Callable[[_ResponseT], Any],
    payload_error_types: tuple[type[Exception], ...],
    terminal_status_codes: frozenset[int],
    response_error_handler: Callable[[_ResponseT], ApiResponse],
    close_before_retry: Callable[[_ResponseT], None],
) -> tuple[_ResponseT, Any | None, int]:
    """Run the one sanctioned HTTP attempt loop for materialized and streamed calls."""
    retry_config = plan.source_config.extraction.retry
    retryable_status_codes = frozenset(retry_config.retryable_status_codes)
    last_transport_error: Exception | None = None

    for attempt in range(1, retry_config.max_attempts + 1):
        throttle.wait_for_turn()
        try:
            response = sender(request)
        except ApiNonRetryableTransportError as exc:
            raise policy.transport_error_factory(str(exc)) from exc
        except (ApiTransportError, AuthResolutionError) as exc:
            last_transport_error = exc
            if attempt == retry_config.max_attempts:
                raise policy.transport_error_factory(str(exc)) from exc
            _sleep_for_retry(
                plan, attempt, response=None, logger=logger, policy=policy, sleeper=sleeper
            )
            continue

        if HTTP_STATUS_SUCCESS <= response.status_code < HTTP_STATUS_REDIRECT:
            try:
                payload = success_handler(response)
            except payload_error_types:
                if attempt == retry_config.max_attempts:
                    raise
                close_before_retry(response)
                _sleep_for_retry(
                    plan,
                    attempt,
                    response=response,
                    logger=logger,
                    policy=policy,
                    sleeper=sleeper,
                )
                continue
            return response, payload, attempt

        if response.status_code in terminal_status_codes:
            if logger is not None:
                logger.info(
                    "http_terminal_status_returned",
                    status_code=response.status_code,
                    attempt=attempt,
                )
            return response, None, attempt

        if (
            response.status_code not in retryable_status_codes
            or attempt == retry_config.max_attempts
        ):
            try:
                error_response = response_error_handler(response)
            except ApiNonRetryableTransportError as exc:
                raise policy.transport_error_factory(str(exc)) from exc
            raise policy.response_error_factory(
                _scrubbed(error_response, request.scrubber)
            )

        close_before_retry(response)
        _sleep_for_retry(
            plan, attempt, response=response, logger=logger, policy=policy, sleeper=sleeper
        )

    if last_transport_error is not None:
        raise policy.transport_error_factory(str(last_transport_error)) from last_transport_error
    raise policy.transport_error_factory("Retry loop exited without a response or error")


def _scrubbed(response: ApiResponse, scrubber: SecretScrubber | None) -> ApiResponse:
    """Return a response safe to render in a persisted or logged family error."""
    if scrubber is None:
        return response
    body = scrubber.scrub_bytes(response.body)
    if body is response.body:
        return response
    return replace(response, body=body)


def _sleep_for_retry(
    plan: ExecutionPlan,
    attempt: int,
    *,
    response: ApiResponse | ApiStreamedResponse | None,
    logger: StructuredLogger | None,
    policy: RetryErrorPolicy,
    sleeper: Callable[[float], None],
) -> None:
    delay = _retry_delay_seconds(plan, attempt, response)
    if logger is not None:
        logger.warning(
            policy.retry_log_event,
            attempt=attempt,
            delay_seconds=delay,
            status_code=response.status_code if response is not None else None,
        )
    sleeper(delay)
