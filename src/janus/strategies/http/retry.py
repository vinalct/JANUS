"""One retry loop for every HTTP-shaped strategy family.

Collapses the three copied ``_send_with_retries`` / ``_sleep_for_retry`` pairs
(api, catalog, file) into a single body. The only verified differences between
the copies become explicit parameters, never a silent fork.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from janus.models import ExecutionPlan
from janus.strategies.common import _retry_delay_seconds
from janus.strategies.http.throttle import HttpRequestThrottle
from janus.strategies.http.transport import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiTransportError,
    AuthResolutionError,
)
from janus.utils.logging import StructuredLogger

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


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
    """
    retry_config = plan.source_config.extraction.retry
    last_transport_error: Exception | None = None

    for attempt in range(1, retry_config.max_attempts + 1):
        throttle.wait_for_turn()
        try:
            response = client.send(request)
        except (ApiTransportError, AuthResolutionError) as exc:
            last_transport_error = exc
            if attempt == retry_config.max_attempts:
                raise policy.transport_error_factory(str(exc)) from exc
            _sleep_for_retry(
                plan, attempt, response=None, logger=logger, policy=policy, sleeper=sleeper
            )
            continue

        if 200 <= response.status_code < 300:
            if decode is None:
                return response, None, attempt
            try:
                payload = decode(response)
            except payload_error_types:
                if attempt == retry_config.max_attempts:
                    raise
                _sleep_for_retry(
                    plan, attempt, response=response, logger=logger, policy=policy, sleeper=sleeper
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
            response.status_code not in RETRYABLE_STATUS_CODES
            or attempt == retry_config.max_attempts
        ):
            raise policy.response_error_factory(response)

        _sleep_for_retry(
            plan, attempt, response=response, logger=logger, policy=policy, sleeper=sleeper
        )

    if last_transport_error is not None:
        raise policy.transport_error_factory(str(last_transport_error)) from last_transport_error
    raise policy.transport_error_factory("Retry loop exited without a response or error")


def _sleep_for_retry(
    plan: ExecutionPlan,
    attempt: int,
    *,
    response: ApiResponse | None,
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
