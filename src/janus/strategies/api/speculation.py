"""Speculative concurrent pagination policy.

Concurrent extraction submits page N+1..N+k before page N has answered, predicting a full page
each time (see ``_predicted_next_pagination_state``). This module owns the three questions that
prediction raises: which responses mean "you went past the end", which request indexes are
speculative (and therefore allowed to end the stream quietly), and how far ahead it is legitimate
to guess.

The look-ahead ceiling caps *speculation*, it never terminates *extraction*. A ceiling derived
from a payload-reported total forbids submitting an index above it ahead of time; if the loop
legitimately reaches that index because the previous page came back full, the page is still
fetched. A stale, cached or plainly wrong ``total`` can therefore cost a little throughput, but
can never truncate a dataset.

The policy half of this module is a pure function of the config, the paginator and one
payload. The mechanics half below it — the in-flight window, its cancellation, and the
past-end conflict check — operates on already-submitted futures, but still owns no transport
and no writer: it is handed the futures, it never creates them.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard

from janus.models import ExecutionPlan
from janus.strategies.api.errors import ApiPastEndConflictError
from janus.strategies.api.pagination import (
    OffsetPaginator,
    PageNumberPaginator,
    PaginationState,
)
from janus.strategies.api.records import _default_records_from_payload
from janus.strategies.http import (
    HTTP_STATUS_REDIRECT,
    HTTP_STATUS_SUCCESS,
    ApiRequest,
    ApiResponse,
)
from janus.utils.logging import StructuredLogger, redact_url

#: Root/container keys read as a total-record count when no explicit field is configured.
#: ``count`` is deliberately absent: several APIs use it for *records on this page*, which would
#: yield a one-page ceiling and silently serialize the run. A source whose total really is ``count``
#: opts in with ``total_count_field: count``.
DEFAULT_TOTAL_COUNT_HINT_KEYS: tuple[str, ...] = (
    "total",
    "total_count",
    "totalCount",
    "total_records",
    "totalRecords",
    "total_items",
    "totalItems",
    "totalElements",
)
TOTAL_COUNT_CONTAINER_KEYS: tuple[str, ...] = ("meta", "metadata", "pagination")


@dataclass(frozen=True, slots=True)
class SpeculativePaginationPolicy:
    """End-of-stream and look-ahead rules for one request input of one source."""

    max_in_flight: int
    past_end_status_codes: frozenset[int]
    total_count_field: str | None
    page_size: int
    first_request_index: int
    first_page_number: int | None = None
    first_offset: int | None = None

    @property
    def enabled(self) -> bool:
        """Return whether this request input is paginated speculatively at all."""
        return self.max_in_flight > 1

    def is_past_end_status(self, status_code: int) -> bool:
        """Return whether ``status_code`` is evidence that the stream ended."""
        return status_code in self.past_end_status_codes

    def is_speculative(self, request_index: int) -> bool:
        """Return whether ``request_index`` exists only because a full page was guessed."""
        return request_index > self.first_request_index

    def last_request_index_for_total(self, total_records: int) -> int | None:
        """Return the highest request index worth requesting for ``total_records``.

        ``None`` means "no ceiling": the total is nonsense, or the paginator carries neither a page
        number nor an offset to convert it against.
        """
        if total_records < 0:
            return None

        pages = _ceil_div(total_records, self.page_size)
        if self.first_page_number is not None:
            last_index = pages - (self.first_page_number - 1)
        elif self.first_offset is not None:
            last_index = _ceil_div(total_records - self.first_offset, self.page_size)
        else:
            return None
        return max(last_index, self.first_request_index)


def resolve_speculative_policy(
    plan: ExecutionPlan,
    paginator: PageNumberPaginator | OffsetPaginator,
    initial_state: PaginationState,
) -> SpeculativePaginationPolicy:
    """Build the policy for the request input about to start at ``initial_state``.

    The first index is read from ``initial_state`` rather than hard-coded to 1: a resumed input
    starts at ``request_index=1`` with ``page_number=last+1`` (see ``_resume_pagination_state``),
    and the speculation rules must anchor on the page actually handed to the extraction loop.
    """
    pagination = plan.source_config.access.pagination
    return SpeculativePaginationPolicy(
        max_in_flight=plan.source_config.access.rate_limit.concurrency,
        past_end_status_codes=frozenset(pagination.past_end_status_codes),
        total_count_field=pagination.total_count_field,
        page_size=paginator.page_size,
        first_request_index=initial_state.request_index,
        first_page_number=initial_state.page_number,
        first_offset=initial_state.offset,
    )


def total_records_from_payload(payload: Any, *, total_count_field: str | None) -> int | None:
    """Return the total record count advertised by ``payload``, when it advertises one.

    Precedence: the configured dotted path first, then the root hint keys, then the same hints
    inside a ``meta`` / ``metadata`` / ``pagination`` container. Anything that is not a
    non-negative integer (or an all-digit string, as several Brazilian federal APIs return) is
    ignored rather than raised on — a page may legitimately omit or garble the count.
    """
    if total_count_field:
        resolved = _coerce_total(_resolve_dotted_path(payload, total_count_field))
        if resolved is not None:
            return resolved
    return _total_from_hint_keys(payload)


def _total_from_hint_keys(payload: Any) -> int | None:
    if not isinstance(payload, Mapping):
        return None

    for key in DEFAULT_TOTAL_COUNT_HINT_KEYS:
        total = _coerce_total(payload.get(key))
        if total is not None:
            return total

    for container_key in TOTAL_COUNT_CONTAINER_KEYS:
        nested = payload.get(container_key)
        if isinstance(nested, Mapping):
            nested_total = _total_from_hint_keys(nested)
            if nested_total is not None:
                return nested_total
    return None


def _resolve_dotted_path(payload: Any, path: str) -> Any:
    current = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _coerce_total(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        candidate = value.strip()
        return int(candidate) if candidate.isdigit() else None
    return None


def _ceil_div(dividend: int, divisor: int) -> int:
    """Ceiling division on integers — page counts at CNPJ scale must not touch binary floats."""
    if dividend <= 0:
        return 0
    return -(-dividend // divisor)


# ---------------------------------------------------------------------------
# The in-flight window
# ---------------------------------------------------------------------------


class TotalRecordsResolver(Protocol):
    """The single hook method ``_resolve_total_records`` needs.

    Structural rather than an ``ApiHook`` import: ``ApiHook`` is declared in ``core.py``,
    which imports this module, so naming the class here would close the cycle that
    ``errors.py`` was split out to break. ``ApiHook`` satisfies this protocol as written.
    """

    def resolve_total_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> int | None: ...


@dataclass(frozen=True, slots=True)
class SubmittedApiRequest:
    pagination_state: PaginationState
    request: ApiRequest
    future: Future[tuple[ApiResponse, Any, int]]


def _supports_concurrent_pagination(
    paginator: Any,
    concurrency: int,
) -> TypeGuard[PageNumberPaginator | OffsetPaginator]:
    return concurrency > 1 and isinstance(paginator, PageNumberPaginator | OffsetPaginator)


def _resolve_total_records(
    plan: ExecutionPlan,
    request: ApiRequest,
    response: ApiResponse,
    payload: Any,
    *,
    api_hook: TotalRecordsResolver | None,
    policy: SpeculativePaginationPolicy | None,
    logger: StructuredLogger | None,
) -> tuple[int | None, str | None]:
    """Return the total this page advertises and where it came from."""
    if policy is None:
        return None, None

    if api_hook is not None:
        hook_total = api_hook.resolve_total_records(plan, request, response, payload)
        if hook_total is not None:
            if isinstance(hook_total, int) and not isinstance(hook_total, bool) and hook_total >= 0:
                return hook_total, "hook"
            if logger is not None:
                logger.warning(
                    "api_total_records_invalid",
                    hook_value=repr(hook_total),
                    hook_value_type=type(hook_total).__name__,
                    request_url=redact_url(request.full_url()),
                )

    payload_total = total_records_from_payload(
        payload,
        total_count_field=policy.total_count_field,
    )
    if payload_total is None:
        return None, None
    return payload_total, "payload"


def _may_submit(request_index: int, ceiling: int | None, next_request_index: int) -> bool:
    """Allow submission ahead of evidence only up to the reported ceiling.

    Above the ceiling we do not stop — we degrade to an in-flight window of one, i.e. the index
    is submitted only when it is the very next one to commit. So a stale or wrong total costs a
    little parallelism at the tail and can never truncate the dataset: the authority on "the
    stream ended" stays with the paginator's short/empty page rule and the past-end status.
    """
    if ceiling is None or request_index <= ceiling:
        return True
    return request_index == next_request_index


def _cancel_pending(
    pending: dict[int, SubmittedApiRequest],
    logger: StructuredLogger | None,
    *,
    next_request_index: int,
) -> int:
    """Cancel every outstanding speculative request and report how many were discarded."""
    discarded = 0
    for submitted in pending.values():
        if submitted.future.cancel() or not submitted.future.done():
            discarded += 1
    pending.clear()
    if discarded and logger is not None:
        logger.info(
            "api_pagination_speculation_cancelled",
            cancelled_count=discarded,
            next_request_index=next_request_index,
        )
    return discarded


def _raise_on_past_end_conflict(
    pending: dict[int, SubmittedApiRequest],
    response: ApiResponse,
    past_end_index: int,
    logger: StructuredLogger | None,
) -> None:
    """Fail loudly when a *later* page already proved the stream did not end here."""
    for index, submitted in sorted(pending.items()):
        if index <= past_end_index or not submitted.future.done() or submitted.future.cancelled():
            continue
        try:
            later_response, later_payload, _ = submitted.future.result(timeout=0)
        except Exception:  # a later failure proves nothing; the past-end read stands
            continue
        if (
            HTTP_STATUS_SUCCESS <= later_response.status_code < HTTP_STATUS_REDIRECT
            and _default_records_from_payload(later_payload)
        ):
            if logger is not None:
                logger.warning(
                    "api_pagination_past_end_conflict",
                    request_index=past_end_index,
                    status_code=response.status_code,
                    request_url=redact_url(response.request.full_url()),
                    conflicting_request_index=index,
                )
            raise ApiPastEndConflictError(response, conflicting_request_index=index)


def _predicted_next_pagination_state(
    paginator: PageNumberPaginator | OffsetPaginator,
    pagination_state: PaginationState,
) -> PaginationState | None:
    return paginator.next_state(
        pagination_state,
        records_extracted=paginator.page_size,
        payload=None,
    )
