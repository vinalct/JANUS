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

Every decision here is a pure function of the config, the paginator and one payload: no
transport, no writer, no executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from janus.models import ExecutionPlan
from janus.strategies.api.pagination import (
    OffsetPaginator,
    PageNumberPaginator,
    PaginationState,
)

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
