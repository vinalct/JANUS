"""Unit tests for the speculative pagination policy.

Pure decision logic: no transport, no writer, no executor. The only I/O-adjacent object here is an
``ExecutionPlan``, and only where the test is about reading the source contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import build_concurrent_plan
from janus.strategies.api.pagination import (
    OffsetPaginator,
    PageNumberPaginator,
    PaginationState,
)
from janus.strategies.api.speculation import (
    SpeculativePaginationPolicy,
    resolve_speculative_policy,
    total_records_from_payload,
)


def build_policy(
    *,
    max_in_flight: int = 4,
    past_end_status_codes: frozenset[int] = frozenset({404, 416}),
    total_count_field: str | None = None,
    page_size: int = 100,
    first_request_index: int = 1,
    first_page_number: int | None = 1,
    first_offset: int | None = None,
) -> SpeculativePaginationPolicy:
    return SpeculativePaginationPolicy(
        max_in_flight=max_in_flight,
        past_end_status_codes=past_end_status_codes,
        total_count_field=total_count_field,
        page_size=page_size,
        first_request_index=first_request_index,
        first_page_number=first_page_number,
        first_offset=first_offset,
    )


# --- Policy resolution -------------------------------------------------------------------


def test_policy_reads_concurrency_and_past_end_codes_from_config(tmp_path):
    plan = build_concurrent_plan(
        tmp_path,
        source_id="speculation_config_source",
        page_size=500,
        concurrency=6,
        past_end_status_codes=[416, 404],
        total_count_field="meta.total",
    )
    paginator = PageNumberPaginator(page_param="page", size_param="page_size", page_size=500)

    policy = resolve_speculative_policy(
        plan,
        paginator,
        PaginationState(request_index=1, page_number=1),
    )

    assert policy.enabled is True
    assert policy.max_in_flight == 6
    assert policy.past_end_status_codes == frozenset({404, 416})
    assert policy.total_count_field == "meta.total"
    assert policy.page_size == 500


def test_policy_is_disabled_when_concurrency_is_one(tmp_path):
    plan = build_concurrent_plan(
        tmp_path,
        source_id="sequential_source",
        concurrency=1,
    )
    paginator = PageNumberPaginator(page_param="page", size_param="page_size", page_size=2)

    policy = resolve_speculative_policy(
        plan,
        paginator,
        PaginationState(request_index=1, page_number=1),
    )

    assert policy.enabled is False


def test_policy_defaults_the_past_end_codes_when_the_source_is_silent(tmp_path):
    plan = build_concurrent_plan(tmp_path, source_id="default_past_end_source")
    paginator = PageNumberPaginator(page_param="page", size_param="page_size", page_size=2)

    policy = resolve_speculative_policy(
        plan,
        paginator,
        PaginationState(request_index=1, page_number=1),
    )

    assert policy.past_end_status_codes == frozenset({404, 416})
    assert policy.total_count_field is None


def test_policy_captures_the_initial_state_for_a_resumed_input(tmp_path):
    """A resumed input starts at request_index 1 but at page 7 — anchor on the state, not on 1."""
    plan = build_concurrent_plan(tmp_path, source_id="resumed_source", concurrency=3)
    paginator = PageNumberPaginator(page_param="page", size_param="page_size", page_size=2)

    policy = resolve_speculative_policy(
        plan,
        paginator,
        PaginationState(request_index=1, page_number=7),
    )

    assert policy.first_page_number == 7
    assert policy.first_request_index == 1
    assert policy.is_speculative(1) is False
    assert policy.is_speculative(2) is True


def test_policy_captures_the_starting_offset_for_an_offset_source(tmp_path):
    plan = build_concurrent_plan(
        tmp_path,
        source_id="offset_source",
        variant="offset_api",
        pagination_type="offset",
        page_size=100,
        concurrency=3,
    )
    paginator = OffsetPaginator(offset_param="offset", limit_param="limit", page_size=100)

    policy = resolve_speculative_policy(
        plan,
        paginator,
        PaginationState(request_index=1, offset=200),
    )

    assert policy.first_offset == 200
    assert policy.first_page_number is None


# --- Past-end classification -------------------------------------------------------------


def test_configured_status_is_past_end():
    policy = build_policy(past_end_status_codes=frozenset({404, 416}))

    assert policy.is_past_end_status(404) is True
    assert policy.is_past_end_status(416) is True


def test_unconfigured_status_is_not():
    policy = build_policy(past_end_status_codes=frozenset({404}))

    assert policy.is_past_end_status(416) is False
    assert policy.is_past_end_status(500) is False
    assert policy.is_past_end_status(200) is False


def test_empty_past_end_set_disables_classification():
    """An explicit `[]` restores raise-on-4xx for a source that wants it."""
    policy = build_policy(past_end_status_codes=frozenset())

    assert policy.is_past_end_status(404) is False
    assert policy.is_past_end_status(416) is False


# --- Ceiling math ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total_records", "expected"),
    [
        pytest.param(0, 1, id="empty-total-floors-at-the-first-index"),
        pytest.param(1, 1, id="one-record-is-one-page"),
        pytest.param(100, 1, id="exactly-one-full-page"),
        pytest.param(101, 2, id="one-record-into-the-second-page"),
        pytest.param(1_000_000, 10_000, id="ten-thousand-pages"),
    ],
)
def test_page_number_ceiling_counts_whole_pages(total_records: int, expected: int):
    policy = build_policy(page_size=100, first_page_number=1, first_request_index=1)

    assert policy.last_request_index_for_total(total_records) == expected


def test_page_number_ceiling_is_relative_to_the_starting_page():
    """Pages 3, 4, 5 of a 500-record dataset are request indexes 1, 2, 3."""
    policy = build_policy(page_size=100, first_page_number=3, first_request_index=1)

    assert policy.last_request_index_for_total(500) == 3


def test_offset_ceiling_is_relative_to_the_starting_offset():
    policy = build_policy(
        page_size=100,
        first_page_number=None,
        first_offset=200,
        first_request_index=1,
    )

    assert policy.last_request_index_for_total(450) == 3


def test_ceiling_never_drops_below_the_current_first_index():
    policy = build_policy(page_size=100, first_page_number=1, first_request_index=4)

    assert policy.last_request_index_for_total(0) == 4
    assert policy.last_request_index_for_total(150) == 4


def test_ceiling_caps_speculation_but_never_terminates_extraction():
    """A stale or wrong total can cost throughput; it can never truncate the dataset."""
    policy = build_policy(page_size=100, first_page_number=3, first_request_index=3)

    # The total says the dataset ended before the page we are already holding: the ceiling
    # still admits that page instead of ordering the loop to stop short of it.
    assert policy.last_request_index_for_total(0) == 3
    assert policy.last_request_index_for_total(10) == 3
    # And an index above the ceiling stays a classifiable index rather than a rejected one —
    # the ceiling bounds what may be *submitted ahead of time*, not what may be fetched.
    assert policy.is_speculative(4) is True


def test_negative_total_is_ignored():
    policy = build_policy(page_size=100)

    assert policy.last_request_index_for_total(-1) is None


def test_ceiling_is_undefined_without_a_page_number_or_offset():
    policy = build_policy(first_page_number=None, first_offset=None)

    assert policy.last_request_index_for_total(1_000) is None


def test_ceiling_uses_integer_arithmetic():
    """Page counts at CNPJ scale must not round-trip through binary floating point."""
    policy = build_policy(page_size=3, first_page_number=1, first_request_index=1)

    assert policy.last_request_index_for_total(10**15 + 1) == 333_333_333_333_334


# --- Total discovery ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param({"total": 42}, 42, id="root-total"),
        pytest.param({"total": "42"}, 42, id="root-total-as-string"),
        pytest.param({"total": " 42 "}, 42, id="root-total-as-padded-string"),
        pytest.param({"totalRecords": 7}, 7, id="root-camel-case-hint"),
        pytest.param({"meta": {"total": 42}}, 42, id="meta-container"),
        pytest.param({"metadata": {"totalItems": 9}}, 9, id="metadata-container"),
        pytest.param({"pagination": {"totalElements": 7}}, 7, id="pagination-container"),
        pytest.param({"total": 0}, 0, id="zero-is-a-real-total"),
    ],
)
def test_total_is_discovered_from_the_default_hints(payload: Any, expected: int):
    assert total_records_from_payload(payload, total_count_field=None) == expected


def test_explicit_total_count_field_wins_over_a_root_hint():
    payload = {"total": 99, "meta": {"total": 42}}

    assert total_records_from_payload(payload, total_count_field="meta.total") == 42


def test_count_is_not_a_default_hint_but_can_be_configured():
    """`count` is records-on-this-page for several APIs, so it is opt-in only."""
    payload = {"count": 10}

    assert total_records_from_payload(payload, total_count_field=None) is None
    assert total_records_from_payload(payload, total_count_field="count") == 10


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"total": -1}, id="negative"),
        pytest.param({"total": 1.5}, id="float"),
        pytest.param({"total": True}, id="bool"),
        pytest.param({"total": "abc"}, id="non-numeric-string"),
        pytest.param({"total": None}, id="null"),
        pytest.param([1, 2, 3], id="list-payload"),
        pytest.param(None, id="no-payload"),
    ],
)
def test_malformed_totals_are_ignored(payload: Any):
    assert total_records_from_payload(payload, total_count_field=None) is None


def test_missing_total_count_field_segment_is_not_an_error():
    assert total_records_from_payload({"meta": []}, total_count_field="meta.total") is None
    assert total_records_from_payload({}, total_count_field="meta.total") is None
