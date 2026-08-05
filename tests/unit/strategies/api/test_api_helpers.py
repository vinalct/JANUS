"""
Characterization tests for private helpers in janus.strategies.api.core.

These tests lock in current behavior before the helpers are extracted into
strategies/common.py.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import (
    ApiRequest,
    ApiResponse,
    ApiTransportError,
    UrllibApiTransport,
)
from janus.strategies.api.artifacts import _raw_relative_path
from janus.strategies.api.core import (
    _request_input_key,
)
from janus.strategies.api.pagination import (
    OffsetPaginator,
    PageNumberPaginator,
    PaginationState,
    _resume_pagination_state,
)
from janus.strategies.common import (
    _compare_checkpoint_values,
    _freeze_string_mapping,
    _retry_delay_seconds,
    _stringify_mapping,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_plan(
    tmp_path: Path,
    *,
    retry_backoff_strategy: str = "fixed",
    retry_backoff_seconds: int = 2,
    retry_max_attempts: int = 3,
    rate_limit_backoff_seconds: int | None = None,
) -> ExecutionPlan:
    source_config = SourceConfig.from_mapping(
        {
            "source_id": "helper_test",
            "name": "helper_test",
            "owner": "janus",
            "enabled": True,
            "source_type": "api",
            "strategy": "api",
            "strategy_variant": "page_number_api",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "base_url": "https://example.invalid",
                "path": "/records",
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "pagination": {
                    "type": "page_number",
                    "page_param": "page",
                    "size_param": "size",
                    "page_size": 100,
                },
                "rate_limit": {
                    "requests_per_minute": 10,
                    "concurrency": 1,
                    "backoff_seconds": rate_limit_backoff_seconds,
                },
            },
            "extraction": {
                "mode": "full_refresh",
                "retry": {
                    "max_attempts": retry_max_attempts,
                    "backoff_strategy": retry_backoff_strategy,
                    "backoff_seconds": retry_backoff_seconds,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "json", "write_mode": "append"},
            "outputs": {
                "raw": {"path": "data/raw/example/helper_test", "format": "json"},
                "bronze": {"path": "data/bronze/example/helper_test", "format": "iceberg"},
                "metadata": {"path": "data/metadata/example/helper_test", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / "helper_test.yaml",
    )
    run_context = RunContext.create(
        run_id="run-helper-test",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _make_response(retry_after: str | None = None) -> ApiResponse:
    headers: tuple[tuple[str, str], ...] = ()
    if retry_after is not None:
        headers = (("Retry-After", retry_after),)
    return ApiResponse(
        request=ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30),
        status_code=429,
        body=b"",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# _retry_delay_seconds — fixed backoff
# ---------------------------------------------------------------------------


def test_retry_delay_fixed_backoff_returns_same_delay_for_all_attempts(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="fixed",
        retry_backoff_seconds=2,
        retry_max_attempts=5,
    )

    delays = [_retry_delay_seconds(plan, attempt, None) for attempt in range(1, 5)]

    assert all(d == 2.0 for d in delays)


# ---------------------------------------------------------------------------
# _retry_delay_seconds — exponential backoff
# ---------------------------------------------------------------------------


def test_retry_delay_exponential_backoff_doubles_on_each_attempt(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="exponential",
        retry_backoff_seconds=2,
        retry_max_attempts=5,
    )

    assert _retry_delay_seconds(plan, 1, None) == 2.0
    assert _retry_delay_seconds(plan, 2, None) == 4.0
    assert _retry_delay_seconds(plan, 3, None) == 8.0


def test_retry_delay_exponential_is_capped_by_rate_limit_backoff(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="exponential",
        retry_backoff_seconds=2,
        retry_max_attempts=10,
        rate_limit_backoff_seconds=6,
    )

    assert _retry_delay_seconds(plan, 4, None) == 6.0
    assert _retry_delay_seconds(plan, 5, None) == 6.0


# ---------------------------------------------------------------------------
# _retry_delay_seconds — Retry-After header
# ---------------------------------------------------------------------------


def test_retry_delay_valid_retry_after_extends_delay(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="fixed",
        retry_backoff_seconds=1,
        rate_limit_backoff_seconds=60,
    )
    response = _make_response(retry_after="30")

    delay = _retry_delay_seconds(plan, 1, response)

    assert delay == 30.0


def test_retry_delay_invalid_retry_after_is_ignored(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="fixed",
        retry_backoff_seconds=2,
        rate_limit_backoff_seconds=60,
    )
    response = _make_response(retry_after="not-a-number")

    delay = _retry_delay_seconds(plan, 1, response)

    assert delay == 2.0


def test_retry_delay_none_response_uses_backoff_config(tmp_path):
    plan = _build_plan(tmp_path, retry_backoff_strategy="fixed", retry_backoff_seconds=3)

    assert _retry_delay_seconds(plan, 1, None) == 3.0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values — datetime
# ---------------------------------------------------------------------------


def test_compare_checkpoint_values_earlier_datetime_is_less(tmp_path):
    result = _compare_checkpoint_values("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z")
    assert result == -1


def test_compare_checkpoint_values_later_datetime_is_greater(tmp_path):
    result = _compare_checkpoint_values("2026-06-01T00:00:00Z", "2026-01-01T00:00:00Z")
    assert result == 1


def test_compare_checkpoint_values_equal_datetimes_returns_zero(tmp_path):
    result = _compare_checkpoint_values("2026-04-10T12:00:00Z", "2026-04-10T12:00:00Z")
    assert result == 0


def test_compare_checkpoint_values_timezone_aware_datetimes_compare_correctly(tmp_path):
    # Same instant expressed in different timezones must compare as equal.
    result = _compare_checkpoint_values("2026-04-10T12:00:00+00:00", "2026-04-10T09:00:00-03:00")
    assert result == 0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values — decimal
# ---------------------------------------------------------------------------


def test_compare_checkpoint_values_smaller_decimal_is_less(tmp_path):
    result = _compare_checkpoint_values("10", "20")
    assert result == -1


def test_compare_checkpoint_values_larger_decimal_is_greater(tmp_path):
    result = _compare_checkpoint_values("100.5", "99.9")
    assert result == 1


def test_compare_checkpoint_values_equal_decimal_strings_return_zero(tmp_path):
    result = _compare_checkpoint_values("42", "42.0")
    assert result == 0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values — text fallback
# ---------------------------------------------------------------------------


def test_compare_checkpoint_values_text_uses_lexicographic_order(tmp_path):
    assert _compare_checkpoint_values("alpha", "beta") == -1
    assert _compare_checkpoint_values("beta", "alpha") == 1
    assert _compare_checkpoint_values("same", "same") == 0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values — mixed types fall back to string comparison
# ---------------------------------------------------------------------------


def test_compare_checkpoint_values_mixed_types_fall_back_to_string(tmp_path):
    # "2026-04-10T12:00:00Z" is a datetime; "not-a-date" is text.
    # Types differ → string comparison is used.
    result = _compare_checkpoint_values("2026-04-10T12:00:00Z", "not-a-date")
    assert result == ("2026-04-10T12:00:00Z" > "not-a-date") - (
        "2026-04-10T12:00:00Z" < "not-a-date"
    )


# ---------------------------------------------------------------------------
# _freeze_string_mapping
# ---------------------------------------------------------------------------


def test_freeze_string_mapping_empty_input_returns_empty_tuple():
    assert _freeze_string_mapping(None) == ()
    assert _freeze_string_mapping({}) == ()


def test_freeze_string_mapping_returns_sorted_tuple_of_pairs():
    result = _freeze_string_mapping({"z": "last", "a": "first"})
    assert result == (("a", "first"), ("z", "last"))


def test_freeze_string_mapping_raises_on_empty_key():
    with pytest.raises(ValueError, match="mapping keys must be non-empty strings"):
        _freeze_string_mapping({"": "value"})


def test_freeze_string_mapping_raises_on_empty_value():
    with pytest.raises(ValueError, match="mapping values must be non-empty strings"):
        _freeze_string_mapping({"key": ""})


def test_freeze_string_mapping_strips_whitespace():
    result = _freeze_string_mapping({"  key  ": "  value  "})
    assert result == (("key", "value"),)


# ---------------------------------------------------------------------------
# _stringify_mapping
# ---------------------------------------------------------------------------


def test_stringify_mapping_skips_none_values():
    result = _stringify_mapping({"a": "x", "b": None})
    assert result == {"a": "x"}


def test_stringify_mapping_skips_empty_string_values():
    result = _stringify_mapping({"a": "x", "b": "  "})
    assert result == {"a": "x"}


def test_stringify_mapping_raises_on_empty_key():
    with pytest.raises(ValueError, match="mapping keys must be non-empty strings"):
        _stringify_mapping({"": "value"})


def test_stringify_mapping_empty_input_returns_empty_dict():
    assert _stringify_mapping({}) == {}


def test_stringify_mapping_converts_non_string_values_to_strings():
    result = _stringify_mapping({"n": 42, "f": 3.14})
    assert result == {"n": "42", "f": "3.14"}


# ---------------------------------------------------------------------------
# ApiRequest.with_header and with_params use _freeze_string_mapping / _stringify_mapping
# ---------------------------------------------------------------------------


def test_api_request_with_header_normalizes_and_stores_header():
    req = ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30)
    updated = req.with_header("Authorization", "Bearer token")
    assert updated.headers_as_dict()["Authorization"] == "Bearer token"


def test_api_request_with_header_raises_on_empty_value():
    req = ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30)
    with pytest.raises(ValueError):
        req.with_header("X-Empty", "")


def test_api_request_with_params_skips_none_values():
    req = ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30)
    updated = req.with_params({"a": "1", "b": None})
    assert updated.params_as_dict() == {"a": "1"}


def test_api_request_with_params_merges_onto_existing_params():
    req = ApiRequest(
        method="GET",
        url="https://example.invalid/",
        timeout_seconds=30,
        params=(("existing", "yes"),),
    )
    updated = req.with_params({"new": "also"})
    assert updated.params_as_dict() == {"existing": "yes", "new": "also"}


def test_urllib_transport_raises_explicit_error_when_opener_is_missing():
    class BrokenTransport(UrllibApiTransport):
        def open(self) -> None:
            self.opener = None

    req = ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30)

    with pytest.raises(ApiTransportError, match="failed to initialize urllib opener"):
        BrokenTransport().send(req)


def test_page_number_paginator_rejects_offset_state():
    req = ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30)
    paginator = PageNumberPaginator(page_param="page", size_param="size", page_size=100)

    with pytest.raises(ValueError, match="requires PaginationState.page_number"):
        paginator.apply(req, PaginationState(request_index=1, offset=0))


def test_offset_paginator_rejects_page_number_state():
    req = ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30)
    paginator = OffsetPaginator(offset_param="offset", limit_param="limit", page_size=100)

    with pytest.raises(ValueError, match="requires PaginationState.offset"):
        paginator.apply(req, PaginationState(request_index=1, page_number=1))


def test_resume_pagination_rejects_mismatched_progress_state():
    paginator = OffsetPaginator(offset_param="offset", limit_param="limit", page_size=100)

    with pytest.raises(ValueError, match="last_page_number"):
        _resume_pagination_state(
            paginator,
            PaginationState(request_index=1, offset=0),
            {"last_page_number": 3},
        )


# ---------------------------------------------------------------------------
# _raw_relative_path
# ---------------------------------------------------------------------------


def test_raw_relative_path_page_number():
    state = PaginationState(request_index=1, page_number=1)
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/page-0001.json")


def test_raw_relative_path_page_number_zero_padded():
    state = PaginationState(request_index=10, page_number=10)
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/page-0010.json")


def test_raw_relative_path_offset():
    state = PaginationState(request_index=1, offset=0)
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/offset-00000000.json")


def test_raw_relative_path_offset_nonzero():
    state = PaginationState(request_index=2, offset=100)
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/offset-00000100.json")


def test_raw_relative_path_cursor():
    state = PaginationState(request_index=1, cursor="some-cursor-token")
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/cursor-0001.json")


def test_raw_relative_path_no_pagination():
    state = PaginationState(request_index=1)
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/response-0001.json")


def test_raw_relative_path_multi_input_uses_request_input_directory():
    state = PaginationState(request_index=1, page_number=1)
    result = _raw_relative_path("json", state, request_input_index=1, request_input_count=3)
    assert result == Path("request-input-000001/page-0001.json")


def test_raw_relative_path_multi_input_second_input():
    state = PaginationState(request_index=1, page_number=1)
    result = _raw_relative_path("json", state, request_input_index=2, request_input_count=3)
    assert result == Path("request-input-000002/page-0001.json")


def test_raw_relative_path_binary_suffix():
    state = PaginationState(request_index=1, page_number=1)
    result = _raw_relative_path("binary", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/page-0001.bin")


def test_raw_relative_path_jsonl_suffix():
    state = PaginationState(request_index=1, page_number=1)
    result = _raw_relative_path("jsonl", state, request_input_index=1, request_input_count=1)
    assert result == Path("pages/page-0001.jsonl")


# ---------------------------------------------------------------------------
# _request_input_key
# ---------------------------------------------------------------------------


def test_request_input_key_none_returns_sentinel():
    assert _request_input_key(None) == "__none__"


def test_request_input_key_dict_returns_sorted_pairs():
    result = _request_input_key({"b": "2", "a": "1"})
    assert result == "a=1|b=2"


def test_request_input_key_single_entry():
    assert _request_input_key({"window_start": "2026-01-01"}) == "window_start=2026-01-01"


def test_request_input_key_is_deterministic_regardless_of_insertion_order():
    key1 = _request_input_key({"z": "last", "a": "first"})
    key2 = _request_input_key({"a": "first", "z": "last"})
    assert key1 == key2
