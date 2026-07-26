"""Concurrent pagination is a declared, validated capability — not an implicit one."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from janus.models import SourceConfig
from janus.models.source_config import (
    DEFAULT_PAST_END_STATUS_CODES,
    RETRYABLE_CLIENT_STATUS_CODES,
    SourceConfigValidationError,
)

CONFIG_PATH = Path("conf/sources/example.yaml")


def _base_payload() -> dict[str, Any]:
    """A minimal, valid page_number API contract that individual tests mutate."""
    return {
        "source_id": "concurrency_contract_example",
        "name": "Concurrency Contract Example",
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
            "path": "/events",
            "method": "GET",
            "format": "json",
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {"concurrency": 1, "backoff_seconds": 5},
        },
        "extraction": {
            "mode": "full_refresh",
            "checkpoint_strategy": "none",
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "json",
            "write_mode": "append",
            "repartition": 1,
            "partition_by": [],
        },
        "outputs": {
            "raw": {"path": "data/raw/example/contract", "format": "json"},
            "bronze": {"path": "data/bronze/example/contract", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/contract", "format": "json"},
        },
        "quality": {
            "required_fields": ["event_id"],
            "allow_schema_evolution": True,
        },
    }


def _issues_for(payload: dict[str, Any]) -> dict[str, str]:
    """Load an invalid payload and return its issues as a path -> message mapping."""
    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(payload, CONFIG_PATH)
    return {issue.path: issue.message for issue in exc_info.value.issues}


def test_past_end_status_codes_default_to_404_and_416() -> None:
    config = SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)

    assert config.access.pagination.past_end_status_codes == (404, 416)
    assert DEFAULT_PAST_END_STATUS_CODES == (404, 416)


def test_past_end_status_codes_are_sorted_and_deduplicated() -> None:
    """The normalized value is deterministic, so run summaries stay reproducible."""
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = [416, 404, 404]

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.access.pagination.past_end_status_codes == (404, 416)


def test_explicit_empty_past_end_list_disables_the_feature() -> None:
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = []

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.access.pagination.past_end_status_codes == ()


def test_past_end_status_codes_reject_non_4xx() -> None:
    """A 5xx is a server fault, never evidence that the stream ended."""
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = [503]

    issues = _issues_for(payload)

    assert (
        issues["access.pagination.past_end_status_codes[0]"]
        == "must be a 4xx client-error status code"
    )


def test_past_end_status_codes_reject_retryable_client_codes() -> None:
    """A status cannot mean both "try again" and "the stream ended"."""
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = [429]

    issues = _issues_for(payload)

    message = issues["access.pagination.past_end_status_codes[0]"]
    assert "retryable" in message
    assert "408, 429" in message


@pytest.mark.parametrize("entry", ["404", True])
def test_past_end_status_codes_reject_non_integer_entries(entry: object) -> None:
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = [entry]

    issues = _issues_for(payload)

    assert issues["access.pagination.past_end_status_codes[0]"] == "must be an integer"


def test_past_end_status_codes_reject_a_non_list_value() -> None:
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = 404

    issues = _issues_for(payload)

    assert issues["access.pagination.past_end_status_codes"] == "must be a list"


def test_total_count_field_is_optional_and_defaults_to_none() -> None:
    config = SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)

    assert config.access.pagination.total_count_field is None


def test_total_count_field_accepts_a_dotted_path() -> None:
    payload = _base_payload()
    payload["access"]["pagination"]["total_count_field"] = "meta.total"

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.access.pagination.total_count_field == "meta.total"


@pytest.mark.parametrize("path", ["meta..total", ".total", "total."])
def test_total_count_field_rejects_malformed_paths(path: str) -> None:
    payload = _base_payload()
    payload["access"]["pagination"]["total_count_field"] = path

    issues = _issues_for(payload)

    assert "empty segments" in issues["access.pagination.total_count_field"]


def test_concurrency_above_one_requires_page_or_offset_pagination() -> None:
    payload = _base_payload()
    payload["strategy_variant"] = "cursor_api"
    payload["access"]["pagination"] = {"type": "cursor", "cursor_param": "next"}
    payload["access"]["rate_limit"]["concurrency"] = 2

    issues = _issues_for(payload)

    message = issues["access.rate_limit.concurrency"]
    assert "must be 1 unless access.pagination.type is 'page_number' or 'offset'" in message
    assert "'cursor'" in message


def test_concurrency_above_one_is_allowed_for_offset_pagination() -> None:
    payload = _base_payload()
    payload["strategy_variant"] = "offset_api"
    payload["access"]["pagination"] = {
        "type": "offset",
        "offset_param": "offset",
        "limit_param": "limit",
        "page_size": 100,
    }
    payload["access"]["rate_limit"]["concurrency"] = 4

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.access.rate_limit.concurrency == 4


@pytest.mark.parametrize(
    ("source_type", "strategy_variant"),
    [("catalog", "metadata_catalog"), ("file", "static_file")],
)
def test_concurrency_above_one_allowed_for_catalog_and_file_sources(
    source_type: str, strategy_variant: str
) -> None:
    """Documented inertness stays legal: those families never read ``concurrency``.

    Failing them here would break shipped configs for a debt this contract does not own.
    """
    payload = _base_payload()
    payload["source_type"] = source_type
    payload["strategy"] = source_type
    payload["strategy_variant"] = strategy_variant
    payload["access"]["pagination"] = {"type": "none"}
    payload["access"]["rate_limit"]["concurrency"] = 5

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.access.rate_limit.concurrency == 5


def test_concurrency_of_one_is_allowed_for_any_pagination_type() -> None:
    payload = _base_payload()
    payload["strategy_variant"] = "cursor_api"
    payload["access"]["pagination"] = {"type": "cursor", "cursor_param": "next"}

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.access.rate_limit.concurrency == 1


def test_concurrency_and_past_end_issues_are_collected_together() -> None:
    """The new checks append to the collected issues instead of short-circuiting."""
    payload = _base_payload()
    payload["strategy_variant"] = "cursor_api"
    payload["access"]["pagination"] = {
        "type": "cursor",
        "cursor_param": "next",
        "past_end_status_codes": [503],
    }
    payload["access"]["rate_limit"]["concurrency"] = 3

    issues = _issues_for(payload)

    assert "access.rate_limit.concurrency" in issues
    assert "access.pagination.past_end_status_codes[0]" in issues


def test_retryable_client_status_codes_match_the_shared_retry_policy() -> None:
    """Guardrail: the local literal must not drift from the one retry loop.

    ``models`` cannot import ``strategies`` without inverting the dependency, so the
    4xx retryable set is duplicated as a literal. This test is what keeps the copy honest.
    """
    from janus.strategies.http.retry import RETRYABLE_STATUS_CODES

    assert RETRYABLE_CLIENT_STATUS_CODES <= RETRYABLE_STATUS_CODES
    retryable_4xx = frozenset(code for code in RETRYABLE_STATUS_CODES if 400 <= code < 500)
    assert retryable_4xx == RETRYABLE_CLIENT_STATUS_CODES


def test_base_payload_is_valid() -> None:
    """Guards the fixture itself so the tests above fail for the intended reason."""
    assert copy.deepcopy(_base_payload()) == _base_payload()
    SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)
