"""``extraction.retry.retryable_status_codes`` is a validated, per-source declaration.

The option exists because "this status is transient" is a fact about one upstream API, not
about HTTP. These tests pin its parsing, its defaults, its rejections, and the cross-block
rule that a status may not simultaneously mean "the stream ended" and "send that again".
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from janus.models import SourceConfig
from janus.models.source_config import (
    DEFAULT_RETRYABLE_STATUS_CODES,
    SourceConfigValidationError,
)

CONFIG_PATH = Path("conf/sources/example.yaml")


def _base_payload() -> dict[str, Any]:
    """A minimal, valid page_number API contract that individual tests mutate."""
    return {
        "source_id": "retry_status_contract_example",
        "name": "Retry Status Contract Example",
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
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example", "format": "json"},
            "bronze": {"path": "data/bronze/example", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example", "format": "json"},
        },
        "quality": {"allow_schema_evolution": True},
    }


def _issues_for(payload: dict[str, Any]) -> dict[str, str]:
    """Load an invalid payload and return its issues as a path -> message mapping."""
    with pytest.raises(SourceConfigValidationError) as error:
        SourceConfig.from_mapping(payload, CONFIG_PATH)
    return {issue.path: issue.message for issue in error.value.issues}


def test_an_undeclared_source_keeps_the_default_set() -> None:
    config = SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)

    assert config.extraction.retry.retryable_status_codes == DEFAULT_RETRYABLE_STATUS_CODES


def test_a_declared_set_is_normalized_to_a_sorted_deduplicated_tuple() -> None:
    payload = _base_payload()
    payload["extraction"]["retry"]["retryable_status_codes"] = [503, 400, 400, 429]

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.extraction.retry.retryable_status_codes == (400, 429, 503)


def test_an_empty_declaration_is_honoured_rather_than_treated_as_absent() -> None:
    """``[]`` says "never retry on status" — a policy, not an omission."""
    payload = _base_payload()
    payload["extraction"]["retry"]["retryable_status_codes"] = []

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.extraction.retry.retryable_status_codes == ()


@pytest.mark.parametrize("code", [200, 204, 302, 399, 600, 0])
def test_a_non_error_status_is_rejected(code: int) -> None:
    """2xx never reaches the classification branch and 3xx is resolved by the transport."""
    payload = _base_payload()
    payload["extraction"]["retry"]["retryable_status_codes"] = [code]

    issues = _issues_for(payload)

    assert "extraction.retry.retryable_status_codes[0]" in issues


def test_a_non_integer_entry_is_rejected() -> None:
    payload = _base_payload()
    payload["extraction"]["retry"]["retryable_status_codes"] = ["400"]

    issues = _issues_for(payload)

    assert "extraction.retry.retryable_status_codes[0]" in issues


def test_a_status_cannot_be_both_retryable_and_past_end() -> None:
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = [404, 410]
    payload["extraction"]["retry"]["retryable_status_codes"] = [410, 503]

    issues = _issues_for(payload)

    assert "410" in issues["extraction.retry.retryable_status_codes"]
    assert "404" not in issues["extraction.retry.retryable_status_codes"]


def test_the_default_sets_do_not_overlap_out_of_the_box() -> None:
    """A source declaring neither key must never trip the cross-block rule."""
    config = SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)

    assert not set(config.extraction.retry.retryable_status_codes) & set(
        config.access.pagination.past_end_status_codes
    )


def test_retry_and_past_end_issues_are_collected_together() -> None:
    """The new check appends to the collected issues instead of short-circuiting."""
    payload = _base_payload()
    payload["access"]["pagination"]["past_end_status_codes"] = [410]
    payload["extraction"]["retry"]["retryable_status_codes"] = [410, 200]

    issues = _issues_for(payload)

    assert "extraction.retry.retryable_status_codes[1]" in issues
    assert "extraction.retry.retryable_status_codes" in issues


def test_base_payload_is_valid() -> None:
    """Guards the fixture itself so the tests above fail for the intended reason."""
    assert copy.deepcopy(_base_payload()) == _base_payload()
    SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)
