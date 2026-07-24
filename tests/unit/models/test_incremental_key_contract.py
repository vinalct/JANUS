"""Incremental mode must declare idempotency keys.

Incremental writes are upserted on ``quality.unique_fields``; without a key there is no
definable "same row", so the loader could only duplicate silently. The rule is enforced at
*load* time in ``SourceConfig.from_mapping`` — before a single HTTP request is sent — and it
must participate in the collected-issues ergonomics rather than short-circuiting.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from janus.models import SourceConfig
from janus.models.source_config import SourceConfigValidationError

CONFIG_PATH = Path("conf/sources/example.yaml")


def _base_payload() -> dict[str, Any]:
    """A minimal, valid full_refresh API contract that individual tests mutate."""
    return {
        "source_id": "incremental_contract_example",
        "name": "Incremental Contract Example",
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
            "required_fields": ["event_id", "event_date"],
            "allow_schema_evolution": True,
        },
    }


def _incremental_payload() -> dict[str, Any]:
    """The base contract switched to a valid incremental extraction block."""
    payload = _base_payload()
    payload["extraction"]["mode"] = "incremental"
    payload["extraction"]["checkpoint_field"] = "event_date"
    payload["extraction"]["checkpoint_strategy"] = "max_value"
    return payload


def test_incremental_without_unique_fields_is_rejected() -> None:
    payload = _incremental_payload()

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(payload, CONFIG_PATH)

    paths = {issue.path for issue in exc_info.value.issues}
    assert "quality.unique_fields" in paths
    message = str(exc_info.value)
    assert "extraction.mode is 'incremental'" in message
    assert "no defined idempotency" in message


def test_incremental_with_unique_fields_loads_clean() -> None:
    payload = _incremental_payload()
    payload["quality"]["unique_fields"] = ["event_id"]

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.extraction.mode == "incremental"
    assert config.quality.unique_fields == ("event_id",)


@pytest.mark.parametrize("mode", ["full_refresh", "snapshot"])
def test_non_incremental_modes_do_not_require_unique_fields(mode: str) -> None:
    payload = _base_payload()
    payload["extraction"]["mode"] = mode

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.extraction.mode == mode
    assert config.quality.unique_fields == ()


def test_missing_unique_fields_and_checkpoint_field_are_both_reported() -> None:
    """The rule appends to the collected issues rather than short-circuiting.

    An incremental config that lacks both the idempotency key and the checkpoint field must
    surface both problems in one raise — proving the new cross-block check composes with the
    existing incremental checks instead of masking them.
    """
    payload = _incremental_payload()
    del payload["extraction"]["checkpoint_field"]

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(payload, CONFIG_PATH)

    paths = {issue.path for issue in exc_info.value.issues}
    assert "quality.unique_fields" in paths
    assert "extraction.checkpoint_field" in paths


def test_base_payload_is_valid() -> None:
    """Guards the fixture itself so the tests above fail for the intended reason."""
    assert copy.deepcopy(_base_payload()) == _base_payload()
    SourceConfig.from_mapping(_base_payload(), CONFIG_PATH)
