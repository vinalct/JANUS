"""Characterization: the api and catalog module-level binding helpers.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from janus.checkpoints import CheckpointState

from .conftest import build_plan, build_source_config


def _checkpoint_state(value: str) -> CheckpointState:
    return CheckpointState(
        run_id="run-previous",
        source_id="binding_source",
        strategy_family="api",
        strategy_variant="page_number_api",
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
        checkpoint_value=value,
        updated_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# _split_path_and_query_params
# ---------------------------------------------------------------------------


def test_split_separates_path_placeholders_from_query_params(binding):
    path_params, query_params = binding.split_path_and_query_params(
        "https://x/{id}/items",
        {"id": "7", "q": "a"},
    )
    assert path_params == {"id": "7"}
    assert query_params == {"q": "a"}


def test_split_without_placeholders_sends_everything_to_query(binding):
    path_params, query_params = binding.split_path_and_query_params(
        "https://x/items",
        {"id": "7", "q": "a"},
    )
    assert path_params == {}
    assert query_params == {"id": "7", "q": "a"}


def test_split_only_recognizes_word_character_placeholders(binding):
    path_params, query_params = binding.split_path_and_query_params(
        "https://x/{data-id}/items",
        {"data-id": "7"},
    )
    assert path_params == {}
    assert query_params == {"data-id": "7"}


# ---------------------------------------------------------------------------
# _resolve_url
# ---------------------------------------------------------------------------


def test_resolve_url_prefers_explicit_access_url(binding, tmp_path):
    config = build_source_config(
        binding.name,
        tmp_path,
        source_id=f"{binding.name}_explicit_url",
        url="https://explicit.invalid/v2/things",
        base_url="https://example.invalid",
        path="/records",
    )
    assert binding.resolve_url(config) == "https://explicit.invalid/v2/things"


@pytest.mark.parametrize(
    ("base_url", "path"),
    [
        ("https://example.invalid/api/", "/v1/records"),
        ("https://example.invalid/api", "v1/records"),
        ("https://example.invalid/api/", "v1/records"),
        ("https://example.invalid/api", "/v1/records"),
    ],
)
def test_resolve_url_joins_base_url_and_path_with_exactly_one_slash(
    binding, tmp_path, base_url, path
):
    config = build_source_config(
        binding.name,
        tmp_path,
        source_id=f"{binding.name}_join",
        base_url=base_url,
        path=path,
    )
    assert binding.resolve_url(config) == "https://example.invalid/api/v1/records"


def test_resolve_url_returns_base_url_verbatim_when_path_is_missing(binding, tmp_path):
    config = build_source_config(
        binding.name,
        tmp_path,
        source_id=f"{binding.name}_base_only",
        base_url="https://example.invalid/api/",
        path=None,
    )
    assert binding.resolve_url(config) == "https://example.invalid/api/"


def test_resolve_url_without_base_url_raises_family_specific_message(binding, tmp_path):
    config = build_source_config(binding.name, tmp_path, source_id=f"{binding.name}_no_base")
    config = replace(config, access=replace(config.access, base_url=None, url=None, path=None))

    with pytest.raises(ValueError) as excinfo:
        binding.resolve_url(config)

    # Verified difference: the message names the family. Task 06 parameterizes
    # the label; the two strings below must survive the refactor unchanged.
    assert str(excinfo.value) == binding.resolve_url_missing_base_message


# ---------------------------------------------------------------------------
# _default_checkpoint_params
# ---------------------------------------------------------------------------


def test_default_checkpoint_params_empty_when_value_is_none(binding, tmp_path):
    plan = build_plan(
        binding.name,
        tmp_path,
        source_id=f"{binding.name}_ckpt_none_value",
        extraction_mode="incremental",
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
    )
    assert binding.default_checkpoint_params(plan, None) == {}


def test_default_checkpoint_params_empty_when_checkpoint_field_is_none(binding, tmp_path):
    plan = build_plan(binding.name, tmp_path, source_id=f"{binding.name}_ckpt_no_field")
    assert binding.default_checkpoint_params(plan, "2026-04-01") == {}


def test_default_checkpoint_params_maps_field_to_value(binding, tmp_path):
    plan = build_plan(
        binding.name,
        tmp_path,
        source_id=f"{binding.name}_ckpt_mapped",
        extraction_mode="incremental",
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
    )
    assert binding.default_checkpoint_params(plan, "2026-04-01") == {"updated_at": "2026-04-01"}


# ---------------------------------------------------------------------------
# _checkpoint_request_value
# ---------------------------------------------------------------------------


def _incremental_plan(binding, tmp_path, *, source_id, lookback_days=None):
    return build_plan(
        binding.name,
        tmp_path,
        source_id=source_id,
        extraction_mode="incremental",
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
        lookback_days=lookback_days,
    )


def test_checkpoint_request_value_none_outside_incremental_mode(binding, tmp_path):
    plan = build_plan(binding.name, tmp_path, source_id=f"{binding.name}_full_refresh")
    state = _checkpoint_state("2026-04-10T00:00:00Z")
    assert binding.checkpoint_request_value(plan, state) is None


def test_checkpoint_request_value_none_without_checkpoint_state(binding, tmp_path):
    plan = _incremental_plan(binding, tmp_path, source_id=f"{binding.name}_no_state")
    assert binding.checkpoint_request_value(plan, None) is None


def test_checkpoint_request_value_verbatim_without_lookback(binding, tmp_path):
    plan = _incremental_plan(binding, tmp_path, source_id=f"{binding.name}_no_lookback")
    state = _checkpoint_state("2026-04-10T00:00:00Z")
    assert binding.checkpoint_request_value(plan, state) == "2026-04-10T00:00:00Z"


def test_checkpoint_request_value_subtracts_lookback_days_from_datetimes(binding, tmp_path):
    plan = _incremental_plan(
        binding, tmp_path, source_id=f"{binding.name}_lookback", lookback_days=2
    )
    state = _checkpoint_state("2026-04-10T00:00:00Z")
    assert binding.checkpoint_request_value(plan, state) == "2026-04-08T00:00:00Z"


def test_checkpoint_request_value_returns_unparseable_values_verbatim(binding, tmp_path):
    plan = _incremental_plan(
        binding, tmp_path, source_id=f"{binding.name}_unparseable", lookback_days=2
    )
    state = _checkpoint_state("release-42")
    assert binding.checkpoint_request_value(plan, state) == "release-42"


def test_checkpoint_request_value_ignores_falsy_lookback(binding, tmp_path):
    plan = _incremental_plan(
        binding, tmp_path, source_id=f"{binding.name}_zero_lookback", lookback_days=0
    )
    state = _checkpoint_state("2026-04-10T00:00:00Z")
    assert binding.checkpoint_request_value(plan, state) == "2026-04-10T00:00:00Z"
