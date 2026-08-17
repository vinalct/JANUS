
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from janus.models.config.policy import DEFAULT_VALIDATION_POLICY
from janus.models.config.strategy_registry import STRATEGY_REGISTRY
from janus.models.source_config import SourceConfig, SourceConfigValidationError

CONFIG_PATH = Path("conf/sources/example/default_policy_parity.yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_HEADER = f"Invalid source config: {CONFIG_PATH}"

PUBLIC_ACCESS_MESSAGE = (
    "public_access: must be true because JANUS only supports public federal sources in phase 1"
)


def _base_mapping(**overrides: Any) -> dict[str, Any]:
    """A source that loads cleanly, so every failure below is the override's doing."""
    mapping: dict[str, Any] = {
        "source_id": "default_policy_parity",
        "name": "default_policy_parity",
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
                "size_param": "page_size",
                "page_size": 100,
            },
            "rate_limit": {"requests_per_minute": 10, "concurrency": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/default_policy_parity", "format": "json"},
            "bronze": {"path": "data/bronze/example/default_policy_parity", "format": "iceberg"},
            "metadata": {
                "path": "data/metadata/example/default_policy_parity",
                "format": "json",
            },
        },
        "quality": {"allow_schema_evolution": True},
    }
    mapping.update(overrides)
    return mapping


def _without(field_name: str, **overrides: Any) -> dict[str, Any]:
    mapping = _base_mapping(**overrides)
    del mapping[field_name]
    return mapping


# ── the differential ─────────────────────────────────────────────────────────

BROKEN_CASES: dict[str, tuple[dict[str, Any], str]] = {
    "private_source": (
        _base_mapping(public_access=False),
        f"{_HEADER}\n- {PUBLIC_ACCESS_MESSAGE}",
    ),
    # The subtle one. ``_require_bool`` appends "is required" and returns ``False``, so the
    # policy's ``is False`` identity check then appends the phase-1 message too. Two issues,
    # in that order. A refactor that swapped the identity check for truthiness reports one.
    "missing_public_access": (
        _without("public_access"),
        f"{_HEADER}\n- public_access: is required\n- {PUBLIC_ACCESS_MESSAGE}",
    ),
    "public_access_wrong_type": (
        _base_mapping(public_access="yes"),
        f"{_HEADER}\n- public_access: must be a boolean\n- {PUBLIC_ACCESS_MESSAGE}",
    ),
    "strategy_mismatch": (
        _base_mapping(source_type="api", strategy="catalog"),
        f"{_HEADER}\n"
        "- strategy: must match source_type 'api' for the current JANUS strategy families\n"
        "- strategy_variant: must be one of: metadata_catalog, resource_catalog",
    ),
    # An unrecognised source_type is still a non-empty string, so the pairing rule does see
    # it and does fire. Pinned because it is counter-intuitive, and because a "tidy-up" that
    # hoisted the guard to the call site or widened it would change this output.
    "unknown_source_type": (
        _base_mapping(source_type="graphql"),
        f"{_HEADER}\n"
        "- source_type: must be one of: api, catalog, file\n"
        "- strategy: must match source_type 'graphql' for the current JANUS strategy families",
    ),
    # The empty-guard case proper: an *absent* source_type must not also collect a pairing
    # message on top of "is required". This is what ``source_type and strategy`` buys.
    "missing_source_type": (
        _without("source_type"),
        f"{_HEADER}\n- source_type: is required",
    ),
    "state_federation": (
        _base_mapping(federation_level="state"),
        f"{_HEADER}\n- federation_level: must be one of: federal",
    ),
    "unknown_variant": (
        _base_mapping(strategy_variant="nope"),
        f"{_HEADER}\n"
        "- strategy_variant: must be one of: "
        "cursor_api, date_window_api, offset_api, page_number_api",
    ),
    # Pins the empty rendering an unregistered family produces — the ``.get(..., frozenset())``
    # behaviour the registry preserved.
    "variant_for_unknown_family": (
        _base_mapping(source_type="graphql", strategy="graphql"),
        f"{_HEADER}\n"
        "- source_type: must be one of: api, catalog, file\n"
        "- strategy: must be one of: api, catalog, file\n"
        "- strategy_variant: must be one of: ",
    ),
    # The ordering test. Every single-issue case above passes even if the policy calls were
    # reordered; only a config that trips several rules at once can see it.
    "several_at_once": (
        _base_mapping(
            source_type="api",
            strategy="catalog",
            strategy_variant="nope",
            public_access=False,
            extraction={
                "mode": "nonsense",
                "retry": {
                    "max_attempts": 3,
                    "backoff_strategy": "fixed",
                    "backoff_seconds": 1,
                },
            },
        ),
        f"{_HEADER}\n"
        "- strategy: must match source_type 'api' for the current JANUS strategy families\n"
        "- strategy_variant: must be one of: metadata_catalog, resource_catalog\n"
        f"- {PUBLIC_ACCESS_MESSAGE}\n"
        "- extraction.mode: must be one of: full_refresh, incremental, snapshot",
    ),
}


@pytest.mark.parametrize("case", sorted(BROKEN_CASES))
def test_rendered_validation_error_is_unchanged(case: str):
    """Full-text comparison: message, field path and append order in one assertion."""
    mapping, expected = BROKEN_CASES[case]

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(mapping, CONFIG_PATH)

    assert str(exc_info.value) == expected


def test_the_differential_covers_every_rule_the_policy_owns():
    """A table nobody extends stops testing the thing it was written for."""
    covered = " ".join(expected for _, expected in BROKEN_CASES.values())

    assert "public_access" in covered
    assert "must match source_type" in covered
    assert "federation_level" in covered
    assert "strategy_variant" in covered


# ── the injection seam is call-compatible ────────────────────────────────────


def test_from_mapping_accepts_a_policy_keyword():
    """Passing the default explicitly is indistinguishable from omitting it."""
    implicit = SourceConfig.from_mapping(_base_mapping(), CONFIG_PATH)
    explicit = SourceConfig.from_mapping(
        _base_mapping(), CONFIG_PATH, policy=DEFAULT_VALIDATION_POLICY
    )

    assert implicit == explicit


def test_policy_argument_is_keyword_only():
    """AC-1 rests on every pre-existing call site being untouched — pin that.

    A ``policy`` that could be passed positionally would let a third argument mean
    something new to code that never asked for it.
    """
    parameters = inspect.signature(SourceConfig.from_mapping).parameters

    assert parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["registry"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["policy"].default is DEFAULT_VALIDATION_POLICY
    assert parameters["registry"].default is STRATEGY_REGISTRY

    with pytest.raises(TypeError):
        SourceConfig.from_mapping(  # type: ignore[call-arg]
            _base_mapping(), CONFIG_PATH, DEFAULT_VALIDATION_POLICY
        )


def test_valid_config_still_round_trips():
    """A real checked-in source parses to the same object down to every field.

    Built from the fixture on disk rather than a hand-written literal, so the assertion
    tracks the config rather than a snapshot of what someone believed it said.
    """
    config_path = PROJECT_ROOT / "conf/sources/inep/inep.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config = SourceConfig.from_mapping(data, config_path)
    explicit = SourceConfig.from_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        config_path,
        policy=DEFAULT_VALIDATION_POLICY,
        registry=STRATEGY_REGISTRY,
    )

    assert config == explicit
    assert config.source_type == config.strategy
    assert config.federation_level == "federal"
    assert config.public_access is True
