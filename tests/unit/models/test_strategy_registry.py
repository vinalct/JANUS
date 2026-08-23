"""AC-3/AC-4: strategy families and variants are declared once, and both consumers read it.

The registry is the shared knowledge behind two otherwise unrelated checks — the model
rejecting an unknown ``strategy_variant`` and the planner building its dispatch bindings.
These tests pin the registry's behaviour (immutability, the empty-set answer the model's
message depends on, the exact rendering of that message) and then demonstrate the payoff:
a variant added to an injected registry is accepted by *both* consumers with no edit to
``source_config.py`` or to ``planner/core.py``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from janus.models.config.strategy_registry import STRATEGY_REGISTRY, StrategyRegistry
from janus.models.source_config import (
    SUPPORTED_SOURCE_TYPES,
    SUPPORTED_STRATEGY_VARIANTS,
    SourceConfig,
    SourceConfigValidationError,
)
from janus.planner import PlannerError, StrategyCatalog

CONFIG_PATH = Path("conf/sources/example/strategy_registry_source.yaml")

EXPECTED_DISPATCH_KEY_COUNT = 9

API_VARIANTS = frozenset({"cursor_api", "date_window_api", "offset_api", "page_number_api"})


def test_registry_exposes_every_family_and_variant():
    """The registry is a view over the one literal, not a second copy of it."""
    assert STRATEGY_REGISTRY.families == SUPPORTED_SOURCE_TYPES
    assert STRATEGY_REGISTRY.variants_for("api") == API_VARIANTS
    assert len(STRATEGY_REGISTRY.dispatch_keys()) == EXPECTED_DISPATCH_KEY_COUNT
    assert STRATEGY_REGISTRY.dispatch_keys() == tuple(
        (family, variant)
        for family in sorted(SUPPORTED_STRATEGY_VARIANTS)
        for variant in sorted(SUPPORTED_STRATEGY_VARIANTS[family])
    )


def test_variants_for_unknown_family_is_empty_not_an_error():
    """The model asks about a family that may itself be invalid, and must not blow up.

    ``from_mapping`` collects issues; a raise here would report one problem where the
    contract promises all of them.
    """
    assert STRATEGY_REGISTRY.variants_for("nope") == frozenset()
    assert STRATEGY_REGISTRY.supports("nope", "page_number_api") is False


def test_registry_is_immutable():
    """A shared module-level default that a caller can mutate is global mutable state."""
    with pytest.raises(TypeError):
        STRATEGY_REGISTRY.variants_by_family["graphql"] = frozenset({"x"})  # type: ignore[index]

    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is a dataclass detail
        STRATEGY_REGISTRY.variants_by_family = {}  # type: ignore[misc]


def test_a_caller_supplied_mapping_is_frozen_on_construction():
    """Constructing from a live dict must not leave the caller holding a mutation handle."""
    mutable = {"api": frozenset({"page_number_api"})}

    registry = StrategyRegistry(variants_by_family=mutable)
    mutable["api"] = frozenset({"anything_api"})

    assert registry.variants_for("api") == frozenset({"page_number_api"})
    with pytest.raises(TypeError):
        registry.variants_by_family["api"] = frozenset()  # type: ignore[index]


def test_with_family_does_not_mutate_the_default():
    """NFR-2 in miniature: injection produces a new registry, it does not edit the shared one."""
    before = copy.deepcopy({f: set(v) for f, v in STRATEGY_REGISTRY.variants_by_family.items()})

    extended = STRATEGY_REGISTRY.with_family("api", API_VARIANTS | {"tsv_api"})

    assert extended is not STRATEGY_REGISTRY
    assert extended.variants_for("api") == API_VARIANTS | {"tsv_api"}
    assert {
        family: set(variants) for family, variants in STRATEGY_REGISTRY.variants_by_family.items()
    } == before


def test_describe_variants_matches_the_model_message():
    """The message rendering has one definition; parity with today's inline join is the point."""
    assert STRATEGY_REGISTRY.describe_variants("api") == ", ".join(sorted(API_VARIANTS))
    assert STRATEGY_REGISTRY.describe_variants("nope") == ""


def test_model_rejects_a_variant_the_default_registry_does_not_know():
    """AC-1: the default posture is unchanged, message included."""
    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(_source_mapping(strategy_variant="tsv_api"), CONFIG_PATH)

    assert f"strategy_variant: must be one of: {', '.join(sorted(API_VARIANTS))}" in str(
        exc_info.value
    )


def test_model_accepts_a_variant_added_only_to_an_injected_registry():
    """AC-4: widening the accepted set needs no edit to ``source_config.py``."""
    extended = STRATEGY_REGISTRY.with_family("api", API_VARIANTS | {"tsv_api"})

    config = SourceConfig.from_mapping(
        _source_mapping(strategy_variant="tsv_api"), CONFIG_PATH, registry=extended
    )

    assert config.strategy_variant == "tsv_api"


def test_planner_catalog_accepts_the_same_extended_registry():
    """AC-3: the planner reads the same registry, so one edit site serves both consumers."""
    extended = STRATEGY_REGISTRY.with_family("api", API_VARIANTS | {"tsv_api"})
    config = SourceConfig.from_mapping(
        _source_mapping(strategy_variant="tsv_api"), CONFIG_PATH, registry=extended
    )

    binding = StrategyCatalog.with_defaults(extended).resolve(config)

    assert (binding.family, binding.variant) == ("api", "tsv_api")
    assert binding.strategy.strategy_family == "api"


def test_with_defaults_raises_a_named_error_for_a_family_with_no_implementation():
    """Registry ahead of planner used to be a bare ``KeyError`` at every construction."""
    extended = STRATEGY_REGISTRY.with_family("graphql", frozenset({"graphql_api"}))

    with pytest.raises(PlannerError) as exc_info:
        StrategyCatalog.with_defaults(extended)

    message = str(exc_info.value)
    assert "graphql" in message
    assert "StrategyCatalog.with_defaults" in message
    assert "SUPPORTED_STRATEGY_VARIANTS" in message


def _source_mapping(*, strategy_variant: str) -> dict[str, Any]:
    """A source that loads cleanly, so a failure is about the registry, not the fixture."""
    return {
        "source_id": "strategy_registry_source",
        "name": "strategy_registry_source",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": strategy_variant,
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
            "raw": {"path": "data/raw/example/strategy_registry_source", "format": "json"},
            "bronze": {
                "path": "data/bronze/example/strategy_registry_source",
                "format": "iceberg",
            },
            "metadata": {
                "path": "data/metadata/example/strategy_registry_source",
                "format": "json",
            },
        },
        "quality": {"allow_schema_evolution": True},
    }
