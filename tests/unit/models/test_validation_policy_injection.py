"""AC-2: an alternate policy relaxes a phase-1 rule without editing ``source_config.py``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from janus.models.config.issues import ValidationIssue
from janus.models.config.policy import DEFAULT_VALIDATION_POLICY, ValidationPolicy
from janus.models.source_config import SourceConfig, SourceConfigValidationError

CONFIG_PATH = Path("conf/sources/example/policy_injection.yaml")

PUBLIC_SOURCES_ALLOWED = replace(DEFAULT_VALIDATION_POLICY, require_public_access=False)

SPLIT_FAMILIES_ALLOWED = replace(
    DEFAULT_VALIDATION_POLICY, require_strategy_matches_source_type=False
)

STATE_SOURCES_ALLOWED = replace(
    DEFAULT_VALIDATION_POLICY, federation_levels=frozenset({"federal", "state"})
)

NOTHING_RELAXABLE_LEFT_STRICT = replace(
    DEFAULT_VALIDATION_POLICY,
    require_public_access=False,
    require_strategy_matches_source_type=False,
    federation_levels=frozenset({"federal", "state", "municipal"}),
    source_types=frozenset({"api", "catalog", "file", "graphql"}),
    strategies=frozenset({"api", "catalog", "file", "graphql"}),
)


def _access_block(**overrides: Any) -> dict[str, Any]:
    """The access block of the base mapping, with `overrides` applied at the top level."""
    access: dict[str, Any] = {
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
    }
    access.update(overrides)
    return access


def _base_mapping(**overrides: Any) -> dict[str, Any]:
    """A source that loads cleanly under the default policy.

    Every failure below is therefore the override's doing, not the fixture's — which is
    what lets each test assert on the *complete* issue list rather than on a substring.
    """
    mapping: dict[str, Any] = {
        "source_id": "policy_injection_source",
        "name": "policy_injection_source",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": _access_block(),
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
            "raw": {"path": "data/raw/example/policy_injection_source", "format": "json"},
            "bronze": {
                "path": "data/bronze/example/policy_injection_source",
                "format": "iceberg",
            },
            "metadata": {
                "path": "data/metadata/example/policy_injection_source",
                "format": "json",
            },
        },
        "quality": {"allow_schema_evolution": True},
    }
    mapping.update(overrides)
    return mapping


def test_permissive_public_access_policy_accepts_a_private_source():
    """The rule the PRD names first, relaxed by one field on one object."""
    mapping = _base_mapping(public_access=False)

    with pytest.raises(SourceConfigValidationError):
        SourceConfig.from_mapping(mapping, CONFIG_PATH)

    config = SourceConfig.from_mapping(mapping, CONFIG_PATH, policy=PUBLIC_SOURCES_ALLOWED)

    assert config.public_access is False


def test_permissive_pairing_policy_accepts_a_split_family():
    """A source whose strategy family differs from its source type."""
    mapping = _base_mapping(
        source_type="api", strategy="catalog", strategy_variant="metadata_catalog"
    )

    with pytest.raises(SourceConfigValidationError):
        SourceConfig.from_mapping(mapping, CONFIG_PATH)

    config = SourceConfig.from_mapping(mapping, CONFIG_PATH, policy=SPLIT_FAMILIES_ALLOWED)

    assert (config.source_type, config.strategy) == ("api", "catalog")
    assert config.strategy_variant == "metadata_catalog"


def test_broadened_federation_policy_accepts_a_state_source():
    """PRD Q2: federation level folded into the same seam as the other two axes.

    Without the fold this would be the one phase-scope rule still needing a type-layer
    edit, and "one evolution seam" would be three quarters true.
    """
    mapping = _base_mapping(federation_level="state")

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(mapping, CONFIG_PATH)
    assert "federation_level: must be one of: federal" in str(exc_info.value)

    config = SourceConfig.from_mapping(mapping, CONFIG_PATH, policy=STATE_SOURCES_ALLOWED)

    assert config.federation_level == "state"


def test_a_hand_written_policy_object_works():
    """The seam is a protocol, not a base class — proven by not inheriting from one.

    ``replace(DEFAULT, ...)`` would keep passing even if the seam quietly became a
    subclass hierarchy, because the default *is* a ``PhaseValidationPolicy``. This class
    is unrelated to it by inheritance and only agrees on shape, so it can only work if
    ``from_mapping`` depends on the protocol and nothing more.
    """

    class FederatedProfile:
        """A profile someone might write to onboard a state-level catalog source."""

        allowed_source_types = frozenset({"api", "catalog"})
        allowed_strategies = frozenset({"api", "catalog"})
        allowed_federation_levels = frozenset({"federal", "state"})

        def validate_strategy_pairing(
            self, source_type: str, strategy: str, issues: list[ValidationIssue]
        ) -> None:
            return None

        def validate_public_access(
            self, public_access: bool, issues: list[ValidationIssue]
        ) -> None:
            return None

    profile = FederatedProfile()

    assert not isinstance(profile, type(DEFAULT_VALIDATION_POLICY))
    assert isinstance(profile, ValidationPolicy)

    config = SourceConfig.from_mapping(
        _base_mapping(
            source_type="api",
            strategy="catalog",
            strategy_variant="metadata_catalog",
            federation_level="state",
            public_access=False,
        ),
        CONFIG_PATH,
        policy=profile,
    )

    assert (config.federation_level, config.public_access) == ("state", False)


def test_relaxing_one_rule_leaves_the_others_strict():
    """A policy is a scalpel. Relaxing one axis must not blunt the rest.

    The config breaks two rules; the policy names one. The surviving report must name
    only the other — no trace of the relaxed rule, and no new issue either.
    """
    mapping = _base_mapping(public_access=False, strategy_variant="telepathy_api")

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(mapping, CONFIG_PATH, policy=PUBLIC_SOURCES_ALLOWED)

    assert [issue.path for issue in exc_info.value.issues] == ["strategy_variant"]
    assert str(exc_info.value) == (
        f"Invalid source config: {CONFIG_PATH}\n"
        "- strategy_variant: must be one of: "
        "cursor_api, date_window_api, offset_api, page_number_api"
    )


@pytest.mark.parametrize(
    ("case", "overrides", "expected_issue_path"),
    [
        (
            "incremental_without_unique_fields",
            {
                "extraction": {
                    "mode": "incremental",
                    "checkpoint_field": "updated_at",
                    "checkpoint_strategy": "max_value",
                    "retry": {
                        "max_attempts": 3,
                        "backoff_strategy": "fixed",
                        "backoff_seconds": 1,
                    },
                },
                "quality": {"allow_schema_evolution": True},
            },
            "quality.unique_fields",
        ),
        (
            "concurrency_with_cursor_pagination",
            {
                "access": _access_block(
                    pagination={"type": "cursor", "cursor_param": "cursor", "page_size": 100},
                    rate_limit={"requests_per_minute": 10, "concurrency": 4},
                ),
                "strategy_variant": "cursor_api",
            },
            "access.rate_limit.concurrency",
        ),
    ],
    ids=["incremental_without_unique_fields", "concurrency_with_cursor_pagination"],
)
def test_structural_rules_are_not_relaxable_by_policy(
    case: str, overrides: dict[str, Any], expected_issue_path: str
):
    mapping = _base_mapping(**overrides)

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(mapping, CONFIG_PATH, policy=NOTHING_RELAXABLE_LEFT_STRICT)

    assert expected_issue_path in [issue.path for issue in exc_info.value.issues], (
        f"{case} validated under a policy that relaxed everything it can. "
        f"{expected_issue_path} is a structural rule — it must not be reachable from the "
        "ValidationPolicy seam."
    )
