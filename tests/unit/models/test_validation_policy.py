"""The phase-scope policy object, before anything consults it.

These tests answer one question only — *does the policy produce the right issues?* — so
that when ``from_mapping`` starts calling it, a parity failure there has exactly one
possible cause. The messages are compared verbatim because they are the contract: the
default posture is unchanged only if a config that failed yesterday fails today with the
same words, in the same order.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from janus.models.config.issues import ValidationIssue
from janus.models.config.policy import (
    DEFAULT_VALIDATION_POLICY,
    PhaseValidationPolicy,
    ValidationPolicy,
)
from janus.models.config.strategy_registry import STRATEGY_REGISTRY
from janus.models.source_config import SUPPORTED_SOURCE_TYPES, SUPPORTED_STRATEGIES

PAIRING_MESSAGE = "must match source_type 'api' for the current JANUS strategy families"

PUBLIC_ACCESS_MESSAGE = (
    "must be true because JANUS only supports public federal sources in phase 1"
)

RELAXED_POLICY = replace(
    DEFAULT_VALIDATION_POLICY,
    require_public_access=False,
    require_strategy_matches_source_type=False,
)


def test_default_policy_is_phase_one_strict():
    """AC-1 at the object level: the shipped default relaxes nothing."""
    assert DEFAULT_VALIDATION_POLICY.require_public_access is True
    assert DEFAULT_VALIDATION_POLICY.require_strategy_matches_source_type is True
    assert DEFAULT_VALIDATION_POLICY.allowed_federation_levels == frozenset({"federal"})
    assert DEFAULT_VALIDATION_POLICY.allowed_source_types == SUPPORTED_SOURCE_TYPES
    assert DEFAULT_VALIDATION_POLICY.allowed_strategies == SUPPORTED_STRATEGIES


def test_policy_enum_sets_match_the_registry():
    """One definition of "which families exist" — the policy is a view, not a copy."""
    assert PhaseValidationPolicy().allowed_strategies == STRATEGY_REGISTRY.families
    assert PhaseValidationPolicy().allowed_source_types == STRATEGY_REGISTRY.families


def test_pairing_message_is_verbatim():
    """The rendered message is what tests and operators read; it must not drift by a quote."""
    issues: list[ValidationIssue] = []

    DEFAULT_VALIDATION_POLICY.validate_strategy_pairing("api", "catalog", issues)

    assert len(issues) == 1
    assert issues[0] == ValidationIssue("strategy", PAIRING_MESSAGE)
    assert issues[0].render() == f"strategy: {PAIRING_MESSAGE}"


@pytest.mark.parametrize(("source_type", "strategy"), [("", "catalog"), ("api", ""), ("", "")])
def test_pairing_is_silent_when_either_side_is_empty(source_type: str, strategy: str):
    """A config that already failed the enum check must not collect a second, confusing issue.

    The guard lives inside the policy precisely so the call site cannot forget it.
    """
    issues: list[ValidationIssue] = []

    DEFAULT_VALIDATION_POLICY.validate_strategy_pairing(source_type, strategy, issues)

    assert issues == []


def test_pairing_is_silent_when_they_match():
    issues: list[ValidationIssue] = []

    DEFAULT_VALIDATION_POLICY.validate_strategy_pairing("api", "api", issues)

    assert issues == []


def test_public_access_message_is_verbatim():
    issues: list[ValidationIssue] = []

    DEFAULT_VALIDATION_POLICY.validate_public_access(False, issues)

    assert len(issues) == 1
    assert issues[0] == ValidationIssue("public_access", PUBLIC_ACCESS_MESSAGE)


def test_public_access_true_is_silent():
    issues: list[ValidationIssue] = []

    DEFAULT_VALIDATION_POLICY.validate_public_access(True, issues)

    assert issues == []


def test_relaxed_policy_appends_nothing():
    """AC-2 in miniature: a different posture is a different object, not a code edit."""
    issues: list[ValidationIssue] = []

    RELAXED_POLICY.validate_strategy_pairing("api", "catalog", issues)
    RELAXED_POLICY.validate_public_access(False, issues)

    assert issues == []


def test_policy_is_frozen_and_shareable():
    """NFR-2: no global state means no test can change what another test validates against."""
    with pytest.raises(FrozenInstanceError):
        DEFAULT_VALIDATION_POLICY.require_public_access = False  # type: ignore[misc]

    relaxed = replace(DEFAULT_VALIDATION_POLICY, require_public_access=False)

    assert relaxed is not DEFAULT_VALIDATION_POLICY
    assert relaxed.require_public_access is False
    assert DEFAULT_VALIDATION_POLICY.require_public_access is True
    assert DEFAULT_VALIDATION_POLICY.require_strategy_matches_source_type is True


def test_a_minimal_stand_in_satisfies_the_protocol():
    """What makes AC-2 cheap: conformance by shape, with no base class to inherit."""

    class PermissivePolicy:
        allowed_source_types = frozenset({"api", "graphql"})
        allowed_strategies = frozenset({"api", "graphql"})
        allowed_federation_levels = frozenset({"federal", "state"})

        def validate_strategy_pairing(
            self, source_type: str, strategy: str, issues: list[ValidationIssue]
        ) -> None:
            return None

        def validate_public_access(
            self, public_access: bool, issues: list[ValidationIssue]
        ) -> None:
            return None

    stand_in = PermissivePolicy()

    assert isinstance(stand_in, ValidationPolicy)
    assert isinstance(DEFAULT_VALIDATION_POLICY, ValidationPolicy)


def test_issue_list_is_appended_to_never_replaced():
    """Issue order is contract — ``SourceConfigValidationError`` renders in append order."""
    earlier = ValidationIssue("source_id", "is required")
    issues = [earlier]

    DEFAULT_VALIDATION_POLICY.validate_strategy_pairing("api", "catalog", issues)
    DEFAULT_VALIDATION_POLICY.validate_public_access(False, issues)

    assert issues[0] is earlier
    assert [issue.path for issue in issues] == ["source_id", "strategy", "public_access"]
