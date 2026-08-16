"""Phase-scope validation policy. The one place product-scope decisions live."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from janus.models.config.constants import SUPPORTED_FEDERATION_LEVELS
from janus.models.config.issues import ValidationIssue
from janus.models.config.strategy_registry import STRATEGY_REGISTRY


@runtime_checkable
class ValidationPolicy(Protocol):
    """What ``from_mapping`` needs from a phase-scope policy."""

    @property
    def allowed_source_types(self) -> frozenset[str]: ...

    @property
    def allowed_strategies(self) -> frozenset[str]: ...

    @property
    def allowed_federation_levels(self) -> frozenset[str]: ...

    def validate_strategy_pairing(
        self, source_type: str, strategy: str, issues: list[ValidationIssue]
    ) -> None: ...

    def validate_public_access(
        self, public_access: bool, issues: list[ValidationIssue]
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PhaseValidationPolicy:
    """The default, phase-1 strict policy. Also the base for any relaxed profile."""

    source_types: frozenset[str] = STRATEGY_REGISTRY.families
    strategies: frozenset[str] = STRATEGY_REGISTRY.families
    federation_levels: frozenset[str] = SUPPORTED_FEDERATION_LEVELS
    require_strategy_matches_source_type: bool = True
    require_public_access: bool = True

    @property
    def allowed_source_types(self) -> frozenset[str]:
        """Source-type values the contract accepts — which families exist as a product."""
        return self.source_types

    @property
    def allowed_strategies(self) -> frozenset[str]:
        """Strategy values the contract accepts.

        Defaults to the same set as ``allowed_source_types`` because phase 1 pairs them;
        a policy that relaxes the pairing rule can supply two different sets, which is the
        whole point of keeping them as separate fields.
        """
        return self.strategies

    @property
    def allowed_federation_levels(self) -> frozenset[str]:
        """Federation levels JANUS has chosen to onboard (PRD Q2)."""
        return self.federation_levels

    def validate_strategy_pairing(
        self, source_type: str, strategy: str, issues: list[ValidationIssue]
    ) -> None:
        """Phase-1 requires one strategy family per source type.

        The ``source_type and strategy`` guard belongs here, not at the call site: a config
        that already failed the enum check must not also collect a confusing pairing
        message, and that is today's behaviour.
        """
        if not self.require_strategy_matches_source_type:
            return
        if source_type and strategy and source_type != strategy:
            issues.append(
                ValidationIssue(
                    "strategy",
                    (
                        f"must match source_type {source_type!r} "
                        "for the current JANUS strategy families"
                    ),
                )
            )

    def validate_public_access(
        self, public_access: bool, issues: list[ValidationIssue]
    ) -> None:
        """Phase-1 onboards public sources only.

        ``is False`` is identity on purpose. ``_require_bool`` returns ``False`` both for a
        missing field and for a wrong type, having already appended its own issue, so a
        config missing ``public_access`` collects two issues today and must keep collecting
        two.
        """
        if not self.require_public_access:
            return
        if public_access is False:
            issues.append(
                ValidationIssue(
                    "public_access",
                    "must be true because JANUS only supports public federal sources in phase 1",
                )
            )


DEFAULT_VALIDATION_POLICY: Final = PhaseValidationPolicy()
