"""The canonical strategy family/variant registry.

One definition of "which strategy families and variants exist", read by the config model
(to validate ``strategy_variant``) and by the planner (to build its dispatch bindings). The
registry names families and variants only — it never references a strategy *class*, so
``janus.models`` stays independent of ``janus.strategies`` and the planner keeps importing
downward.

The literal stays in ``constants.py``: that module is the bottom layer where every closed
value set of the contract lives, and it imports nothing from this package. Data there,
behaviour here — one definition either way.

Adding a variant is one edit in ``constants.py``. Adding a *family* is that edit plus its
implementation binding in ``StrategyCatalog.with_defaults``; the two are kept in step by
``tests/unit/planner/test_strategy_registry_drift.py``, which asserts set equality in both
directions, and by the named ``PlannerError`` ``with_defaults`` raises for a family with no
implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from janus.models.config.constants import SUPPORTED_STRATEGY_VARIANTS


@dataclass(frozen=True, slots=True)
class StrategyRegistry:
    """The closed set of strategy families and the variants each one supports."""

    variants_by_family: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        """Freeze the mapping so a caller cannot mutate a shared registry in place.

        The module-level default is shared by every consumer; a plain ``dict`` field
        would make it global mutable state wearing a dataclass costume (NFR-2).
        """
        frozen = {
            family: frozenset(variants)
            for family, variants in self.variants_by_family.items()
        }
        object.__setattr__(self, "variants_by_family", MappingProxyType(frozen))

    @property
    def families(self) -> frozenset[str]:
        """Every registered family."""
        return frozenset(self.variants_by_family)

    def variants_for(self, family: str) -> frozenset[str]:
        """Variants registered for `family`; empty frozenset for an unknown family.

        Empty rather than raising, because the model's validation path asks about a
        family that may itself be invalid and must report both problems, not the first.
        """
        return self.variants_by_family.get(family, frozenset())

    def supports(self, family: str, variant: str) -> bool:
        """Whether `variant` is registered for `family`."""
        return variant in self.variants_for(family)

    def dispatch_keys(self) -> tuple[tuple[str, str], ...]:
        """Every (family, variant) pair, sorted — the planner's binding source."""
        return tuple(
            (family, variant)
            for family in sorted(self.variants_by_family)
            for variant in sorted(self.variants_by_family[family])
        )

    def describe_variants(self, family: str) -> str:
        """The rendering the model's ``strategy_variant`` message uses.

        Formatted here so the message has one definition; the planner deliberately keeps
        its own ``StrategyResolutionError`` wording, which addresses a different reader.
        """
        return ", ".join(sorted(self.variants_for(family)))

    def with_family(self, family: str, variants: Iterable[str]) -> Self:
        """Return a copy with one family added or replaced, leaving this one untouched."""
        extended = dict(self.variants_by_family)
        extended[family] = frozenset(variants)
        return type(self)(variants_by_family=extended)


STRATEGY_REGISTRY = StrategyRegistry(variants_by_family=SUPPORTED_STRATEGY_VARIANTS)
