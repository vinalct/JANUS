from __future__ import annotations

from janus.models.source_config import SUPPORTED_STRATEGY_VARIANTS
from janus.planner import StrategyCatalog

EXPECTED_BINDING_COUNT = 9


def _model_pairs() -> set[tuple[str, str]]:
    return {
        (family, variant)
        for family, variants in SUPPORTED_STRATEGY_VARIANTS.items()
        for variant in variants
    }


def test_planner_catalog_covers_every_registered_family_and_variant():
    """Every (family, variant) the model accepts must be dispatchable by the planner.

    Set equality, in both directions, because each direction fails differently:

    - a pair present in the model but missing from the planner turns a load-time config
      validation error into a plan-time ``StrategyResolutionError``, which the operator
      only sees after the config passed;
    - a pair bound in the planner but absent from the model is completely silent today —
      nothing reads it, nothing reports it, and it rots.

    Containment in either direction would let one of those through.
    """
    catalog = StrategyCatalog.with_defaults()

    registered = {(binding.family, binding.variant) for binding in catalog.bindings}
    expected = _model_pairs()

    assert registered == expected, (
        "the planner's strategy catalog has drifted from the model's variant registry.\n"
        f"  accepted by the model, not dispatchable: {sorted(expected - registered)}\n"
        f"  dispatchable, not accepted by the model: {sorted(registered - expected)}\n"
        "Adding a family or a variant must update both, and makes the "
        "planner derive its bindings from the registry so it cannot not."
    )


def test_the_coverage_sweep_is_not_comparing_two_empty_sets():
    """An empty registry matching an empty catalog would satisfy the test above."""
    expected = _model_pairs()

    assert len(expected) == EXPECTED_BINDING_COUNT, (
        f"the model registry describes {len(expected)} (family, variant) pairs, expected "
        f"{EXPECTED_BINDING_COUNT}. If a family or variant was deliberately added or "
        "removed, update EXPECTED_BINDING_COUNT — that edit is the review conversation."
    )
    assert set(SUPPORTED_STRATEGY_VARIANTS) == {"api", "catalog", "file"}, (
        f"the model registry describes families {sorted(SUPPORTED_STRATEGY_VARIANTS)}; the "
        "planner binds an implementation per family in StrategyCatalog.with_defaults, so a "
        "new family needs a binding there too."
    )


def test_the_public_catalog_surface_reports_the_same_variants_per_family():
    """The same rule through ``registered_variants_for_family``, the public accessor.

    Reading it here keeps the guarantee expressible without reaching into
    ``_bindings_by_key``, and pins the accessor as the supported way to ask.
    """
    catalog = StrategyCatalog.with_defaults()

    for family, variants in SUPPORTED_STRATEGY_VARIANTS.items():
        assert catalog.registered_variants_for_family(family) == tuple(sorted(variants)), (
            f"family {family!r} resolves variants "
            f"{catalog.registered_variants_for_family(family)} but the model accepts "
            f"{tuple(sorted(variants))}"
        )
