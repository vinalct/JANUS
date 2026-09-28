"""The one evolution decision: what may this table become under this contract?"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from janus.models.data_contracts import DataContract, load_data_contract
from janus.normalizers import NORMALIZATION_METADATA_COLUMNS

pytestmark = pytest.mark.xfail(strict=True, reason="red until TASK-09 (order-19)")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
MODULE_PATH = PROJECT_ROOT / "src" / "janus" / "writers" / "evolution.py"
MODES = ("additive", "backward", "frozen")

RECORDED = "1.0.0"

VERIFIED_PROMOTIONS = frozenset({("integer", "long"), ("float", "double")})

DIFFERENCES: dict[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {
    "identical": ("base", "base", ()),
    "add_nullable": ("base", "base_plus_nullable", ()),
    "add_required": ("base", "base_plus_required", ()),
    "int_to_long": ("base_int", "base", ()),
    "float_to_double": ("base_float", "base_double", ()),
    "decimal_widening": ("base_decimal_10_2", "base_decimal_18_2", ()),
    "decimal_scale_change": ("base_decimal_18_2", "base_decimal_18_4", ()),
    "long_to_int": ("base", "base_narrowed", ()),
    "string_to_long": ("base_string_amount", "base", ()),
    "drop_column": ("base", "base_dropped", ()),
    "rename_column": ("base", "base_renamed", ()),
    "undeclared_live_column": ("base", "base", (("legacy_col", "string"),)),
}

FROZEN = ("refused", {"frozen"})
PROMOTABLE = {"additive": ("refused", {"retyped"}), "backward": ("evolve", set()), "frozen": FROZEN}


def _refused_in_place(kind: str) -> dict[str, tuple[str, set[str]]]:
    return {"additive": ("refused", {kind}), "backward": ("refused", {kind}), "frozen": FROZEN}


EXPECTED: dict[str, dict[str, tuple[str, set[str]]]] = {
    "identical": {mode: ("noop", set()) for mode in MODES},
    "add_nullable": {
        "additive": ("evolve", set()),
        "backward": ("evolve", set()),
        "frozen": FROZEN,
    },
    "add_required": _refused_in_place("newly_required"),
    "int_to_long": PROMOTABLE,
    "float_to_double": PROMOTABLE,
    "decimal_widening": PROMOTABLE,
    "decimal_scale_change": _refused_in_place("retyped"),
    "long_to_int": _refused_in_place("narrowed"),
    "string_to_long": _refused_in_place("retyped"),
    "drop_column": _refused_in_place("dropped"),
    "rename_column": _refused_in_place("renamed_or_dropped"),
    "undeclared_live_column": _refused_in_place("undeclared_live_column"),
}

MATRIX = [
    pytest.param(difference, mode, id=f"{difference}-{mode}")
    for difference in DIFFERENCES
    for mode in MODES
]


# ── builders ─────────────────────────────────────────────────────────────────


def _contract(name: str, *, compatibility: str | None = None, version: str | None = None):
    contract = load_data_contract(HOSTILE / f"{name}.yaml")
    if compatibility is not None:
        contract = replace(contract, janus=replace(contract.janus, compatibility=compatibility))
    if version is not None:
        contract = replace(contract, version=version)
    return contract


def _live(name: str, extra: tuple[tuple[str, str], ...] = ()) -> tuple[Any, ...]:
    """A live table shaped like one fixture; every bronze column is Iceberg-optional (D-3)."""
    from janus.writers.evolution import LiveColumn

    shape = load_data_contract(HOSTILE / f"{name}.yaml")
    columns = [(prop.name, prop.physical_type) for prop in shape.schema.properties]
    return tuple(LiveColumn(name, physical, False) for name, physical in (*columns, *extra))


def _plan(
    contract: DataContract,
    live: tuple[Any, ...],
    *,
    recorded: str | None = RECORDED,
    strategy: str = "insert",
    batch_index: int = 1,
) -> Any:
    from janus.writers.evolution import plan_schema_evolution

    return plan_schema_evolution(
        contract=contract,
        live_columns=live,
        recorded_contract_version=recorded,
        write_strategy=strategy,
        batch_index=batch_index,
    )


def _refusal_kinds(plan: Any) -> set[str]:
    return {refusal.kind for refusal in plan.refusals}


# ── the matrix ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("difference", "mode"), MATRIX)
def test_the_evolution_matrix(difference, mode):
    live_shape, next_contract, extra = DIFFERENCES[difference]
    plan = _plan(_contract(next_contract, compatibility=mode), _live(live_shape, extra))

    expected_outcome, expected_kinds = EXPECTED[difference][mode]
    assert plan.outcome == expected_outcome
    assert _refusal_kinds(plan) == expected_kinds


@pytest.mark.parametrize("mode", MODES)
def test_a_major_bump_on_the_first_full_refresh_batch_replaces_the_table(mode):
    """D-10: a MAJOR bump declares a new table, under every mode — ``frozen`` included."""
    plan = _plan(_contract("base_v2", compatibility=mode), _live("base"), strategy="replace_table")

    assert plan.outcome == "breaking_replace"
    assert (plan.recorded_major, plan.contract_major) == (1, 2)
    assert plan.render() == "breaking_replace"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("strategy", "batch_index"),
    [("insert", 1), ("merge_on_keys", 1), ("replace_table", 2)],
    ids=["insert", "merge_on_keys", "replace_table-batch-2"],
)
def test_a_major_bump_outside_the_first_full_refresh_batch_is_refused(mode, strategy, batch_index):
    plan = _plan(
        _contract("base_v2", compatibility=mode),
        _live("base"),
        strategy=strategy,
        batch_index=batch_index,
    )

    assert plan.outcome == "refused"
    assert "MAJOR" in plan.reason


@pytest.mark.parametrize("mode", MODES)
def test_a_breaking_change_without_a_major_bump_is_refused_even_on_a_full_refresh(mode):
    plan = _plan(
        _contract("base_dropped", compatibility=mode), _live("base"), strategy="replace_table"
    )

    assert plan.outcome == "refused"
    assert "full refresh" in plan.reason


# ── the version rule (D-8) ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("version", "major"), [("0.0.0", 0), ("0.1.0", 0), ("1.0.0", 1), ("1.2.3", 1), ("2.0.0", 2)]
)
def test_contract_major_reads_the_first_semver_component(version, major):
    from janus.writers.evolution import contract_major

    assert contract_major(version) == major


def test_an_unstamped_table_counts_as_major_zero_and_admits_one_declared_breaking_change():
    plan = _plan(
        _contract("base_dropped"), _live("base"), recorded=None, strategy="replace_table"
    )

    assert (plan.recorded_major, plan.contract_major) == (0, 1)
    assert plan.outcome == "breaking_replace"


def test_a_legacy_zero_contract_never_authorises_a_breaking_replace():
    legacy = _contract("base_dropped", version="0.0.0")

    plan = _plan(legacy, _live("base"), recorded=None, strategy="replace_table")

    assert plan.outcome == "refused"


@pytest.mark.parametrize(
    ("version", "outcome"), [("1.2.3", "refused"), ("2.0.0", "breaking_replace")]
)
def test_only_a_major_increase_counts_as_a_bump(version, outcome):
    plan = _plan(
        _contract("base_dropped", version=version), _live("base"), strategy="replace_table"
    )

    assert plan.outcome == outcome


# ── what the plan carries ────────────────────────────────────────────────────


def test_promotions_equal_the_table_verified_on_the_pinned_pair():
    from janus.writers.evolution import PROMOTIONS

    assert frozenset(PROMOTIONS) == VERIFIED_PROMOTIONS


def test_the_metadata_key_is_spelled_once():
    from janus.writers.evolution import PLAN_METADATA_KEY

    assert PLAN_METADATA_KEY == "schema_evolution"


def test_the_eight_normalization_columns_are_never_a_difference():
    metadata = tuple((column, "string") for column in NORMALIZATION_METADATA_COLUMNS)

    plan = _plan(_contract("base", compatibility="frozen"), _live("base", metadata))

    assert len(NORMALIZATION_METADATA_COLUMNS) == 8
    assert plan.outcome == "noop"
    assert plan.is_noop is True


def test_an_addition_carries_the_column_and_its_vocabulary_type():
    plan = _plan(_contract("base_plus_nullable"), _live("base"))

    assert plan.add_columns == (("note", "string"),)
    assert plan.promote_columns == ()
    assert plan.evolves is True
    assert plan.render() == "added:note"


def test_a_promotion_carries_both_vocabulary_types():
    plan = _plan(_contract("base_backward"), _live("base_int"))

    assert plan.promote_columns == (("amount", "integer", "long"),)
    assert plan.render() == "promoted:amount(integer->long)"


def test_additions_and_promotions_render_together():
    note = load_data_contract(HOSTILE / "base_plus_nullable.yaml").schema.properties[-1]
    backward = _contract("base_backward")
    contract = replace(
        backward,
        schema=replace(backward.schema, properties=(*backward.schema.properties, note)),
    )

    plan = _plan(contract, _live("base_int"))

    assert plan.outcome == "evolve"
    assert plan.render() == "added:note;promoted:amount(integer->long)"


def test_noop_and_refusals_render_as_documented():
    noop = _plan(_contract("base"), _live("base"))
    refused = _plan(_contract("base"), _live("base_int"))

    assert noop.render() == "none"
    assert refused.render().startswith("refused:")
    assert "retyped" in refused.render()


def test_refusals_are_listed_by_column_then_kind():
    plan = _plan(_contract("base_plus_required"), _live("base_string_amount"))

    assert [(refusal.column, refusal.kind) for refusal in plan.refusals] == [
        ("amount", "retyped"),
        ("code", "newly_required"),
    ]


def test_only_the_empty_diff_is_a_noop():
    plans = {
        "noop": _plan(_contract("base"), _live("base")),
        "evolve": _plan(_contract("base_plus_nullable"), _live("base")),
        "refused": _plan(_contract("base_frozen"), _live("base_int")),
        "breaking_replace": _plan(
            _contract("base_v2"), _live("base"), strategy="replace_table"
        ),
    }

    assert {outcome: plan.outcome for outcome, plan in plans.items()} == {
        outcome: outcome for outcome in plans
    }
    assert [outcome for outcome, plan in plans.items() if plan.is_noop] == ["noop"]
    assert [outcome for outcome, plan in plans.items() if plan.evolves] == ["evolve"]


def test_the_plan_is_frozen():
    plan = _plan(_contract("base"), _live("base"))

    with pytest.raises(FrozenInstanceError):
        plan.outcome = "evolve"


def test_the_planner_is_total_over_every_fixture_pair():
    names = sorted(path.stem for path in HOSTILE.glob("*.yaml") if path.stem != "base_max3")
    contracts = {name: _contract(name) for name in names}
    lives = {name: _live(name) for name in names}

    outcomes = {
        _plan(contract, live, strategy=strategy).outcome
        for contract in contracts.values()
        for live in lives.values()
        for strategy in ("insert", "merge_on_keys", "replace_table")
    }

    assert outcomes <= {"noop", "evolve", "breaking_replace", "refused"}
    assert len(names) >= 19


# ── the module is engine-free (NFR-3) ────────────────────────────────────────


def test_module_imports_no_engine():
    import_paths = (str(PROJECT_ROOT / "src"), *(entry for entry in sys.path if entry))
    command = (
        "import sys; "
        f"sys.path[:0] = {import_paths!r}; "
        "import janus.writers.evolution; "
        "assert not {'pyspark', 'pyiceberg', 'pyarrow'} & sys.modules.keys()"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "pyspark" not in source
    assert "pyiceberg" not in source
