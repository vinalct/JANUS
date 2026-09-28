"""AC-4 on a real Iceberg table: every cell of the evolution matrix, for every write strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")

pytestmark = pytest.mark.xfail(strict=True, reason="red until finish implementation")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
STRATEGIES = ("insert", "merge_on_keys", "replace_table")

DIFFERENCES: dict[str, tuple[str, str, tuple[str, ...]]] = {
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
    "undeclared_live_column": ("base", "base", ("legacy_col string",)),
    "major_bump": ("base", "base_v2", ()),
}

CELL_SPECS: tuple[tuple[str, str, str, str | None, tuple[str, ...]], ...] = (
    ("identical", "additive", "noop", "none", ()),
    ("add_nullable", "additive", "evolve", "added:note", ("ADD COLUMNS (`note` string)",)),
    ("add_nullable", "frozen", "refused", None, ()),
    ("add_required", "additive", "refused", None, ()),
    ("int_to_long", "additive", "refused", None, ()),
    (
        "int_to_long",
        "backward",
        "evolve",
        "promoted:amount(integer->long)",
        ("ALTER COLUMN `amount` TYPE bigint",),
    ),
    ("int_to_long", "frozen", "refused", None, ()),
    ("float_to_double", "additive", "refused", None, ()),
    (
        "float_to_double",
        "backward",
        "evolve",
        "promoted:amount(float->double)",
        ("ALTER COLUMN `amount` TYPE double",),
    ),
    ("decimal_widening", "additive", "refused", None, ()),
    (
        "decimal_widening",
        "backward",
        "evolve",
        "promoted:amount(decimal(10,2)->decimal(18,2))",
        ("ALTER COLUMN `amount` TYPE decimal(18,2)",),
    ),
    ("decimal_scale_change", "backward", "refused", None, ()),
    ("long_to_int", "backward", "refused", None, ()),
    ("string_to_long", "backward", "refused", None, ()),
    ("drop_column", "additive", "refused", None, ()),
    ("rename_column", "additive", "refused", None, ()),
    ("undeclared_live_column", "additive", "refused", None, ()),
    ("major_bump", "additive", "breaking_replace", "breaking_replace", ("REPLACE TABLE",)),
    ("major_bump", "frozen", "breaking_replace", "breaking_replace", ("REPLACE TABLE",)),
)


@dataclass(frozen=True)
class CellSpec:

    id: str
    difference: str
    compatibility: str
    strategy: str
    live_contract: Path
    next_contract: Path
    live_extra_columns: tuple[str, ...]
    expected_outcome: str
    expected_render: str | None
    expected_ddl: tuple[str, ...]


def _cells() -> tuple[CellSpec, ...]:
    cells = []
    for strategy in STRATEGIES:
        for difference, compatibility, declared_outcome, declared_render, declared_ddl in (
            CELL_SPECS
        ):
            live, following, extra = DIFFERENCES[difference]
            outcome, render, ddl = declared_outcome, declared_render, declared_ddl
            if outcome == "breaking_replace" and strategy != "replace_table":
                outcome, render, ddl = "refused", None, ()
            cells.append(
                CellSpec(
                    id=f"{difference}-{compatibility}-{strategy}",
                    difference=difference,
                    compatibility=compatibility,
                    strategy=strategy,
                    live_contract=HOSTILE / f"{live}.yaml",
                    next_contract=HOSTILE / f"{following}.yaml",
                    live_extra_columns=extra,
                    expected_outcome=outcome,
                    expected_render=render,
                    expected_ddl=ddl,
                )
            )
    return tuple(cells)


CELLS = _cells()


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from tests.support.spark_sessions import build_iceberg_session

    session = build_iceberg_session(
        "janus-evolution-matrix", tmp_path_factory.mktemp("janus-evolution-matrix")
    )
    yield session
    session.stop()


def _run(spark: Any, tmp_path: Path, spec: CellSpec) -> Any:
    from tests.support.evolution_harness import EvolutionCell, run_cell

    return run_cell(
        spark,
        tmp_path,
        EvolutionCell(
            id=spec.id,
            live_contract=spec.live_contract,
            next_contract=spec.next_contract,
            compatibility=spec.compatibility,
            strategy=spec.strategy,
            live_extra_columns=spec.live_extra_columns,
            expected_outcome=spec.expected_outcome,
            expected_ddl=spec.expected_ddl,
        ),
    )


def _in_order(statements: list[str], fragments: tuple[str, ...]) -> bool:
    position = 0
    for fragment in fragments:
        matches = [index for index, sql in enumerate(statements) if fragment in sql]
        later = [index for index in matches if index >= position]
        if not later:
            return False
        position = later[0] + 1
    return True


@pytest.mark.parametrize("spec", CELLS, ids=[cell.id for cell in CELLS])
def test_the_evolution_matrix_cell(spark, tmp_path, spec):
    result = _run(spark, tmp_path, spec)
    next_version = result.next_contract_version

    if spec.expected_outcome == "refused":
        assert type(result.error).__name__ == "SchemaEvolutionRefusedError"
        assert result.snapshots_after == result.snapshots_before
        assert result.schema_after == result.schema_before
        assert result.stamp_after == result.stamp_before
        return

    assert result.error is None, result.error
    assert result.write_metadata["schema_evolution"] == spec.expected_render
    assert _in_order(result.statements, spec.expected_ddl)
    assert result.snapshots_after == result.snapshots_before + 1
    assert result.uuid_after == result.uuid_before
    assert result.stamp_after["janus.contract_version"] == next_version

    if spec.expected_outcome == "noop":
        assert not any("ALTER COLUMN" in sql or "ADD COLUMNS" in sql for sql in result.statements)
        assert result.schema_after == result.schema_before
    elif spec.expected_outcome == "evolve":
        assert result.run_1_rows_after == result.run_1_rows_expected
    else:
        metadata = result.write_metadata
        assert metadata["overwrite_mechanism"] == "replace_table"
        assert metadata["history_reset_reason"].startswith("contract major version 1 -> 2")
        assert result.run_1_is_current_ancestor is False
        assert "not an ancestor" in result.rollback_to_run_1_error


def test_the_matrix_covers_every_difference_for_every_strategy():
    """The parametrisation is the spine of AC-4; it must not shrink silently."""
    assert len(CELLS) == len(CELL_SPECS) * len(STRATEGIES) == 57
    for strategy in STRATEGIES:
        covered = {cell.difference for cell in CELLS if cell.strategy == strategy}
        assert covered == set(DIFFERENCES)
    from tests.support import evolution_harness

    assert evolution_harness.run_cell


def test_a_legacy_zero_contract_cannot_authorise_a_breaking_replace(spark, tmp_path):
    """Against an unstamped table a ``0.0.0`` contract is major 0 on both sides (D-8)."""
    from tests.support.evolution_harness import EvolutionCell, run_cell

    if not (PROJECT_ROOT / "src" / "janus" / "models" / "data_contracts" / "legacy.py").exists():
        pytest.skip("legacy contracts were retired; the rule is pinned on the host")

    result = run_cell(
        spark,
        tmp_path,
        EvolutionCell(
            id="legacy_zero-additive-replace_table",
            live_contract=HOSTILE / "base.yaml",
            next_contract=HOSTILE / "base_dropped.yaml",
            compatibility="additive",
            strategy="replace_table",
            next_version="0.0.0",
            stamp_live_table=False,
            expected_outcome="refused",
            expected_ddl=(),
        ),
    )

    assert type(result.error).__name__ == "SchemaEvolutionRefusedError"


def test_an_unstamped_table_admits_one_declared_breaking_refresh_then_refuses(spark, tmp_path):
    from tests.support.evolution_harness import EvolutionCell, run_cell

    first = run_cell(
        spark,
        tmp_path / "first",
        EvolutionCell(
            id="unstamped-additive-replace_table",
            live_contract=HOSTILE / "base.yaml",
            next_contract=HOSTILE / "base_dropped.yaml",
            compatibility="additive",
            strategy="replace_table",
            stamp_live_table=False,
            expected_outcome="breaking_replace",
            expected_ddl=("REPLACE TABLE",),
        ),
    )

    again = first.write_again(next_contract=HOSTILE / "base_narrowed.yaml")

    assert first.error is None, first.error
    assert first.write_metadata["overwrite_mechanism"] == "replace_table"
    assert first.stamp_after["janus.contract_version"] == "1.1.0"
    assert type(again.error).__name__ == "SchemaEvolutionRefusedError"
