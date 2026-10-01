"""One search resolves a declared contract file.

The four steps — absolute as written, project-relative when it exists, each parent of the
source config, and the runtime path anyway when nothing exists — are what an operator
relies on when they write ``conf/contracts/<domain>/<table>.yaml`` into a source. They
keep source declarations stable across project and config-relative paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.registry import load_registry
from janus.schema_contracts import (
    resolve_contract_path_for_plan,
    resolve_declared_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STARTED_AT = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


@pytest.fixture
def source_config() -> SourceConfig:
    registry = load_registry(PROJECT_ROOT)
    return registry.get_source("federal_open_data_example")


def _plan(source_config: SourceConfig, project_root: Path) -> ExecutionPlan:
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-paths-001",
            environment="local",
            project_root=project_root,
            started_at=STARTED_AT,
        ),
    )


# ── the four steps ────────────────────────────────────────────────────────────


def test_an_absolute_declaration_is_used_as_written(tmp_path: Path) -> None:
    declared = tmp_path / "elsewhere" / "contract.yaml"

    resolved = resolve_declared_path(tmp_path, tmp_path / "conf" / "s.yaml", str(declared))

    assert resolved == declared


def test_a_project_relative_declaration_wins_when_it_exists(tmp_path: Path) -> None:
    declared = tmp_path / "conf" / "contracts" / "example" / "c.yaml"
    declared.parent.mkdir(parents=True)
    declared.write_text("contract", encoding="utf-8")

    resolved = resolve_declared_path(
        tmp_path,
        tmp_path / "conf" / "sources" / "example" / "s.yaml",
        "conf/contracts/example/c.yaml",
    )

    assert resolved == declared


def test_a_parent_of_the_config_is_searched_next(tmp_path: Path) -> None:
    config_path = tmp_path / "conf" / "sources" / "example" / "s.yaml"
    config_path.parent.mkdir(parents=True)
    beside = config_path.parent / "c.yaml"
    beside.write_text("contract", encoding="utf-8")

    resolved = resolve_declared_path(tmp_path / "other-root", config_path, "c.yaml")

    assert resolved == beside


def test_nothing_found_still_returns_the_runtime_path(tmp_path: Path) -> None:
    """A missing file must surface as "does not exist", never as a silent None."""
    resolved = resolve_declared_path(
        tmp_path, tmp_path / "conf" / "s.yaml", "conf/contracts/example/missing.yaml"
    )

    assert resolved == tmp_path / "conf" / "contracts" / "example" / "missing.yaml"
    assert not resolved.exists()


def test_an_undeclared_file_resolves_to_nothing(tmp_path: Path) -> None:
    assert resolve_declared_path(tmp_path, tmp_path / "conf" / "s.yaml", None) is None
    assert resolve_declared_path(tmp_path, tmp_path / "conf" / "s.yaml", "") is None


# ── the plan-scoped entry point ───────────────────────────────────────────────


def test_the_plan_free_entry_point_agrees_with_the_plan_one(
    source_config: SourceConfig,
) -> None:
    """Registry resolves without a plan; it must land on the same file."""
    plan = _plan(source_config, PROJECT_ROOT)

    assert resolve_declared_path(
        PROJECT_ROOT, source_config.config_path, source_config.schema.contract
    ) == resolve_contract_path_for_plan(plan)
