from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.checkpoints import CheckpointStore
from janus.lineage import MetadataZonePaths
from janus.models import ExecutionPlan
from janus.planner import Planner, PlanningRequest
from tests.support.operator_cli import OPERATOR_ENV, arm_spark_tripwire, run_janus
from tests.support.semantics_fixtures import (
    CLEAN,
    CLEAN_CONSUMER,
    CLEAN_PRODUCER,
    materialize,
    tree_snapshot,
)

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="janus.cli.dispatch registers no `checkpoint` verb yet "
    "(and CheckpointStore.reset/clear_state)",
)

SEEDED = (
    ("run-a", "2026-09-01T00:00:00Z", datetime(2026, 9, 1, 6, 0, tzinfo=UTC)),
    ("run-b", "2026-09-02T00:00:00Z", datetime(2026, 9, 2, 6, 0, tzinfo=UTC)),
    ("run-c", "2026-09-03T00:00:00Z", datetime(2026, 9, 3, 6, 0, tzinfo=UTC)),
)
LATEST = SEEDED[-1][1]
BACKWARDS = "2026-08-15T00:00:00Z"
OPERATOR = {OPERATOR_ENV: "ops-tester"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return materialize(CLEAN, tmp_path / "project")


def _plan(root: Path, source_id: str = CLEAN_PRODUCER, run_id: str = "run-seed") -> ExecutionPlan:
    request = PlanningRequest.create(
        source_id=source_id,
        environment="local",
        project_root=root,
        run_id=run_id,
        started_at=datetime(2026, 9, 18, 6, 0, tzinfo=UTC),
        include_disabled=True,
    )
    return Planner().plan(request).plan


def _seed(root: Path) -> ExecutionPlan:
    store = CheckpointStore()
    for run_id, value, recorded_at in SEEDED:
        store.save(_plan(root, run_id=run_id), value, updated_at=recorded_at)
    return _plan(root)


def _checkpoint(root: Path, action: str, *extra: str, source_id: str = CLEAN_PRODUCER):
    return run_janus(
        ("checkpoint", action, "--project-root", str(root), "--source-id", source_id, *extra),
        env=OPERATOR,
    )


def _manual_history(plan: ExecutionPlan) -> dict:
    (path,) = MetadataZonePaths.from_plan(plan).checkpoint_history_dir.glob("manual-*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _current_value(plan: ExecutionPlan) -> str:
    state = CheckpointStore().load(plan)
    assert state is not None
    return state.checkpoint_value


# ---------------------------------------------------------------------------------------
# show


def test_show_prints_the_current_state_and_the_last_n_history_entries(root: Path) -> None:
    _seed(root)

    result = _checkpoint(root, "show", "--history", "2", "--format", "json")
    payload = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert payload["source_id"] == CLEAN_PRODUCER
    assert payload["checkpoint_field"] == "updated_at"
    assert payload["checkpoint_strategy"] == "max_value"
    assert payload["value"] == LATEST
    assert [entry["run_id"] for entry in payload["history"]] == ["run-c", "run-b"]
    assert result.stdout == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_show_text_names_the_field_strategy_value_and_run(root: Path) -> None:
    _seed(root)

    result = _checkpoint(root, "show")

    assert result.exit_code == 0, result.output
    for expected in (CLEAN_PRODUCER, "updated_at", "max_value", LATEST, "run-c"):
        assert expected in result.stdout


def test_history_zero_prints_no_history(root: Path) -> None:
    _seed(root)

    result = _checkpoint(root, "show", "--history", "0", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["history"] == []


def test_show_on_a_source_without_a_checkpoint_is_a_clear_answer(root: Path) -> None:
    result = _checkpoint(root, "show", source_id=CLEAN_CONSUMER)

    assert result.exit_code == 0, result.output
    assert f"{CLEAN_CONSUMER} declares no checkpoint (strategy: none)" in result.stdout


def test_show_with_no_state_is_a_clear_answer(root: Path) -> None:
    result = _checkpoint(root, "show")

    assert result.exit_code == 0, result.output
    assert f"no checkpoint recorded for {CLEAN_PRODUCER}" in result.stdout


def test_show_refuses_a_stored_state_that_no_longer_matches_the_plan(root: Path) -> None:
    """The contract changed under a stored checkpoint: say so, never print a confusing value."""
    plan = _plan(root)
    CheckpointStore().save(replace(plan, checkpoint_field="published_at"), LATEST)

    result = _checkpoint(root, "show")

    assert result.exit_code == 2
    assert "does not match the current execution plan" in result.stderr


def test_a_malformed_history_file_does_not_hide_the_current_state(root: Path) -> None:
    plan = _seed(root)
    history_dir = MetadataZonePaths.from_plan(plan).checkpoint_history_dir
    (history_dir / "zzz-broken.json").write_text("{not json", encoding="utf-8")

    result = _checkpoint(root, "show", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["value"] == LATEST


# ---------------------------------------------------------------------------------------
# set and clear


@pytest.mark.parametrize(
    "argv", [("set", "--to", BACKWARDS), ("clear",)], ids=["set", "clear"]
)
def test_state_changes_refuse_to_run_without_a_reason(root: Path, argv: tuple[str, ...]) -> None:
    plan = _seed(root)
    before = tree_snapshot(root)

    result = _checkpoint(root, *argv)

    assert result.exit_code == 2
    assert "--reason" in result.stderr
    assert tree_snapshot(root) == before
    assert _current_value(plan) == LATEST


def test_set_refuses_a_value_of_another_kind(root: Path) -> None:
    """D-14: a `datetime` checkpoint set to text would compare lexicographically forever."""
    plan = _seed(root)
    before = tree_snapshot(root)

    result = _checkpoint(root, "set", "--to", "latest", "--reason", "wrong kind")

    assert result.exit_code == 2
    for expected in ("datetime", "text", LATEST):
        assert expected in result.stderr
    assert tree_snapshot(root) == before
    assert _current_value(plan) == LATEST


@pytest.mark.parametrize("value", ["", "   "], ids=["empty", "blank"])
def test_set_refuses_an_empty_value(root: Path, value: str) -> None:
    plan = _seed(root)
    before = tree_snapshot(root)

    result = _checkpoint(root, "set", "--to", value, "--reason", "empty")

    assert result.exit_code == 2
    assert "--to" in result.stderr
    assert tree_snapshot(root) == before
    assert _current_value(plan) == LATEST


def test_set_prints_the_backwards_transition_and_records_a_reset(root: Path) -> None:
    """PRD §8 risk 4: the previous value and the direction are printed before the write."""
    plan = _seed(root)

    result = _checkpoint(root, "set", "--to", BACKWARDS, "--reason", "upstream republished")

    assert result.exit_code == 0, result.output
    assert f"{LATEST} -> {BACKWARDS}" in result.stdout
    assert "backwards" in result.stdout
    assert _current_value(plan) == BACKWARDS
    history = _manual_history(plan)
    assert history["decision"] == "reset"
    assert history["advanced"] is False
    assert history["previous_value"] == LATEST
    assert history["metadata"]["operator"] == "ops-tester"
    assert history["metadata"]["reason"] == "upstream republished"


@pytest.mark.parametrize(
    ("next_value", "decision"),
    [("2026-08-16T00:00:00Z", "advanced"), ("2026-08-14T00:00:00Z", "retained")],
)
def test_the_next_run_advances_from_the_value_set(
    root: Path, next_value: str, decision: str
) -> None:
    _seed(root)
    assert _checkpoint(root, "set", "--to", BACKWARDS, "--reason", "backfill").exit_code == 0

    result = CheckpointStore().save(_plan(root, run_id="run-after-set"), next_value)

    assert result.decision == decision


def test_set_with_no_stored_state_accepts_any_non_empty_value(root: Path) -> None:
    plan = _plan(root)

    result = _checkpoint(root, "set", "--to", "202401", "--reason", "seed")

    assert result.exit_code == 0, result.output
    assert _current_value(plan) == "202401", "--to is a string: a leading zero must survive"


def test_set_on_a_source_without_a_checkpoint_exits_2(root: Path) -> None:
    result = _checkpoint(root, "set", "--to", BACKWARDS, "--reason", "nothing reads it",
                         source_id=CLEAN_CONSUMER)

    assert result.exit_code == 2
    assert "none" in result.stderr


def test_clear_records_what_it_forgets_and_deletes_the_state(root: Path) -> None:
    plan = _seed(root)

    result = _checkpoint(root, "clear", "--reason", "re-extract from scratch")

    assert result.exit_code == 0, result.output
    assert LATEST in result.stdout
    assert MetadataZonePaths.from_plan(plan).checkpoint_state_path.exists() is False
    history = _manual_history(plan)
    assert history["metadata"]["cleared"] == "true"
    assert history["stored_value"] == LATEST


def test_clear_with_no_state_writes_nothing(root: Path) -> None:
    before = tree_snapshot(root)

    result = _checkpoint(root, "clear", "--reason", "nothing stored")

    assert result.exit_code == 0, result.output
    assert tree_snapshot(root) == before


@pytest.mark.parametrize(
    "argv",
    [("show",), ("set", "--to", BACKWARDS, "--reason", "tripwire"), ("clear", "--reason", "x")],
    ids=["show", "set", "clear"],
)
def test_checkpoint_verbs_never_acquire_a_spark_session(
    root: Path, monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...]
) -> None:
    _seed(root)
    arm_spark_tripwire(monkeypatch)

    assert _checkpoint(root, *argv).exit_code == 0
