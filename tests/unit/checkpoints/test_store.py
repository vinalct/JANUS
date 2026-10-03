import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import janus.checkpoints.store as store_module
from janus.checkpoints import SUPPORTED_CHECKPOINT_DECISIONS, CheckpointStore
from janus.lineage import MetadataZonePaths
from janus.models import ExecutionPlan, RunContext
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RUN_DECISIONS = frozenset({"advanced", "retained", "reused", "skipped"})
STORED_VALUE = "2026-04-08T12:00:00Z"
RESET_VALUE = "2026-04-01T00:00:00Z"
RESET_AT = datetime(2026, 9, 21, 9, 15, tzinfo=UTC)


def test_checkpoint_store_persists_and_loads_current_state(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-checkpoint-001",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    store = CheckpointStore()

    result = store.save(
        plan,
        "2026-04-08T12:00:00Z",
        metadata={"records_extracted": "100"},
    )

    assert result.decision == "advanced"
    assert result.advanced is True
    assert result.state is not None
    assert result.state.checkpoint_value == "2026-04-08T12:00:00Z"
    assert result.current_path == MetadataZonePaths.from_plan(plan).checkpoint_state_path
    assert result.history_path == MetadataZonePaths.from_plan(plan).checkpoint_history_path(
        "run-checkpoint-001"
    )

    loaded = store.load(plan)
    assert loaded == result.state

    payload = json.loads(result.current_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_field"] == "updated_at"
    assert payload["metadata"] == {"records_extracted": "100"}


def test_checkpoint_store_keeps_newer_value_during_reruns(tmp_path):
    initial_plan = _build_plan(
        tmp_path,
        run_id="run-checkpoint-002",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    store = CheckpointStore()
    store.save(initial_plan, "2026-04-08T12:00:00Z")

    rerun_plan = replace(
        initial_plan,
        run_context=RunContext.create(
            run_id="run-checkpoint-003",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 9, 10, 0, tzinfo=UTC),
        ),
    )
    retained = store.save(rerun_plan, "2026-04-07T12:00:00Z")

    assert retained.decision == "retained"
    assert retained.advanced is False
    assert retained.state is not None
    assert retained.state.run_id == "run-checkpoint-002"
    assert retained.state.checkpoint_value == "2026-04-08T12:00:00Z"

    current_payload = json.loads(
        MetadataZonePaths.from_plan(rerun_plan)
        .checkpoint_state_path.read_text(encoding="utf-8")
    )
    assert current_payload["run_id"] == "run-checkpoint-002"
    assert current_payload["checkpoint_value"] == "2026-04-08T12:00:00Z"

    history_payload = json.loads(retained.history_path.read_text(encoding="utf-8"))
    assert history_payload["candidate_value"] == "2026-04-07T12:00:00Z"
    assert history_payload["stored_value"] == "2026-04-08T12:00:00Z"
    assert history_payload["decision"] == "retained"


def test_checkpoint_store_skips_when_checkpointing_is_disabled(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-checkpoint-004",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    disabled_plan = replace(plan, checkpoint_strategy="none", checkpoint_field=None)
    store = CheckpointStore()

    result = store.save(disabled_plan, "2026-04-08T12:00:00Z")

    assert result.decision == "skipped"
    assert result.state is None
    assert result.current_path is None
    assert result.history_path is None


def _build_plan(tmp_path: Path, *, run_id: str, started_at: datetime) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


# ---------------------------------------------------------------------------------------
# FR-6 / AC-7: an operator reset is a fifth decision with a history entry


def _plan_with_stored_value(tmp_path: Path) -> ExecutionPlan:
    plan = _build_plan(
        tmp_path,
        run_id="run-checkpoint-101",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    CheckpointStore().save(plan, STORED_VALUE, updated_at=datetime(2026, 4, 8, 12, 5, tzinfo=UTC))
    return plan


def _rerun(plan: ExecutionPlan, tmp_path: Path, run_id: str) -> ExecutionPlan:
    return replace(
        plan,
        run_context=RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 22, 10, 0, tzinfo=UTC),
        ),
    )


def test_save_never_decides_reset(tmp_path):
    """`reset` is an operator act; no run path may claim one (it would stop the runs table's
    `checkpoint_decision` from meaning "what the run decided")."""
    plan = _plan_with_stored_value(tmp_path)
    store = CheckpointStore()
    decisions = {
        store.save(_rerun(plan, tmp_path, f"run-checkpoint-1{index:02d}"), value).decision
        for index, value in enumerate(
            (STORED_VALUE, RESET_VALUE, "2026-05-01T00:00:00Z", ""), start=2
        )
    }

    assert decisions == RUN_DECISIONS


def test_reset_is_the_fifth_and_last_checkpoint_decision():
    assert RUN_DECISIONS | {"reset"} == SUPPORTED_CHECKPOINT_DECISIONS


def test_reset_moves_the_checkpoint_backwards_and_records_who_why_and_from_what(tmp_path):
    plan = _plan_with_stored_value(tmp_path)

    result = CheckpointStore().reset(
        plan,
        RESET_VALUE,
        operator="ops-tester",
        reason="upstream republished April",
        recorded_at=RESET_AT,
    )

    assert result.decision == "reset"
    assert result.advanced is False
    assert result.state is not None and result.state.checkpoint_value == RESET_VALUE
    current = json.loads(result.current_path.read_text(encoding="utf-8"))
    assert current["checkpoint_value"] == RESET_VALUE
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history["decision"] == "reset"
    assert history["advanced"] is False
    assert history["candidate_value"] == history["stored_value"] == RESET_VALUE
    assert history["previous_value"] == STORED_VALUE
    assert history["metadata"] == {
        "operator": "ops-tester",
        "reason": "upstream republished April",
        "source": "cli",
        "previous_value": STORED_VALUE,
    }


def test_reset_records_under_a_synthesized_manual_run_id(tmp_path):
    """D-12: a real-looking run id would make an operator act indistinguishable from a run."""
    plan = _plan_with_stored_value(tmp_path)

    result = CheckpointStore().reset(
        plan, RESET_VALUE, operator="ops-tester", reason="backfill", recorded_at=RESET_AT
    )

    assert result.state is not None and result.state.run_id.startswith("manual-")
    assert result.history_path.name.startswith("manual-")
    assert result.history_path.parent == MetadataZonePaths.from_plan(plan).checkpoint_history_dir


@pytest.mark.parametrize(
    ("next_value", "decision"),
    [("2026-04-02T00:00:00Z", "advanced"), ("2026-03-31T00:00:00Z", "retained")],
)
def test_the_next_run_compares_against_the_reset_value(tmp_path, next_value, decision):
    plan = _plan_with_stored_value(tmp_path)
    store = CheckpointStore()
    store.reset(plan, RESET_VALUE, operator="ops-tester", reason="backfill", recorded_at=RESET_AT)

    result = store.save(_rerun(plan, tmp_path, "run-checkpoint-after-reset"), next_value)

    assert result.decision == decision


def test_a_reset_to_the_stored_value_still_writes_history(tmp_path):
    """An operator's no-op is still an act, and still on the record."""
    plan = _plan_with_stored_value(tmp_path)

    result = CheckpointStore().reset(
        plan, STORED_VALUE, operator="ops-tester", reason="confirming", recorded_at=RESET_AT
    )

    assert result.decision == "reset"
    assert json.loads(result.history_path.read_text("utf-8"))["previous_value"] == STORED_VALUE


def test_a_reset_with_no_stored_state_records_no_previous_value(tmp_path):
    plan = _build_plan(tmp_path, run_id="run-checkpoint-102", started_at=RESET_AT)

    result = CheckpointStore().reset(
        plan, RESET_VALUE, operator="ops-tester", reason="seed", recorded_at=RESET_AT
    )

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert result.decision == "reset"
    assert "previous_value" not in history
    assert "previous_value" not in history["metadata"]


def test_reset_is_skipped_for_a_source_without_a_checkpoint(tmp_path):
    plan = _build_plan(tmp_path, run_id="run-checkpoint-103", started_at=RESET_AT)
    disabled_plan = replace(plan, checkpoint_strategy="none", checkpoint_field=None)

    result = CheckpointStore().reset(
        disabled_plan, RESET_VALUE, operator="ops-tester", reason="nothing", recorded_at=RESET_AT
    )

    assert result.decision == "skipped"
    assert result.state is None
    assert not MetadataZonePaths.from_plan(plan).checkpoints_dir.exists()


def test_reset_refuses_a_stored_state_that_belongs_to_another_plan(tmp_path):
    plan = _plan_with_stored_value(tmp_path)
    renamed = replace(plan, checkpoint_field="published_at")

    with pytest.raises(ValueError, match="does not match the current execution plan"):
        CheckpointStore().reset(
            renamed, RESET_VALUE, operator="ops-tester", reason="drift", recorded_at=RESET_AT
        )


def test_reset_writes_its_history_before_moving_the_checkpoint(tmp_path, monkeypatch):
    """A failed history write must leave the checkpoint where it was, never moved silently."""
    plan = _plan_with_stored_value(tmp_path)
    paths = MetadataZonePaths.from_plan(plan)
    write = store_module.write_json_atomic

    def refusing_history(path, payload):
        if paths.checkpoint_history_dir in Path(path).parents:
            raise OSError("history volume is read-only")
        return write(path, payload)

    monkeypatch.setattr(store_module, "write_json_atomic", refusing_history)

    with pytest.raises(OSError, match="read-only"):
        CheckpointStore().reset(
            plan, RESET_VALUE, operator="ops-tester", reason="backfill", recorded_at=RESET_AT
        )

    assert json.loads(paths.checkpoint_state_path.read_text("utf-8"))["checkpoint_value"] == (
        STORED_VALUE
    )


@pytest.mark.parametrize(
    ("value", "operator", "reason"),
    [
        ("  ", "ops-tester", "backfill"),
        (RESET_VALUE, " ", "backfill"),
        (RESET_VALUE, "ops-tester", ""),
    ],
    ids=["value", "operator", "reason"],
)
def test_reset_refuses_a_blank_value_operator_or_reason_before_writing(
    tmp_path, value, operator, reason
):
    plan = _plan_with_stored_value(tmp_path)
    paths = MetadataZonePaths.from_plan(plan)
    before = sorted(path.name for path in paths.checkpoint_history_dir.iterdir())

    with pytest.raises(ValueError, match="must be a non-empty string|must not be empty"):
        CheckpointStore().reset(plan, value, operator=operator, reason=reason, recorded_at=RESET_AT)

    assert sorted(path.name for path in paths.checkpoint_history_dir.iterdir()) == before
    assert CheckpointStore().load(plan).checkpoint_value == STORED_VALUE


def test_an_operator_change_never_overwrites_another_history_record(tmp_path):
    """Two changes in the same second would share a `manual-…` id; the second is refused
    rather than allowed to erase the record of the first."""
    plan = _plan_with_stored_value(tmp_path)
    store = CheckpointStore()
    first = store.reset(
        plan, RESET_VALUE, operator="ops-tester", reason="backfill", recorded_at=RESET_AT
    )
    first_history = first.history_path.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        store.clear_state(plan, operator="ops-tester", reason="re-extract", recorded_at=RESET_AT)

    assert first.history_path.read_bytes() == first_history
    assert store.load(plan).checkpoint_value == RESET_VALUE


def test_clear_state_records_what_it_forgets_and_then_deletes_it(tmp_path):
    """D-13: `stored_value` cannot be blank, so the entry carries the forgotten value twice."""
    plan = _plan_with_stored_value(tmp_path)
    current_path = MetadataZonePaths.from_plan(plan).checkpoint_state_path

    result = CheckpointStore().clear_state(
        plan, operator="ops-tester", reason="re-extract from scratch", recorded_at=RESET_AT
    )

    assert result.decision == "reset"
    assert result.state is None
    assert current_path.exists() is False
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history["candidate_value"] == history["stored_value"] == STORED_VALUE
    assert history["previous_value"] == STORED_VALUE
    assert history["advanced"] is False
    assert history["metadata"]["cleared"] == "true"
    assert history["metadata"]["operator"] == "ops-tester"
    assert history["metadata"]["reason"] == "re-extract from scratch"


def test_clear_state_writes_its_history_before_deleting(tmp_path, monkeypatch):
    plan = _plan_with_stored_value(tmp_path)
    paths = MetadataZonePaths.from_plan(plan)
    write = store_module.write_json_atomic

    def refusing_history(path, payload):
        if paths.checkpoint_history_dir in Path(path).parents:
            raise OSError("history volume is read-only")
        return write(path, payload)

    monkeypatch.setattr(store_module, "write_json_atomic", refusing_history)

    with pytest.raises(OSError, match="read-only"):
        CheckpointStore().clear_state(
            plan, operator="ops-tester", reason="re-extract", recorded_at=RESET_AT
        )

    assert json.loads(paths.checkpoint_state_path.read_text("utf-8"))["checkpoint_value"] == (
        STORED_VALUE
    )


def test_clear_state_with_nothing_stored_writes_nothing(tmp_path):
    plan = _build_plan(tmp_path, run_id="run-checkpoint-104", started_at=RESET_AT)

    result = CheckpointStore().clear_state(
        plan, operator="ops-tester", reason="nothing", recorded_at=RESET_AT
    )

    assert result.decision == "skipped"
    assert not MetadataZonePaths.from_plan(plan).checkpoints_dir.exists()


def test_the_comparator_is_public_and_the_private_names_still_resolve():
    """D-14: `checkpoint set` validates through the store's own comparator, not a similar one."""
    from janus.checkpoints import compare_checkpoint_values, normalize_checkpoint_value

    assert store_module._compare_checkpoint_values is compare_checkpoint_values
    assert store_module._normalize_checkpoint_value is normalize_checkpoint_value
    kinds = [normalize_checkpoint_value(value)[0] for value in (RESET_VALUE, "202401", "latest")]
    assert kinds == ["datetime", "decimal", "text"]
    assert compare_checkpoint_values(STORED_VALUE, RESET_VALUE) == 1
