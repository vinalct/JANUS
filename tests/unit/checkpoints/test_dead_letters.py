import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

import janus.checkpoints.dead_letters as dead_letters_module
from janus.checkpoints import DeadLetterState, DeadLetterStore
from janus.lineage import MetadataZonePaths
from janus.models import ExecutionPlan, RunContext
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RELEASE_RUN_ID = "run-dead-letter-release"
RELEASED_AT = datetime(2026, 9, 20, 8, 30, tzinfo=UTC)
SEEDED_KEYS = ("orgao_codigo=1", "orgao_codigo=2", "orgao_codigo=3")


def test_dead_letter_store_records_and_loads_entries(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-dead-letter-001",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    store = DeadLetterStore()

    state = store.record(
        plan,
        item_key="entity_id=123",
        item_type="request_input",
        error=RuntimeError("missing entity"),
        metadata={"request_url": "https://example.invalid/entities/123"},
    )

    assert state.entry_count == 1
    assert state.item_keys == frozenset({"entity_id=123"})

    loaded = store.load(plan)
    assert loaded == state

    payload = json.loads(store.path(plan).read_text(encoding="utf-8"))
    assert payload["entries"][0]["error_type"] == "RuntimeError"
    assert payload["entries"][0]["metadata"]["request_url"] == "https://example.invalid/entities/123"


def test_dead_letter_store_deduplicates_and_clears_entries(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-dead-letter-002",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    store = DeadLetterStore()

    store.record(
        plan,
        item_key="entity_id=123",
        item_type="request_input",
        error=RuntimeError("missing entity"),
        metadata={"request_url": "https://example.invalid/entities/123"},
    )
    state = store.record(
        plan,
        item_key="entity_id=123",
        item_type="request_input",
        error=RuntimeError("still missing"),
        metadata={"request_url": "https://example.invalid/entities/123"},
    )

    assert state.entry_count == 1

    store.clear(plan)

    assert store.load(plan) is None
    assert store.path(plan).exists() is False


def test_dead_letter_store_loads_existing_state_for_a_new_resume_run_id(tmp_path):
    initial_plan = _build_plan(
        tmp_path,
        run_id="run-dead-letter-003",
        started_at=datetime(2026, 4, 8, 10, 0, tzinfo=UTC),
    )
    store = DeadLetterStore()
    store.record(
        initial_plan,
        item_key="entity_id=123",
        item_type="request_input",
        error=RuntimeError("missing entity"),
        metadata={"request_url": "https://example.invalid/entities/123"},
    )

    resume_plan = _build_plan(
        tmp_path,
        run_id="run-dead-letter-004",
        started_at=datetime(2026, 4, 9, 10, 0, tzinfo=UTC),
    )

    loaded = store.load(resume_plan)

    assert loaded is not None
    assert loaded.item_keys == frozenset({"entity_id=123"})


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
# FR-5 / AC-6: per-entry release with a history record 


def _seed_three(tmp_path: Path) -> tuple[ExecutionPlan, DeadLetterStore]:
    plan = _build_plan(
        tmp_path, run_id=RELEASE_RUN_ID, started_at=datetime(2026, 9, 19, 10, 0, tzinfo=UTC)
    )
    store = DeadLetterStore()
    for minute, key in enumerate(SEEDED_KEYS, start=1):
        store.record(
            plan,
            item_key=key,
            item_type="request_input",
            error=RuntimeError(f"API request failed with status 400 for {key}"),
            metadata={"request_url": f"https://example.invalid/orgaos?{key}"},
            recorded_at=datetime(2026, 9, 19, 10, minute, tzinfo=UTC),
        )
    return plan, store


def _history_files(plan: ExecutionPlan) -> list[Path]:
    history_dir = MetadataZonePaths.from_plan(plan).dead_letters_dir / "history"
    return sorted(history_dir.glob("*.json")) if history_dir.exists() else []


def _file_state(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def test_metadata_zone_paths_name_the_dead_letter_history_location(tmp_path):
    """Mirrors `checkpoint_history_dir`, so both histories share one containment rule."""
    paths = MetadataZonePaths.from_plan(
        _build_plan(tmp_path, run_id=RELEASE_RUN_ID, started_at=RELEASED_AT)
    )

    assert paths.dead_letter_history_dir == paths.dead_letters_dir / "history"
    assert paths.dead_letter_history_path("x") == paths.dead_letters_dir / "history" / "x.json"


def test_release_removes_exactly_the_named_key_and_keeps_the_rest_in_order(tmp_path):
    plan, store = _seed_three(tmp_path)

    record = store.release(
        plan,
        item_keys=["orgao_codigo=2"],
        operator="ops-tester",
        reason="upstream fixed",
        released_at=RELEASED_AT,
    )

    remaining = store.load(plan)
    assert remaining is not None
    assert [entry.item_key for entry in remaining.entries] == ["orgao_codigo=1", "orgao_codigo=3"]
    assert remaining.run_id == RELEASE_RUN_ID, "D-15: the state keeps naming the run that wrote it"
    assert remaining.updated_at == RELEASED_AT
    assert DeadLetterState.from_dict(json.loads(store.path(plan).read_text("utf-8"))) == remaining
    assert [entry.item_key for entry in record.released_entries] == ["orgao_codigo=2"]
    assert record.remaining_item_keys == ("orgao_codigo=1", "orgao_codigo=3")
    assert record.state_run_id == RELEASE_RUN_ID


def test_release_writes_a_history_record_with_the_whole_entries_operator_and_reason(tmp_path):
    """The released entries are kept whole: their error is the only record of *why* they were
    given up on, and `current.json` is about to lose it."""
    plan, store = _seed_three(tmp_path)

    store.release(
        plan,
        item_keys=["orgao_codigo=2"],
        operator="ops-tester",
        reason="upstream fixed",
        released_at=RELEASED_AT,
    )

    (history,) = _history_files(plan)
    assert history.name.lower() == f"20260920t083000z-{RELEASE_RUN_ID}.json"
    payload = json.loads(history.read_text(encoding="utf-8"))
    assert payload["source_id"] == plan.source.source_id
    assert payload["operator"] == "ops-tester"
    assert payload["reason"] == "upstream fixed"
    assert payload["state_run_id"] == RELEASE_RUN_ID
    assert payload["remaining_item_keys"] == ["orgao_codigo=1", "orgao_codigo=3"]
    (released,) = payload["released_entries"]
    assert released["item_key"] == "orgao_codigo=2"
    assert released["item_type"] == "request_input"
    assert released["error_type"] == "RuntimeError"
    assert "status 400" in released["error_message"]
    assert released["metadata"] == {"request_url": "https://example.invalid/orgaos?orgao_codigo=2"}


def test_releasing_every_key_deletes_the_state_file_like_clear_does(tmp_path):
    """D-15: `load` returns None, so a resuming run sees no skip set."""
    plan, store = _seed_three(tmp_path)

    record = store.release(
        plan, item_keys=None, operator="ops-tester", reason="retry all", released_at=RELEASED_AT
    )

    assert store.path(plan).exists() is False
    assert store.load(plan) is None
    assert [entry.item_key for entry in record.released_entries] == list(SEEDED_KEYS)
    assert record.remaining_item_keys == ()
    assert len(_history_files(plan)) == 1


def test_releasing_an_unknown_key_names_both_sets_and_changes_nothing(tmp_path):
    """A typo must never release the wrong entry, or half of a request."""
    from janus.checkpoints import DeadLetterReleaseError

    plan, store = _seed_three(tmp_path)
    before = _file_state(store.path(plan))

    with pytest.raises(DeadLetterReleaseError) as refused:
        store.release(
            plan,
            item_keys=["orgao_codigo=2", "orgao_codigo=9"],
            operator="ops-tester",
            reason="typo",
            released_at=RELEASED_AT,
        )

    assert isinstance(refused.value, ValueError)
    assert "orgao_codigo=9" in str(refused.value)
    assert all(key in str(refused.value) for key in SEEDED_KEYS)
    assert _file_state(store.path(plan)) == before
    assert _history_files(plan) == []


def test_releasing_with_no_recorded_state_is_an_error_and_writes_nothing(tmp_path):
    from janus.checkpoints import DeadLetterReleaseError

    plan = _build_plan(tmp_path, run_id=RELEASE_RUN_ID, started_at=RELEASED_AT)

    with pytest.raises(DeadLetterReleaseError, match=plan.source.source_id):
        DeadLetterStore().release(
            plan, item_keys=None, operator="ops-tester", reason="nothing", released_at=RELEASED_AT
        )

    assert not MetadataZonePaths.from_plan(plan).dead_letters_dir.exists()


def test_the_history_record_is_written_before_the_state_is_touched(tmp_path, monkeypatch):
    """If the record cannot be written, the release did not happen: entries never leave
    `current.json` without a trace of who let them go."""
    plan, store = _seed_three(tmp_path)
    before = _file_state(store.path(plan))
    history_dir = MetadataZonePaths.from_plan(plan).dead_letters_dir / "history"
    write = dead_letters_module.write_json_atomic

    def refusing_history(path, payload):
        if history_dir in Path(path).parents:
            raise OSError("history volume is read-only")
        return write(path, payload)

    monkeypatch.setattr(dead_letters_module, "write_json_atomic", refusing_history)

    with pytest.raises(OSError, match="read-only"):
        store.release(
            plan,
            item_keys=["orgao_codigo=2"],
            operator="ops-tester",
            reason="upstream fixed",
            released_at=RELEASED_AT,
        )

    assert _file_state(store.path(plan)) == before


def test_a_state_run_id_cannot_steer_the_history_file_out_of_its_directory(tmp_path):
    """The file name passes through the planner's `normalize_run_id_segment`."""
    plan, store = _seed_three(tmp_path)
    tampered = json.loads(store.path(plan).read_text(encoding="utf-8"))
    tampered["run_id"] = "../../../escape/run"
    store.path(plan).write_text(json.dumps(tampered), encoding="utf-8")

    store.release(
        plan, item_keys=None, operator="ops-tester", reason="containment", released_at=RELEASED_AT
    )

    (history,) = _history_files(plan)
    assert history.parent == MetadataZonePaths.from_plan(plan).dead_letters_dir / "history"
    assert ".." not in history.name and "/" not in history.name


def test_released_at_must_be_timezone_aware(tmp_path):
    plan, store = _seed_three(tmp_path)
    before = _file_state(store.path(plan))

    with pytest.raises(ValueError, match="timezone-aware"):
        store.release(
            plan,
            item_keys=None,
            operator="ops-tester",
            reason="naive clock",
            released_at=datetime(2026, 9, 20, 8, 30),
        )

    assert _file_state(store.path(plan)) == before


def test_a_released_key_is_recorded_afresh_the_next_time_it_fails(tmp_path):
    """`record` stays idempotent only for keys still present; a released key that fails
    again must produce a new entry, not be swallowed."""
    plan, store = _seed_three(tmp_path)
    store.release(
        plan,
        item_keys=["orgao_codigo=2"],
        operator="ops-tester",
        reason="upstream fixed",
        released_at=RELEASED_AT,
    )

    state = store.record(
        plan,
        item_key="orgao_codigo=2",
        item_type="request_input",
        error=RuntimeError("API request failed with status 400 again"),
    )

    assert [entry.item_key for entry in state.entries] == [
        "orgao_codigo=1",
        "orgao_codigo=3",
        "orgao_codigo=2",
    ]
    assert "again" in state.entries[-1].error_message


def test_a_crash_while_rewriting_the_state_leaves_the_old_state_whole(tmp_path, monkeypatch):
    """The rewrite is a `replace` of a finished temp file: a crash before it lands leaves the
    previous `current.json` intact, and the history record (written first) says what was
    meant, so a retry is safe."""
    plan, store = _seed_three(tmp_path)
    before = _file_state(store.path(plan))
    state_path = store.path(plan)
    replace = Path.replace

    def crashing_replace(self, target):
        if Path(target) == state_path:
            raise OSError("power lost mid-write")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", crashing_replace)

    with pytest.raises(OSError, match="power lost"):
        store.release(
            plan,
            item_keys=["orgao_codigo=2"],
            operator="ops-tester",
            reason="upstream fixed",
            released_at=RELEASED_AT,
        )

    assert _file_state(state_path) == before
    assert store.load(plan).item_keys == frozenset(SEEDED_KEYS)
    (history,) = _history_files(plan)
    assert json.loads(history.read_text("utf-8"))["remaining_item_keys"] == [
        "orgao_codigo=1",
        "orgao_codigo=3",
    ]


def test_a_second_release_in_the_same_second_never_overwrites_the_first_record(tmp_path):
    """Both releases name `<second>-<state run id>.json`; the second is refused rather than
    erase the only record of the first."""
    from janus.checkpoints import DeadLetterReleaseError

    plan, store = _seed_three(tmp_path)
    store.release(
        plan,
        item_keys=["orgao_codigo=1"],
        operator="ops-tester",
        reason="first",
        released_at=RELEASED_AT,
    )
    (history,) = _history_files(plan)
    first_record = history.read_bytes()
    state_before = _file_state(store.path(plan))

    with pytest.raises(DeadLetterReleaseError, match="already on record"):
        store.release(
            plan,
            item_keys=["orgao_codigo=2"],
            operator="ops-tester",
            reason="second",
            released_at=RELEASED_AT.replace(microsecond=500_000),
        )

    assert history.read_bytes() == first_record
    assert _file_state(store.path(plan)) == state_before


def test_a_state_recorded_for_another_source_is_refused_and_left_alone(tmp_path):
    """Rule 1: `load` disowns a file whose `source_id` is not the plan's, and so does
    `release`."""
    from janus.checkpoints import DeadLetterReleaseError

    plan, store = _seed_three(tmp_path)
    foreign = json.loads(store.path(plan).read_text(encoding="utf-8"))
    foreign["source_id"] = "some_other_source"
    store.path(plan).write_text(json.dumps(foreign), encoding="utf-8")
    before = _file_state(store.path(plan))

    with pytest.raises(DeadLetterReleaseError, match=plan.source.source_id):
        store.release(
            plan,
            item_keys=None,
            operator="ops-tester",
            reason="wrong file",
            released_at=RELEASED_AT,
        )

    assert _file_state(store.path(plan)) == before
    assert _history_files(plan) == []


@pytest.mark.parametrize("item_keys", [[], ["  "]], ids=["empty", "blank"])
def test_an_empty_selection_is_not_a_release_of_everything(tmp_path, item_keys):
    """Only `None` means every entry; an empty list is a caller bug, never a wildcard."""
    plan, store = _seed_three(tmp_path)
    before = _file_state(store.path(plan))

    with pytest.raises(ValueError, match="pass None to release every entry"):
        store.release(
            plan,
            item_keys=item_keys,
            operator="ops-tester",
            reason="empty",
            released_at=RELEASED_AT,
        )

    assert _file_state(store.path(plan)) == before
    assert _history_files(plan) == []


def test_a_release_through_a_later_plan_keeps_the_recording_run_as_the_author(tmp_path):
    """D-15, as the command will exercise it: the operator's plan carries a fresh run id,
    and neither the state nor the history file name may take it. The file name is UTC
    whatever zone `released_at` arrives in."""
    plan, store = _seed_three(tmp_path)
    operator_plan = _build_plan(
        tmp_path, run_id="run-operator-plan", started_at=datetime(2026, 9, 20, 8, 0, tzinfo=UTC)
    )
    brasilia = timezone(timedelta(hours=-3))

    record = store.release(
        operator_plan,
        item_keys=["orgao_codigo=3"],
        operator="ops-tester",
        reason="upstream fixed",
        released_at=datetime(2026, 9, 20, 5, 30, tzinfo=brasilia),
    )

    remaining = store.load(operator_plan)
    assert remaining is not None and remaining.run_id == RELEASE_RUN_ID
    assert record.state_run_id == RELEASE_RUN_ID
    (history,) = _history_files(plan)
    assert history.name == f"20260920T083000Z-{RELEASE_RUN_ID}.json"
    assert store.history_path(operator_plan, record) == history
