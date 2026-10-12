"""Mutations refuse overlap; inspection remains available while a source runs."""

import pytest

from janus.checkpoints import CheckpointStore, DeadLetterStore
from janus.lineage import MetadataZonePaths
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.support.parallel_execution import file_snapshot, held_lock, probe_lock
from tests.support.semantics_fixtures import CLEAN, CLEAN_PRODUCER, install_profile, materialize
from tests.unit.cli.test_checkpoint_cli import _seed as seed_checkpoint
from tests.unit.cli.test_dead_letters_cli import _ExecutedRun
from tests.unit.cli.test_dead_letters_cli import _seed as seed_dead_letters


@pytest.fixture
def seeded(tmp_path):
    root = materialize(CLEAN, tmp_path / "project")
    install_profile(root, "local")
    plan = seed_checkpoint(root)
    seed_dead_letters(root)
    return root, plan, MetadataZonePaths.from_plan(plan).base_dir


def invoke(root, *argv):
    return run_janus((*argv, "--project-root", str(root), "--source-id", CLEAN_PRODUCER))


@pytest.mark.parametrize(
    "argv",
    [
        ("dead-letters", "release", "--all", "--reason", "retry"),
        ("checkpoint", "set", "--to", "2026-09-01T00:00:00Z", "--reason", "retry"),
        ("checkpoint", "clear", "--reason", "retry"),
        ("dead-letters", "replay", "--all", "--reason", "retry", "--execute"),
    ],
)
def test_mutating_verb_refuses_without_history_or_current_changes(seeded, monkeypatch, argv):
    arm_spark_tripwire(monkeypatch)
    root, _plan, metadata = seeded
    with held_lock(metadata / ".lock", source_id=CLEAN_PRODUCER) as (_process, _release, holder):
        before = file_snapshot(metadata)
        result = invoke(root, *argv)
        assert result.exit_code == 2
        assert result.stdout == ""
        assert result.stderr.rstrip().endswith("Nothing was changed.")
        for value in (holder["pid"], holder["hostname"], holder["run_id"], holder["acquired_at"]):
            assert str(value) in result.stderr
        assert file_snapshot(metadata) == before


@pytest.mark.parametrize(
    "argv",
    [
        ("dead-letters", "list"),
        ("dead-letters", "replay", "--all", "--reason", "preview"),
        ("checkpoint", "show"),
    ],
)
def test_read_only_verb_succeeds_while_source_locked(seeded, monkeypatch, argv):
    arm_spark_tripwire(monkeypatch)
    root, _plan, metadata = seeded
    with held_lock(metadata / ".lock", source_id=CLEAN_PRODUCER):
        before = file_snapshot(metadata)
        result = invoke(root, *argv)
        assert result.exit_code == 0, result.output
        assert file_snapshot(metadata) == before


def test_replay_holds_one_lock_across_release_and_execution(seeded, monkeypatch):
    from janus.cli import run
    from janus.runtime import SourceExecutor
    from janus.runtime.locking import COMMAND_DEAD_LETTERS_REPLAY, source_state_lock_path

    root, plan, metadata = seeded
    probes = []
    release = DeadLetterStore.release
    run_command = run.run_command

    def checked_release(self, *args, **kwargs):
        probes.append(probe_lock(metadata / ".lock"))
        result = release(self, *args, **kwargs)
        probes.append(probe_lock(metadata / ".lock"))
        return result

    def checked_run(argv):
        probes.append(probe_lock(metadata / ".lock"))
        return run_command(argv)

    def checked_execute(self, planned_run, resolved_plan, layout, *args):
        assert source_state_lock_path(resolved_plan, layout) == metadata / ".lock"
        probes.append(probe_lock(metadata / ".lock"))
        return _ExecutedRun(planned_run)

    monkeypatch.setattr(DeadLetterStore, "release", checked_release)
    monkeypatch.setattr(run, "run_command", checked_run)
    monkeypatch.setattr(SourceExecutor, "_execute_locked", checked_execute)
    result = invoke(root, "dead-letters", "replay", "--all", "--reason", "retry", "--execute")
    assert result.exit_code == 0, result.output
    assert len(probes) == 4 and all(probe["acquired"] is False for probe in probes)
    assert all(probe["holder"] == probes[0]["holder"] for probe in probes)
    assert probes[0]["holder"]["command"] == COMMAND_DEAD_LETTERS_REPLAY
    assert probes[0]["holder"]["run_id"] is None
    assert probes[0]["holder"]["pipeline_run_id"] is None
    assert probe_lock(metadata / ".lock")["acquired"] is True
    assert CheckpointStore().load(plan) is not None


@pytest.mark.parametrize(
    "store,argv",
    [
        (
            CheckpointStore,
            ("checkpoint", "set", "--to", "2026-09-01T00:00:00Z", "--reason", "retry"),
        ),
        (CheckpointStore, ("checkpoint", "clear", "--reason", "retry")),
        (DeadLetterStore, ("dead-letters", "release", "--all", "--reason", "retry")),
        (DeadLetterStore, ("dead-letters", "replay", "--all", "--reason", "retry", "--execute")),
    ],
)
def test_competing_writer_is_refused_after_the_validation_read(seeded, monkeypatch, store, argv):
    from janus.runtime import SourceExecutor

    arm_spark_tripwire(monkeypatch)
    root, _plan, metadata = seeded
    load = store.load
    probes = []

    def checked_load(self, *args, **kwargs):
        state = load(self, *args, **kwargs)
        probes.append(probe_lock(metadata / ".lock"))
        return state

    monkeypatch.setattr(store, "load", checked_load)
    monkeypatch.setattr(
        SourceExecutor, "_execute_locked", lambda self, planned, *a: _ExecutedRun(planned)
    )
    result = invoke(root, *argv)
    assert result.exit_code == 0, result.output
    assert probes and all(probe["acquired"] is False for probe in probes)
    assert probe_lock(metadata / ".lock")["acquired"] is True


def test_replay_release_refusal_unlocks_without_starting_the_run(seeded, monkeypatch):
    from janus.cli import run

    root, _plan, metadata = seeded
    before = file_snapshot(metadata)

    def unexpected_run(*args):
        pytest.fail("a refused release started a run")

    monkeypatch.setattr(run, "run_command", unexpected_run)
    result = invoke(
        root, "dead-letters", "replay", "--item-key", "absent", "--reason", "retry", "--execute"
    )
    assert result.exit_code == 2
    assert {
        key: value for key, value in file_snapshot(metadata).items() if key != ".lock"
    } == before
    assert probe_lock(metadata / ".lock")["acquired"] is True


@pytest.mark.parametrize("failure", [RuntimeError("run failed"), KeyboardInterrupt()])
def test_replay_run_exception_releases_the_outer_lock(seeded, monkeypatch, failure):
    from janus.runtime import SourceExecutor

    root, _plan, metadata = seeded

    def failed_run(*args):
        raise failure

    monkeypatch.setattr(SourceExecutor, "_execute_locked", failed_run)
    with pytest.raises(type(failure)):
        invoke(root, "dead-letters", "replay", "--all", "--reason", "retry", "--execute")
    assert probe_lock(metadata / ".lock")["acquired"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ("checkpoint", "set", "--to", "2026-09-01T00:00:00Z", "--reason", "retry"),
        ("checkpoint", "clear", "--reason", "retry"),
        ("dead-letters", "release", "--all", "--reason", "retry"),
        ("dead-letters", "replay", "--all", "--reason", "retry", "--execute"),
    ],
)
def test_unsupported_lock_refuses_without_mutation(seeded, monkeypatch, argv):
    from janus.cli import common
    from janus.runtime.locking import LockUnsupportedError

    root, _plan, metadata = seeded
    before = file_snapshot(metadata)

    def unsupported(*args, **kwargs):
        raise LockUnsupportedError("source-state locking requires POSIX fcntl")

    monkeypatch.setattr(common, "hold_source_state_lock", unsupported)
    result = invoke(root, *argv)
    assert result.exit_code == 2
    assert result.stderr.rstrip().endswith("Nothing was changed.")
    assert "POSIX fcntl" in result.stderr
    assert file_snapshot(metadata) == before
