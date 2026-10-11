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

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="operator verbs do not hold the source-state lock",
)


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
        ("checkpoint", "set", "--value", "2026-09-01", "--reason", "retry"),
        ("checkpoint", "clear", "--reason", "retry"),
    ],
)
def test_mutating_verb_refuses_without_history_or_current_changes(seeded, monkeypatch, argv):
    from janus.runtime.locking import SourceStateLock  # noqa: F401

    arm_spark_tripwire(monkeypatch)
    root, _plan, metadata = seeded
    with held_lock(metadata / ".lock"):
        before = file_snapshot(metadata)
        result = invoke(root, *argv)
        assert result.exit_code == 2
        assert "nothing was changed" in result.stderr
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
    from janus.runtime.locking import SourceStateLock  # noqa: F401

    arm_spark_tripwire(monkeypatch)
    root, _plan, metadata = seeded
    with held_lock(metadata / ".lock"):
        before = file_snapshot(metadata)
        result = invoke(root, *argv)
        assert result.exit_code == 0, result.output
        assert file_snapshot(metadata) == before


def test_replay_holds_one_lock_across_release_and_execution(seeded, monkeypatch):
    from janus.runtime import SourceExecutor
    from janus.runtime.locking import SourceStateLock  # noqa: F401

    root, plan, metadata = seeded
    probes = []
    release = DeadLetterStore.release

    def checked_release(self, *args, **kwargs):
        result = release(self, *args, **kwargs)
        probes.append(probe_lock(metadata / ".lock"))
        return result

    def checked_execute(self, planned_run, *args):
        probes.append(probe_lock(metadata / ".lock"))
        return _ExecutedRun(planned_run)

    monkeypatch.setattr(DeadLetterStore, "release", checked_release)
    monkeypatch.setattr(SourceExecutor, "execute", checked_execute)
    result = invoke(root, "dead-letters", "replay", "--all", "--reason", "retry", "--execute")
    assert result.exit_code == 0, result.output
    assert len(probes) == 2 and all(probe["acquired"] is False for probe in probes)
    assert probes[0]["holder"] == probes[1]["holder"]
    assert probe_lock(metadata / ".lock")["acquired"] is True
    assert CheckpointStore().load(plan) is not None
