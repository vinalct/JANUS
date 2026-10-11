"""FR-1 and AC-2 against real spawned contenders, never inherited Python state."""

import json
import multiprocessing
import os
import signal
import socket
import sys
import time
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from tests.support.parallel_execution import (
    exec_lock_holder,
    held_lock,
    lock_at,
    planned_project,
    probe_lock,
    try_lock,
)

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="source-state flock primitive is not implemented",
)


def test_acquire_writes_exact_utc_holder_and_holds_kernel_lock(tmp_path):
    path = tmp_path / "fresh/.lock"
    with lock_at(path):
        record = json.loads(path.read_bytes())
        assert set(record) == {
            "pid",
            "hostname",
            "run_id",
            "pipeline_run_id",
            "source_id",
            "acquired_at",
            "command",
        }
        assert record["pid"] == os.getpid()
        assert record["hostname"] == socket.gethostname()
        assert record["source_id"] == "A" and record["run_id"] == "holder-run"
        assert record["pipeline_run_id"] is None
        assert datetime.fromisoformat(record["acquired_at"]).utcoffset() == UTC.utcoffset(None)
        refused = probe_lock(path)
        assert refused["acquired"] is False
        assert refused["holder"] == record
        assert refused["elapsed"] < 0.1
        for field in ("pid", "hostname", "run_id", "acquired_at"):
            assert str(record[field]) in refused["message"]


@pytest.mark.parametrize("exception", [None, RuntimeError, KeyboardInterrupt])
def test_context_releases_after_every_exit(tmp_path, exception):
    path = tmp_path / ".lock"
    if exception is None:
        with lock_at(path):
            assert probe_lock(path)["acquired"] is False
    else:
        with pytest.raises(exception), lock_at(path):
            raise exception("scripted context exit")
    assert probe_lock(path)["acquired"] is True


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_kernel_releases_when_holder_dies(tmp_path, sig):
    path = tmp_path / ".lock"
    with held_lock(path) as (process, _release, _holder):
        os.kill(process.pid, sig)
        process.join(5)
        assert process.exitcode == -sig
        with lock_at(path):
            assert path.is_file()


def test_waiter_acquires_after_event_driven_release(tmp_path):
    from janus.runtime.locking import SourceStateLock  # noqa: F401

    path = tmp_path / ".lock"
    context = multiprocessing.get_context("spawn")
    attempting = context.Event()
    reader, writer = context.Pipe(duplex=False)
    with held_lock(path) as (_holder, release, _record):
        contender = context.Process(target=try_lock, args=(path, writer, 1.5, attempting))
        contender.start()
        writer.close()
        try:
            # Timed waiting is the behavior under test; Event is the release handshake.
            assert attempting.wait(10)
            assert not release.wait(0.5)
            release.set()
            assert reader.poll(5)
            assert reader.recv()["acquired"] is True
            contender.join(5)
            assert contender.exitcode == 0
        finally:
            if contender.is_alive():
                contender.kill()
                contender.join(5)
            reader.close()


def test_wait_timeout_is_bounded(tmp_path):
    from janus.runtime.locking import SourceLockedError

    path = tmp_path / ".lock"
    with held_lock(path):
        started = time.monotonic()
        with pytest.raises(SourceLockedError), lock_at(path, wait_seconds=1.5):
            pytest.fail("a contended lock was acquired")
        assert 1.5 <= time.monotonic() - started < 2.5


def test_exec_does_not_inherit_lock(tmp_path):
    from janus.runtime.locking import SourceStateLock  # noqa: F401

    context = multiprocessing.get_context("spawn")
    ready, proceed = context.Event(), context.Event()
    reader, writer = context.Pipe(duplex=False)
    path = tmp_path / ".lock"
    process = context.Process(target=exec_lock_holder, args=(path, ready, proceed, writer))
    process.start()
    writer.close()
    try:
        assert ready.wait(15) and reader.poll(1)
        pid = reader.recv()
        assert probe_lock(path)["acquired"] is False
        proceed.set()
        # EOF on the pipe proves exec closed its descriptors; sleep is still alive.
        assert reader.poll(5)
        with pytest.raises(EOFError):
            reader.recv()
        with lock_at(path):
            assert process.is_alive() and process.pid == pid
    finally:
        proceed.set()
        if process.is_alive():
            process.kill()
        process.join(5)
        reader.close()


def test_inode_survives_release_and_second_acquisition(tmp_path):
    path = tmp_path / ".lock"
    path.touch()
    inode = path.stat().st_ino
    for _ in range(2):
        with lock_at(path):
            assert path.stat().st_ino == inode
        assert path.exists() and path.stat().st_ino == inode


def test_nested_acquire_keeps_outer_record_and_kernel_lock(tmp_path):
    path = tmp_path / ".lock"
    with lock_at(path):
        outer = path.read_bytes()
        with lock_at(path, source_id="C", run_id="inner"):
            assert path.read_bytes() == outer
        assert path.read_bytes() == outer
        assert probe_lock(path)["acquired"] is False
    assert probe_lock(path)["acquired"] is True


def test_unreadable_holder_still_refuses(tmp_path):
    path = tmp_path / ".lock"
    with lock_at(path):
        with path.open("r+b") as stream:
            stream.write(b"\x00{")
            stream.truncate()
        result = probe_lock(path)
        assert result["acquired"] is False and result["holder"] is None
        assert "unreadable" in result["message"]


def test_missing_fcntl_fails_closed(tmp_path, monkeypatch):
    from janus.runtime.locking import LockUnsupportedError

    monkeypatch.setitem(sys.modules, "fcntl", None)
    with pytest.raises(LockUnsupportedError, match="fcntl"), lock_at(tmp_path / ".lock"):
        pytest.fail("unsupported lock acquired")


def test_metadata_uri_fails_closed(tmp_path):
    from dataclasses import replace

    from janus.runtime.locking import LockUnsupportedError, source_state_lock_path
    from janus.utils.storage import StorageLayout

    _root, _registry, batch, config, _paths = planned_project(tmp_path)
    plan = batch.source("A").require_planned_run().plan
    changed = replace(plan, metadata_output=replace(plan.metadata_output, path="s3://bucket/state"))
    layout = StorageLayout.from_environment_config(config, batch.request.project_root)
    with pytest.raises(LockUnsupportedError, match="URI"):
        source_state_lock_path(changed, layout)


def test_lock_events_include_diagnostics_without_credentials(tmp_path):
    logger = Mock()
    path = tmp_path / ".lock"
    with held_lock(path):
        from janus.runtime.locking import SourceLockedError

        with pytest.raises(SourceLockedError), lock_at(path, logger=logger):
            pytest.fail("held lock acquired")
    with lock_at(path, logger=logger):
        pass
    calls = [call for call in logger.mock_calls if call.args]
    assert {call.args[0] for call in calls} >= {
        "source_lock_acquired",
        "source_lock_refused",
        "source_lock_released",
    }
    for call in calls:
        assert {"source_id", "lock_path", "holder_pid"} <= call.kwargs.keys()
        assert not any(
            word in key.lower()
            for key in call.kwargs
            for word in ("password", "credential", "token", "authorization")
        )
