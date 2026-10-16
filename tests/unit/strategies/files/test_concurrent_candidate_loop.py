from __future__ import annotations

import threading
from io import StringIO
from pathlib import Path

import pytest

import janus.strategies.files.concurrent_extraction as concurrent
import janus.strategies.files.core as core
from janus.strategies.files.core import FileHook
from janus.utils.logging import StructuredLogger, build_structured_logger
from janus.writers.raw import StagedWrite
from tests.support.file_downloads import build_cnpj_fixture
from tests.unit.strategies.files.test_concurrent_dead_letter_order import (
    assert_no_staging,
    committed_archives,
    run_leg,
    seed_dead_letters,
)


@pytest.fixture(autouse=True)
def small_spool(monkeypatch):
    import janus.strategies.files.download as download
    import janus.strategies.files.download_loop as loop

    monkeypatch.setattr(download, "SPOOL_THRESHOLD_BYTES", 32)
    monkeypatch.setattr(loop, "SPOOL_THRESHOLD_BYTES", 32)


@pytest.mark.parametrize(
    "concurrency,skipped,empty,maximum",
    [(1, 0, False, None), (4, 0, False, 4), (4, 9, False, 1),
     (4, 10, False, 0), (4, 0, True, 0), (1, 0, True, None)],
)
def test_outcome_reports_only_eligible_download_slots(
    tmp_path, monkeypatch, concurrency, skipped, empty, maximum,
):
    class EmptyHook(FileHook):
        def discovered_files(self, plan, files):
            return ()

    fixture = build_cnpj_fixture(tmp_path)
    seed_dead_letters(fixture, range(1, skipped + 1))
    outcomes = []
    original = core.run_file_extraction

    def capture(context, **kwargs):
        outcome = original(context, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(core, "run_file_extraction", capture)
    strategy = fixture.build_strategy()
    strategy.extract(
        fixture.build_plan("stats", concurrency=concurrency, resume=True),
        hook=EmptyHook() if empty else None,
    )
    outcome, = outcomes
    if maximum is None:
        assert outcome.download_window is None
    else:
        assert outcome.download_window.download_concurrency == concurrency
        assert outcome.download_window.max_in_flight_observed == maximum
    assert outcome.dead_letter_skip_count == skipped
    assert outcome.totals.persisted_file_count == (0 if empty else 10 - skipped)
    assert fixture.server.opened == fixture.server.closed
    assert_no_staging(fixture)


def test_interrupt_after_take_discards_the_taken_and_pending_payloads(tmp_path, monkeypatch):
    original = StructuredLogger.info

    def info(self, event, **fields):
        if event == "file_download_completed" and fields["file_index"] == 2:
            raise KeyboardInterrupt("interrupt after take")
        return original(self, event, **fields)

    monkeypatch.setattr(StructuredLogger, "info", info)
    fixture = build_cnpj_fixture(tmp_path)
    logger = build_structured_logger("janus.tests.file.take-interrupt", stream=StringIO())
    strategy = fixture.build_strategy(logger=logger)
    plan = fixture.build_plan("con", concurrency=4)
    with pytest.raises(KeyboardInterrupt, match="interrupt after take"):
        strategy.extract(plan)
    assert fixture.server.opened == fixture.server.closed
    assert strategy.dead_letter_store.load(plan) is None
    assert_no_staging(fixture)
    assert len(list(fixture.storage_layout.project_root.rglob("Empresas*.zip"))) == 1


def test_budget_failure_reraises_the_worker_exception_object(tmp_path, monkeypatch):
    error = RuntimeError("worker failure identity")
    original = concurrent._download_candidate

    def download(downloader, plan, discovered_file, **kwargs):
        if discovered_file.filename == "Empresas1.zip":
            raise error
        return original(downloader, plan, discovered_file, **kwargs)

    monkeypatch.setattr(concurrent, "_download_candidate", download)
    fixture = build_cnpj_fixture(tmp_path, dead_letter_max_items=0)
    leg = run_leg(fixture, 4)
    assert leg.error is error
    frames, traceback = [], error.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert {"complete", "_download_in_worker", "download"} <= set(frames)
    assert len(committed_archives(leg)) == 1
    assert fixture.server.opened == fixture.server.closed
    assert_no_staging(fixture)


@pytest.mark.parametrize(
    "race", ["before-write", "before-open", "during-mkdir", "recreated-during-mkdir"],
)
def test_commit_during_another_downloads_staging_setup_keeps_both_candidates(
    tmp_path, monkeypatch, race,
):
    first_staged, second_ready = threading.Event(), threading.Event()
    first_committed, second_finished = threading.Event(), threading.Event()
    worker = threading.local()
    original_download = concurrent._download_candidate
    original_write = StagedWrite.write_from
    original_commit = StagedWrite.commit
    original_open, original_mkdir = Path.open, Path.mkdir
    removed_during_open, removed_during_mkdir = [], []

    def download(downloader, plan, discovered_file, **kwargs):
        worker.second = discovered_file.filename == "Empresas1.zip"
        if worker.second:
            assert first_staged.wait(5)
        return original_download(downloader, plan, discovered_file, **kwargs)

    def write(self, stream):
        if worker.second:
            second_ready.set()
            assert first_committed.wait(5)
            try:
                return original_write(self, stream)
            finally:
                second_finished.set()
        original_write(self, stream)
        first_staged.set()

    def commit(self, relative_path, **kwargs):
        if str(relative_path).endswith("Empresas0.zip"):
            assert second_ready.wait(5)
            result = original_commit(self, relative_path, **kwargs)
            first_committed.set()
            assert second_finished.wait(5)
            return result
        return original_commit(self, relative_path, **kwargs)

    def open_payload(self, mode="r", *args, **kwargs):
        if (
            race == "before-open" and getattr(worker, "second", False)
            and mode == "xb" and self.suffix == ".partial" and not removed_during_open
        ):
            self.parent.rmdir()
            removed_during_open.append(self)
        return original_open(self, mode, *args, **kwargs)

    def mkdir(self, *args, **kwargs):
        if (
            race in {"during-mkdir", "recreated-during-mkdir"}
            and getattr(worker, "second", False)
            and self.name == ".staging" and not removed_during_mkdir
        ):
            original_mkdir(self, *args, **kwargs)
            self.rmdir()
            removed_during_mkdir.append(self)
            if race == "recreated-during-mkdir":
                original_mkdir(self, *args, **kwargs)
            raise FileExistsError("directory removed after mkdir observed it")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Path, "open", open_payload)
    monkeypatch.setattr(concurrent, "_download_candidate", download)
    monkeypatch.setattr(StagedWrite, "write_from", write)
    monkeypatch.setattr(StagedWrite, "commit", commit)
    fixture = build_cnpj_fixture(tmp_path, archive_count=2)
    leg = run_leg(fixture, 2)
    assert leg.error is None
    assert leg.result.metadata_as_dict()["dead_letter_count"] == "0"
    assert len(committed_archives(leg)) == 2
    assert bool(removed_during_open) == (race == "before-open")
    assert bool(removed_during_mkdir) == (race in {"during-mkdir", "recreated-during-mkdir"})
    assert fixture.server.opened == fixture.server.closed
    assert_no_staging(fixture)
