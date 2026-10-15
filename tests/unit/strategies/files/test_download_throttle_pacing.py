from __future__ import annotations

import threading
from itertools import pairwise

import pytest

from janus.strategies.files.extraction import FileExtractionContext
from janus.strategies.http import HttpRequestThrottle
from tests.support.file_downloads import build_cnpj_fixture

CONCURRENCIES = [1, 3]


def _measure_slots(tmp_path, monkeypatch, concurrency):
    slots, built = [], []
    spy_lock = threading.Lock()
    original_wait = HttpRequestThrottle.wait_for_turn
    original_build = FileExtractionContext.build.__func__

    def wait_for_turn(self):
        with spy_lock:
            original_wait(self)
            slots.append(
                (
                    id(self),
                    threading.current_thread().name,
                    self._next_allowed_at - 60 / self.requests_per_minute,
                )
            )

    def build(cls, *args, **kwargs):
        context = original_build(cls, *args, **kwargs)
        built.append(id(context.throttle))
        return context

    monkeypatch.setattr(HttpRequestThrottle, "wait_for_turn", wait_for_turn)
    monkeypatch.setattr(FileExtractionContext, "build", classmethod(build))
    fixture = build_cnpj_fixture(tmp_path, retrying={5: 1})
    fixture.source_mapping["access"]["rate_limit"]["requests_per_minute"] = 6
    result = fixture.build_strategy(clock=lambda: 0.0, sleeper=lambda seconds: None).extract(
        fixture.build_plan("paced", concurrency=concurrency),
    )
    assert result.metadata_as_dict()["persisted_file_count"] == "10"
    assert fixture.server.opened == fixture.server.closed
    return fixture, slots, built


def _assert_worker_slots(slots, concurrency):
    if concurrency > 1:
        assert all(name.startswith("janus-file") for _, name, _ in slots), (
            "concurrent download slots were granted on the loop thread"
        )


@pytest.mark.parametrize("concurrency", CONCURRENCIES)
def test_slots_obey_rate_limit_and_match_sequential_retry_schedule(
    tmp_path, monkeypatch, concurrency
):
    fixture, recorded, _ = _measure_slots(tmp_path / "candidate", monkeypatch, concurrency)
    slots = sorted(slot for _, _, slot in recorded)
    assert slots == list(range(0, 101, 10))
    assert len(slots) == 11
    assert all(right - left >= 10 for left, right in pairwise(slots))
    assert all(sum(start <= slot < start + 60 for slot in slots) <= 6 for start in slots)
    assert all(slots[index + 6] - slots[index] >= 60 for index in range(len(slots) - 6))
    # Restore the spies before installing another set: wrapping a spy twice changes counts.
    monkeypatch.undo()
    _, sequential, _ = _measure_slots(tmp_path / "seq", monkeypatch, 1)
    assert slots == sorted(slot for _, _, slot in sequential)
    assert fixture.server.opened == (11 if concurrency > 1 else 1)
    _assert_worker_slots(recorded, concurrency)


@pytest.mark.parametrize("concurrency", CONCURRENCIES)
def test_all_attempts_use_the_throttle_created_by_context_build(tmp_path, monkeypatch, concurrency):
    _, slots, built = _measure_slots(tmp_path, monkeypatch, concurrency)
    assert len(slots) == 11
    assert len(built) == 1
    assert {ident for ident, _, _ in slots} == set(built)
    _assert_worker_slots(slots, concurrency)
