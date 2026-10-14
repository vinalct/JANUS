from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, reason="ordered download window missing"
)
TIMEOUT = 5


class Payload:
    def __init__(self, index, discarded):
        self.index, self.discarded = index, discarded

    def discard(self):
        self.discarded[self.index] += 1


def _download(index, discarded, *, gate=None, answered=None, order=None, error=None):
    if gate is not None:
        assert gate.wait(TIMEOUT), "download gate was never released"
    if order is not None:
        order.append(index)
    if answered is not None:
        answered.set()
    if error is not None:
        raise error
    return SimpleNamespace(payload=Payload(index, discarded))


def _window(capacity):
    from janus.strategies.files.download_window import OrderedDownloadWindow

    return OrderedDownloadWindow(capacity)


def test_reversed_completions_are_taken_in_index_order():
    discarded, order = Counter(), []
    events = {index: threading.Event() for index in range(1, 5)}
    with _window(4) as window:
        for index in range(1, 5):
            window.submit(
                index,
                partial(
                    _download,
                    index,
                    discarded,
                    gate=events[index + 1] if index < 4 else None,
                    answered=events[index],
                    order=order,
                ),
            )
        outcomes = [window.take(index) for index in range(1, 5)]
        assert [outcome.file_index for outcome in outcomes] == [1, 2, 3, 4]
        assert order == [4, 3, 2, 1]
        assert all(outcome.error is None for outcome in outcomes)
        assert window.in_flight == 0
    assert discarded == Counter()


def test_completed_but_untaken_payloads_still_occupy_window_slots():
    discarded, release_first, second_done = Counter(), threading.Event(), threading.Event()
    submitted_unconsumed, observed = set(), []
    with _window(2) as window:
        try:
            window.submit(1, partial(_download, 1, discarded, gate=release_first))
            submitted_unconsumed.add(1)
            observed.append(len(submitted_unconsumed))
            window.submit(2, partial(_download, 2, discarded, answered=second_done))
            submitted_unconsumed.add(2)
            observed.append(len(submitted_unconsumed))
            assert second_done.wait(TIMEOUT)
            assert window.in_flight == 2 and not window.has_room
            with pytest.raises(RuntimeError):
                window.submit(3, partial(_download, 3, discarded))
            assert not window.is_pending(3)
        finally:
            release_first.set()
        window.take(1)
        submitted_unconsumed.remove(1)
        assert window.has_room and window.in_flight == len(submitted_unconsumed)
        window.submit(3, partial(_download, 3, discarded))
        submitted_unconsumed.add(3)
        observed.append(len(submitted_unconsumed))
        assert window.max_in_flight_observed == max(observed) == 2
        window.take(2)
        window.take(3)


def test_ordinary_exception_is_an_outcome_value():
    error = ValueError("candidate failure")
    with _window(1) as window:
        window.submit(1, partial(_download, 1, Counter(), error=error))
        outcome = window.take(1)
        assert outcome.file_index == 1
        assert outcome.error is error and outcome.downloaded is None
        assert outcome.download_seconds >= 0
        outcome.discard()


def test_worker_base_exception_propagates_and_exit_discards_remaining_payloads():
    discarded = Counter()
    error = KeyboardInterrupt("worker interrupt")
    with pytest.raises(KeyboardInterrupt, match="worker interrupt"), _window(4) as window:
        for index in range(1, 5):
            window.submit(
                index, partial(_download, index, discarded, error=error if index == 2 else None)
            )
        window.take(1)
        window.take(2)
    assert discarded == Counter({3: 1, 4: 1})
    assert window.in_flight == 0


def test_stop_and_drain_wait_for_every_future_and_discard_once(monkeypatch):
    discarded, order = Counter(), []
    gate, entered, release = threading.Event(), threading.Event(), threading.Event()
    original_result = Future.result

    def result(self, *args, **kwargs):
        if not entered.is_set():
            assert not self.done(), "drain never waited for in-flight downloads"
            entered.set()
        return original_result(self, *args, **kwargs)

    def release_when_drain_starts():
        assert entered.wait(TIMEOUT)
        release.set()
        gate.set()

    releaser = threading.Thread(target=release_when_drain_starts, daemon=True)
    with _window(4) as window:
        for index in range(1, 4):
            window.submit(index, partial(_download, index, discarded, gate=gate, order=order))
        assert window.has_room  # Stop must refuse dispatch even when capacity remains.
        window.stop_dispatch()
        assert not window.has_room
        with pytest.raises(RuntimeError):
            window.submit(4, partial(_download, 4, discarded))
        releaser.start()
        # Release only when drain actually waits on a future, never before the call.
        with monkeypatch.context() as spy:
            spy.setattr(Future, "result", result)
            outcomes = window.drain()
        assert release.is_set() and set(order) == {1, 2, 3}
        assert [outcome.file_index for outcome in outcomes] == [1, 2, 3]
        assert discarded == Counter({1: 1, 2: 1, 3: 1})
        assert window.in_flight == 0
    releaser.join(TIMEOUT)
    assert not releaser.is_alive()
    assert discarded == Counter({1: 1, 2: 1, 3: 1})


def test_context_exception_discards_untaken_payloads_and_shuts_pool_down():
    discarded = Counter()
    with pytest.raises(ValueError, match="loop error"), _window(3) as window:
        for index in range(1, 4):
            window.submit(index, partial(_download, index, discarded))
        raise ValueError("loop error")
    assert discarded == Counter({1: 1, 2: 1, 3: 1})
    assert window.in_flight == 0
    assert not window.has_room
    with pytest.raises(RuntimeError):
        window.submit(4, partial(_download, 4, discarded))


def test_taking_an_undispatched_index_fails_immediately():
    with _window(1) as window:
        with pytest.raises(RuntimeError, match="file index 42.*never dispatched") as caught:
            window.take(42)
        assert isinstance(caught.value.__cause__, KeyError)


def test_capacity_one_is_a_strict_sequence():
    with _window(1) as window:
        for index in range(1, 5):
            assert window.has_room
            window.submit(index, partial(_download, index, Counter()))
            assert window.in_flight == 1 and not window.has_room
            assert window.take(index).file_index == index
            assert window.in_flight == 0
        assert window.max_in_flight_observed == 1


@pytest.mark.parametrize("downloaded,error", [(None, None), (object(), ValueError("both"))])
def test_outcome_requires_exactly_one_of_value_or_error(downloaded, error):
    from janus.strategies.files.download_window import DownloadOutcome

    with pytest.raises(ValueError, match="exactly one"):
        DownloadOutcome(1, downloaded, error, 0.0)
