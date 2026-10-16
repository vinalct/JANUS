from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import Future
from functools import partial
from itertools import permutations
from random import Random
from types import SimpleNamespace

import pytest

TIMEOUT = 5
COMPLETION_ORDERS = Random(25).sample(list(permutations(range(1, 7))), 50)


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
    answered = {index: threading.Event() for index in range(1, 5)}
    with pytest.raises(KeyboardInterrupt, match="worker interrupt") as caught, _window(4) as window:
        for index in range(1, 5):
            window.submit(
                index, partial(
                    _download, index, discarded, answered=answered[index],
                    error=error if index == 2 else None,
                ),
            )
        # Ensure every download started before shutdown can cancel queued futures.
        assert all(event.wait(TIMEOUT) for event in answered.values())
        window.take(1)
        window.take(2)
    assert caught.value is error
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
    answered = {index: threading.Event() for index in range(1, 4)}
    with pytest.raises(ValueError, match="loop error"), _window(3) as window:
        for index in range(1, 4):
            window.submit(index, partial(_download, index, discarded, answered=answered[index]))
        assert all(event.wait(TIMEOUT) for event in answered.values())
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


@pytest.mark.parametrize("completion_order", COMPLETION_ORDERS)
def test_seeded_completion_permutations_preserve_order_and_bound(completion_order):
    discarded, order = Counter(), []
    answered = {index: threading.Event() for index in completion_order}
    predecessors = dict(zip(completion_order[1:], completion_order[:-1], strict=True))
    with _window(6) as window:
        for index in range(1, 7):
            window.submit(
                index, partial(
                    _download, index, discarded,
                    gate=answered[predecessors[index]] if index in predecessors else None,
                    answered=answered[index], order=order,
                ),
            )
            assert window.in_flight == index <= 6
        assert not window.has_room and window.max_in_flight_observed == 6
        outcomes = []
        for index in range(1, 7):
            outcomes.append(window.take(index))
            assert window.in_flight == 6 - index
        assert [outcome.file_index for outcome in outcomes] == list(range(1, 7))
        assert all(outcome.error is None for outcome in outcomes)
        assert order == list(completion_order)
    assert discarded == Counter()


@pytest.mark.parametrize("error", [None, ValueError("candidate failure")])
def test_worker_returns_the_value_or_error_with_injected_elapsed_time(error):
    from janus.strategies.files.download_window import _download_in_worker

    outcome = _download_in_worker(
        7, partial(_download, 7, Counter(), error=error),
        monotonic=partial(next, iter((10.0, 12.5))),
    )
    assert outcome.file_index == 7
    assert outcome.download_seconds == 2.5
    assert outcome.error is error
    assert (outcome.downloaded is None) == (error is not None)


def test_drain_cleans_all_outcomes_before_raising_the_first_worker_interrupt():
    discarded = Counter()
    first_error, later_error = KeyboardInterrupt("first"), SystemExit("later")
    errors = {2: first_error, 3: later_error}
    with _window(4) as window:
        # Drain sorts by index even when submissions and completions differ.
        for index in (4, 3, 2, 1):
            window.submit(index, partial(_download, index, discarded, error=errors.get(index)))
        with pytest.raises(KeyboardInterrupt) as caught:
            window.drain()
        assert caught.value is first_error
        assert discarded == Counter({1: 1, 4: 1})
        assert window.in_flight == 0
        assert window.drain() == ()
    assert discarded == Counter({1: 1, 4: 1})


@pytest.mark.parametrize("finished_before_interrupt", [False, True])
def test_caller_interrupt_during_take_keeps_its_payload_reachable_for_cleanup(
    monkeypatch, finished_before_interrupt,
):
    discarded, gate, answered = Counter(), threading.Event(), threading.Event()
    error = KeyboardInterrupt("caller interrupt")
    original_result = Future.result
    interrupted = False

    def result(self, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            if finished_before_interrupt:
                gate.set()
                original_result(self, *args, **kwargs)
            raise error
        return original_result(self, *args, **kwargs)

    with pytest.raises(KeyboardInterrupt) as caught, _window(2) as window:
        window.submit(1, partial(_download, 1, discarded, gate=gate))
        window.submit(2, partial(_download, 2, discarded, answered=answered))
        assert answered.wait(TIMEOUT)
        monkeypatch.setattr(Future, "result", result)
        try:
            window.take(1)
        finally:
            gate.set()
    assert caught.value is error
    assert discarded == Counter({1: 1, 2: 1})
    assert window.in_flight == 0


def test_exit_preserves_the_caller_error_when_an_untaken_worker_also_interrupts():
    discarded, answered = Counter(), threading.Event()
    caller_error = ValueError("caller failure")
    with pytest.raises(ValueError) as caught, _window(2) as window:
        window.submit(1, partial(_download, 1, discarded, error=KeyboardInterrupt("worker")))
        window.submit(2, partial(_download, 2, discarded, answered=answered))
        assert answered.wait(TIMEOUT)
        raise caller_error
    assert caught.value is caller_error
    assert discarded == Counter({2: 1})
    assert window.in_flight == 0


def test_duplicate_pending_index_cannot_overwrite_a_payload():
    discarded, answered = Counter(), threading.Event()
    with _window(2) as window:
        window.submit(1, partial(_download, 1, discarded, answered=answered))
        assert answered.wait(TIMEOUT)
        with pytest.raises(RuntimeError, match="already dispatched"):
            window.submit(1, partial(_download, 1, discarded))
        assert window.in_flight == window.max_in_flight_observed == 1
    assert discarded == Counter({1: 1})


@pytest.mark.parametrize("thread_name_prefix", ["janus-file", "custom-file"])
def test_pool_uses_configured_capacity_and_thread_names(monkeypatch, thread_name_prefix):
    from janus.strategies.files import download_window

    original_pool = download_window.ThreadPoolExecutor
    constructed, worker_names = [], []

    def pool(**kwargs):
        constructed.append(kwargs)
        return original_pool(**kwargs)

    def download():
        worker_names.append(threading.current_thread().name)
        return _download(1, Counter())

    monkeypatch.setattr(download_window, "ThreadPoolExecutor", pool)
    with download_window.OrderedDownloadWindow(3, thread_name_prefix=thread_name_prefix) as window:
        window.submit(1, download)
        window.take(1)
    assert constructed == [{"max_workers": 3, "thread_name_prefix": thread_name_prefix}]
    assert len(worker_names) == 1 and worker_names[0].startswith(thread_name_prefix)
