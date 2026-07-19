"""Dedicated tests for the unified thread-safe ``HttpRequestThrottle``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from janus.strategies.http import HttpRequestThrottle

REQUESTS_PER_MINUTE = 60  # interval of exactly 1.0 second


def test_shared_throttle_grants_unique_spaced_slots_under_concurrency():
    workers = 8
    calls_per_worker = 16
    total_calls = workers * calls_per_worker
    interval = 60 / REQUESTS_PER_MINUTE

    recorded = Lock()
    sleeps: list[float] = []

    def sleeper(delay: float) -> None:
        with recorded:
            sleeps.append(delay)

    throttle = HttpRequestThrottle(
        requests_per_minute=REQUESTS_PER_MINUTE,
        clock=lambda: 0.0,
        sleeper=sleeper,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(throttle.wait_for_turn) for _ in range(total_calls)]
        for future in futures:
            future.result()

    # (c) No lost update on the shared cursor: every call advanced
    #     `_next_allowed_at` exactly once. An unlocked race would drop
    #     increments and leave this below the expected total.
    assert throttle._next_allowed_at == total_calls * interval

    # (a) Each reserved slot is unique and spaced by a full interval. The very
    #     first slot lands at delay 0, so it is never handed to the sleeper;
    #     every later slot is a distinct whole-interval delay.
    assert sorted(sleeps) == [slot * interval for slot in range(1, total_calls)]

    # (b) Total granted slots == N*M — the zero-delay slot plus every sleep.
    assert len(sleeps) + 1 == total_calls


def test_shared_throttle_without_rate_limit_is_a_noop_under_concurrency():
    throttle = HttpRequestThrottle(
        requests_per_minute=None,
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not be read")),
        sleeper=lambda _delay: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(throttle.wait_for_turn) for _ in range(64)]
        for future in futures:
            future.result()

    assert throttle._next_allowed_at is None
