"""Characterization: request-throttle pacing for the api, catalog, and file families.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .conftest import FAMILIES, ScriptedClock

RECEITA_FEDERAL_REQUESTS_PER_MINUTE = 6  # → interval of exactly 10.0 seconds


def test_no_rate_limit_never_sleeps_and_never_reads_the_clock(family):
    clock = ScriptedClock([])
    sleeps: list[float] = []
    throttle = family.throttle_cls(
        requests_per_minute=None,
        clock=clock,
        sleeper=sleeps.append,
    )

    for _ in range(3):
        throttle.wait_for_turn()

    assert sleeps == []
    assert clock.reads == 0


def test_first_call_does_not_sleep(family):
    clock = ScriptedClock([0.0])
    sleeps: list[float] = []
    throttle = family.throttle_cls(
        requests_per_minute=RECEITA_FEDERAL_REQUESTS_PER_MINUTE,
        clock=clock,
        sleeper=sleeps.append,
    )

    throttle.wait_for_turn()

    assert sleeps == []


def test_immediate_second_call_sleeps_one_full_interval(family):
    clock = ScriptedClock([0.0, 0.0])
    sleeps: list[float] = []
    throttle = family.throttle_cls(
        requests_per_minute=RECEITA_FEDERAL_REQUESTS_PER_MINUTE,
        clock=clock,
        sleeper=sleeps.append,
    )

    throttle.wait_for_turn()
    throttle.wait_for_turn()

    assert sleeps == [10.0]


def test_pacing_sequence_partial_wait_then_reanchor_after_idle_window(family):
    # Call 1 at t=0 → no sleep, next slot at t=10.
    # Call 2 at t=4 → sleeps the remaining 6.0 s, next slot at t=20.
    # Call 3 at t=50 (past the window) → no sleep, re-anchors: next slot at t=60.
    # Call 4 still at t=50 → sleeps the full 10.0 s again.
    clock = ScriptedClock([0.0, 4.0, 50.0, 50.0])
    sleeps: list[float] = []
    throttle = family.throttle_cls(
        requests_per_minute=RECEITA_FEDERAL_REQUESTS_PER_MINUTE,
        clock=clock,
        sleeper=sleeps.append,
    )

    for _ in range(4):
        throttle.wait_for_turn()

    assert sleeps == [6.0, 10.0]


def test_clock_is_read_exactly_once_per_throttled_call(family):
    clock = ScriptedClock([0.0, 4.0, 50.0, 50.0])
    throttle = family.throttle_cls(
        requests_per_minute=RECEITA_FEDERAL_REQUESTS_PER_MINUTE,
        clock=clock,
        sleeper=lambda seconds: None,
    )

    for _ in range(4):
        throttle.wait_for_turn()

    assert clock.reads == 4


def test_api_throttle_grants_unique_slots_under_concurrent_callers():
    call_count = 32
    sleeps: list[float] = []
    throttle = FAMILIES["api"].throttle_cls(
        requests_per_minute=60,  # interval of exactly 1.0 second
        clock=lambda: 0.0,
        sleeper=sleeps.append,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(throttle.wait_for_turn) for _ in range(call_count)]
        for future in futures:
            future.result()

    # The first slot is at delay 0 (no sleeper call); every later slot is a
    # unique whole-interval delay.
    assert sorted(sleeps) == [float(slot) for slot in range(1, call_count)]
