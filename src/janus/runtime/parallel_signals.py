"""Temporary parent signal handling for a parallel dispatch loop."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any


def raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


@contextmanager
def parent_signal_handlers(log: Callable[..., Any]) -> Iterator[None]:
    """Translate SIGTERM while dispatching; embedding threads keep their handlers."""
    previous = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    except ValueError:
        log("parallel_signal_handlers_not_installed")
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
