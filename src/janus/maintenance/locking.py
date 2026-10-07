"""Visible null locking until source-state locking is available."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class MaintenanceLock(Protocol):
    """Per-source advisory lock held while maintenance reads and mutates state."""

    @property
    def name(self) -> str: ...

    def acquire(self, source_id: str) -> bool:
        """Return true when this process holds the source lock; false on contention."""

    def release(self, source_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NullMaintenanceLock:
    """Acquires nothing and says so. Its name is what the record reports."""

    name: str = "none"

    def acquire(self, source_id: str) -> bool:
        del source_id
        return True

    def release(self, source_id: str) -> None:
        del source_id


LOCKED_SOURCE_REASON = "source_locked"


@contextmanager
def source_lock(lock: MaintenanceLock, source_id: str) -> Iterator[bool]:
    """Release each acquired source in finally, including interrupted execution."""
    acquired = lock.acquire(source_id)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release(source_id)
