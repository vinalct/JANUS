"""Adapt the shared source-state lock to maintenance's per-source lock seam."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from janus.models import ExecutionPlan
from janus.runtime.locking import (
    COMMAND_MAINTAIN,
    LockHolder,
    SourceLockedError,
    SourceStateLock,
    source_state_lock_path,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout


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


@dataclass(slots=True)
class SourceStateMaintenanceLock:
    plans: Mapping[str, ExecutionPlan]
    storage_layout: StorageLayout
    wait_seconds: float = 0.0
    logger: StructuredLogger | None = None
    name: str = "flock"
    _held: dict[str, SourceStateLock] = field(default_factory=dict, init=False, repr=False)

    def acquire(self, source_id: str) -> bool:
        lock = SourceStateLock(
            path=source_state_lock_path(self.plans[source_id], self.storage_layout),
            holder=LockHolder.now(
                source_id=source_id, command=COMMAND_MAINTAIN, run_id=None, pipeline_run_id=None
            ),
            wait_seconds=self.wait_seconds,
            logger=self.logger,
        )
        try:
            lock.acquire()
        except SourceLockedError:
            return False
        self._held[source_id] = lock
        return True

    def release(self, source_id: str) -> None:
        held = self._held.pop(source_id, None)
        if held is not None:
            held.release()


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
