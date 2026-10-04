"""Visible null locking until source-state locking is available."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class MaintenanceLock(Protocol):
    name: str

    def acquire(self, source_id: str) -> bool: ...

    def release(self, source_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NullMaintenanceLock:
    """Acquires nothing; its name is recorded and the CLI warns on every invocation."""

    name: str = "none"

    def acquire(self, source_id: str) -> bool:
        return True

    def release(self, source_id: str) -> None:
        pass
