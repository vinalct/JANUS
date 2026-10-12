"""The process-scoped advisory lock for a source's local metadata directory."""

from __future__ import annotations

import errno
import json
import os
import socket
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from janus.models import ExecutionPlan
from janus.utils.catalog_options import is_location_uri
from janus.utils.logging import StructuredLogger, sanitize_log_payload
from janus.utils.storage import StorageLayout

LOCK_FILE_NAME = ".lock"
_HOLDER_READ_LIMIT = 64 * 1024


class LockUnsupportedError(RuntimeError):
    """Source-state locking cannot work here; refuse to run unprotected."""


def source_state_lock_path(plan: ExecutionPlan, storage_layout: StorageLayout) -> Path:
    """Resolve the writer's metadata directory; sources sharing it share one lock."""
    target = storage_layout.resolve_output(plan, "metadata")
    if is_location_uri(target.configured_path):
        raise LockUnsupportedError(
            f"Source {plan.source.source_id!r} has a metadata zone at a location URI "
            f"({target.configured_path}); source-state locking needs a local POSIX path"
        )
    return target.resolved_path / LOCK_FILE_NAME


@dataclass(frozen=True, slots=True)
class LockHolder:
    pid: int
    hostname: str
    source_id: str
    run_id: str | None
    pipeline_run_id: str | None
    acquired_at: datetime
    command: str

    @classmethod
    def now(
        cls, *, source_id: str, command: str, run_id: str | None, pipeline_run_id: str | None
    ) -> Self:
        """Construct diagnostics using a fixed command label, never operator argv."""
        return cls(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            source_id=source_id,
            run_id=run_id,
            pipeline_run_id=pipeline_run_id,
            acquired_at=datetime.now(UTC),
            command=command,
        )

    def to_json(self) -> bytes:
        payload = asdict(self)
        payload["acquired_at"] = self.acquired_at.isoformat()
        return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> Self | None:
        """Malformed diagnostics never turn contention into permission to run."""
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or payload.keys() != cls.__dataclass_fields__.keys():
                return None
            if (
                type(payload["pid"]) is not int
                or payload["pid"] <= 0
                or any(
                    not isinstance(payload[key], str) or not payload[key]
                    for key in ("hostname", "source_id", "command", "acquired_at")
                )
            ):
                return None
            if any(
                payload[key] is not None and not isinstance(payload[key], str)
                for key in ("run_id", "pipeline_run_id")
            ):
                return None
            acquired_at = datetime.fromisoformat(payload["acquired_at"])
            if acquired_at.utcoffset() is None:
                return None
            return cls(**(payload | {"acquired_at": acquired_at.astimezone(UTC)}))
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            return None

    def describe(self) -> str:
        return (
            f"held by pid {self.pid} on {self.hostname} for {self.source_id!r} "
            f"(run {self.run_id or '-'}, pipeline {self.pipeline_run_id or '-'}, "
            f"{self.command}) since {self.acquired_at.isoformat()}"
        )


class SourceLockedError(RuntimeError):
    """Another process holds this source's state lock; nothing was executed."""

    def __init__(self, path: Path, holder: LockHolder | None, *, wanted: LockHolder):
        self.path, self.holder, self.wanted = path, holder, wanted
        description = (
            holder.describe() if holder else "by another process whose holder record is unreadable"
        )
        super().__init__(
            f"Source {wanted.source_id!r} is locked: {description}; "
            f"nothing was executed. Lock: {path}"
        )


@dataclass(slots=True)
class _Holding:
    pid: int
    fd: int
    depth: int
    holder: LockHolder


_HELD: dict[Path, _Holding] = {}


@dataclass(slots=True)
class SourceStateLock:
    path: Path
    holder: LockHolder
    wait_seconds: float = 0.0
    poll_interval_seconds: float = 0.1
    logger: StructuredLogger | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _key: Path | None = field(default=None, init=False, repr=False)
    _depth: int = field(default=0, init=False, repr=False)
    _pid: int | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def acquire(self) -> None:
        try:
            import fcntl
        except ImportError as error:
            raise LockUnsupportedError(
                "source-state locking requires POSIX fcntl; this platform has none"
            ) from error
        if os.name != "posix":
            raise LockUnsupportedError("source-state locking requires a POSIX platform")
        key = Path(os.path.realpath(self.path))
        entry = _HELD.get(key)
        if entry is not None and entry.pid == os.getpid():
            self._log("source_lock_acquired", entry.holder, reentrant=True)
            entry.depth += 1
            self._remember_acquisition(key)
            return
        if entry is not None:
            # A forked copy must neither reuse nor unlock the parent's open description.
            with suppress(OSError):
                os.close(entry.fd)
            del _HELD[key]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            deadline = self.clock() + self.wait_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if error.errno not in {errno.EWOULDBLOCK, errno.EAGAIN}:
                        raise
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        holder = LockHolder.from_json(os.pread(fd, _HOLDER_READ_LIMIT, 0))
                        self._log("source_lock_refused", holder)
                        raise SourceLockedError(self.path, holder, wanted=self.holder) from error
                    self.sleep(min(self.poll_interval_seconds, remaining))
            self.holder = replace(self.holder, acquired_at=datetime.now(UTC))
            os.ftruncate(fd, 0)
            raw = self.holder.to_json()
            offset = 0
            while offset < len(raw):
                written = os.pwrite(fd, raw[offset:], offset)
                if written == 0:
                    raise OSError(errno.EIO, "could not write source lock holder")
                offset += written
            os.fsync(fd)
            self._log("source_lock_acquired", self.holder)
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            raise
        _HELD[key] = _Holding(pid=os.getpid(), fd=fd, depth=1, holder=self.holder)
        self._remember_acquisition(key)

    def _remember_acquisition(self, key: Path) -> None:
        if self._pid != os.getpid():
            self._depth = 0
        self._key, self._pid = key, os.getpid()
        self._depth += 1

    def release(self) -> None:
        entry = _HELD.get(self._key) if self._key is not None else None
        if (
            entry is None
            or entry.pid != os.getpid()
            or self._pid != os.getpid()
            or self._depth == 0
        ):
            return
        import fcntl

        self._depth -= 1
        entry.depth -= 1
        if entry.depth > 0:
            self._log("source_lock_released", entry.holder, reentrant=True)
            return
        with suppress(OSError):
            os.ftruncate(entry.fd, 0)
        with suppress(OSError):
            fcntl.flock(entry.fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(entry.fd)
        assert self._key is not None
        del _HELD[self._key]
        self._log("source_lock_released", entry.holder)

    def _log(self, event: str, holder: LockHolder | None, *, reentrant: bool = False) -> None:
        if self.logger is not None:
            payload = sanitize_log_payload(
                {
                    "source_id": self.holder.source_id,
                    "lock_path": str(self.path),
                    "holder_pid": holder.pid if holder else None,
                    "holder_hostname": holder.hostname if holder else None,
                    "holder_source_id": holder.source_id if holder else None,
                    "run_id": holder.run_id if holder else None,
                    "pipeline_run_id": holder.pipeline_run_id if holder else None,
                    "acquired_at": holder.acquired_at.isoformat() if holder else None,
                    "command": holder.command if holder else None,
                    "reentrant": reentrant,
                }
            )
            log = self.logger.warning if event == "source_lock_refused" else self.logger.info
            log(event, **payload)


@contextmanager
def hold_source_state_lock(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    *,
    command: str,
    run_id: str | None,
    pipeline_run_id: str | None,
    wait_seconds: float = 0.0,
    logger: StructuredLogger | None = None,
) -> Iterator[SourceStateLock]:
    """Resolve the one lock path for this plan, take it, and always release it."""
    holder = LockHolder.now(
        source_id=plan.source.source_id,
        command=command,
        run_id=run_id,
        pipeline_run_id=pipeline_run_id,
    )
    with SourceStateLock(
        source_state_lock_path(plan, storage_layout),
        holder,
        wait_seconds=wait_seconds,
        logger=logger,
    ) as lock:
        yield lock
