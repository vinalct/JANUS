from __future__ import annotations

import time
import typing
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .download_loop import DownloadedCandidate


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    """What a worker hands back: a value, never a state decision."""

    file_index: int
    downloaded: DownloadedCandidate | None
    error: Exception | None
    download_seconds: float

    def __post_init__(self) -> None:
        if (self.downloaded is None) == (self.error is None):
            raise ValueError("a download outcome carries exactly one of downloaded / error")

    def discard(self) -> None:
        if self.downloaded is not None:
            self.downloaded.payload.discard()


def _download_in_worker(
    file_index: int,
    download: typing.Callable[[], DownloadedCandidate],
    monotonic: typing.Callable[[], float] = time.monotonic,
) -> DownloadOutcome:
    """The only callable submitted to a pool; ordinary failures are values.

    ``BaseException`` propagates through the future to the caller taking this index.
    """
    started = monotonic()
    try:
        downloaded = download()
    except Exception as exc:
        return DownloadOutcome(file_index, None, exc, monotonic() - started)
    return DownloadOutcome(file_index, downloaded, None, monotonic() - started)


class OrderedDownloadWindow:
    """Up to ``capacity`` downloads submitted and untaken; the caller owns ordering."""

    def __init__(self, capacity: int, *, thread_name_prefix: str = "janus-file") -> None:
        if capacity < 1:
            raise ValueError("download window capacity must be positive")
        self._capacity = capacity
        self._thread_name_prefix = thread_name_prefix
        self._pool: ThreadPoolExecutor | None = None
        self._pending: dict[int, Future[DownloadOutcome]] = {}
        self._stopped = False
        self._max_in_flight_observed = 0

    def __enter__(self) -> Self:
        if self._pool is not None or self._stopped:
            raise RuntimeError("download window cannot be entered more than once")
        self._pool = ThreadPoolExecutor(
            max_workers=self._capacity, thread_name_prefix=self._thread_name_prefix,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.stop_dispatch()
        shutdown_error: BaseException | None = None
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            shutdown_error = error
        finally:
            self._pool = None

        # Cancelled futures have no payload; all running downloads must be consumed.
        for index in tuple(self._pending):
            if self._pending[index].cancelled():
                self._pending.pop(index)
        try:
            self.drain()
        except BaseException:
            if exc_type is None and shutdown_error is None:
                raise
        if shutdown_error is not None and exc_type is None:
            raise shutdown_error

    @property
    def has_room(self) -> bool:
        return not self._stopped and len(self._pending) < self._capacity

    @property
    def in_flight(self) -> int:
        """Count submitted and untaken outcomes, including completed ones."""
        return len(self._pending)

    @property
    def max_in_flight_observed(self) -> int:
        return self._max_in_flight_observed

    def submit(
        self, file_index: int, download: typing.Callable[[], DownloadedCandidate],
    ) -> None:
        if not self.has_room:
            raise RuntimeError("download window is full or dispatch has stopped")
        if self._pool is None:
            raise RuntimeError("download window must be entered before dispatch")
        if file_index in self._pending:
            raise RuntimeError(f"download for file index {file_index} was already dispatched")
        self._pending[file_index] = self._pool.submit(_download_in_worker, file_index, download)
        self._max_in_flight_observed = max(self._max_in_flight_observed, self.in_flight)

    def is_pending(self, file_index: int) -> bool:
        return file_index in self._pending

    def take(self, file_index: int) -> DownloadOutcome:
        """Wait on this index only; transfer its payload's ownership to the caller."""
        try:
            future = self._pending.pop(file_index)
        except KeyError as exc:
            raise RuntimeError(
                f"download for file index {file_index} was never dispatched"
            ) from exc
        try:
            return future.result()
        except BaseException as exc:
            # A caller interrupt while waiting must leave the future reachable for
            # cleanup, even if it finishes successfully just before this check.
            worker_failed = future.cancelled() or (
                future.done() and future.exception() is exc
            )
            if not worker_failed:
                self._pending[file_index] = future
            raise

    def stop_dispatch(self) -> None:
        self._stopped = True

    def drain(self) -> tuple[DownloadOutcome, ...]:
        """Wait in index order and discard every untaken payload before returning.

        If interrupted, finish cleanup before re-raising the first exception.
        """
        outcomes: list[DownloadOutcome] = []
        first_error: BaseException | None = None
        for index in sorted(self._pending):
            while self.is_pending(index):
                try:
                    outcome = self.take(index)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    continue
                outcomes.append(outcome)
                try:
                    outcome.discard()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error
        return tuple(outcomes)
