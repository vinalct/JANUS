from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from janus.checkpoints import DeadLetterState, DeadLetterStore, can_continue_after_dead_letter
from janus.writers import RawArtifactWriter

from .artifacts import _discovered_file_dead_letter_key, _discovered_file_dead_letter_metadata
from .download_loop import FileCandidateResult, _complete_candidate, _download_candidate
from .download_window import DownloadOutcome, OrderedDownloadWindow

if TYPE_CHECKING:
    from .core import DiscoveredFile
    from .extraction import FileExtractionContext, RunTotals


@dataclass(frozen=True, slots=True)
class DownloadWindowStats:
    download_concurrency: int
    max_in_flight_observed: int


@dataclass(slots=True)
class CandidateLedger:
    totals: RunTotals
    dead_letter_state: DeadLetterState | None
    dead_letter_keys: set[str]
    dead_letter_skip_count: int = 0

    def is_dead_lettered(self, discovered_file: DiscoveredFile) -> bool:
        return _discovered_file_dead_letter_key(discovered_file) in self.dead_letter_keys


def _skip_dead_lettered(
    context: FileExtractionContext, ledger: CandidateLedger, discovered_file: DiscoveredFile,
    *, file_index: int,
) -> bool:
    """Count and log a resume-time skip when the candidate is already dead-lettered."""
    if not ledger.is_dead_lettered(discovered_file):
        return False
    ledger.dead_letter_skip_count += 1
    if context.logger is not None:
        context.logger.info(
            "file_candidate_skipped_dead_letter",
            file_index=file_index, filename=discovered_file.filename,
            dead_letter_count=len(ledger.dead_letter_keys),
        )
    return True


def _consume_candidate(
    context: FileExtractionContext, ledger: CandidateLedger, discovered_file: DiscoveredFile,
    *, file_index: int, selected_file_count: int, dead_letter_store: DeadLetterStore,
    complete: Callable[[], FileCandidateResult | None],
) -> None:
    """Run a completion thunk and apply the dead-letter policy on the loop's thread.
    Exceptions, budget decisions and result bookkeeping stay in candidate order.
    """
    logger = context.logger
    try:
        candidate = complete()
    except Exception as exc:
        if logger is not None:
            logger.exception("file_candidate_execution_failed")
        ledger.dead_letter_state = dead_letter_store.record(
            context.plan,
            item_key=_discovered_file_dead_letter_key(discovered_file),
            item_type="file_candidate", error=exc,
            metadata=_discovered_file_dead_letter_metadata(
                discovered_file, file_index=file_index, selected_file_count=selected_file_count,
            ),
        )
        ledger.dead_letter_keys = set(ledger.dead_letter_state.item_keys)
        if logger is not None:
            logger.info(
                "file_candidate_dead_lettered",
                file_index=file_index, filename=discovered_file.filename,
                dead_letter_count=ledger.dead_letter_state.entry_count,
                dead_letter_max_items=context.dead_letter_max_items,
                error_type=type(exc).__name__,
            )
        if not can_continue_after_dead_letter(
            total_item_count=selected_file_count,
            dead_letter_count=ledger.dead_letter_state.entry_count,
            dead_letter_max_items=context.dead_letter_max_items,
        ):
            raise
        ledger.dead_letter_skip_count += 1
        return
    if candidate is None:
        ledger.totals.skipped_file_count += 1
        return
    ledger.totals.absorb(candidate)


def _fill_window(
    context: FileExtractionContext,
    ledger: CandidateLedger,
    window: OrderedDownloadWindow,
    selected_files: Sequence[DiscoveredFile],
    *,
    start: int,
    raw_writer: RawArtifactWriter,
) -> int:
    """Dispatch the next eligible indices without releasing untaken window slots."""
    count = len(selected_files)
    while start <= count and window.has_room:
        discovered_file = selected_files[start - 1]
        context.log_candidate_started(
            discovered_file, file_index=start, selected_file_count=count,
        )
        if not ledger.is_dead_lettered(discovered_file):
            window.submit(
                start,
                partial(
                    _download_candidate, context.downloader, context.plan, discovered_file,
                    client=None, throttle=context.throttle, logger=context.logger,
                    raw_writer=raw_writer,
                ),
            )
            if context.logger is not None:
                context.logger.info(
                    "file_download_dispatched", file_index=start,
                    selected_file_count=count, in_flight=window.in_flight,
                    download_concurrency=context.download_concurrency,
                )
        start += 1
    return start


def _completion_thunk(
    context: FileExtractionContext,
    discovered_file: DiscoveredFile,
    outcome: DownloadOutcome,
    *,
    raw_writer: RawArtifactWriter,
    file_index: int,
) -> Callable[[], FileCandidateResult | None]:
    def complete() -> FileCandidateResult | None:
        if outcome.error is not None:
            raise outcome.error
        assert outcome.downloaded is not None  # DownloadOutcome guarantees exactly one value.
        return _complete_candidate(
            context, discovered_file, outcome.downloaded,
            raw_writer=raw_writer, file_index=file_index,
        )

    return complete


def _run_concurrent_candidates(
    context: FileExtractionContext,
    ledger: CandidateLedger,
    selected_files: Sequence[DiscoveredFile],
    *,
    raw_writer: RawArtifactWriter,
    dead_letter_store: DeadLetterStore,
) -> DownloadWindowStats:
    """Consume by index; stop and discard the speculative tail when the budget fails."""
    count = len(selected_files)
    next_to_dispatch = 1
    with OrderedDownloadWindow(context.download_concurrency) as window:
        for file_index, discovered_file in enumerate(selected_files, start=1):
            next_to_dispatch = _fill_window(
                context, ledger, window, selected_files,
                start=next_to_dispatch, raw_writer=raw_writer,
            )
            if not window.is_pending(file_index):
                _skip_dead_lettered(context, ledger, discovered_file, file_index=file_index)
                continue
            outcome = window.take(file_index)
            try:
                if context.logger is not None:
                    context.logger.info(
                        "file_download_completed", file_index=file_index,
                        in_flight=window.in_flight + 1,
                    )
                if ledger.is_dead_lettered(discovered_file):
                    _skip_dead_lettered(context, ledger, discovered_file, file_index=file_index)
                    continue
                try:
                    _consume_candidate(
                        context, ledger, discovered_file,
                        file_index=file_index, selected_file_count=count,
                        dead_letter_store=dead_letter_store,
                        complete=_completion_thunk(
                            context, discovered_file, outcome,
                            raw_writer=raw_writer, file_index=file_index,
                        ),
                    )
                except Exception:
                    window.stop_dispatch()
                    for drained in window.drain():
                        if context.logger is not None:
                            context.logger.info(
                                "file_download_drained", file_index=drained.file_index,
                                outcome="failed" if drained.error is not None else "downloaded",
                                discarded=True,
                            )
                    raise
            finally:
                # take() transfers ownership; also clean it on a loop-thread interrupt.
                outcome.discard()
                del outcome  # Release the consumed buffer before refilling the window.
        return DownloadWindowStats(
            download_concurrency=context.download_concurrency,
            max_in_flight_observed=window.max_in_flight_observed,
        )
