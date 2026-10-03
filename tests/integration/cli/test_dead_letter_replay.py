"""Release then resume retries exactly the released key.

The claim an operator bets a multi-hour extraction on rests on three mechanisms written for
other reasons: `DeadLetterStore.release` drops K from the skip set; `ResumeState.load` reads
the stores only for a `resume=true` run and clears both otherwise; and the progress record
keeps completed inputs so they are rehydrated, not re-requested. That record names an input
complete before the next one starts (`ExtractionProgressStore.save_between_inputs`), so an
input that dead-letters on its first request cannot leave the one before it unfinished.

The transport is the assertion: a request for a window the run must not touch fails the
test, it does not merely cost time or land in the dead-letter store as one more failure.

Extraction, the stores and the CLI are session-free, so everything here runs in the fast
job with an injected transport; only the bronze assertion needs Spark and skips without it.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from janus.checkpoints import DeadLetterStore, ExtractionProgressStore
from janus.models import ExecutionPlan, ExtractionResult
from janus.planner import (
    PlannedRun,
    Planner,
    PlanningRequest,
    StrategyBinding,
    StrategyCatalog,
)
from janus.runtime.executor import ExecutedRun, SourceExecutor
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.logging import StructuredLogger, build_structured_logger
from janus.utils.storage import StorageLayout
from tests.support.operator_cli import OPERATOR_ENV, run_janus
from tests.support.semantics_fixtures import CLEAN, CLEAN_PRODUCER, install_profile, materialize
from tests.support.spark_sessions import build_iceberg_session, require_iceberg_runtime

FIRST, SECOND = "2026-09-01", "2026-09-02"
OPERATOR = {OPERATOR_ENV: "ops-tester"}
STARTED_AT = datetime(2026, 9, 20, 6, 0, tzinfo=UTC)
BRONZE_TABLE = "semantics.clean_producer"

ALREADY_COMPLETED = "that input already completed"
STILL_DEAD_LETTERED = "that input is still dead-lettered"


def _record(code: str, day: str) -> dict[str, Any]:
    return {"code": code, "updated_at": f"{day}T10:00:00Z", "payload": {"label": code.lower()}}


@dataclass
class WindowTransport:
    """Answers by `window_start` and page.

    A window in `forbidden`, or a page with no scripted answer, fails the test when asked
    for. `pytest.fail` is not an `Exception`, so the extraction's dead-letter handler cannot
    record either as one more failed request input and carry on.
    """

    pages: dict[tuple[str, int], tuple[int, Any]]
    forbidden: Mapping[str, str] = field(default_factory=dict)
    requests: list[tuple[str, int]] = field(default_factory=list)

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request):
        query = parse_qs(urlsplit(request.full_url()).query)
        window, page = query["window_start"][0], int(query["page"][0])
        self.requests.append((window, page))
        if window in self.forbidden:
            pytest.fail(f"{window} page {page} was re-requested: {self.forbidden[window]}")
        if (window, page) not in self.pages:
            pytest.fail(f"{window} page {page} was requested: no answer is scripted for it")
        status, payload = self.pages[(window, page)]
        return ApiResponse(request=request, status_code=status, body=json.dumps(payload).encode())


# Bare arrays, as the endpoint the Spark half reads them from serves them: the JSON reader
# makes one row per element, where an envelope would read as one empty row per page.
FIRST_RUN_PAGES = {
    (FIRST, 1): (200, [_record("A", FIRST), _record("B", FIRST)]),
    (FIRST, 2): (200, [_record("C", FIRST)]),
    (SECOND, 1): (400, {"message": "upstream rejected the window"}),
}
RESUME_PAGES = {(SECOND, 1): (200, [_record("D", SECOND)])}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return materialize(CLEAN, tmp_path / "project")


def _environment(root: Path) -> dict[str, Any]:
    data = root / "data"
    return {
        "storage": {
            "root_dir": str(data),
            "raw_dir": str(data / "raw"),
            "bronze_dir": str(data / "bronze"),
            "metadata_dir": str(data / "metadata"),
        }
    }


def _strategy(
    root: Path, transport: WindowTransport, *, logger: StructuredLogger | None = None
) -> ApiStrategy:
    layout = StorageLayout.from_environment_config(_environment(root), root)
    return ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
        logger=logger,
    )


def _request(root: Path, run_id: str, *, resume: bool = False) -> PlanningRequest:
    return PlanningRequest.create(
        source_id=CLEAN_PRODUCER,
        environment="local",
        project_root=root,
        run_id=run_id,
        started_at=STARTED_AT,
        attributes={"resume": "true"} if resume else None,
    )


def _plan(root: Path, run_id: str, *, resume: bool = False) -> ExecutionPlan:
    return Planner().plan(_request(root, run_id, resume=resume)).plan


def _first_run(
    root: Path, pages=FIRST_RUN_PAGES, *, logger: StructuredLogger | None = None
) -> tuple[ExecutionPlan, ExtractionResult]:
    plan = _plan(root, "run-first")
    return plan, _strategy(root, WindowTransport(dict(pages)), logger=logger).extract(plan)


def _capturing_logger(name: str) -> tuple[StructuredLogger, io.StringIO]:
    stream = io.StringIO()
    return build_structured_logger(f"janus.tests.dead_letter_replay.{name}", stream=stream), stream


def _events(stream: io.StringIO) -> list[str]:
    return [json.loads(line)["event"] for line in stream.getvalue().splitlines()]


def _second_input_key(plan: ExecutionPlan) -> str:
    """Read from the recorded state, never built here: the key format is not the contract."""
    state = DeadLetterStore().load(plan)
    assert state is not None
    (key,) = [entry.item_key for entry in state.entries if SECOND in entry.item_key]
    return key


def _release(root: Path, key: str):
    return run_janus(
        (
            "dead-letters", "release", "--project-root", str(root), "--source-id", CLEAN_PRODUCER,
            "--item-key", key, "--reason", "upstream fixed",
        ),
        env=OPERATOR,
    )


def test_progress_is_retained_when_an_input_was_dead_lettered(root: Path) -> None:
    logger, stream = _capturing_logger("dead_lettered")

    plan, result = _first_run(root, logger=logger)

    assert result.metadata_as_dict()["dead_letter_count"] == "1"
    assert DeadLetterStore().load(plan).entry_count == 1
    assert ExtractionProgressStore().load(plan) is not None
    assert "api_extraction_progress_retained" in _events(stream)


def test_progress_is_retained_when_an_input_was_skipped(root: Path) -> None:
    plan, _first = _first_run(root)
    transport = WindowTransport({})
    logger, stream = _capturing_logger("skipped")

    skipped = _strategy(root, transport, logger=logger).extract(
        _plan(root, "run-skip", resume=True)
    )

    assert transport.requests == []
    assert skipped.metadata_as_dict()["dead_letter_skipped_count"] == "1"
    assert "api_extraction_progress_retained" in _events(stream)
    progress = ExtractionProgressStore().load(plan)
    assert progress is not None
    assert [FIRST in entry["key"] for entry in progress["completed_inputs"]] == [True]
    assert DeadLetterStore().load(plan).entry_count == 1


def test_an_input_that_completed_is_recorded_before_the_next_one_dead_letters(root: Path) -> None:
    """The cause of F-12 in isolation: input 1 ran to its last page, input 2 failed on its
    first request, and the progress record still names input 1 as unfinished."""
    plan, _result = _first_run(root)

    progress = ExtractionProgressStore().load(plan)

    assert progress is not None
    completed = [entry["key"] for entry in progress["completed_inputs"]]
    assert [FIRST in key for key in completed] == [True], progress


def test_release_then_resume_retries_only_the_released_input(root: Path) -> None:
    plan, first = _first_run(root)
    key = _second_input_key(plan)
    assert first.records_extracted == 3

    released = _release(root, key)
    assert released.exit_code == 0, released.output

    transport = WindowTransport(dict(RESUME_PAGES), forbidden={FIRST: ALREADY_COMPLETED})
    resumed = _strategy(root, transport).extract(_plan(root, "run-resume", resume=True))

    assert transport.requests == [(SECOND, 1)]
    assert DeadLetterStore().path(plan).exists() is False
    resumed_inputs = {Path(artifact.path).parent.name for artifact in resumed.artifacts}
    assert resumed_inputs == {"request-input-000001", "request-input-000002"}
    # Input 1 is the first run's pages, read back from where it wrote them; input 2 is new.
    first_paths = [artifact.path for artifact in first.artifacts]
    resumed_paths = [artifact.path for artifact in resumed.artifacts]
    assert resumed_paths[: len(first_paths)] == first_paths
    assert all(Path(path).is_file() for path in resumed_paths)
    assert resumed.records_extracted == 1
    assert ExtractionProgressStore().load(plan) is None


def test_release_of_the_only_dead_letter_deletes_the_state_file(root: Path) -> None:
    plan, _first = _first_run(root)

    released = _release(root, _second_input_key(plan))

    assert released.exit_code == 0, released.output
    assert DeadLetterStore().path(plan).exists() is False
    assert DeadLetterStore().load(plan) is None


def test_replay_execute_carries_the_resume_attribute(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `resume=true`, `ResumeState.load` clears both stores: the unreleased dead
    letter would be forgotten and its input re-requested. Here that request fails the test."""
    both_fail = {
        (FIRST, 1): (400, {"message": "upstream rejected the window"}),
        (SECOND, 1): (400, {"message": "upstream rejected the window"}),
    }
    plan, _first = _first_run(root, both_fail)
    key = _second_input_key(plan)
    transport = WindowTransport(dict(RESUME_PAGES), forbidden={FIRST: STILL_DEAD_LETTERED})

    @dataclass
    class Executed:
        planned_run: PlannedRun
        is_successful: bool = True

        def to_summary(self) -> dict[str, Any]:
            return {"status": "succeeded"}

    def execute(self, planned_run, spark_provider, environment_config):
        _strategy(root, transport).extract(planned_run.plan)
        return Executed(planned_run)

    monkeypatch.setattr(SourceExecutor, "execute", execute)
    install_profile(root, "local")

    replayed = run_janus(
        (
            "dead-letters", "replay", "--project-root", str(root), "--environment", "local",
            "--source-id", CLEAN_PRODUCER, "--item-key", key, "--reason", "upstream fixed",
            "--execute",
        ),
        env=OPERATOR,
    )

    assert replayed.exit_code == 0, replayed.output
    assert transport.requests == [(SECOND, 1)]
    remaining = DeadLetterStore().load(plan)
    assert remaining is not None
    assert [FIRST in item_key for item_key in remaining.item_keys] == [True]


def test_bronze_holds_each_input_exactly_once_after_release_and_resume(
    root: Path, tmp_path: Path
) -> None:
    """The narrative through `SourceExecutor`, so both runs materialize as an operator's do.

    The first run commits input 1 and dead-letters input 2. The resume reads input 1's pages
    back from the raw zone, fetches input 2, and writes all four records again; the contract's
    primaryKey makes that write a MERGE on `code`, so bronze holds each record once.
    """
    require_iceberg_runtime()

    def session_factory():
        return build_iceberg_session("janus-dead-letter-replay", tmp_path / "catalog")

    first = _execute(root, "run-first", WindowTransport(dict(FIRST_RUN_PAGES)), session_factory)

    assert first.is_successful, first.failure_reason
    assert first.extraction_result is not None
    assert first.extraction_result.metadata_as_dict()["dead_letter_count"] == "1"
    assert _bronze_rows(session_factory) == [("A", FIRST), ("B", FIRST), ("C", FIRST)]

    released = _release(root, _second_input_key(_plan(root, "run-first")))
    assert released.exit_code == 0, released.output

    transport = WindowTransport(dict(RESUME_PAGES), forbidden={FIRST: ALREADY_COMPLETED})
    resumed = _execute(root, "run-resume", transport, session_factory, resume=True)

    assert resumed.is_successful, resumed.failure_reason
    assert transport.requests == [(SECOND, 1)]
    assert _bronze_rows(session_factory) == [
        ("A", FIRST),
        ("B", FIRST),
        ("C", FIRST),
        ("D", SECOND),
    ]


def _execute(
    root: Path,
    run_id: str,
    transport: WindowTransport,
    session_factory: Callable[[], Any],
    *,
    resume: bool = False,
) -> ExecutedRun:
    """One `run --execute [--resume]`, with the strategy bound to the scripted transport."""
    binding = StrategyBinding("api", "page_number_api", _strategy(root, transport))
    planned_run = Planner(strategy_catalog=StrategyCatalog((binding,))).plan(
        _request(root, run_id, resume=resume)
    )
    provider = SparkSessionProvider({}, {}, session_factory=session_factory)
    return SourceExecutor().execute(planned_run, provider, _environment(root))


def _bronze_rows(session_factory: Callable[[], Any]) -> list[tuple[str, str]]:
    """Every bronze row as (code, day), sorted: a list, so a duplicate shows as one."""
    session = session_factory()
    try:
        rows = session.table(BRONZE_TABLE).select("code", "updated_at").collect()
    finally:
        session.stop()
    return sorted((row["code"], row["updated_at"][:10]) for row in rows)
