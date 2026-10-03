"""Release then resume retries exactly the released key.

The claim an operator bets a multi-hour extraction on rests on three mechanisms written for
other reasons: `DeadLetterStore.release` drops K from the skip set; `ResumeState.load` reads
the stores only for a `resume=true` run and clears both otherwise; and the progress record
keeps completed inputs so they are rehydrated, not re-requested. The transport is the
assertion: a request for the completed input fails the test, it does not merely cost time.

Extraction, the stores and the CLI are session-free, so everything here runs in the fast
job with an injected transport; only the bronze assertion needs Spark and skips without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from janus.checkpoints import DeadLetterStore, ExtractionProgressStore
from janus.models import ExecutionPlan, ExtractionResult
from janus.planner import PlannedRun, Planner, PlanningRequest
from janus.runtime.executor import SourceExecutor
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.storage import StorageLayout
from tests.support.operator_cli import OPERATOR_ENV, run_janus
from tests.support.semantics_fixtures import CLEAN, CLEAN_PRODUCER, install_profile, materialize

RED = pytest.mark.xfail(
    strict=True,
    reason="a completed input is re-requested on resume when the next input dead-letters on "
    "its first request (F-12)",
)

FIRST, SECOND = "2026-09-01", "2026-09-02"
OPERATOR = {OPERATOR_ENV: "ops-tester"}


def _record(code: str, day: str) -> dict[str, Any]:
    return {"code": code, "updated_at": f"{day}T10:00:00Z", "payload": {"label": code.lower()}}


@dataclass
class WindowTransport:
    """Answers by `window_start` and page; a window in `completed` must never be asked for."""

    pages: dict[tuple[str, int], tuple[int, dict[str, Any]]]
    completed: frozenset[str] = frozenset()
    requests: list[tuple[str, int]] = field(default_factory=list)

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request):
        query = parse_qs(urlsplit(request.full_url()).query)
        window, page = query["window_start"][0], int(query["page"][0])
        self.requests.append((window, page))
        if window in self.completed:
            pytest.fail(f"{window} page {page} was re-requested: that input already completed")
        status, payload = self.pages[(window, page)]
        return ApiResponse(request=request, status_code=status, body=json.dumps(payload).encode())


FIRST_RUN_PAGES = {
    (FIRST, 1): (200, {"records": [_record("A", FIRST), _record("B", FIRST)]}),
    (FIRST, 2): (200, {"records": [_record("C", FIRST)]}),
    (SECOND, 1): (400, {"message": "upstream rejected the window"}),
}
RESUME_PAGES = {(SECOND, 1): (200, {"records": [_record("D", SECOND)]})}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return materialize(CLEAN, tmp_path / "project")


def _strategy(root: Path, transport: WindowTransport) -> ApiStrategy:
    data = root / "data"
    layout = StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": str(data),
                "raw_dir": str(data / "raw"),
                "bronze_dir": str(data / "bronze"),
                "metadata_dir": str(data / "metadata"),
            }
        },
        root,
    )
    return ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )


def _plan(root: Path, run_id: str, *, resume: bool = False) -> ExecutionPlan:
    request = PlanningRequest.create(
        source_id=CLEAN_PRODUCER,
        environment="local",
        project_root=root,
        run_id=run_id,
        started_at=datetime(2026, 9, 20, 6, 0, tzinfo=UTC),
        attributes={"resume": "true"} if resume else None,
    )
    return Planner().plan(request).plan


def _first_run(root: Path, pages=FIRST_RUN_PAGES) -> tuple[ExecutionPlan, ExtractionResult]:
    plan = _plan(root, "run-first")
    return plan, _strategy(root, WindowTransport(dict(pages))).extract(plan)


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
    plan, result = _first_run(root)

    assert result.metadata_as_dict()["dead_letter_count"] == "1"
    assert DeadLetterStore().load(plan).entry_count == 1
    assert ExtractionProgressStore().load(plan) is not None


@RED
def test_an_input_that_completed_is_recorded_before_the_next_one_dead_letters(root: Path) -> None:
    """The cause of F-12 in isolation: input 1 ran to its last page, input 2 failed on its
    first request, and the progress record still names input 1 as unfinished."""
    plan, _result = _first_run(root)

    progress = ExtractionProgressStore().load(plan)

    assert progress is not None
    completed = [entry["key"] for entry in progress["completed_inputs"]]
    assert [FIRST in key for key in completed] == [True], progress


@RED
def test_release_then_resume_retries_only_the_released_input(root: Path) -> None:
    plan, first = _first_run(root)
    key = _second_input_key(plan)
    assert first.records_extracted == 3

    released = _release(root, key)
    assert released.exit_code == 0, released.output

    transport = WindowTransport(dict(RESUME_PAGES), completed=frozenset({FIRST}))
    resumed = _strategy(root, transport).extract(_plan(root, "run-resume", resume=True))

    assert transport.requests == [(SECOND, 1)]
    assert DeadLetterStore().path(plan).exists() is False
    resumed_inputs = {Path(artifact.path).parent.name for artifact in resumed.artifacts}
    assert resumed_inputs == {"request-input-000001", "request-input-000002"}


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
    transport = WindowTransport(dict(RESUME_PAGES), completed=frozenset({FIRST}))

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


@RED
def test_bronze_holds_each_input_exactly_once_after_release_and_resume() -> None:
    pytest.importorskip("pyspark")
    pytest.fail("the bronze half is written where PySpark is available")
