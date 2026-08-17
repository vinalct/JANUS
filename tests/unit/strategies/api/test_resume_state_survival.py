"""Resume state must survive the runs that cannot use it, and read where the writer wrote."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

# The api suite is not a package (no ``__init__.py``), so pytest puts this directory on
# ``sys.path`` and the sibling module is imported by plain name.
from test_api_strategy import (
    ResponseSpec,
    _build_source_config,
    _build_strategy,
    _storage_layout,
)

from janus.checkpoints import DeadLetterStore, ExtractionProgressStore
from janus.models import ExecutionPlan, RunContext
from janus.strategies.api.artifacts import (
    _pages_dir,
    _rediscover_all_artifacts_for_input,
    _rediscover_raw_artifacts,
)

RAW_PREFIX = Path("runs/ingestion_date=2026-08-16/run_id=run-interrupted")


def _plan(tmp_path: Path, source_config, *, run_id: str, resume: bool = False) -> ExecutionPlan:
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
            attributes={"resume": "true"} if resume else {},
        ),
    )


def _write_pages(directory: Path, page_numbers: range, *, marker: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for page in page_numbers:
        payload = {"records": [{"id": f"{marker}-{page}"}]}
        (directory / f"page-{page:04d}.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix 1: the end-of-loop cleanup must not fire for a run that extracted nothing


def test_progress_survives_a_run_whose_only_input_was_dead_letter_skipped(tmp_path):
    """The regression itself: a no-op run must not delete the resume position."""
    source_config = _build_source_config(
        tmp_path,
        source_id="progress_survives_skip",
        requests_per_minute=None,
        dead_letter_max_items=100,
    )
    seed_plan = _plan(tmp_path, source_config, run_id="run-interrupted")
    DeadLetterStore().record(
        seed_plan,
        item_key="__none__",
        item_type="request_input",
        error=RuntimeError("API request failed with status 400"),
        metadata={"request_url": "https://example.invalid/records"},
    )
    ExtractionProgressStore().save(
        seed_plan,
        page_number=9686,
        artifact_count=9686,
        current_input_key="__none__",
        current_input_index=1,
        request_input_count=1,
        raw_path_prefix=str(RAW_PREFIX),
    )

    resume_plan = _plan(tmp_path, source_config, run_id="run-resume", resume=True)
    strategy, transport = _build_strategy(tmp_path, [])
    result = strategy.extract(resume_plan)

    assert result.records_extracted == 0
    assert transport.requests == []
    assert result.metadata_as_dict()["dead_letter_skipped_count"] == "1"

    survived = ExtractionProgressStore().load(resume_plan)
    assert survived is not None, (
        "the skipped-input run deleted the progress record — this is the bug that cost a "
        "9,686-page extraction its resume position"
    )
    assert survived["last_page_number"] == 9686
    assert survived["raw_path_prefix"] == str(RAW_PREFIX)


def test_a_clean_run_still_clears_progress(tmp_path):
    """The cleanup must keep happening when there is genuinely nothing left to resume."""
    source_config = _build_source_config(
        tmp_path,
        source_id="progress_cleared_when_complete",
        requests_per_minute=None,
    )
    seed_plan = _plan(tmp_path, source_config, run_id="run-interrupted")
    ExtractionProgressStore().save(
        seed_plan,
        page_number=1,
        artifact_count=1,
        current_input_key="__none__",
        current_input_index=1,
        request_input_count=1,
    )

    resume_plan = _plan(tmp_path, source_config, run_id="run-resume", resume=True)
    strategy, _transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"records": []})],
    )
    strategy.extract(resume_plan)

    assert ExtractionProgressStore().load(resume_plan) is None


# ---------------------------------------------------------------------------
# Fix 2: rediscovery must read the directory the writer wrote to


def test_pages_dir_applies_the_run_scoped_raw_prefix(tmp_path):
    source_config = _build_source_config(
        tmp_path, source_id="pages_dir_prefix", requests_per_minute=None
    )
    plan = _plan(tmp_path, source_config, run_id="run-any")
    layout = _storage_layout(tmp_path)

    flat = _pages_dir(plan, layout, 1, 1)
    prefixed = _pages_dir(plan, layout, 1, 1, RAW_PREFIX)

    assert flat.name == "pages"
    assert prefixed == flat.parent / RAW_PREFIX / "pages"


def test_resume_rediscovers_pages_written_under_the_run_prefix(tmp_path):
    """The pages the interrupted attempt wrote are the ones recovered."""
    source_config = _build_source_config(
        tmp_path, source_id="rediscover_prefixed", requests_per_minute=None
    )
    plan = _plan(tmp_path, source_config, run_id="run-any")
    layout = _storage_layout(tmp_path)
    raw_root = layout.resolve_output(plan, "raw").resolved_path
    _write_pages(raw_root / RAW_PREFIX / "pages", range(1, 5), marker="august")

    artifacts = _rediscover_raw_artifacts(
        plan,
        layout,
        {"last_page_number": 4, "raw_path_prefix": str(RAW_PREFIX)},
        1,
        1,
    )

    assert [Path(a.path).name for a in artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
        "page-0004.json",
    ]
    assert all(str(RAW_PREFIX) in a.path for a in artifacts)


def test_a_prefixed_resume_ignores_leftovers_from_the_flat_layout(tmp_path):
    """The dangerous half: stale pages must not be rehydrated into an unrelated run.

    A raw zone that predates the run-prefix layout still holds ``<raw>/pages``. Reading that
    directory for a prefixed run would stitch a months-old extraction onto a fresh one and
    write the mixture to bronze as a single run.
    """
    source_config = _build_source_config(
        tmp_path, source_id="ignores_stale_flat_pages", requests_per_minute=None
    )
    plan = _plan(tmp_path, source_config, run_id="run-any")
    layout = _storage_layout(tmp_path)
    raw_root = layout.resolve_output(plan, "raw").resolved_path
    _write_pages(raw_root / "pages", range(1, 100), marker="stale-april")

    artifacts = _rediscover_raw_artifacts(
        plan,
        layout,
        {"last_page_number": 50, "raw_path_prefix": str(RAW_PREFIX)},
        1,
        1,
    )

    assert artifacts == [], (
        "rediscovery reached outside the run prefix and picked up a different extraction"
    )


def test_progress_without_a_prefix_still_reads_the_flat_layout(tmp_path):
    """Resume state written before the prefix existed must keep resolving as it always did."""
    source_config = _build_source_config(
        tmp_path, source_id="legacy_flat_progress", requests_per_minute=None
    )
    plan = _plan(tmp_path, source_config, run_id="run-any")
    layout = _storage_layout(tmp_path)
    raw_root = layout.resolve_output(plan, "raw").resolved_path
    _write_pages(raw_root / "pages", range(1, 4), marker="legacy")

    artifacts = _rediscover_raw_artifacts(plan, layout, {"last_page_number": 3}, 1, 1)

    assert [Path(a.path).name for a in artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]


def test_completed_input_recovery_also_honours_the_prefix(tmp_path):
    """The other rediscovery entry point reads the same layout, so it takes the same prefix."""
    source_config = _build_source_config(
        tmp_path, source_id="completed_input_prefix", requests_per_minute=None
    )
    plan = _plan(tmp_path, source_config, run_id="run-any")
    layout = _storage_layout(tmp_path)
    raw_root = layout.resolve_output(plan, "raw").resolved_path
    _write_pages(raw_root / "pages", range(1, 6), marker="stale")
    _write_pages(raw_root / RAW_PREFIX / "pages", range(1, 3), marker="current")

    recovered = _rediscover_all_artifacts_for_input(plan, layout, 1, 1, RAW_PREFIX)

    assert len(recovered) == 2
    assert all(str(RAW_PREFIX) in a.path for a in recovered)
