from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.checkpoints.progress import ExtractionProgressStore
from janus.models import ExecutionPlan, RunContext
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_plan(tmp_path: Path, *, run_id: str = "run-progress-001") -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 14, 10, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


# ---------------------------------------------------------------------------
# save / load round-trip — pagination position
# ---------------------------------------------------------------------------


def test_progress_store_saves_and_loads_page_number(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, page_number=42, artifact_count=42)
    progress = store.load(plan)

    assert progress is not None
    assert progress["source_id"] == plan.source.source_id
    assert progress["last_page_number"] == 42
    assert progress["artifact_count"] == 42
    assert "last_offset" not in progress
    assert "last_cursor" not in progress


def test_progress_store_saves_and_loads_offset(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, offset=1000, artifact_count=10)
    progress = store.load(plan)

    assert progress is not None
    assert progress["last_offset"] == 1000
    assert "last_page_number" not in progress


def test_progress_store_saves_and_loads_cursor(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, cursor="abc123", request_index=5, artifact_count=5)
    progress = store.load(plan)

    assert progress is not None
    assert progress["last_cursor"] == "abc123"
    assert progress["request_index"] == 5


def test_progress_store_overwrites_on_subsequent_saves(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, page_number=10, artifact_count=10)
    store.save(plan, page_number=20, artifact_count=20)

    progress = store.load(plan)
    assert progress is not None
    assert progress["last_page_number"] == 20
    assert progress["artifact_count"] == 20


# ---------------------------------------------------------------------------
# save / load round-trip — content-based input tracking
# ---------------------------------------------------------------------------


def test_progress_store_saves_completed_inputs(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    completed = [
        ("orgao_codigo=170010|window_end=2025-01-31|window_start=2025-01-01", 1),
        ("orgao_codigo=170010|window_end=2025-02-28|window_start=2025-02-01", 2),
    ]
    store.save(
        plan,
        page_number=3,
        artifact_count=20,
        completed_inputs=completed,
        current_input_key="orgao_codigo=170010|window_end=2025-03-31|window_start=2025-03-01",
        current_input_index=3,
    )
    progress = store.load(plan)

    assert progress is not None
    assert len(progress["completed_inputs"]) == 2
    assert progress["completed_inputs"][0]["key"] == completed[0][0]
    assert progress["completed_inputs"][0]["index"] == 1
    assert progress["completed_inputs"][1]["index"] == 2
    assert progress["current_input_key"] == "orgao_codigo=170010|window_end=2025-03-31|window_start=2025-03-01"
    assert progress["current_input_index"] == 3


def test_progress_store_completed_inputs_defaults_to_empty_list(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, page_number=1, artifact_count=5)
    progress = store.load(plan)

    assert progress is not None
    assert progress["completed_inputs"] == []
    assert progress["current_input_key"] is None


def test_progress_store_completed_inputs_none_treated_as_empty(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, page_number=1, artifact_count=5, completed_inputs=None)
    progress = store.load(plan)

    assert progress is not None
    assert progress["completed_inputs"] == []


# ---------------------------------------------------------------------------
# load — edge cases
# ---------------------------------------------------------------------------


def test_progress_store_load_returns_none_when_no_file(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    assert store.load(plan) is None


def test_progress_store_load_returns_none_for_wrong_source_id(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, page_number=5, artifact_count=5)

    # Manually corrupt the source_id
    from janus.lineage.persistence import MetadataZonePaths

    progress_path = MetadataZonePaths.from_plan(plan).base_dir / "extraction_progress.json"
    payload = json.loads(progress_path.read_text())
    payload["source_id"] = "some_other_source"
    progress_path.write_text(json.dumps(payload))

    assert store.load(plan) is None


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_progress_store_clear_removes_file(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(plan, page_number=99, artifact_count=99)
    assert store.load(plan) is not None

    store.clear(plan)
    assert store.load(plan) is None


def test_progress_store_clear_is_safe_when_no_file(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    # Should not raise
    store.clear(plan)


# ---------------------------------------------------------------------------
# file is written atomically
# ---------------------------------------------------------------------------


def test_progress_store_persists_raw_path_prefix_for_resume(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    store.save(
        plan,
        page_number=2,
        artifact_count=2,
        raw_path_prefix="runs/ingestion_date=2026-04-14/run_id=run-progress-001",
    )
    progress = store.load(plan)

    assert progress is not None
    assert progress["raw_path_prefix"] == (
        "runs/ingestion_date=2026-04-14/run_id=run-progress-001"
    )


def test_progress_store_writes_valid_json(tmp_path):
    plan = _build_plan(tmp_path)
    store = ExtractionProgressStore()

    path = store.save(plan, page_number=7, artifact_count=7)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["source_id"] == plan.source.source_id
    assert payload["last_page_number"] == 7
    assert "updated_at" in payload
    assert "completed_inputs" in payload
    assert "current_input_key" in payload
