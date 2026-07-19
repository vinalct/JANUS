from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.lineage.models import LineageRecord
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_from_runtime_metadata_round_trips(tmp_path):
    plan = _build_plan(tmp_path)

    record = LineageRecord.from_runtime(
        plan,
        status="succeeded",
        extraction_result=_build_extraction_result(plan),
        metadata={"strategy.foo": "bar"},
    )

    assert record.metadata_as_dict() == {"strategy.foo": "bar"}
    assert record.to_dict()["metadata"] == {"strategy.foo": "bar"}


def test_from_runtime_without_metadata_defaults_empty(tmp_path):
    plan = _build_plan(tmp_path)

    record = LineageRecord.from_runtime(plan, status="succeeded")

    assert record.metadata == ()
    assert record.metadata_as_dict() == {}
    assert record.to_dict()["metadata"] == {}


def test_metadata_and_extraction_metadata_are_independent(tmp_path):
    plan = _build_plan(tmp_path)

    record = LineageRecord.from_runtime(
        plan,
        status="succeeded",
        extraction_result=_build_extraction_result(plan),
        metadata={"strategy.foo": "bar"},
    )

    payload = record.to_dict()
    assert payload["extraction_metadata"] == {"http_status": "200"}
    assert payload["metadata"] == {"strategy.foo": "bar"}


@pytest.mark.parametrize("invalid", [{"": "x"}, {"k": ""}])
def test_from_runtime_rejects_invalid_metadata(tmp_path, invalid):
    plan = _build_plan(tmp_path)

    with pytest.raises(ValueError):
        LineageRecord.from_runtime(plan, status="succeeded", metadata=invalid)


def _build_plan(tmp_path: Path) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-lineage-metadata",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _build_extraction_result(plan: ExecutionPlan) -> ExtractionResult:
    return ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(
                path=f"{plan.raw_output.path}/page-0001.json",
                format="json",
            ),
        ),
        records_extracted=100,
        checkpoint_value="2026-04-08T12:00:00Z",
        metadata={"http_status": "200"},
    )
