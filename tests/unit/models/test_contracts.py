from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    SourceReference,
    WriteResult,
)
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_source_reference_projects_runtime_identity_from_source_config():
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")

    source = SourceReference.from_source_config(source_config)

    assert source.source_id == "federal_open_data_example"
    assert source.name == source_config.name
    assert source.source_type == "api"
    assert source.strategy == "api"
    assert source.strategy_variant == "page_number_api"


def test_execution_plan_lifts_planner_fields_from_source_config():
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-20260408-001",
        environment="local",
        project_root=PROJECT_ROOT,
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
        attributes={"trigger": "unit-test"},
    )

    plan = ExecutionPlan.from_source_config(source_config, run_context).with_note("validated")

    assert plan.run_context.attributes_as_dict() == {"trigger": "unit-test"}
    assert plan.source.source_id == source_config.source_id
    assert plan.extraction_mode == source_config.extraction.mode
    assert plan.checkpoint_strategy == source_config.extraction.checkpoint_strategy
    assert plan.raw_output.path == source_config.outputs.raw.path
    assert plan.metadata_output.format == source_config.outputs.metadata.format
    assert plan.notes == ("validated",)


def test_extraction_result_and_write_result_keep_runtime_handoffs_typed():
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-20260408-002",
        environment="local",
        project_root=PROJECT_ROOT,
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)

    extraction_result = ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(
                path="data/raw/example/federal_open_data_example/page-0001.json",
                format="json",
            ),
        ),
        records_extracted=100,
        checkpoint_value="2026-04-08T12:00:00Z",
        metadata={"page_count": "1"},
    ).with_metadata("http_status", "200")
    write_result = WriteResult.from_plan(
        plan,
        "bronze",
        records_written=100,
        partition_by=("ingestion_date",),
        metadata={"writer": "spark"},
    ).with_metadata("compression", "snappy")

    assert extraction_result.run_id == "run-20260408-002"
    assert extraction_result.artifact_paths == (
        "data/raw/example/federal_open_data_example/page-0001.json",
    )
    assert extraction_result.metadata_as_dict() == {
        "http_status": "200",
        "page_count": "1",
    }
    assert write_result.zone == "bronze"
    assert write_result.path == source_config.outputs.bronze.path
    assert write_result.partition_by == ("ingestion_date",)
    assert write_result.metadata_as_dict() == {
        "compression": "snappy",
        "writer": "spark",
    }


def test_write_result_rejects_unknown_output_zone():
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-20260408-003",
        environment="local",
        project_root=PROJECT_ROOT,
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)

    with pytest.raises(ValueError, match="zone must be one of"):
        WriteResult.from_plan(plan, "silver")


def test_execution_plan_requires_matching_source_reference():
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-20260408-004",
        environment="local",
        project_root=PROJECT_ROOT,
    )
    wrong_source = replace(
        SourceReference.from_source_config(source_config),
        source_id="different_source",
    )

    with pytest.raises(ValueError, match="must match source_config.source_id"):
        ExecutionPlan(
            run_context=run_context,
            source=wrong_source,
            source_config=source_config,
            extraction_mode=source_config.extraction.mode,
            checkpoint_strategy=source_config.extraction.checkpoint_strategy,
            checkpoint_field=source_config.extraction.checkpoint_field,
            raw_output=source_config.outputs.raw,
            bronze_output=source_config.outputs.bronze,
            metadata_output=source_config.outputs.metadata,
        )
