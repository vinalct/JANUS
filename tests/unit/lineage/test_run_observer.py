import itertools
import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from janus.lineage import RunObserver, compute_config_version
from janus.lineage.store import _prepare_strategy_metadata
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext, WriteResult
from janus.registry import load_registry
from janus.utils.logging import StructuredLogger, build_structured_logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_LOGGER_SEQUENCE = itertools.count()


def _recording_logger() -> tuple[StructuredLogger, StringIO]:
    stream = StringIO()
    logger = build_structured_logger(
        f"janus.tests.lineage.run_observer.{next(_LOGGER_SEQUENCE)}",
        stream=stream,
    )
    return logger, stream


def _warning_events(stream: StringIO) -> list[dict]:
    events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [event for event in events if event["level"] == "WARNING"]


def test_run_observer_persists_running_success_lineage_and_checkpoint_artifacts(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-lineage-001",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    extraction_result = _build_extraction_result(plan)
    write_results = _build_write_results(plan)
    observer = RunObserver()

    started = observer.start_run(plan)
    assert started.run_metadata.status == "running"

    persisted = observer.record_success(
        plan,
        extraction_result,
        write_results,
        finished_at=datetime(2026, 4, 8, 12, 5, tzinfo=UTC),
    )

    run_payload = json.loads(started.run_metadata_path.read_text(encoding="utf-8"))
    assert run_payload["status"] == "succeeded"
    assert run_payload["run_id"] == "run-lineage-001"
    assert run_payload["duration_seconds"] == 300.0
    assert run_payload["records_extracted"] == 100
    assert run_payload["checkpoint_value"] == "2026-04-08T12:00:00Z"

    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))
    assert lineage_payload["status"] == "succeeded"
    assert lineage_payload["strategy_variant"] == "page_number_api"
    assert lineage_payload["config_version"] == compute_config_version(
        plan.source_config.config_path
    )
    assert lineage_payload["configured_outputs"][2]["path"] == plan.metadata_output.path
    assert lineage_payload["artifacts"][0]["path"].endswith("page-0001.json")

    assert persisted.checkpoint_result is not None
    assert persisted.checkpoint_result.decision == "advanced"
    checkpoint_payload = json.loads(
        persisted.checkpoint_result.current_path.read_text(encoding="utf-8")
    )
    assert checkpoint_payload["checkpoint_value"] == "2026-04-08T12:00:00Z"


def test_run_observer_persists_failure_reason_without_writing_checkpoint(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-lineage-002",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    extraction_result = _build_extraction_result(plan)
    write_results = _build_write_results(plan)[:1]
    observer = RunObserver()

    persisted = observer.record_failure(
        plan,
        RuntimeError("upstream returned HTTP 500"),
        extraction_result,
        write_results,
        finished_at=datetime(2026, 4, 8, 12, 7, tzinfo=UTC),
    )

    run_payload = json.loads(persisted.run_metadata_path.read_text(encoding="utf-8"))
    assert run_payload["status"] == "failed"
    assert run_payload["error_type"] == "RuntimeError"
    assert run_payload["failure_reason"] == "upstream returned HTTP 500"

    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))
    assert lineage_payload["status"] == "failed"
    assert lineage_payload["failure_reason"] == "upstream returned HTTP 500"
    assert lineage_payload["materialized_outputs"][0]["zone"] == "raw"
    assert persisted.checkpoint_result is None


def test_prepare_strategy_metadata_coerces_scalars_and_namespaces():
    prepared = _prepare_strategy_metadata(
        {
            "artifact_count": 12,
            "request_timeout_seconds": 3.5,
            "used_cache": True,
            "verbose": False,
            "note": "  ok  ",
        }
    )

    assert prepared == {
        "strategy.artifact_count": "12",
        "strategy.request_timeout_seconds": "3.5",
        "strategy.used_cache": "true",
        "strategy.verbose": "false",
        "strategy.note": "ok",
    }


def test_prepare_strategy_metadata_drops_invalid_entries_and_warns_without_raising():
    logger, stream = _recording_logger()

    prepared = _prepare_strategy_metadata(
        {
            "kept": 7,
            "missing": None,
            "blank": "   ",
            "nested": {"x": 1},
            "listed": [1, 2, 3],
        },
        logger=logger,
    )

    assert prepared == {"strategy.kept": "7"}

    warnings = _warning_events(stream)
    dropped = {(event["fields"]["key"], event["fields"]["reason"]) for event in warnings}
    assert dropped == {
        ("missing", "none_value"),
        ("blank", "empty_value"),
        ("nested", "non_scalar_value"),
        ("listed", "non_scalar_value"),
    }
    for event in warnings:
        assert event["event"] == "strategy_metadata_entry_dropped"


def test_prepare_strategy_metadata_empty_and_none_return_empty():
    assert _prepare_strategy_metadata(None) == {}
    assert _prepare_strategy_metadata({}) == {}


def test_record_success_forwards_strategy_metadata_to_both_records(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-lineage-meta-001",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    extraction_result = _build_extraction_result(plan)
    write_results = _build_write_results(plan)
    observer = RunObserver()

    persisted = observer.record_success(
        plan,
        extraction_result,
        write_results,
        finished_at=datetime(2026, 4, 8, 12, 5, tzinfo=UTC),
        strategy_metadata={"records_extracted": 999, "hook_owner": "ibge"},
    )

    expected_metadata = {
        "strategy.records_extracted": "999",
        "strategy.hook_owner": "ibge",
    }
    assert persisted.run_metadata.metadata_as_dict() == expected_metadata
    assert persisted.lineage_record is not None
    assert persisted.lineage_record.metadata_as_dict() == expected_metadata

    run_payload = json.loads(persisted.run_metadata_path.read_text(encoding="utf-8"))
    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))
    assert run_payload["metadata"] == expected_metadata
    assert lineage_payload["metadata"] == expected_metadata


def test_record_success_namespaced_metadata_cannot_clobber_reserved_fields(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-lineage-meta-002",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    extraction_result = _build_extraction_result(plan)
    write_results = _build_write_results(plan)
    observer = RunObserver()

    persisted = observer.record_success(
        plan,
        extraction_result,
        write_results,
        finished_at=datetime(2026, 4, 8, 12, 5, tzinfo=UTC),
        strategy_metadata={"status": "spoofed", "records_extracted": 42},
    )

    run_payload = json.loads(persisted.run_metadata_path.read_text(encoding="utf-8"))
    # Real reserved fields keep their runtime values ...
    assert run_payload["status"] == "succeeded"
    assert run_payload["records_extracted"] == 100
    # ... and the shadowing keys survive only under the strategy namespace.
    assert run_payload["metadata"]["strategy.status"] == "spoofed"
    assert run_payload["metadata"]["strategy.records_extracted"] == "42"


def test_record_failure_persists_surviving_metadata_without_masking_failure(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-lineage-meta-003",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    extraction_result = _build_extraction_result(plan)
    logger, stream = _recording_logger()
    observer = RunObserver(logger=logger)

    persisted = observer.record_failure(
        plan,
        RuntimeError("upstream returned HTTP 500"),
        extraction_result,
        finished_at=datetime(2026, 4, 8, 12, 7, tzinfo=UTC),
        strategy_metadata={"attempts": 3, "broken": None, "payload": {"x": 1}},
    )

    run_payload = json.loads(persisted.run_metadata_path.read_text(encoding="utf-8"))
    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))

    # The failure is still reported faithfully ...
    assert run_payload["status"] == "failed"
    assert run_payload["failure_reason"] == "upstream returned HTTP 500"
    assert lineage_payload["status"] == "failed"

    # ... and only the coercible metadata entry survives, the rest dropped with warnings.
    assert run_payload["metadata"] == {"strategy.attempts": "3"}
    assert lineage_payload["metadata"] == {"strategy.attempts": "3"}
    dropped_keys = {event["fields"]["key"] for event in _warning_events(stream)}
    assert dropped_keys == {"broken", "payload"}


def test_record_success_without_strategy_metadata_leaves_metadata_empty(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-lineage-meta-004",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    extraction_result = _build_extraction_result(plan)
    write_results = _build_write_results(plan)
    observer = RunObserver()

    persisted = observer.record_success(
        plan,
        extraction_result,
        write_results,
        finished_at=datetime(2026, 4, 8, 12, 5, tzinfo=UTC),
    )

    assert persisted.run_metadata.metadata == ()
    assert persisted.lineage_record is not None
    assert persisted.lineage_record.metadata == ()


def _build_plan(tmp_path: Path, *, run_id: str, started_at: datetime) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
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


def _build_write_results(plan: ExecutionPlan) -> tuple[WriteResult, ...]:
    return (
        WriteResult.from_plan(
            plan,
            "raw",
            path=f"{plan.raw_output.path}/page-0001.json",
            format_name="json",
            mode="append",
            records_written=1,
        ),
        WriteResult.from_plan(
            plan,
            "bronze",
            path=f"{plan.bronze_output.path}/ingestion_date=2026-04-08",
            format_name="parquet",
            mode="append",
            records_written=100,
            partition_by=("ingestion_date",),
            metadata={"writer": "spark"},
        ),
    )
