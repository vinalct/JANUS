"""Duration expectations are evaluated once and persisted before lifecycle emission."""

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from janus.lineage import RunMetadataStore, RunObserver
from janus.models import ExecutionPlan, ExtractionResult, RunContext
from janus.models.config.types import FreshnessConfig, OperationsConfig
from janus.observability import RunRecord
from janus.observability.openlineage import OpenLineageDatasetContext, build_openlineage_run_event
from janus.registry import load_registry
from janus.utils.logging import StructuredLogger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STARTED_AT = datetime(2026, 7, 4, 12, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 4, 12, 0, 5, tzinfo=UTC)


def _plan(root, budget, freshness=86400):
    config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    config = replace(
        config,
        operations=OperationsConfig(
            max_duration_seconds=budget,
            freshness=FreshnessConfig(freshness) if freshness is not None else None,
        ),
    )
    return ExecutionPlan.from_source_config(
        config,
        RunContext.create(
            run_id="order23-budget-persistence",
            environment="local",
            project_root=root,
            started_at=STARTED_AT,
        ),
    )


@pytest.mark.parametrize("budget", [None, 4])
@pytest.mark.parametrize("freshness", [None, 86400])
def test_start_persists_declarations_without_evaluating(tmp_path, monkeypatch, budget, freshness):
    inspected = []

    def unexpected_evaluation(self, duration_seconds):
        raise AssertionError("START cannot compare a duration")

    monkeypatch.setattr(OperationsConfig, "evaluate_duration", unexpected_evaluation)

    class InspectingEmitter:
        def emit_started(self, plan, persisted):
            payload = json.loads(persisted.run_metadata_path.read_bytes())
            assert payload["duration_budget_seconds"] == budget
            assert payload["duration_budget_outcome"] is None
            assert payload["freshness_max_age_seconds"] == freshness
            inspected.append(payload)
            raise RuntimeError("injected START emitter failure")

    persisted = RunObserver(emitter=InspectingEmitter()).start_run(
        _plan(tmp_path, budget, freshness)
    )
    assert len(inspected) == 1
    assert persisted.run_metadata.status == "running"
    assert persisted.lineage_record is None


@pytest.mark.parametrize("status", ["succeeded", "failed"])
@pytest.mark.parametrize(
    ("budget", "outcome"), [(None, "undeclared"), (6, "within"), (5, "within"), (4, "exceeded")]
)
def test_terminal_persistence_evaluates_once_before_writing_and_emitting(
    tmp_path, monkeypatch, caplog, status, budget, outcome
):
    plan = _plan(tmp_path, budget)
    calls, emitted = [], []
    evaluate = OperationsConfig.evaluate_duration

    def count_evaluation(self, duration_seconds):
        calls.append(duration_seconds)
        return evaluate(self, duration_seconds)

    monkeypatch.setattr(OperationsConfig, "evaluate_duration", count_evaluation)
    expected = {
        "duration_budget_seconds": budget,
        "duration_budget_outcome": outcome,
        "freshness_max_age_seconds": 86400,
    }

    class InspectingStore:
        def write(self, plan, record):
            assert calls == [5.0]
            assert {key: record.to_dict()[key] for key in expected} == expected
            return RunMetadataStore().write(plan, record)

    class InspectingEmitter:
        def inspect_and_raise(self, plan, persisted):
            assert calls == [5.0]
            for path in (persisted.run_metadata_path, persisted.lineage_path):
                payload = json.loads(path.read_bytes())
                assert payload["status"] == status
                assert {key: payload[key] for key in expected} == expected
            record = RunRecord.from_run(
                persisted.run_metadata, persisted.lineage_record, emitted_at=FINISHED_AT
            )
            event = build_openlineage_run_event(
                persisted.run_metadata,
                OpenLineageDatasetContext("janus", "/warehouse"),
                lineage_record=persisted.lineage_record,
                run_record=record,
            )
            facet = event["run"]["facets"]["janusRun"]
            assert {key: facet[key] for key in expected} == expected
            schema = PROJECT_ROOT / "docs/schemas/openlineage/JanusRunFacet.json"
            Draft202012Validator(
                json.loads(schema.read_bytes()), format_checker=FormatChecker()
            ).validate(facet)
            assert calls == [5.0], "projection and OpenLineage must lift the recorded outcome"
            emitted.append(status)
            raise RuntimeError("injected terminal emitter failure")

        emit_succeeded = inspect_and_raise
        emit_failed = inspect_and_raise

    caplog.set_level(logging.WARNING)
    observer = RunObserver(
        run_metadata_store=InspectingStore(),
        emitter=InspectingEmitter(),
        logger=StructuredLogger(logging.getLogger("budget")),
    )
    extraction = ExtractionResult.from_plan(plan, ())
    persisted = (
        observer.record_success(plan, extraction, finished_at=FINISHED_AT)
        if status == "succeeded"
        else observer.record_failure(plan, RuntimeError("run failure"), finished_at=FINISHED_AT)
    )
    assert calls == [5.0]
    assert emitted == [status]
    assert persisted.run_metadata.status == status
    assert persisted.run_metadata.duration_budget_outcome == outcome
    assert persisted.lineage_record.duration_budget_outcome == outcome
    warnings = [
        record for record in caplog.records if record.message == "run_duration_budget_exceeded"
    ]
    assert len(warnings) == (1 if outcome == "exceeded" else 0)
    if warnings:
        assert warnings[0].levelno == logging.WARNING
        assert warnings[0].event_fields == {
            "source_id": plan.source.source_id,
            "run_id": plan.run_context.run_id,
            "duration_seconds": 5.0,
            "duration_budget_seconds": budget,
        }
    assert len([r for r in caplog.records if r.message == "run_event_emission_failed"]) == 1
