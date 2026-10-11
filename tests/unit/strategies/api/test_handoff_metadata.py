"""Persist handoff evidence through the real live/replay observers, without Spark I/O."""

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, ValidationReport
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.runtime.materialize import BronzeMaterializer
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.strategies.api import ApiStrategy
from janus.utils.logging import build_structured_logger
from tests.fixtures.handoff.cases import (
    ENVIRONMENT_CONFIG,
    HandoffHook,
    RecordsHook,
    build_raw_zone,
    page_snapshot,
    source_mapping,
)


@pytest.mark.parametrize("reason", ["jsonl", "pages", "none", "hook", "hook_records", "empty"])
@pytest.mark.parametrize("entrypoint", ["live", "replay"])
def test_run_metadata_persists_handoff_evidence(tmp_path, monkeypatch, reason, entrypoint):
    zone = build_raw_zone(
        tmp_path,
        raw_format="text" if reason == "none" else "json",
        spark={"handoff": "pages"} if reason == "pages" else None,
    )
    zone.plan.source_config.config_path.write_text(
        json.dumps(
            source_mapping(
                raw_format=zone.plan.source_config.outputs.raw.format,
                spark={"handoff": "pages"} if reason == "pages" else None,
            )
        ),
        encoding="utf-8",
    )
    extraction = zone.extraction
    if reason == "empty":
        extraction = replace(extraction, artifacts=(), records_extracted=0)
    hook = {"hook": HandoffHook(zone), "hook_records": RecordsHook()}.get(reason)
    strategy = zone.strategy()
    planned = PlannedRun(plan=zone.plan, strategy=strategy, hook=hook)
    before = page_snapshot(zone)
    stream = StringIO()
    logger = build_structured_logger("janus.tests.handoff.metadata", stream=stream)
    provider = SparkSessionProvider(
        {}, {}, logger, session_factory=lambda: SimpleNamespace(stop=lambda: None)
    )
    captured = []
    original_handoff = ApiStrategy.build_normalization_handoff

    def capture_handoff(self, plan, extraction_result, hook=None):
        assert provider.was_started is False, "handoff must remain session-free"
        handoff = original_handoff(self, plan, extraction_result, hook)
        captured.append(handoff)
        assert provider.was_started is False
        return handoff

    monkeypatch.setattr(ApiStrategy, "build_normalization_handoff", capture_handoff)
    # Replace bronze I/O only. The real materialize_handoff owns the session boundary,
    # and the real observer persists both authoritative JSON artifacts.
    monkeypatch.setattr(
        BronzeMaterializer, "materialize", lambda *args, **kwargs: ((), None, None, ())
    )
    quality_gate = Mock()
    quality_gate.validate_and_store.return_value = PersistedValidationReport(
        report=ValidationReport.from_plan(zone.plan, []),
        path=tmp_path / "validation.json",
    )
    options = {
        "logger": logger,
        "quality_gate": quality_gate,
        "storage_layout_resolver": lambda plan, config: zone.layout,
    }
    if entrypoint == "live":
        monkeypatch.setattr(ApiStrategy, "extract", lambda *args, **kwargs: extraction)
        result = SourceExecutor(**options).execute(planned, provider, ENVIRONMENT_CONFIG)
    else:
        if reason == "empty":
            monkeypatch.setattr(
                "janus.scripts.raw_to_bronze._build_extraction_result_from_raw",
                lambda *args, **kwargs: extraction,
            )
        result = RawToBronzeLoader(**options).ingest(
            planned,
            provider,
            ENVIRONMENT_CONFIG,
            bronze_table="bronze.handoff_evidence",
        )

    assert result.is_successful, result.failure_reason
    assert len(captured) == 1
    handoff = captured[0]
    expected = {
        "handoff_compaction": reason,
        "handoff_source_page_count": str(len(result.extraction_result.artifacts)),
        "handoff_artifact_count": str(len(handoff.artifacts)),
        "handoff_bytes": str(sum(Path(a.path).stat().st_size for a in handoff.artifacts)),
    }
    assert {key: handoff.metadata_as_dict()[key] for key in expected} == expected
    for path in (result.run_metadata_path, result.lineage_path):
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert {f"strategy.{key}": value for key, value in expected.items()}.items() <= (
            persisted["metadata"].items()
        )
        assert all(key not in persisted and key not in persisted["metadata"] for key in expected)
        assert persisted["metadata"]["strategy.artifact_count"] == str(
            len(result.extraction_result.artifacts)
        )
    assert page_snapshot(zone) == before
    assert result.extraction_result.records_extracted == handoff.records_extracted
    assert result.extraction_result.checkpoint_value == handoff.checkpoint_value
    if reason == "empty":
        assert handoff.is_empty
        assert provider.was_started is False
        assert provider.session_info is None
        events = [json.loads(line)["event"] for line in stream.getvalue().splitlines()]
        assert "spark_session_skipped" in events
    else:
        assert provider.was_started is True
    print(f"{entrypoint}: strategy.handoff_compaction={reason} metadata={expected}")
