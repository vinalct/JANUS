"""FR-4 evidence for a mapping defect at the full observer seam."""

from __future__ import annotations

import json
from typing import Any

import janus.observability.openlineage.sink as openlineage_sink
from janus.lineage import RunObserver
from janus.observability import RunEmissionOutcome, build_run_event_emitter
from janus.observability.openlineage import OpenLineageEmissionOutcome
from tests.unit.observability import test_openlineage_transports as transport_fixtures


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.events.append(("info", event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self.events.append(("warning", event, fields))

    @property
    def text(self) -> str:
        return json.dumps(self.events, default=str)


def test_mapping_failure_is_redacted_and_cannot_change_the_run(monkeypatch, tmp_path):
    logger = _Logger()

    def fail_mapping(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("secret mapping detail")

    monkeypatch.setattr(openlineage_sink, "build_openlineage_run_event", fail_mapping)
    emitter = build_run_event_emitter(
        transport_fixtures._config(transport="file"),
        transport_fixtures._paths(tmp_path),
        logger=logger,
        runs_table_sink=transport_fixtures._emitting_sink,
    )
    observer = RunObserver(emitter=emitter)
    plan = transport_fixtures._plan(tmp_path, "task10-mapping-failure")

    persisted = observer.record_success(
        plan,
        transport_fixtures._extraction(plan),
        transport_fixtures._writes(plan),
        finished_at=transport_fixtures.FINISHED_AT,
    )

    assert persisted.run_metadata.status == "succeeded"
    assert persisted.run_metadata_path.is_file()
    assert persisted.lineage_path is not None and persisted.lineage_path.is_file()
    assert emitter.last_result is not None
    assert emitter.last_result.outcome is RunEmissionOutcome.EMITTED
    assert emitter.last_result.openlineage is not None
    assert emitter.last_result.openlineage.outcome is OpenLineageEmissionOutcome.FAILED
    assert emitter.last_result.openlineage.step == "mapping"
    warnings = [fields for level, _event, fields in logger.events if level == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["step"] == "mapping"
    assert warnings[0]["exception_type"] == "RuntimeError"
    assert "secret" not in logger.text
