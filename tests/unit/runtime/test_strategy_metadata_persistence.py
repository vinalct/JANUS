"""End-to-end persistence guard for strategy metadata in the run lifecycle."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext, WriteResult
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, ValidationCheck, ValidationReport
from janus.registry import load_registry
from janus.runtime import SourceExecutor
from janus.runtime.executor import ExecutedRun
from janus.strategies.base import SourceHook
from janus.utils.logging import StructuredLogger, build_structured_logger
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_LOGGER_SEQUENCE = itertools.count()


# --------------------------------------------------------------------------------------
# Configurable strategy / hook doubles
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class MetadataStrategy:
    """Stub API strategy that emits a caller-supplied metadata map.

    ``emit_metadata`` merges any hook ``metadata_fields`` flat into the strategy map,
    exactly like the real :class:`~janus.strategies.api.core.ApiStrategy`
    (``api/core.py`` — ``metadata.update(hook.metadata_fields(...))``), so an assertion
    on the persisted metadata is a faithful end-to-end check of the hook channel.
    """

    calls: list[str]
    emitted_metadata: Mapping[str, Any] = field(default_factory=dict)
    raise_on_extract: bool = False

    @property
    def strategy_family(self) -> str:
        return "api"

    def plan(self, source_config, run_context, hook=None):  # pragma: no cover - unused
        raise NotImplementedError

    def extract(self, plan, hook=None, *, spark=None):
        del hook
        del spark
        self.calls.append("extract")
        if self.raise_on_extract:
            raise RuntimeError("extract boom")
        return ExtractionResult.from_plan(
            plan,
            artifacts=(
                ExtractedArtifact(
                    path=f"{plan.raw_output.path}/page-0001.json",
                    format="json",
                    checksum="abc123",
                ),
            ),
            records_extracted=100,
            metadata={"http_status": "200"},
        )

    def build_normalization_handoff(self, plan, extraction_result, hook=None):
        del plan
        del hook
        self.calls.append("handoff")
        return extraction_result

    def emit_metadata(self, plan, extraction_result, write_results=(), hook=None):
        self.calls.append("metadata")
        merged: dict[str, Any] = dict(self.emitted_metadata)
        if hook is not None:
            merged.update(hook.metadata_fields(plan, extraction_result, write_results))
        return merged


class ProbeHook(SourceHook):
    """Minimal source-local hook whose ``metadata_fields`` returns a sentinel."""

    def metadata_fields(self, plan, extraction_result, write_results=()):
        del plan
        del extraction_result
        del write_results
        return {"hook_probe": "present"}


# --------------------------------------------------------------------------------------
# Fake Spark-facing collaborators (Spark is unavailable in the dev container)
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FakeReader:
    calls: list[str]

    def read_extraction_result(
        self,
        spark,
        extraction_result,
        format_name=None,
        schema=None,
        options=None,
    ):
        del spark
        del extraction_result
        del format_name
        del schema
        del options
        self.calls.append("read")
        return object()


@dataclass(slots=True)
class FakeNormalizer:
    calls: list[str]

    def normalize(self, dataframe, plan):
        del dataframe
        del plan
        self.calls.append("normalize")
        return object()


@dataclass(slots=True)
class FakeWriter:
    calls: list[str]
    bronze_path: Path

    def write(self, dataframe, plan, zone, **kwargs):
        del dataframe
        del kwargs
        self.calls.append("write")
        return WriteResult.from_plan(
            plan,
            zone,
            path=str(self.bronze_path),
            format_name="parquet",
            mode="append",
            records_written=100,
            partition_by=("ingestion_date",),
            metadata={"writer": "fake"},
        )


@dataclass(slots=True)
class FakeQualityGate:
    calls: list[str]
    report: ValidationReport
    path: Path

    def validate_and_store(self, plan, **kwargs):
        del plan
        del kwargs
        self.calls.append("validate")
        return PersistedValidationReport(report=self.report, path=self.path)


# --------------------------------------------------------------------------------------
# success persistence (run-metadata AND lineage)
# --------------------------------------------------------------------------------------


def test_success_persists_namespaced_strategy_metadata_to_both_zones(tmp_path):
    executed = _run(
        tmp_path,
        emitted_metadata={"artifact_count": 12, "owner": "  data-eng  "},
        quality_passes=True,
    )

    assert executed.is_successful is True

    run_payload = _read(executed.run_metadata_path)
    lineage_payload = _read(executed.lineage_path)

    # Both zones carry the namespaced strategy metadata ...
    for payload in (run_payload, lineage_payload):
        assert payload["metadata"]["strategy.owner"] == "data-eng"
        # ... and the non-string (int) value was coerced to its string form.
        assert payload["metadata"]["strategy.artifact_count"] == "12"


# --------------------------------------------------------------------------------------
# failure persistence (best-effort)
# --------------------------------------------------------------------------------------


def test_quality_failure_persists_strategy_metadata_computed_before_failure(tmp_path):
    executed = _run(
        tmp_path,
        emitted_metadata={"attempts": 3, "owner": "data-eng"},
        quality_passes=False,
    )

    assert executed.is_successful is False
    assert executed.error_type == "RuntimeError"

    run_payload = _read(executed.run_metadata_path)
    lineage_payload = _read(executed.lineage_path)

    # The metadata computed before the quality gate rejected the run is still on disk ...
    expected = {"strategy.attempts": "3", "strategy.owner": "data-eng"}
    assert run_payload["metadata"] == expected
    assert lineage_payload["metadata"] == expected
    # ... on records that faithfully report the failure.
    assert run_payload["status"] == "failed"
    assert lineage_payload["status"] == "failed"
    assert "Quality validation failed" in run_payload["failure_reason"]


def test_early_exception_persists_empty_metadata_and_original_error(tmp_path):
    executed = _run(
        tmp_path,
        emitted_metadata={"never": "reached"},
        quality_passes=True,
        raise_on_extract=True,
    )

    assert executed.is_successful is False
    assert executed.failure_reason == "extract boom"
    assert executed.error_type == "RuntimeError"

    run_payload = _read(executed.run_metadata_path)
    lineage_payload = _read(executed.lineage_path)

    # emit_metadata never ran, so the metadata block is empty — no crash, no masking ...
    assert run_payload["metadata"] == {}
    assert lineage_payload["metadata"] == {}
    # ... and the original error is the recorded failure.
    assert run_payload["status"] == "failed"
    assert run_payload["failure_reason"] == "extract boom"
    assert run_payload["error_type"] == "RuntimeError"
    assert lineage_payload["failure_reason"] == "extract boom"


# --------------------------------------------------------------------------------------
# hook metadata_fields durability
# --------------------------------------------------------------------------------------


def test_hook_metadata_fields_are_persisted_not_just_returned(tmp_path):
    executed = _run(
        tmp_path,
        emitted_metadata={"owner": "data-eng"},
        quality_passes=True,
        hook=ProbeHook(),
    )

    assert executed.is_successful is True

    run_payload = _read(executed.run_metadata_path)
    lineage_payload = _read(executed.lineage_path)

    # The hook value is durably on disk (namespaced), not merely returned — this is the
    # assertion that would have caught the original bug.
    assert run_payload["metadata"]["strategy.hook_probe"] == "present"
    assert lineage_payload["metadata"]["strategy.hook_probe"] == "present"


# --------------------------------------------------------------------------------------
# collision / namespacing guard
# --------------------------------------------------------------------------------------


def test_reserved_field_names_cannot_be_clobbered_by_strategy_metadata(tmp_path):
    executed = _run(
        tmp_path,
        emitted_metadata={"status": "SHOULD_NOT_WIN", "records_extracted": "999999"},
        quality_passes=True,
    )

    assert executed.is_successful is True

    run_payload = _read(executed.run_metadata_path)

    # Reserved top-level fields keep their real runtime values ...
    assert run_payload["status"] == "succeeded"
    assert run_payload["records_extracted"] == 100
    # ... while the shadowing keys survive only under the strategy namespace.
    assert run_payload["metadata"]["strategy.status"] == "SHOULD_NOT_WIN"
    assert run_payload["metadata"]["strategy.records_extracted"] == "999999"


# --------------------------------------------------------------------------------------
# malformed / non-string coercion (drop-with-warning, never fatal)
# --------------------------------------------------------------------------------------


def test_malformed_entries_are_dropped_with_warning_and_do_not_fail_the_run(tmp_path):
    logger, stream = _recording_logger()
    executed = _run(
        tmp_path,
        emitted_metadata={
            "kept": 7,
            "missing": None,
            "blank": "   ",
            "nested": {"x": 1},
            "listed": [1, 2, 3],
        },
        quality_passes=True,
        observer_logger=logger,
    )

    # The malformed payload never aborted the run ...
    assert executed.is_successful is True

    run_payload = _read(executed.run_metadata_path)
    lineage_payload = _read(executed.lineage_path)

    # ... only the coercible scalar survived, the invalid entries are absent ...
    assert run_payload["metadata"] == {"strategy.kept": "7"}
    assert lineage_payload["metadata"] == {"strategy.kept": "7"}

    # ... and each drop was logged as a warning (not raised).
    dropped = {event["fields"]["key"] for event in _warning_events(stream)}
    assert dropped == {"missing", "blank", "nested", "listed"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _run(
    tmp_path: Path,
    *,
    emitted_metadata: Mapping[str, Any],
    quality_passes: bool,
    hook: SourceHook | None = None,
    raise_on_extract: bool = False,
    observer_logger: StructuredLogger | None = None,
) -> ExecutedRun:
    calls: list[str] = []
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-metadata-persistence-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=MetadataStrategy(
            calls,
            emitted_metadata=emitted_metadata,
            raise_on_extract=raise_on_extract,
        ),
        hook=hook,
    )

    check = (
        ValidationCheck.passed("output", "materialized_outputs", "ok")
        if quality_passes
        else ValidationCheck.failed("data", "required_fields", "missing fields")
    )
    validation_report = ValidationReport.from_plan(planned_run.plan, [check])
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"

    executor = SourceExecutor(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-metadata-persistence-001.json",
        ),
        observer=RunObserver(logger=observer_logger),
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    return executor.execute(planned_run, spark=object(), environment_config={})


def _read(path: Path | None) -> dict[str, Any]:
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": str(tmp_path / "data"),
                "raw_dir": str(tmp_path / "data" / "raw"),
                "bronze_dir": str(tmp_path / "data" / "bronze"),
                "metadata_dir": str(tmp_path / "data" / "metadata"),
            }
        },
        tmp_path,
    )


def _recording_logger() -> tuple[StructuredLogger, StringIO]:
    stream = StringIO()
    logger = build_structured_logger(
        f"janus.tests.runtime.metadata_persistence.{next(_LOGGER_SEQUENCE)}",
        stream=stream,
    )
    return logger, stream


def _warning_events(stream: StringIO) -> list[dict]:
    events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return [event for event in events if event["level"] == "WARNING"]
