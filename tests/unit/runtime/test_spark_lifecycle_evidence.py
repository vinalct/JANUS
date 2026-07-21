"""Whole-pipeline evidence for the lazy Spark lifecycle."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import janus.runtime.spark_lifecycle as spark_lifecycle
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext, WriteResult
from janus.models.source_config import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    RequestInputsConfig,
)
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, ValidationCheck, ValidationReport
from janus.registry import load_registry
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.runtime.spark_lifecycle import scoped_request_input_session
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NONE_INPUTS = RequestInputsConfig(type="none")
DATE_WINDOW_INPUTS = DateWindowRequestInputsConfig(
    type="date_window",
    start=date(2026, 1, 1),
    end=date(2026, 1, 3),
    step="day",
)
ICEBERG_ROWS_INPUTS = IcebergRowsRequestInputsConfig(
    type="iceberg_rows",
    namespace="bronze",
    table_name="empresas",
    columns={"cnpj": "cnpj_basico"},
)
COMBINED_WITH_ICEBERG_INPUTS = CombinedRequestInputsConfig(
    type="combined",
    inputs=(DATE_WINDOW_INPUTS, ICEBERG_ROWS_INPUTS),
)

# Names that would betray a provider-specific assumption in the lifecycle module.
PROVIDER_SPECIFIC_TOKENS = (
    "yarn",
    "k8s",
    "kubernetes",
    "emr",
    "dataproc",
    "databricks",
    "mesos",
    "local[",
)


@dataclass(slots=True)
class _StubSession:
    """Stands in for a built SparkSession: describable and stoppable, nothing else."""

    master: str
    stop_calls: int = 0

    @property
    def sparkContext(self) -> SimpleNamespace:  # matches the PySpark attribute name
        return SimpleNamespace(appName="janus", master=self.master)

    def stop(self) -> None:
        self.stop_calls += 1


# ── the spy: a session cannot be obtained before the boundary ────────────────


class ArmedSpyProvider(SparkSessionProvider):
    """A real provider that refuses to hand out a session until it is armed."""

    def __init__(self) -> None:
        super().__init__({}, {}, session_factory=lambda: SimpleNamespace())
        self.armed = False
        self.start_count = 0
        self.stop_count = 0
        self._live = False

    def arm(self) -> None:
        self.armed = True

    def get(self) -> Any:
        if not self.armed:
            raise AssertionError("session requested during extraction")
        session = super().get()
        if not self._live:
            self._live = True
            self.start_count += 1
        return session

    def stop(self) -> None:
        super().stop()
        if self._live:
            self._live = False
            self.stop_count += 1


@dataclass(slots=True)
class RequestInputAwareStrategy:
    """Stub strategy that loads request inputs the way the real families do."""

    calls: list[str]
    provider: ArmedSpyProvider
    loaded_inputs: tuple[dict[str, Any], ...] = ()

    @property
    def strategy_family(self) -> str:
        return "api"

    def plan(self, source_config, run_context, hook=None):
        raise NotImplementedError

    def extract(self, plan, hook=None, *, spark=None):
        del hook
        self.calls.append("extract_started")
        request_inputs = plan.source_config.access.request_inputs
        with scoped_request_input_session(spark, request_inputs) as session:
            self.calls.append("request_inputs_loaded")
            # Stand-in for the real loaders: a session-free type never touches `session`.
            self.loaded_inputs = () if session is None else ({"row": "upstream"},)

        self.calls.append("download")
        self.calls.append("extract_finished")
        return ExtractionResult.from_plan(
            plan,
            artifacts=(
                ExtractedArtifact(
                    path=str(Path(plan.raw_output.path) / "page-0001.json"),
                    format="json",
                    checksum="abc123",
                ),
            ),
            records_extracted=2,
        )

    def build_normalization_handoff(self, plan, extraction_result, hook=None):
        del plan
        del hook
        # The materialization boundary: extraction and the raw write are behind us,
        # and the executor is about to ask the provider for a session.
        self.calls.append("handoff")
        self.provider.arm()
        return extraction_result

    def emit_metadata(self, plan, extraction_result, write_results=(), hook=None):
        del plan
        del extraction_result
        del write_results
        del hook
        return {}


# ── minimal downstream stubs ─────────────────────────────────────────────────


@dataclass(slots=True)
class StubReader:
    calls: list[str]

    def read_extraction_result(self, spark, extraction_result, **kwargs):
        del spark
        del extraction_result
        del kwargs
        self.calls.append("read")
        return object()


@dataclass(slots=True)
class StubNormalizer:
    calls: list[str]

    def normalize(self, dataframe, plan):
        del dataframe
        del plan
        self.calls.append("normalize")
        return object()


@dataclass(slots=True)
class StubWriter:
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
            records_written=2,
            partition_by=(),
        )


@dataclass(slots=True)
class StubQualityGate:
    calls: list[str]
    report_path: Path
    report: ValidationReport | None = None

    def validate_and_store(self, plan, **kwargs):
        del kwargs
        self.calls.append("validate")
        report = self.report or ValidationReport.from_plan(
            plan,
            [ValidationCheck.passed("output", "materialized_outputs", "ok")],
        )
        return PersistedValidationReport(report=report, path=self.report_path)


@dataclass(slots=True)
class StubObserver:
    calls: list[str]
    metadata_root: Path
    recorded_failure: Exception | None = None

    def start_run(self, plan):
        del plan
        self.calls.append("observe_start")
        return SimpleNamespace()

    def record_success(self, plan, extraction_result, write_results=(), **kwargs):
        del plan
        del extraction_result
        del write_results
        del kwargs
        self.calls.append("observe_success")
        return self._persisted()

    def record_failure(self, plan, error, extraction_result=None, write_results=(), **kwargs):
        del plan
        del extraction_result
        del write_results
        del kwargs
        self.recorded_failure = error
        self.calls.append("observe_failure")
        return self._persisted()

    def _persisted(self):
        return SimpleNamespace(
            run_metadata_path=self.metadata_root / "runs" / "run-001.json",
            lineage_path=self.metadata_root / "lineage" / "run-001.json",
            checkpoint_result=None,
        )


# ── AC-1: extraction cannot obtain a session ─────────────────────────────────


@pytest.mark.parametrize(
    ("input_label", "request_inputs"),
    [
        ("none", NONE_INPUTS),
        ("date_window", DATE_WINDOW_INPUTS),
    ],
)
def test_extraction_completes_without_being_able_to_obtain_a_session(
    tmp_path,
    input_label,
    request_inputs,
):
    """AC-1: a `none`/`date_window` run reaches materialization session-free.

    The spy would raise on any `get()` before the boundary, so reaching a successful
    materialization is proof that nothing in planning, request-input loading, the
    download, or the raw write asked for compute.
    """

    del input_label
    calls: list[str] = []
    provider = ArmedSpyProvider()
    planned_run = _planned_run(tmp_path, calls, provider, request_inputs)
    executor = _executor(tmp_path, calls)

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is True, executed_run.failure_reason
    # The lookup never reached the provider, so the download held no compute.
    assert planned_run.strategy.loaded_inputs == ()
    # Exactly one session, and it opens only after the handoff is prepared.
    assert provider.start_count == 1
    assert provider.stop_count == 1
    assert calls.index("extract_finished") < calls.index("handoff")
    assert calls.index("handoff") < calls.index("read")


def test_the_spy_actually_trips_when_a_session_is_requested_during_extraction(tmp_path):
    """The trap is live: an `iceberg_rows` lookup before arming fails the run.

    Without this, the two AC-1 assertions above would pass just as happily against a
    spy that could never raise.
    """

    calls: list[str] = []
    provider = ArmedSpyProvider()
    planned_run = _planned_run(tmp_path, calls, provider, ICEBERG_ROWS_INPUTS)
    executor = _executor(tmp_path, calls)

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is False
    assert executed_run.failure_reason == "session requested during extraction"
    assert executed_run.error_type == "AssertionError"
    assert "handoff" not in calls


def test_the_provider_not_a_live_session_is_what_crosses_into_extract(tmp_path):
    """AC-1: the executor hands `extract()` the provider, never a started session."""

    calls: list[str] = []
    provider = ArmedSpyProvider()
    seen: list[Any] = []

    @dataclass(slots=True)
    class CapturingStrategy(RequestInputAwareStrategy):
        def extract(self, plan, hook=None, *, spark=None):
            seen.append(spark)
            return RequestInputAwareStrategy.extract(self, plan, hook, spark=spark)

    planned_run = _planned_run(tmp_path, calls, provider, NONE_INPUTS)
    planned_run = replace(planned_run, strategy=CapturingStrategy(calls, provider))
    executor = _executor(tmp_path, calls)

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is True, executed_run.failure_reason
    assert seen == [provider]


def test_combined_with_iceberg_is_the_only_shape_that_needs_spark_during_extraction():
    """FR-2: `requires_spark` is answered from config alone, before any session exists.

    This is the predicate the whole deferral rests on — a `combined` block that hides
    an `iceberg_rows` sub-input must still declare its need.
    """

    assert NONE_INPUTS.requires_spark is False
    assert DATE_WINDOW_INPUTS.requires_spark is False
    assert ICEBERG_ROWS_INPUTS.requires_spark is True
    assert COMBINED_WITH_ICEBERG_INPUTS.requires_spark is True
    assert (
        CombinedRequestInputsConfig(
            type="combined",
            inputs=(DATE_WINDOW_INPUTS, DATE_WINDOW_INPUTS),
        ).requires_spark
        is False
    )


# ── AC-4: config pass-through and no provider-specific code ──────────────────


@pytest.mark.parametrize(
    "master",
    ["local[*]", "spark://cluster-host:7077"],
)
def test_provider_passes_the_environment_config_through_unmodified(monkeypatch, tmp_path, master):
    """AC-4: whatever master the environment declares reaches the builder untouched.

    The provider adds no session-construction logic, so a cluster master is not a
    different code path — it is the same call with a different string.
    """

    recorded: list[tuple[dict[str, Any], dict[str, Path]]] = []

    def fake_build_spark_session(config, resolved_paths):
        recorded.append((config, resolved_paths))
        return _StubSession(master)

    monkeypatch.setattr(spark_lifecycle, "build_spark_session", fake_build_spark_session)

    config = {
        "name": "cluster",
        "spark": {
            "app_name": "janus",
            "master": master,
            "config": {"spark.sql.shuffle.partitions": "200"},
        },
    }
    resolved_paths = {"bronze_dir": tmp_path / "bronze"}
    provider = SparkSessionProvider(config, resolved_paths)

    provider.get()

    assert len(recorded) == 1
    seen_config, seen_paths = recorded[0]
    assert seen_config == config
    assert seen_paths == resolved_paths


def test_the_master_string_changes_nothing_but_the_master_string(monkeypatch, tmp_path):
    """AC-4: `local[*]` and a cluster master produce structurally identical runs.

    The provider echoes the configured master into its log event; it never parses or
    branches on it. Proven by diffing two full lifecycles that differ only there.
    """

    def lifecycle_events(master: str) -> list[dict[str, Any]]:
        monkeypatch.setattr(
            spark_lifecycle,
            "build_spark_session",
            lambda config, paths: _StubSession(master),
        )
        stream = StringIO()
        provider = SparkSessionProvider(
            {"spark": {"app_name": "janus", "master": master}},
            {"bronze_dir": tmp_path / "bronze"},
            build_structured_logger(f"janus.tests.lifecycle.{master}", stream=stream),
        )
        provider.get()
        provider.stop()
        return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]

    local_events = lifecycle_events("local[*]")
    cluster_events = lifecycle_events("spark://cluster-host:7077")

    assert [event["event"] for event in local_events] == [
        "spark_session_starting",
        "spark_session_started",
        "spark_session_stopped",
    ]
    assert [event["event"] for event in local_events] == [
        event["event"] for event in cluster_events
    ]
    # Identical once the master value itself is normalized away.
    assert _without_master(local_events) == _without_master(cluster_events)


def test_spark_lifecycle_module_names_no_specific_runtime():
    """NFR-2: cheap, permanent enforcement that the module stays provider-agnostic.

    Mirrors the catalog-boundary AST guardrail. Docstrings are excluded on purpose —
    prose may legitimately mention `local[*]` to explain that the module does *not*
    care; the guardrail is about executable code and the literals it carries.
    """

    tree = ast.parse(Path(inspect.getfile(spark_lifecycle)).read_text(encoding="utf-8"))
    violations = [
        f"{token!r} in {text!r}"
        for text in _executable_vocabulary(tree)
        for token in PROVIDER_SPECIFIC_TOKENS
        if token in text.lower()
    ]

    assert not violations, (
        "provider-specific names found in runtime/spark_lifecycle.py — the module must "
        "express 'do not hold compute during I/O' independently of where Spark runs:\n"
        + "\n".join(violations)
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _without_master(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize away the master value and the per-run volatile fields."""

    normalized: list[dict[str, Any]] = []
    for event in events:
        fields = dict(event.get("fields", {}))
        if "master" in fields:
            fields["master"] = "<master>"
        normalized.append(
            {
                "event": event["event"],
                "level": event["level"],
                "fields": fields,
            }
        )
    return normalized


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """Identify every string constant that serves as a docstring, by node identity."""

    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return docstrings


def _executable_vocabulary(tree: ast.Module) -> list[str]:
    """Every identifier and non-docstring string literal in the module."""

    docstrings = _docstring_nodes(tree)
    vocabulary: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                vocabulary.append(node.value)
        elif isinstance(node, ast.Name):
            vocabulary.append(node.id)
        elif isinstance(node, ast.Attribute):
            vocabulary.append(node.attr)
        elif isinstance(node, ast.arg):
            vocabulary.append(node.arg)
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            vocabulary.append(node.name)
    return vocabulary


def _planned_run(
    tmp_path: Path,
    calls: list[str],
    provider: ArmedSpyProvider,
    request_inputs: RequestInputsConfig,
) -> PlannedRun:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        access=replace(source_config.access, request_inputs=request_inputs),
    )
    run_context = RunContext.create(
        run_id="run-lifecycle-evidence-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 6, 12, 0, tzinfo=UTC),
    )
    return PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=RequestInputAwareStrategy(calls, provider),
        hook=None,
    )


def _executor(tmp_path: Path, calls: list[str]) -> SourceExecutor:
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"
    return SourceExecutor(
        reader=StubReader(calls),
        normalizer=StubNormalizer(calls),
        quality_gate=StubQualityGate(calls, metadata_root / "validations" / "run-001.json"),
        observer=StubObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: StubWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "data",
                "raw_dir": "data/raw",
                "bronze_dir": "data/bronze",
                "metadata_dir": "data/metadata",
            }
        },
        tmp_path,
    )
