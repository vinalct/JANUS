from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext, WriteResult
from janus.models.source_config import IcebergRowsRequestInputsConfig
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, ValidationCheck, ValidationReport
from janus.registry import load_registry
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.runtime.executor import _plan_with_storage_layout_outputs
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class FakeStrategy:
    calls: list[str]
    seen_metadata_output_path: str | None = None

    @property
    def strategy_family(self) -> str:
        return "api"

    def plan(self, source_config, run_context, hook=None):
        raise NotImplementedError

    def extract(self, plan, hook=None, *, spark=None):
        del hook
        del spark
        self.calls.append("extract")
        self.seen_metadata_output_path = plan.metadata_output.path
        return ExtractionResult.from_plan(
            plan,
            artifacts=(
                ExtractedArtifact(
                    path=str(
                        plan.run_context.project_root
                        / "data"
                        / "raw"
                        / "example"
                        / "federal_open_data_example"
                        / "page-0001.json"
                    ),
                    format="json",
                    checksum="abc123",
                ),
            ),
            records_extracted=2,
            metadata={"request_count": "1"},
        )

    def build_normalization_handoff(self, plan, extraction_result, hook=None):
        del plan
        del hook
        self.calls.append("handoff")
        return extraction_result

    def emit_metadata(self, plan, extraction_result, write_results=(), hook=None):
        del plan
        del extraction_result
        del write_results
        del hook
        self.calls.append("metadata")
        return {"emitted": True}


@dataclass(slots=True)
class FakeReader:
    calls: list[str]
    seen_format_name: str | None = None
    seen_schema: Any | None = None
    seen_options: dict[str, str] | None = None
    read_artifact_counts: list[int] = field(default_factory=list)

    def read_extraction_result(
        self,
        spark,
        extraction_result,
        format_name=None,
        schema=None,
        options=None,
    ):
        del spark
        self.read_artifact_counts.append(len(extraction_result.artifacts))
        self.seen_format_name = format_name
        self.seen_schema = schema
        self.seen_options = None if options is None else dict(options)
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
            records_written=2,
            partition_by=("ingestion_date",),
            metadata={"writer": "fake"},
        )


@dataclass(slots=True)
class RecordingWriter:
    calls: list[str]
    modes: list[str]
    bronze_path: Path

    def write(self, dataframe, plan, zone, **kwargs):
        del dataframe
        self.calls.append("write")
        mode = kwargs.get("mode") or plan.source_config.spark.write_mode
        self.modes.append(mode)
        return WriteResult.from_plan(
            plan,
            zone,
            path=str(self.bronze_path),
            format_name=plan.bronze_output.format,
            mode=mode,
            records_written=None,
            partition_by=plan.source_config.spark.partition_by,
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


@dataclass(slots=True)
class FakeObserver:
    calls: list[str]
    metadata_root: Path
    seen_metadata_output_path: str | None = None
    seen_strategy_metadata: Any | None = None

    def start_run(self, plan):
        self.seen_metadata_output_path = plan.metadata_output.path
        self.calls.append("start")
        return SimpleNamespace()

    def record_success(self, plan, extraction_result, write_results=(), **kwargs):
        del plan
        del extraction_result
        del write_results
        self.seen_strategy_metadata = kwargs.get("strategy_metadata")
        self.calls.append("success")
        return SimpleNamespace(
            run_metadata_path=self.metadata_root / "runs" / "run-001.json",
            lineage_path=self.metadata_root / "lineage" / "run-001.json",
            checkpoint_result=SimpleNamespace(
                current_path=self.metadata_root / "checkpoints" / "current.json",
                history_path=self.metadata_root / "checkpoints" / "history" / "run-001.json",
            ),
        )

    def record_failure(self, *args, **kwargs):
        raise AssertionError("record_failure should not be called in the success test")


@dataclass(slots=True)
class FakeFailingObserver(FakeObserver):
    def record_success(self, *args, **kwargs):
        raise AssertionError("record_success should not be called in the failure test")

    def record_failure(self, plan, error, extraction_result=None, write_results=(), **kwargs):
        del plan
        del error
        del extraction_result
        del write_results
        self.seen_strategy_metadata = kwargs.get("strategy_metadata")
        self.calls.append("failure")
        return SimpleNamespace(
            run_metadata_path=self.metadata_root / "runs" / "run-001.json",
            lineage_path=self.metadata_root / "lineage" / "run-001.json",
            checkpoint_result=None,
        )


def test_source_executor_runs_framework_pipeline_and_returns_summary(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"

    stream = StringIO()
    logger = build_structured_logger("janus.tests.executor.progress", stream=stream)
    observer = FakeObserver(calls, metadata_root)
    executor = SourceExecutor(
        logger=logger,
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=observer,
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    executed_run = executor.execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert calls == [
        "start",
        "extract",
        "handoff",
        "read",
        "normalize",
        "write",
        "metadata",
        "validate",
        "success",
    ]
    assert executed_run.is_successful is True
    assert [result.zone for result in executed_run.write_results] == ["raw", "bronze"]
    assert executed_run.write_results[0].path.endswith("page-0001.json")
    assert executed_run.to_summary()["artifact_count"] == 1
    assert executed_run.to_summary()["validation"]["is_successful"] is True
    assert executed_run.to_summary()["metadata_outputs"]["run_metadata_path"].endswith(
        "runs/run-001.json"
    )
    assert observer.seen_metadata_output_path == str(metadata_root)
    assert planned_run.strategy.seen_metadata_output_path == str(metadata_root)

    log_events = [json.loads(line)["event"] for line in stream.getvalue().splitlines()]
    assert "source_execution_started" in log_events
    assert "source_extraction_started" in log_events
    assert "bronze_write_finished" in log_events
    assert "quality_validation_finished" in log_events
    assert "source_execution_succeeded" in log_events


def test_source_executor_passes_spark_read_options_from_source_config(tmp_path):
    calls: list[str] = []
    read_options = {
        "header": "true",
        "sep": ";",
        "encoding": "ISO-8859-1",
    }
    planned_run = _planned_run(tmp_path, calls, read_options=read_options)
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"
    reader = FakeReader(calls)

    executor = SourceExecutor(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    executed_run = executor.execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is True
    assert reader.seen_options == read_options


def test_source_executor_passes_explicit_schema_for_headerless_csv_sources(tmp_path):
    @dataclass(slots=True)
    class CnpjCsvStrategy(FakeStrategy):
        def extract(self, plan, hook=None, *, spark=None):
            del hook
            del spark
            self.calls.append("extract")
            self.seen_metadata_output_path = plan.metadata_output.path
            return ExtractionResult.from_plan(
                plan,
                artifacts=(
                    ExtractedArtifact(
                        path=str(
                            tmp_path
                            / "data"
                            / "raw"
                            / "receita_federal"
                            / "cnpj"
                            / "empresas"
                            / "extracted"
                            / "current"
                            / "Empresas0"
                            / "K3241.K03200Y0.D30513.EMPRECSV"
                        ),
                        format="csv",
                        checksum="abc123",
                    ),
                ),
                records_extracted=1,
            )

    calls: list[str] = []
    source_config = _source_config_with_absolute_schema(
        load_registry(PROJECT_ROOT).get_source(
            "receita_federal__cnpj__empresas_full_refresh",
            include_disabled=True,
        )
    )
    run_context = RunContext.create(
        run_id="run-executor-cnpj-schema-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=CnpjCsvStrategy(calls),
        hook=None,
    )
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "receita_federal" / "cnpj" / "empresas"
    reader = FakeReader(calls)

    executed_run = SourceExecutor(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-executor-cnpj-schema-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(
            calls,
            tmp_path / "warehouse" / "bronze" / "receita_federal" / "cnpj" / "empresas",
        ),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is True
    assert reader.seen_format_name == "csv"
    assert reader.seen_schema is not None
    assert reader.seen_schema.fieldNames() == [
        "cnpj_basico",
        "razao_social",
        "natureza_juridica",
        "qualificacao_do_responsavel",
        "capital_social",
        "porte_empresa",
        "ente_federativo_responsavel",
    ]
    assert reader.seen_options == source_config.spark.read_options


def test_source_executor_reads_using_handoff_format_instead_of_source_config_format(tmp_path):
    @dataclass(slots=True)
    class JsonlStrategy(FakeStrategy):
        def extract(self, plan, hook=None, *, spark=None):
            del hook
            del spark
            self.calls.append("extract")
            self.seen_metadata_output_path = plan.metadata_output.path
            return ExtractionResult.from_plan(
                plan,
                artifacts=(
                    ExtractedArtifact(
                        path=str(
                            tmp_path
                            / "data"
                            / "raw"
                            / "receita_federal"
                            / "cnpj"
                            / "empresas"
                            / "rows.jsonl"
                        ),
                        format="jsonl",
                        checksum="abc123",
                    ),
                ),
                records_extracted=1,
            )

    calls: list[str] = []
    source_config = load_registry(PROJECT_ROOT).get_source(
        "receita_federal__cnpj__empresas_full_refresh",
        include_disabled=True,
    )
    run_context = RunContext.create(
        run_id="run-executor-jsonl-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=JsonlStrategy(calls),
        hook=None,
    )
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "receita_federal" / "cnpj" / "empresas"
    reader = FakeReader(calls)

    executed_run = SourceExecutor(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-executor-jsonl-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(
            calls,
            tmp_path / "warehouse" / "bronze" / "receita_federal" / "cnpj" / "empresas",
        ),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is True
    assert reader.seen_format_name == "jsonl"
    assert reader.seen_schema is None
    assert reader.seen_options is None


def test_source_executor_batches_file_handoff_artifacts_and_appends_after_overwrite(tmp_path):
    @dataclass(slots=True)
    class MultiFileCnpjStrategy(FakeStrategy):
        @property
        def strategy_family(self) -> str:
            return "file"

        def extract(self, plan, hook=None, *, spark=None):
            del hook
            del spark
            self.calls.append("extract")
            raw_root = (
                tmp_path
                / "data"
                / "raw"
                / "receita_federal"
                / "cnpj"
                / "empresas"
                / "extracted"
                / "current"
            )
            return ExtractionResult.from_plan(
                plan,
                artifacts=(
                    ExtractedArtifact(
                        path=str(raw_root / "Empresas0" / "K3241.K03200Y0.D30513.EMPRECSV"),
                        format="csv",
                        checksum="abc123",
                    ),
                    ExtractedArtifact(
                        path=str(raw_root / "Empresas1" / "K3241.K03200Y1.D30513.EMPRECSV"),
                        format="csv",
                        checksum="def456",
                    ),
                ),
                records_extracted=2,
            )

    calls: list[str] = []
    source_config = _source_config_with_absolute_schema(
        load_registry(PROJECT_ROOT).get_source(
            "receita_federal__cnpj__empresas_full_refresh",
            include_disabled=True,
        )
    )
    run_context = RunContext.create(
        run_id="run-executor-cnpj-batches-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=MultiFileCnpjStrategy(calls),
        hook=None,
    )
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "receita_federal" / "cnpj" / "empresas"
    reader = FakeReader(calls)
    writer = RecordingWriter(
        calls,
        modes=[],
        bronze_path=tmp_path / "warehouse" / "bronze" / "receita_federal" / "cnpj" / "empresas",
    )

    executed_run = SourceExecutor(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-executor-cnpj-batches-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: writer,
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is True
    assert reader.read_artifact_counts == [2]
    assert writer.modes == ["overwrite"]
    assert [write.mode for write in executed_run.write_results if write.zone == "bronze"] == [
        "overwrite",
    ]


def test_source_executor_records_failed_status_when_quality_validation_fails(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.failed("data", "required_fields", "missing fields")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"

    executor = SourceExecutor(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=FakeFailingObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    executed_run = executor.execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is False
    assert executed_run.failure_reason == "Quality validation failed: data.required_fields"
    assert executed_run.error_type == "RuntimeError"
    assert calls[-1] == "failure"
    observer = executor.observer
    assert isinstance(observer, FakeFailingObserver)
    assert observer.seen_strategy_metadata == {"emitted": True}


def test_source_executor_forwards_strategy_metadata_to_observer_on_success(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"
    observer = FakeObserver(calls, metadata_root)

    executor = SourceExecutor(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=observer,
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    executed_run = executor.execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is True
    assert observer.seen_strategy_metadata == {"emitted": True}


def test_source_executor_persists_strategy_metadata_with_real_observer(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"

    executor = SourceExecutor(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=RunObserver(),
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    executed_run = executor.execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is True
    run_metadata = json.loads(executed_run.run_metadata_path.read_text(encoding="utf-8"))
    lineage = json.loads(executed_run.lineage_path.read_text(encoding="utf-8"))
    assert run_metadata["metadata"]["strategy.emitted"] == "true"
    assert lineage["metadata"]["strategy.emitted"] == "true"


def test_source_executor_persists_empty_strategy_metadata_on_early_exception(tmp_path):
    @dataclass(slots=True)
    class ExplodingStrategy(FakeStrategy):
        def extract(self, plan, hook=None, *, spark=None):
            del plan
            del hook
            del spark
            self.calls.append("extract")
            raise RuntimeError("extract boom")

    calls: list[str] = []
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=ExplodingStrategy(calls),
        hook=None,
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"
    observer = FakeFailingObserver(calls, metadata_root)

    executor = SourceExecutor(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            ValidationReport.from_plan(planned_run.plan, []),
            metadata_root / "validations" / "run-001.json",
        ),
        observer=observer,
        writer_factory=lambda storage_layout: FakeWriter(calls, bronze_path),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    executed_run = executor.execute(
        planned_run,
        SparkSessionProvider.wrapping(object()),
        environment_config={},
    )

    assert executed_run.is_successful is False
    assert executed_run.failure_reason == "extract boom"
    assert executed_run.error_type == "RuntimeError"
    assert observer.seen_strategy_metadata == {}


def test_plan_with_storage_layout_outputs_preserves_bronze_namespace_and_table(tmp_path):
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            bronze=replace(
                source_config.outputs.bronze,
                namespace="curated",
                table_name="named_bronze_table",
            ),
        ),
    )
    run_context = RunContext.create(
        run_id="run-namespace-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 15, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)

    resolved_plan = _plan_with_storage_layout_outputs(plan, _storage_layout(tmp_path))

    assert resolved_plan.bronze_output.path == str(
        tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"
    )
    assert resolved_plan.bronze_output.namespace == "curated"
    assert resolved_plan.bronze_output.table_name == "named_bronze_table"


def _planned_run(
    tmp_path: Path,
    calls: list[str],
    *,
    read_options: dict[str, str] | None = None,
) -> PlannedRun:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    if read_options is not None:
        source_config = replace(
            source_config,
            spark=replace(source_config.spark, read_options=read_options),
        )
    run_context = RunContext.create(
        run_id="run-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    return PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=FakeStrategy(calls),
        hook=None,
    )


@dataclass(slots=True)
class SpySparkSessionProvider:
    """Records when the executor asks for and releases a session.

    Mirrors the real provider's semantics — lazy build, same session on repeat
    ``get()``, idempotent ``stop()`` — so call counts mean what they mean in
    production: ``start_count`` is how many sessions the run actually paid for.
    """

    calls: list[str]
    session: Any | None = None
    get_calls: int = 0
    start_count: int = 0
    stop_calls: int = 0

    def get(self) -> Any:
        self.get_calls += 1
        if self.session is None:
            self.session = object()
            self.start_count += 1
            self.calls.append("spark_start")
        return self.session

    def stop(self) -> None:
        if self.session is None:
            return
        self.session = None
        self.stop_calls += 1
        self.calls.append("spark_stop")


def _boundary_executor(tmp_path: Path, calls: list[str], validation_report, **overrides):
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    bronze_path = tmp_path / "data" / "bronze" / "example" / "federal_open_data_example"
    defaults: dict[str, Any] = {
        "reader": FakeReader(calls),
        "normalizer": FakeNormalizer(calls),
        "quality_gate": FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        "observer": FakeObserver(calls, metadata_root),
        "writer_factory": lambda storage_layout: FakeWriter(calls, bronze_path),
        "storage_layout_resolver": lambda plan, config: _storage_layout(tmp_path),
    }
    defaults.update(overrides)
    return SourceExecutor(**defaults)


def _passing_report(plan) -> ValidationReport:
    return ValidationReport.from_plan(
        plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )


def test_executor_acquires_no_session_until_extraction_has_finished(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    provider = SpySparkSessionProvider(calls)
    executor = _boundary_executor(tmp_path, calls, _passing_report(planned_run.plan))

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is True
    # Exactly one session, started only after extraction is complete.
    assert provider.start_count == 1
    assert calls.index("extract") < calls.index("spark_start")
    assert calls.index("spark_start") < calls.index("read")
    assert provider.stop_calls == 1


def test_executor_keeps_the_session_alive_through_quality_validation(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    provider = SpySparkSessionProvider(calls)
    executor = _boundary_executor(tmp_path, calls, _passing_report(planned_run.plan))

    executor.execute(planned_run, provider, environment_config={})

    # The normalized DataFrame belongs to the session, so validation precedes the stop —
    # and the observer runs session-free after it.
    assert calls.index("validate") < calls.index("spark_stop")
    assert calls.index("spark_stop") < calls.index("success")


def test_executor_never_starts_a_session_for_an_empty_handoff(tmp_path):
    @dataclass(slots=True)
    class EmptyHandoffStrategy(FakeStrategy):
        def extract(self, plan, hook=None, *, spark=None):
            del hook
            del spark
            self.calls.append("extract")
            return ExtractionResult.from_plan(plan, artifacts=(), records_extracted=0)

    calls: list[str] = []
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id="run-empty-handoff-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=EmptyHandoffStrategy(calls),
        hook=None,
    )
    provider = SpySparkSessionProvider(calls)
    stream = StringIO()
    executor = _boundary_executor(
        tmp_path,
        calls,
        _passing_report(planned_run.plan),
        logger=build_structured_logger("janus.tests.executor.empty_handoff", stream=stream),
    )

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is True
    assert provider.start_count == 0
    assert executed_run.write_results == ()
    events = [json.loads(line)["event"] for line in stream.getvalue().splitlines() if line.strip()]
    assert "spark_session_skipped" in events


def test_executor_stops_the_session_when_materialization_raises(tmp_path):
    @dataclass(slots=True)
    class ExplodingReader(FakeReader):
        def read_extraction_result(self, *args, **kwargs):
            del args
            del kwargs
            self.calls.append("read")
            raise RuntimeError("materialize boom")

    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    provider = SpySparkSessionProvider(calls)
    executor = _boundary_executor(
        tmp_path,
        calls,
        _passing_report(planned_run.plan),
        reader=ExplodingReader(calls),
        observer=FakeFailingObserver(calls, metadata_root),
    )

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is False
    assert executed_run.failure_reason == "materialize boom"
    assert provider.stop_calls == 1
    # The release happens before the failure is recorded, and the outcome still surfaces.
    assert calls.index("spark_stop") < calls.index("failure")


def test_executor_hands_the_provider_to_extract_without_starting_a_session(tmp_path):
    @dataclass(slots=True)
    class ProviderCapturingStrategy(FakeStrategy):
        seen_spark: Any = None

        def extract(self, plan, hook=None, *, spark=None):
            # Explicit base call: zero-arg super() is unavailable in a slots dataclass.
            self.seen_spark = spark
            return FakeStrategy.extract(self, plan, hook, spark=spark)

    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    strategy = ProviderCapturingStrategy(calls)
    source_config = replace(
        planned_run.plan.source_config,
        access=replace(
            planned_run.plan.source_config.access,
            request_inputs=IcebergRowsRequestInputsConfig(
                type="iceberg_rows",
                namespace="bronze",
                table_name="empresas",
                columns={"cnpj": "cnpj_basico"},
            ),
        ),
    )
    planned_run = replace(
        planned_run,
        plan=replace(planned_run.plan, source_config=source_config),
        strategy=strategy,
    )
    provider = SpySparkSessionProvider(calls)
    executor = _boundary_executor(tmp_path, calls, _passing_report(planned_run.plan))

    executed_run = executor.execute(planned_run, provider, environment_config={})

    assert executed_run.is_successful is True
    assert strategy.seen_spark is provider
    # The only session the executor itself starts is the materialization one.
    assert calls.index("extract") < calls.index("spark_start")
    assert provider.start_count == 1
    assert provider.stop_calls == 1


def _source_config_with_absolute_schema(source_config):
    if source_config.schema.path is None:
        return source_config
    return replace(
        source_config,
        schema=replace(
            source_config.schema,
            path=str(PROJECT_ROOT / source_config.schema.path),
        ),
    )


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
