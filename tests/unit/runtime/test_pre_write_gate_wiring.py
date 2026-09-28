"""The pre-write gate is wired where the write is: no bronze call without a passing check."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from janus.lineage import RunObserver
from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    WriteResult,
)
from janus.models.data_contracts import DataContract, load_data_contract
from janus.planner import PlannedRun
from janus.quality import QualityGate, ValidationReportStore
from janus.registry import load_registry
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.runtime.materialize import BronzeMaterializer
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.utils.storage import StorageLayout, bronze_table_identifier

pytestmark = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
SOURCE_ID = "federal_open_data_example"
STARTED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

GOOD_COLUMNS = (("id", "string"), ("label", "string"), ("amount", "long"), ("when", "timestamp"))
MISSING_AMOUNT = (("id", "string"), ("label", "string"), ("when", "timestamp"))


# ── fakes ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ScriptedSchema:
    """What the structural check reads: the ``jsonValue()`` of a Spark ``StructType``."""

    columns: tuple[tuple[str, str], ...]
    calls: list[str]

    def jsonValue(self) -> dict[str, Any]:
        self.calls.append("check")
        return {
            "type": "struct",
            "fields": [
                {"name": name, "type": spark_type, "nullable": True, "metadata": {}}
                for name, spark_type in self.columns
            ],
        }

    def fieldNames(self) -> list[str]:
        return [name for name, _ in self.columns]


@dataclass(slots=True)
class ScriptedReader:
    """Returns one scripted frame per read, in order; the last one repeats."""

    calls: list[str]
    frames: list[tuple[tuple[str, str], ...]]

    def read_extraction_result(
        self, spark, extraction_result, format_name=None, schema=None, options=None
    ):
        del spark, extraction_result, format_name, schema, options
        self.calls.append("read")
        columns = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return SimpleNamespace(
            schema=ScriptedSchema(columns, self.calls),
            columns=[name for name, _ in columns],
        )


@dataclass(slots=True)
class RecordingNormalizer:
    calls: list[str]

    def normalize(self, dataframe, plan):
        del dataframe, plan
        self.calls.append("normalize")


@dataclass(slots=True)
class RecordingWriter:
    """Answers every bronze write with the table identifier the quality gate expects."""

    calls: list[str]
    writes: list[Any] = field(default_factory=list)

    def write(self, dataframe, plan, zone, *, intent=None, count_records=False, **kwargs):
        del dataframe, count_records, kwargs
        self.calls.append("write")
        self.writes.append(intent)
        return WriteResult.from_plan(
            plan,
            zone,
            path=bronze_table_identifier(
                plan.bronze_output.path,
                fallback_name=plan.source.source_id,
                namespace=plan.bronze_output.namespace,
                table_name=plan.bronze_output.table_name,
            ),
            format_name="iceberg",
            mode=intent.reported_mode if intent is not None else "append",
            records_written=1,
            partition_by=intent.partition_columns if intent is not None else (),
        )


@dataclass(slots=True)
class ParquetHandoffStrategy:
    """An api strategy whose handoff is Parquet: typed by construction, never persisted."""

    calls: list[str]
    artifact_count: int = 1
    family: str = "api"

    @property
    def strategy_family(self) -> str:
        return self.family

    def plan(self, source_config, run_context, hook=None):
        raise NotImplementedError

    def extract(self, plan, hook=None, *, spark=None):
        del hook, spark
        self.calls.append("extract")
        raw_root = Path(plan.raw_output.path)
        return ExtractionResult.from_plan(
            plan,
            artifacts=tuple(
                ExtractedArtifact(path=str(raw_root / f"page-{index:04d}.json"), format="json")
                for index in range(1, self.artifact_count + 1)
            ),
            records_extracted=self.artifact_count,
        )

    def build_normalization_handoff(self, plan, extraction_result, hook=None):
        del plan, hook
        self.calls.append("handoff")
        return replace(
            extraction_result,
            artifacts=tuple(
                replace(artifact, format="parquet") for artifact in extraction_result.artifacts
            ),
        )

    def emit_metadata(self, plan, extraction_result, write_results=(), hook=None):
        del plan, extraction_result, write_results, hook
        return {}


@dataclass(slots=True)
class StubSession:
    """The materialization session: the fakes never touch it beyond ``table``."""

    calls: list[str] = field(default_factory=list)

    def table(self, identifier: str) -> Any:
        del identifier
        return object()

    def stop(self) -> None:
        self.calls.append("spark_stop")


def _provider(calls: list[str]) -> SparkSessionProvider:
    """A real provider over a stub session that records when it is released."""
    return SparkSessionProvider({}, {}, session_factory=lambda: StubSession(calls))


# ── builders ─────────────────────────────────────────────────────────────────


def _contract() -> DataContract:
    return load_data_contract(HOSTILE / "base_lenient.yaml")


def _plan(tmp_path: Path) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID)
    run_context = RunContext.create(
        run_id="run-pre-write-gate-001",
        environment="local",
        project_root=tmp_path,
        started_at=STARTED_AT,
    )
    return ExecutionPlan.from_source_config(
        source_config, run_context, data_contract=_contract()
    )


def _planned_run(tmp_path: Path, calls: list[str], **strategy_options: Any) -> PlannedRun:
    return PlannedRun(
        plan=_plan(tmp_path),
        strategy=ParquetHandoffStrategy(calls, **strategy_options),
        hook=None,
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


def _components(calls: list[str], *frames: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    writer = RecordingWriter(calls)
    return {
        "reader": ScriptedReader(calls, list(frames)),
        "normalizer": RecordingNormalizer(calls),
        "writer_factory": lambda storage_layout: writer,
        "writer": writer,
    }


def _executor(tmp_path: Path, components: dict[str, Any]) -> SourceExecutor:
    return SourceExecutor(
        reader=components["reader"],
        normalizer=components["normalizer"],
        quality_gate=QualityGate(ValidationReportStore()),
        observer=RunObserver(),
        writer_factory=components["writer_factory"],
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )


def _loader(tmp_path: Path, components: dict[str, Any]) -> RawToBronzeLoader:
    return RawToBronzeLoader(
        reader=components["reader"],
        normalizer=components["normalizer"],
        quality_gate=QualityGate(ValidationReportStore()),
        observer=RunObserver(),
        writer_factory=components["writer_factory"],
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )


def _read_json(path: Path | None) -> dict[str, Any]:
    assert path is not None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validation_check(payload: dict[str, Any], phase: str, name: str) -> dict[str, Any]:
    return next(
        check
        for check in payload["checks"]
        if (check["phase"], check["name"]) == (phase, name)
    )


def _seed_raw_zone(plan: ExecutionPlan, tmp_path: Path) -> None:
    raw_root = _storage_layout(tmp_path).resolve_output(plan, "raw").resolved_path
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "page-0001.json").write_text('[{"id": "r1"}]\n', encoding="utf-8")


def _assert_contract_check_failure(run: Any, contract: DataContract, calls: list[str]) -> None:
    """The four things a failed structural check leaves behind, whatever the entry point."""
    assert run.status == "failed"
    assert run.error_type == "ContractViolationError"
    assert run.failure_stage == "contract_check"
    assert run.failure_reason.startswith(f"{contract.id} v{contract.version}")
    assert "amount" in run.failure_reason
    assert run.to_summary()["failure_stage"] == "contract_check"

    run_metadata = _read_json(run.run_metadata_path)
    assert run_metadata["status"] == "failed"
    assert run_metadata["failure_stage"] == "contract_check"

    assert run.validation_report is not None
    validation = _read_json(run.validation_report.path)
    schema_check = _validation_check(validation, "data", "schema_expectations")
    assert schema_check["outcome"] == "failed"
    assert "amount" in schema_check["details"]["mismatches"]

    assert [result.zone for result in run.write_results] == ["raw"]
    assert "write" not in calls
    assert "spark_stop" in calls


# ── the materializer ─────────────────────────────────────────────────────────


def test_the_writer_is_never_called_when_the_check_fails(tmp_path):
    from janus.quality.contract_checks import ContractViolationError

    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    plan = planned_run.plan
    components = _components(calls, MISSING_AMOUNT)
    handoff = planned_run.strategy.build_normalization_handoff(
        plan, planned_run.strategy.extract(plan)
    )

    with pytest.raises(ContractViolationError) as raised:
        BronzeMaterializer(
            reader=components["reader"],
            normalizer=components["normalizer"],
            writer_factory=components["writer_factory"],
        ).materialize(planned_run, plan, StubSession(), handoff, _storage_layout(tmp_path), None)

    assert components["writer"].writes == []
    assert [(m.kind, m.column) for m in raised.value.check.mismatches] == [
        ("missing_column", "amount")
    ]


def test_the_check_runs_between_the_read_and_normalization(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    plan = planned_run.plan
    components = _components(calls, GOOD_COLUMNS)
    handoff = planned_run.strategy.build_normalization_handoff(
        plan, planned_run.strategy.extract(plan)
    )

    BronzeMaterializer(
        reader=components["reader"],
        normalizer=components["normalizer"],
        writer_factory=components["writer_factory"],
    ).materialize(planned_run, plan, StubSession(), handoff, _storage_layout(tmp_path), None)

    assert calls.index("read") < calls.index("check") < calls.index("normalize")
    assert calls.index("normalize") < calls.index("write")


def test_a_failing_second_batch_leaves_the_first_written_and_names_itself(tmp_path):
    from janus.quality.contract_checks import ContractViolationError

    calls: list[str] = []
    # Six file artifacts over the five-per-batch limit: batches of 5 and 1.
    planned_run = _planned_run(tmp_path, calls, artifact_count=6, family="file")
    plan = planned_run.plan
    components = _components(calls, GOOD_COLUMNS, MISSING_AMOUNT)
    handoff = planned_run.strategy.build_normalization_handoff(
        plan, planned_run.strategy.extract(plan)
    )

    with pytest.raises(ContractViolationError) as raised:
        BronzeMaterializer(
            reader=components["reader"],
            normalizer=components["normalizer"],
            writer_factory=components["writer_factory"],
        ).materialize(planned_run, plan, StubSession(), handoff, _storage_layout(tmp_path), None)

    assert len(components["writer"].writes) == 1
    assert raised.value.batch_index == 2
    assert "batch 2/2" in str(raised.value)


def test_a_plan_without_a_contract_cannot_reach_the_writer(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    plan = planned_run.plan.with_data_contract(None)
    components = _components(calls, GOOD_COLUMNS)
    handoff = planned_run.strategy.build_normalization_handoff(
        plan, planned_run.strategy.extract(plan)
    )

    with pytest.raises(Exception) as raised:
        BronzeMaterializer(
            reader=components["reader"],
            normalizer=components["normalizer"],
            writer_factory=components["writer_factory"],
        ).materialize(planned_run, plan, StubSession(), handoff, _storage_layout(tmp_path), None)

    assert type(raised.value).__name__ == "MissingContractError"
    assert components["writer"].writes == []


# ── both entry points turn a violation into a failed run ─────────────────────


def test_the_executor_records_a_contract_check_failure_and_commits_nothing(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    components = _components(calls, MISSING_AMOUNT)

    run = _executor(tmp_path, components).execute(
        planned_run, _provider(calls), environment_config={}
    )

    _assert_contract_check_failure(run, _contract(), calls)


def test_the_replay_loader_records_a_contract_check_failure_and_commits_nothing(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    _seed_raw_zone(planned_run.plan, tmp_path)
    components = _components(calls, MISSING_AMOUNT)

    run = _loader(tmp_path, components).ingest(
        planned_run,
        _provider(calls),
        environment_config={},
        bronze_table=bronze_table_identifier(
            planned_run.plan.bronze_output.path,
            fallback_name=planned_run.plan.source.source_id,
            namespace=planned_run.plan.bronze_output.namespace,
            table_name=planned_run.plan.bronze_output.table_name,
        ),
    )

    _assert_contract_check_failure(run, _contract(), calls)


def test_a_clean_run_reports_the_check_and_the_contracts_modes_not_the_retired_flag(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    components = _components(calls, GOOD_COLUMNS)

    run = _executor(tmp_path, components).execute(
        planned_run, _provider(calls), environment_config={}
    )

    assert run.status == "succeeded", run.failure_reason
    assert run.failure_stage is None
    validation = _read_json(run.validation_report.path)
    assert _validation_check(validation, "data", "schema_expectations")["outcome"] == "passed"
    assert validation["metadata"]["compatibility"] == "additive"
    assert validation["metadata"]["enforcement"] == "lenient"
    assert "allow_schema_evolution" not in validation["metadata"]
    assert "failure_stage" not in _read_json(run.run_metadata_path)
