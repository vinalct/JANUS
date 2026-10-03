"""One terminal run, projected into one flat row."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import janus
import janus.observability.records
from janus.checkpoints import CheckpointWriteResult
from janus.lineage import LineageRecord, RunMetadata, compute_config_version
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext, WriteResult
from janus.observability import RunEvidencePaths, RunRecord, vocabulary
from janus.observability.vocabulary import MAX_FAILURE_REASON_LENGTH, RUN_RECORD_SCHEMA_VERSION
from janus.quality import ValidationCheck, ValidationReport
from janus.quality.malformed_rows import malformed_rows_check
from janus.registry import load_registry
from janus.runtime.contract_preflight import PREFLIGHT_ATTRIBUTE, PREFLIGHT_OUTCOMES
from janus.writers.evolution import PLAN_METADATA_KEY, EvolutionPlan

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "federal_open_data_example"

STARTED_AT = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 4, 12, 0, 5, tzinfo=UTC)
EMITTED_AT = datetime(2026, 7, 4, 12, 0, 6, tzinfo=UTC)

BRONZE_TABLE = "bronze_example.federal_open_data_example"
FORBIDDEN_IMPORT_ROOTS = ("pyspark", "pyiceberg", "pyarrow")


# --------------------------------------------------------------------------------------
# Builders: the real records, built the way the observer builds them
# --------------------------------------------------------------------------------------


def _plan(tmp_path: Path, *, run_id: str, attributes: dict[str, str] | None = None):
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID)
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=STARTED_AT,
        attributes=attributes,
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _extraction_result(
    plan: ExecutionPlan,
    *,
    artifact_count: int = 1,
    records_extracted: int | None = 2,
    checkpoint_value: str | None = "2026-07-02",
) -> ExtractionResult:
    return ExtractionResult.from_plan(
        plan,
        artifacts=tuple(
            ExtractedArtifact(path=f"{plan.raw_output.path}/page-{index:04d}.json", format="json")
            for index in range(1, artifact_count + 1)
        ),
        records_extracted=records_extracted,
        checkpoint_value=checkpoint_value,
    )


def _write_results(
    plan: ExecutionPlan, *, bronze: bool = True, bronze_records: int | None = 2
) -> tuple[WriteResult, ...]:
    raw = WriteResult.from_plan(
        plan,
        "raw",
        path=f"{plan.raw_output.path}/page-0001.json",
        format_name="json",
        mode="append",
        records_written=1,
    )
    if not bronze:
        return (raw,)
    return (
        raw,
        WriteResult.from_plan(
            plan,
            "bronze",
            path=BRONZE_TABLE,
            format_name="iceberg",
            mode="overwrite",
            records_written=bronze_records,
            partition_by=("ingestion_date",),
            metadata={"writer": "spark"},
        ),
    )


def _passing_report(plan: ExecutionPlan) -> ValidationReport:
    return ValidationReport.from_plan(
        plan,
        checks=(
            ValidationCheck.passed("config", "quality_contract", "contract is coherent"),
            ValidationCheck.passed("data", "required_fields", "all present"),
            ValidationCheck.skipped("output", "bronze_key_uniqueness", "no bronze frame"),
        ),
        emitted_at=FINISHED_AT,
    )


def _failing_report(plan: ExecutionPlan) -> ValidationReport:
    return ValidationReport.from_plan(
        plan,
        checks=(
            ValidationCheck.passed("config", "quality_contract", "contract is coherent"),
            ValidationCheck.failed("data", "required_fields", "column 'updated_at' is null"),
            ValidationCheck.failed("output", "unique_fields", "duplicate id 7"),
        ),
        emitted_at=FINISHED_AT,
    )


def _checkpoint_result(
    tmp_path: Path, *, decision: str = "advanced", advanced: bool = True
) -> CheckpointWriteResult:
    history = tmp_path / "metadata" / "checkpoints" / "history" / "run.json"
    return CheckpointWriteResult(
        state=None,
        decision=decision,
        advanced=advanced,
        current_path=tmp_path / "metadata" / "checkpoints" / "current.json",
        history_path=history,
    )


def _evidence(tmp_path: Path, *, validation: bool = True) -> RunEvidencePaths:
    base = tmp_path / "metadata"
    return RunEvidencePaths(
        run_metadata_path=base / "runs" / "run.json",
        lineage_path=base / "lineage" / "run.json",
        validation_report_path=(base / "validations" / "run.json") if validation else None,
    )


def _records(
    plan: ExecutionPlan,
    *,
    status: str,
    extraction_result: ExtractionResult | None,
    write_results: tuple[WriteResult, ...],
    error: Exception | None = None,
) -> tuple[RunMetadata, LineageRecord]:
    """Build the pair exactly as ``RunObserver.record_success``/``record_failure`` does."""
    if status == "succeeded":
        assert extraction_result is not None
        run_metadata = RunMetadata.succeeded(
            plan, extraction_result, write_results, finished_at=FINISHED_AT
        )
    else:
        assert error is not None
        run_metadata = RunMetadata.failed(
            plan, error, extraction_result, write_results, finished_at=FINISHED_AT
        )
    lineage_record = LineageRecord.from_runtime(
        plan,
        status=status,
        extraction_result=extraction_result,
        write_results=write_results,
        failure_reason=run_metadata.failure_reason,
        error_type=run_metadata.error_type,
        emitted_at=FINISHED_AT,
    )
    return run_metadata, lineage_record


def _expected(plan: ExecutionPlan, tmp_path: Path, **overrides: object) -> dict[str, object]:
    """The full column set for the example source, written from literals."""
    base = tmp_path / "metadata"
    expected: dict[str, object] = {
        "run_id": plan.run_context.run_id,
        "source_id": SOURCE_ID,
        "source_name": "Federal Open Data Example Source",
        "environment": "local",
        "strategy_family": "api",
        "strategy_variant": "page_number_api",
        "extraction_mode": "incremental",
        "source_hook": None,
        "pipeline_run_id": None,
        "pipeline_attempt": None,
        "trigger": None,
        "status": "succeeded",
        "started_at": STARTED_AT,
        "ended_at": FINISHED_AT,
        "emitted_at": EMITTED_AT,
        "duration_seconds": 5.0,
        "config_version": compute_config_version(plan.source_config.config_path),
        "schema_version": None,
        "contract_id": None,
        "contract_version": None,
        "source_config_path": str(plan.source_config.config_path),
        "records_extracted": 2,
        "artifact_count": 1,
        "records_written": 2,
        "bronze_table_identifier": BRONZE_TABLE,
        "bronze_write_mode": "overwrite",
        "checkpoint_field": "updated_at",
        "checkpoint_strategy": "max_value",
        "checkpoint_value": "2026-07-02",
        "checkpoint_decision": "advanced",
        "checkpoint_advanced": True,
        "quality_outcome": "passed",
        "quality_checks_passed": 2,
        "quality_checks_failed": 0,
        "quality_checks_skipped": 1,
        "quality_failed_checks": [],
        "failure_reason": None,
        "failure_reason_truncated": None,
        "failure_reason_length": None,
        "error_type": None,
        "run_metadata_path": str(base / "runs" / "run.json"),
        "lineage_path": str(base / "lineage" / "run.json"),
        "checkpoint_history_path": str(base / "checkpoints" / "history" / "run.json"),
        "validation_report_path": str(base / "validations" / "run.json"),
        "record_schema_version": RUN_RECORD_SCHEMA_VERSION,
        "contract_preflight_outcome": None,
        "schema_evolution": None,
        "malformed_rows": None,
    }
    expected.update(overrides)
    return expected


# --------------------------------------------------------------------------------------
# Every terminal shape projects to a valid, fully pinned record
# --------------------------------------------------------------------------------------


def _shape_success(tmp_path: Path):
    plan = _plan(tmp_path, run_id="order15-success")
    extraction_result = _extraction_result(plan)
    write_results = _write_results(plan)
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=extraction_result,
        write_results=write_results,
    )
    record = RunRecord.from_run(
        run_metadata,
        lineage_record,
        emitted_at=EMITTED_AT,
        checkpoint_result=_checkpoint_result(tmp_path),
        validation_report=_passing_report(plan),
        evidence=_evidence(tmp_path),
    )
    return record, _expected(plan, tmp_path)


def _shape_extraction_failure(tmp_path: Path):
    """Failure during extraction: no write results, no report, no checkpoint write."""
    plan = _plan(tmp_path, run_id="order15-extraction-failure")
    run_metadata, lineage_record = _records(
        plan,
        status="failed",
        extraction_result=None,
        write_results=(),
        error=RuntimeError("scripted extraction failure"),
    )
    record = RunRecord.from_run(
        run_metadata,
        lineage_record,
        emitted_at=EMITTED_AT,
        evidence=_evidence(tmp_path, validation=False),
    )
    expected = _expected(
        plan,
        tmp_path,
        status="failed",
        records_extracted=None,
        artifact_count=0,
        records_written=None,
        bronze_table_identifier=None,
        bronze_write_mode=None,
        checkpoint_value=None,
        checkpoint_decision=None,
        checkpoint_advanced=None,
        quality_outcome="not_run",
        quality_checks_passed=None,
        quality_checks_failed=None,
        quality_checks_skipped=None,
        quality_failed_checks=None,
        failure_reason="scripted extraction failure",
        failure_reason_truncated=False,
        failure_reason_length=len("scripted extraction failure"),
        error_type="RuntimeError",
        checkpoint_history_path=None,
        validation_report_path=None,
    )
    return record, expected


def _shape_quality_failure(tmp_path: Path):
    """Failure at the gate: write results *and* a failing report are both present."""
    plan = _plan(tmp_path, run_id="order15-quality-failure")
    extraction_result = _extraction_result(plan)
    write_results = _write_results(plan)
    run_metadata, lineage_record = _records(
        plan,
        status="failed",
        extraction_result=extraction_result,
        write_results=write_results,
        error=RuntimeError("Quality validation failed: data.required_fields"),
    )
    record = RunRecord.from_run(
        run_metadata,
        lineage_record,
        emitted_at=EMITTED_AT,
        validation_report=_failing_report(plan),
        evidence=_evidence(tmp_path),
    )
    expected = _expected(
        plan,
        tmp_path,
        status="failed",
        checkpoint_decision=None,
        checkpoint_advanced=None,
        quality_outcome="failed",
        quality_checks_passed=1,
        quality_checks_failed=2,
        quality_checks_skipped=0,
        quality_failed_checks=["data.required_fields", "output.unique_fields"],
        failure_reason="Quality validation failed: data.required_fields",
        failure_reason_truncated=False,
        failure_reason_length=len("Quality validation failed: data.required_fields"),
        error_type="RuntimeError",
        checkpoint_history_path=None,
    )
    return record, expected


def _shape_empty_handoff(tmp_path: Path):
    """An empty normalization handoff: raw was written, bronze never was."""
    plan = _plan(tmp_path, run_id="order15-empty-handoff")
    extraction_result = _extraction_result(plan)
    write_results = _write_results(plan, bronze=False)
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=extraction_result,
        write_results=write_results,
    )
    record = RunRecord.from_run(
        run_metadata,
        lineage_record,
        emitted_at=EMITTED_AT,
        checkpoint_result=_checkpoint_result(tmp_path),
        validation_report=_passing_report(plan),
        evidence=_evidence(tmp_path),
    )
    expected = _expected(
        plan,
        tmp_path,
        records_written=None,
        bronze_table_identifier=None,
        bronze_write_mode=None,
    )
    return record, expected


def _shape_replay(tmp_path: Path):
    """A replay rehydrates raw: artifacts and bronze rows, but no extraction count."""
    plan = _plan(tmp_path, run_id="order15-replay")
    extraction_result = _extraction_result(plan, records_extracted=None, checkpoint_value=None)
    write_results = _write_results(plan)
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=extraction_result,
        write_results=write_results,
    )
    record = RunRecord.from_run(
        run_metadata,
        lineage_record,
        emitted_at=EMITTED_AT,
        checkpoint_result=_checkpoint_result(tmp_path, decision="skipped", advanced=False),
        validation_report=_passing_report(plan),
        evidence=_evidence(tmp_path),
    )
    expected = _expected(
        plan,
        tmp_path,
        records_extracted=None,
        checkpoint_value=None,
        checkpoint_decision="skipped",
        checkpoint_advanced=False,
    )
    return record, expected


TERMINAL_SHAPES = {
    "success": _shape_success,
    "extraction_failure": _shape_extraction_failure,
    "quality_failure": _shape_quality_failure,
    "empty_handoff": _shape_empty_handoff,
    "replay": _shape_replay,
}


@pytest.mark.parametrize("shape", sorted(TERMINAL_SHAPES))
def test_every_terminal_shape_projects_to_a_pinned_row(shape, tmp_path):
    """The projection's contract is its output, so pin the whole output."""
    record, expected = TERMINAL_SHAPES[shape](tmp_path)

    assert record.to_dict() == expected


def test_every_column_is_present_even_when_it_is_null(tmp_path):
    """A row cannot omit a column: ``NULL`` and "no such column" are different answers."""
    record, _ = _shape_extraction_failure(tmp_path)
    payload = record.to_dict()

    assert None in payload.values()
    assert len(payload) == len(RunRecord.__dataclass_fields__)
    assert set(payload) == set(RunRecord.__dataclass_fields__)


# --------------------------------------------------------------------------------------
# NULL versus zero versus empty list — the three a refactor collapses
# --------------------------------------------------------------------------------------


def test_a_run_that_extracted_nothing_separates_null_from_zero(tmp_path):
    """No reported count is ``NULL``; an empty artifact list is a known ``0``."""
    plan = _plan(tmp_path, run_id="order15-nothing")
    extraction_result = _extraction_result(plan, artifact_count=0, records_extracted=None)
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=extraction_result,
        write_results=_write_results(plan, bronze=True, bronze_records=0),
    )

    record = RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)

    assert record.records_extracted is None
    assert record.artifact_count == 0
    assert record.records_written == 0, "bronze reported zero rows; that is a count, not absence"


def test_bronze_records_written_is_null_when_no_bronze_output_exists(tmp_path):
    """An empty handoff wrote no bronze at all — which is not "wrote zero rows"."""
    record, _ = _shape_empty_handoff(tmp_path)

    assert record.records_written is None
    assert record.bronze_table_identifier is None
    assert record.bronze_write_mode is None


def test_validation_that_did_not_run_separates_null_from_an_empty_list(tmp_path):
    """``not_run`` leaves every detail column ``NULL``; a clean pass reports ``[]`` and zeros."""
    failed_before_validation, _ = _shape_extraction_failure(tmp_path)
    passed, _ = _shape_success(tmp_path)

    assert failed_before_validation.quality_outcome == "not_run"
    assert failed_before_validation.quality_failed_checks is None
    assert failed_before_validation.quality_checks_failed is None

    assert passed.quality_outcome == "passed"
    assert passed.quality_failed_checks == ()
    assert passed.quality_checks_failed == 0
    assert passed.to_dict()["quality_failed_checks"] == []


def test_no_checkpoint_write_is_distinct_from_a_skipped_one(tmp_path):
    """``record_failure`` attempts no write at all; ``skipped`` means one was declined."""
    not_attempted, _ = _shape_quality_failure(tmp_path)
    skipped, _ = _shape_replay(tmp_path)

    assert not_attempted.checkpoint_decision is None
    assert not_attempted.checkpoint_advanced is None
    assert not_attempted.checkpoint_history_path is None

    assert skipped.checkpoint_decision == "skipped"
    assert skipped.checkpoint_advanced is False


# --------------------------------------------------------------------------------------
# Correlation, provenance, and the failure bound
# --------------------------------------------------------------------------------------


def test_batch_correlation_keys_are_lifted_into_their_own_columns(tmp_path):
    """A batch-level question must not require parsing a map column."""
    plan = _plan(
        tmp_path,
        run_id="order15-batch-a1",
        attributes={
            "pipeline_run_id": "order15-pipeline",
            "pipeline_attempt": "2",
            "trigger": "run-all",
        },
    )
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=_extraction_result(plan),
        write_results=_write_results(plan),
    )

    record = RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)

    assert record.pipeline_run_id == "order15-pipeline"
    assert record.pipeline_attempt == 2
    assert record.trigger == "run-all"


def test_a_single_source_run_carries_null_correlation_not_empty_strings(tmp_path):
    """Absence is ``NULL``. An empty string would make every single-source run "a batch"."""
    record, _ = _shape_success(tmp_path)

    assert record.pipeline_run_id is None
    assert record.pipeline_attempt is None
    assert record.trigger is None


def test_a_non_numeric_pipeline_attempt_is_null_rather_than_guessed(tmp_path):
    """Run attributes are a string map; a malformed attempt is not a fabricated number."""
    plan = _plan(
        tmp_path,
        run_id="order15-bad-attempt",
        attributes={"pipeline_run_id": "order15-pipeline", "pipeline_attempt": "second"},
    )
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=_extraction_result(plan),
        write_results=_write_results(plan),
    )

    record = RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)

    assert record.pipeline_run_id == "order15-pipeline"
    assert record.pipeline_attempt is None


def test_config_version_is_carried_byte_for_byte_and_never_recomputed(tmp_path):
    """The lineage record's SHA-256 is the value; this module hashes nothing."""
    plan = _plan(tmp_path, run_id="order15-config-version")
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=_extraction_result(plan),
        write_results=_write_results(plan),
    )

    record = RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)

    assert record.config_version == lineage_record.config_version
    assert record.config_version == compute_config_version(plan.source_config.config_path)


def test_the_projection_module_computes_no_hash_of_its_own():
    """"Never recomputed here" is a property of the module, not only of one call."""
    source = Path(janus.observability.records.__file__).read_text(encoding="utf-8")

    assert "hashlib" not in source
    assert "sha256(" not in source


def test_emitted_at_is_the_value_the_caller_passed(tmp_path):
    """Purity: the partition source comes from the argument, not from a record or a clock."""
    record, _ = _shape_success(tmp_path)

    assert record.emitted_at == EMITTED_AT
    assert record.emitted_at != FINISHED_AT


def test_a_long_failure_reason_is_bounded_and_the_original_length_survives(tmp_path):
    """One pathological error must not bloat every scan of the table."""
    plan = _plan(tmp_path, run_id="order15-long-reason")
    reason = "x" * (MAX_FAILURE_REASON_LENGTH + 517)
    run_metadata, lineage_record = _records(
        plan,
        status="failed",
        extraction_result=None,
        write_results=(),
        error=RuntimeError(reason),
    )

    record = RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)

    assert len(record.failure_reason) == MAX_FAILURE_REASON_LENGTH
    assert record.failure_reason == reason[:MAX_FAILURE_REASON_LENGTH]
    assert record.failure_reason_truncated is True
    assert record.failure_reason_length == MAX_FAILURE_REASON_LENGTH + 517
    assert lineage_record.failure_reason == reason, "the JSON artifact keeps the whole message"


def test_a_short_failure_reason_is_carried_whole_and_says_so(tmp_path):
    record, _ = _shape_extraction_failure(tmp_path)

    assert record.failure_reason == "scripted extraction failure"
    assert record.failure_reason_truncated is False
    assert record.failure_reason_length == len("scripted extraction failure")


def test_the_strategy_metadata_mapping_is_deliberately_not_a_column():
    """Order-05 metadata is per-source and per-hook, so it is unqueryable across sources."""
    columns = set(RunRecord.__dataclass_fields__)

    assert not {column for column in columns if "strategy_metadata" in column}
    assert "lineage_path" in columns, "the row must still point at the JSON that holds it"


# --------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------


def test_projecting_two_different_runs_into_one_row_is_rejected(tmp_path):
    plan_a = _plan(tmp_path, run_id="order15-a")
    plan_b = _plan(tmp_path, run_id="order15-b")
    run_metadata, _ = _records(
        plan_a,
        status="succeeded",
        extraction_result=_extraction_result(plan_a),
        write_results=_write_results(plan_a),
    )
    _, lineage_record = _records(
        plan_b,
        status="succeeded",
        extraction_result=_extraction_result(plan_b),
        write_results=_write_results(plan_b),
    )

    with pytest.raises(ValueError, match="must describe the same run"):
        RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)


def test_a_status_disagreement_between_the_two_records_is_rejected(tmp_path):
    plan = _plan(tmp_path, run_id="order15-status-drift")
    run_metadata, _ = _records(
        plan,
        status="succeeded",
        extraction_result=_extraction_result(plan),
        write_results=_write_results(plan),
    )
    _, lineage_record = _records(
        plan,
        status="failed",
        extraction_result=_extraction_result(plan),
        write_results=_write_results(plan),
        error=RuntimeError("boom"),
    )

    with pytest.raises(ValueError, match="must agree on status"):
        RunRecord.from_run(run_metadata, lineage_record, emitted_at=EMITTED_AT)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "running"}, "status must be one of"),
        ({"quality_outcome": "unknown"}, "quality_outcome must be one of"),
        (
            {"checkpoint_decision": "invented", "checkpoint_advanced": False},
            "checkpoint_decision must be one of",
        ),
        ({"emitted_at": datetime(2026, 7, 4, 12, 0)}, "emitted_at must be timezone-aware"),
        ({"artifact_count": -1}, "artifact_count must not be negative"),
        ({"record_schema_version": 0}, "record_schema_version must be positive"),
        ({"failure_reason": "boom"}, "a failed run must carry a failure_reason"),
        ({"checkpoint_decision": None}, "checkpoint_advanced is present exactly when"),
        ({"quality_outcome": "not_run"}, "quality detail columns are NULL exactly when"),
        (
            {"contract_preflight_outcome": "probably_fine"},
            "contract_preflight_outcome must be one of",
        ),
        ({"schema_evolution": "  "}, "schema_evolution must not be empty"),
        ({"malformed_rows": -1}, "malformed_rows must not be negative"),
        (
            {
                "quality_outcome": "not_run",
                "quality_checks_passed": None,
                "quality_checks_failed": None,
                "quality_checks_skipped": None,
                "quality_failed_checks": None,
                "malformed_rows": 0,
            },
            "malformed_rows is NULL exactly when no malformed_rows check ran",
        ),
    ],
)
def test_the_record_rejects_a_shape_that_cannot_be_read_back(overrides, message, tmp_path):
    """Every invariant here protects a distinction the table would otherwise lose."""
    record, _ = _shape_success(tmp_path)
    fields = {name: getattr(record, name) for name in RunRecord.__dataclass_fields__}
    fields.update(overrides)

    with pytest.raises(ValueError, match=message):
        RunRecord(**fields)


def test_a_failed_quality_outcome_must_name_the_checks_that_failed(tmp_path):
    record, _ = _shape_quality_failure(tmp_path)
    fields = {name: getattr(record, name) for name in RunRecord.__dataclass_fields__}
    fields["quality_failed_checks"] = ()

    with pytest.raises(ValueError, match="must name at least one failed check"):
        RunRecord(**fields)


# --------------------------------------------------------------------------------------
# The import direction
# --------------------------------------------------------------------------------------


def test_importing_the_projection_pulls_in_no_engine():
    """It must be exhaustively testable in the fast CI job, where neither engine exists."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(janus.__file__).parents[1]), env.get("PYTHONPATH")) if part
    )
    program = (
        "import sys, janus.observability.records\n"
        f"roots = {FORBIDDEN_IMPORT_ROOTS!r}\n"
        "print(sorted({m for m in sys.modules for r in roots "
        "if m == r or m.startswith(r + '.')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing janus.observability.records loaded {result.stdout.strip()}. The "
        "projection is a pure data transform and must need no engine to be installed."
    )


def test_the_projection_imports_no_runtime_or_transport_module():
    """The arrow runs runtime → observability → lineage.models, and never back."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(janus.__file__).parents[1]), env.get("PYTHONPATH")) if part
    )
    forbidden = (
        "janus.lineage.store",
        "janus.runtime",
        "janus.strategies",
        "janus.planner",
        "janus.orchestration",
    )
    program = (
        "import sys, janus.observability.records\n"
        f"forbidden = {forbidden!r}\n"
        "print(sorted({m for m in sys.modules for f in forbidden "
        "if m == f or m.startswith(f + '.')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing janus.observability.records loaded {result.stdout.strip()}, which "
        "inverts the layering TASK-03 fixed."
    )


# --------------------------------------------------------------------------------------
# what the contract decided, projected into three columns (FR-8, D-11, D-14)
# --------------------------------------------------------------------------------------


def _bronze_with(plan: ExecutionPlan, *evolutions: str) -> tuple[WriteResult, ...]:
    """Raw plus one bronze result per batch, each carrying the writer's evolution render."""
    raw = _write_results(plan, bronze=False)
    return raw + tuple(
        WriteResult.from_plan(
            plan,
            "bronze",
            path=BRONZE_TABLE,
            format_name="iceberg",
            mode="append",
            records_written=1,
            metadata={"schema_evolution": evolution},
        )
        for evolution in evolutions
    )


def _malformed_report(plan: ExecutionPlan, *checks: ValidationCheck) -> ValidationReport:
    return ValidationReport.from_plan(
        plan,
        checks=(ValidationCheck.passed("data", "required_fields", "all present"), *checks),
        emitted_at=FINISHED_AT,
    )


def _record(
    tmp_path: Path,
    *,
    attributes: dict[str, str] | None = None,
    write_results: tuple[WriteResult, ...] | None = None,
    report: ValidationReport | None = None,
) -> RunRecord:
    plan = _plan(tmp_path, run_id="projection", attributes=attributes)
    extraction_result = _extraction_result(plan)
    run_metadata, lineage_record = _records(
        plan,
        status="succeeded",
        extraction_result=extraction_result,
        write_results=write_results if write_results is not None else _write_results(plan),
    )
    return RunRecord.from_run(
        run_metadata,
        lineage_record,
        emitted_at=EMITTED_AT,
        validation_report=report,
        evidence=_evidence(tmp_path),
    )


def test_the_preflight_outcome_is_lifted_from_its_run_attribute(tmp_path):
    lifted = _record(tmp_path, attributes={"contract_preflight_outcome": "will_evolve"})
    absent = _record(tmp_path)

    assert lifted.contract_preflight_outcome == "will_evolve"
    assert absent.contract_preflight_outcome is None


def test_an_unknown_preflight_outcome_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="contract_preflight_outcome"):
        _record(tmp_path, attributes={"contract_preflight_outcome": "probably_fine"})


def test_schema_evolution_is_the_first_change_any_bronze_batch_made(tmp_path):
    plan = _plan(tmp_path, run_id="projection")

    changed = _record(tmp_path, write_results=_bronze_with(plan, "none", "added:note"))
    unchanged = _record(tmp_path, write_results=_bronze_with(plan, "none", "none"))
    no_bronze = _record(tmp_path, write_results=_write_results(plan, bronze=False))
    pre_order = _record(tmp_path)

    assert changed.schema_evolution == "added:note"
    assert unchanged.schema_evolution == "none"
    assert no_bronze.schema_evolution is None
    assert pre_order.schema_evolution is None


def test_malformed_rows_is_null_until_counted_and_the_sum_once_counted(tmp_path):
    plan = _plan(tmp_path, run_id="projection")

    def malformed(outcome: str, count: str | None) -> ValidationCheck:
        details = {} if count is None else {"count": count}
        factory = getattr(ValidationCheck, outcome)
        return factory("data", "malformed_rows", f"{outcome} malformed rows", details=details)

    counted = _record(tmp_path, report=_malformed_report(plan, malformed("failed", "2")))
    summed = _record(
        tmp_path,
        report=_malformed_report(plan, malformed("passed", "2"), malformed("passed", "3")),
    )
    clean = _record(tmp_path, report=_malformed_report(plan, malformed("passed", "0")))
    parquet = _record(tmp_path, report=_malformed_report(plan, malformed("skipped", None)))
    absent = _record(tmp_path, report=_malformed_report(plan))
    not_validated = _record(tmp_path)

    assert counted.malformed_rows == 2
    assert summed.malformed_rows == 5
    assert clean.malformed_rows == 0, "zero means counted and clean, not absent"
    assert parquet.malformed_rows is None
    assert absent.malformed_rows is None
    assert not_validated.malformed_rows is None


def test_the_row_always_carries_the_three_columns(tmp_path):
    payload = _record(tmp_path).to_dict()

    assert {"contract_preflight_outcome", "schema_evolution", "malformed_rows"} <= set(payload)
    assert payload["record_schema_version"] == 3


def test_the_duplicated_vocabulary_cannot_drift_from_its_owners():
    """Observability may not import runtime or writers, so the strings are spelled twice
    and held together here — the sanctioned single-definition-with-drift-test shape."""
    assert vocabulary.PREFLIGHT_OUTCOMES == PREFLIGHT_OUTCOMES
    assert vocabulary.PREFLIGHT_ATTRIBUTE_NAME == PREFLIGHT_ATTRIBUTE
    assert vocabulary.SCHEMA_EVOLUTION_METADATA_KEY == PLAN_METADATA_KEY


def test_the_unchanged_table_is_spelled_the_way_the_evolution_plan_renders_it():
    noop = EvolutionPlan(
        outcome="noop",
        add_columns=(),
        promote_columns=(),
        refusals=(),
        reason="contract and table agree",
        recorded_major=1,
        contract_major=1,
    )

    assert noop.render() == vocabulary.SCHEMA_EVOLUTION_NONE


@pytest.mark.parametrize(
    ("enforcement", "count"),
    [("strict", 0), ("strict", 3), ("lenient", 4)],
    ids=["strict-clean", "strict-refused", "lenient-warning"],
)
def test_the_projection_sums_the_check_the_quality_layer_renders(tmp_path, enforcement, count):
    """The check is found by the phase, name and detail key ``quality.malformed_rows`` emits."""
    plan = _plan(tmp_path, run_id="projection")
    rendered = malformed_rows_check(count, enforcement=enforcement, threshold=0)

    assert (rendered.phase, rendered.name) == (
        vocabulary.MALFORMED_ROWS_CHECK_PHASE,
        vocabulary.MALFORMED_ROWS_CHECK_NAME,
    )
    assert vocabulary.MALFORMED_ROWS_COUNT_DETAIL in rendered.details_as_dict()
    assert _record(tmp_path, report=_malformed_report(plan, rendered)).malformed_rows == count


def test_a_parquet_handoff_skip_rendered_by_the_quality_layer_projects_to_null(tmp_path):
    plan = _plan(tmp_path, run_id="projection")
    skipped = malformed_rows_check(None, enforcement="strict", threshold=0)

    assert skipped.outcome == "skipped"
    assert _record(tmp_path, report=_malformed_report(plan, skipped)).malformed_rows is None


# --------------------------------------------------------------------------------------


def test_an_operator_reset_moves_the_next_comparison_but_never_becomes_its_decision(tmp_path):
    """`reset` is in the decision vocabulary so the history entry can say what happened.

    The projected decision is the one the run's own `save` made through the observer: a
    reset between two runs changes what the next run compares against, never what it reports.
    """
    from janus.checkpoints import CheckpointStore
    from janus.lineage import RunObserver

    seeded = _plan(tmp_path, run_id="before-reset")
    RunObserver().record_success(
        seeded,
        _extraction_result(seeded, checkpoint_value="2026-07-03"),
        _write_results(seeded),
        finished_at=FINISHED_AT,
    )
    reset = CheckpointStore().reset(
        seeded, "2026-07-01", operator="ops-tester", reason="backfill", recorded_at=FINISHED_AT
    )
    plan = _plan(tmp_path, run_id="after-reset")
    persisted = RunObserver().record_success(
        plan, _extraction_result(plan), _write_results(plan), finished_at=FINISHED_AT
    )

    record = RunRecord.from_run(
        persisted.run_metadata,
        persisted.lineage_record,
        emitted_at=EMITTED_AT,
        checkpoint_result=persisted.checkpoint_result,
    )

    assert reset.decision == "reset"
    assert record.checkpoint_decision == "advanced", "2026-07-02 is behind 07-03, ahead of 07-01"
    assert record.checkpoint_history_path == str(persisted.checkpoint_result.history_path)
    assert Path(record.checkpoint_history_path).name == "after-reset.json"
