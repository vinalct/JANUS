"""AC-2: type drift is counted and refused before the write, never absorbed as nulls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")

from janus.models import ExtractedArtifact
from janus.models.data_contracts import load_data_contract
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.runtime import ExecutedRun
from janus.schema_contracts import spark_schema_from_contract
from tests.support.contract_enforcement import (
    EnforcementCase,
    bronze_rows,
    execute_case_with_pages,
    iceberg_session_factory,
    inspection_session,
    plan_case,
    read_json,
    snapshot_count,
    table_exists,
    table_schema,
    validation_checks,
    write_case_project,
    write_parquet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
MALFORMED = PROJECT_ROOT / "tests" / "fixtures" / "malformed"
INEP_CONTRACT = PROJECT_ROOT / "conf" / "contracts" / "educacao" / "censo_escolar_microdados.yaml"
CNPJ_MOTIVOS_CONTRACT = (
    PROJECT_ROOT / "conf" / "contracts" / "receita_federal" / "cnpj_motivos.yaml"
)
INEP_SOURCE_ID = "inep_censo_escolar_microdados"
CNPJ_MOTIVOS_SOURCE_ID = "receita_federal__cnpj__motivos_full_refresh"
CORRUPT = "_janus_corrupt_record"
STARTED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
SAMPLE_CHARACTER_LIMIT = 500
_CREATED_TABLES: set[str] = set()

DRIFT_PAGE = json.loads((MALFORMED / "page_with_drift.json").read_text(encoding="utf-8"))
CLEAN_PAGE = [
    {"id": "c1", "label": "one", "amount": 1, "when": "2026-09-01T10:00:00Z"},
    {"id": "c2", "label": "two", "amount": 2, "when": "2026-09-02T10:00:00Z"},
]


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    root = tmp_path_factory.mktemp("janus-malformed-rows")
    session_factory = iceberg_session_factory(root / "warehouse", "janus-malformed-rows")
    yield session_factory
    with inspection_session(session_factory) as spark:
        for table in sorted(_CREATED_TABLES):
            if table_exists(spark, table):
                assert CORRUPT not in [name for name, _ in table_schema(spark, table)]


# ── helpers ──────────────────────────────────────────────────────────────────


def _run(
    case: EnforcementCase,
    project: Path,
    factory: Any,
    pages: list[Any],
    *,
    run_id: str,
    **options: Any,
) -> ExecutedRun:
    _CREATED_TABLES.add(case.bronze_table)
    write_case_project(project, case)
    planned = plan_case(project, case, run_id=run_id, started_at=STARTED_AT)
    return execute_case_with_pages(planned, pages, factory, **options)


def _malformed_check(run: ExecutedRun) -> Any:
    return validation_checks(run)["data.malformed_rows"]


def _samples(check: Any) -> list[str]:
    samples = json.loads(check.details_as_dict()["samples"])
    assert all(isinstance(sample, str) for sample in samples)
    assert all(len(sample) <= SAMPLE_CHARACTER_LIMIT for sample in samples)
    return samples


def _table_state(factory: Any, table: str) -> tuple[int | None, list[str]]:
    with inspection_session(factory) as spark:
        if not table_exists(spark, table):
            return None, []
        return snapshot_count(spark, table), [name for name, _ in table_schema(spark, table)]


def _assert_refused_as_malformed(run: ExecutedRun, *, count: int) -> list[str]:
    assert run.status == "failed"
    assert run.error_type == "MalformedRowsError"
    assert run.failure_stage == "malformed_rows"
    assert read_json(run.run_metadata_path)["failure_stage"] == "malformed_rows"
    check = _malformed_check(run)
    assert check.outcome == "failed"
    assert check.details_as_dict()["count"] == str(count)
    return _samples(check)


def _read_options(source_id: str) -> tuple[tuple[str, str], ...]:
    """The checked-in source's own read options: the reader sees what production sends."""
    source = load_registry(PROJECT_ROOT).get_source(source_id, include_disabled=True)
    return tuple(source.spark.read_options.items())


def _csv_case(source_id: str, contract: Path, registry_source_id: str) -> EnforcementCase:
    return EnforcementCase.from_contract_file(
        source_id,
        contract,
        enforcement="strict",
        input_format="csv",
        read_options=_read_options(registry_source_id),
    )


# ── JSON ─────────────────────────────────────────────────────────────────────


def test_a_strict_json_page_with_drift_fails_before_the_commit(factory, tmp_path):
    case = EnforcementCase.from_contract_file("mr_json_page_strict", HOSTILE / "base.yaml")
    first = _run(case, tmp_path, factory, [CLEAN_PAGE], run_id="mr-json-page-1")
    assert first.status == "succeeded", first.failure_reason
    before, columns = _table_state(factory, case.bronze_table)
    assert CORRUPT not in columns

    second = _run(case, tmp_path, factory, [DRIFT_PAGE], run_id="mr-json-page-2")

    samples = _assert_refused_as_malformed(second, count=5)
    assert 1 <= len(samples) <= 5
    assert all('"abc"' in sample for sample in samples)
    after, columns = _table_state(factory, case.bronze_table)
    assert after == before
    assert CORRUPT not in columns


def test_a_strict_jsonl_handoff_counts_each_drifted_record(factory, tmp_path):
    case = EnforcementCase.from_contract_file(
        "mr_jsonl_strict", HOSTILE / "base.yaml", input_format="jsonl", raw_format="jsonl"
    )

    run = _run(case, tmp_path, factory, [DRIFT_PAGE], run_id="mr-jsonl")

    samples = _assert_refused_as_malformed(run, count=2)
    assert len(samples) == 2
    assert any('"abc"' in sample for sample in samples)
    assert any("12.5" in sample for sample in samples)
    assert _table_state(factory, case.bronze_table) == (None, [])


def test_a_lenient_json_page_reproduces_todays_nulls_and_warns(factory, tmp_path):
    case = EnforcementCase.from_contract_file("mr_json_page_lenient", HOSTILE / "base_lenient.yaml")

    run = _run(case, tmp_path, factory, [DRIFT_PAGE], run_id="mr-json-lenient")

    assert run.status == "succeeded", run.failure_reason
    check = _malformed_check(run)
    assert check.outcome == "passed"
    assert check.message.startswith("WARNING:")
    assert check.details_as_dict()["severity"] == "warning"
    assert check.details_as_dict()["count"] == "5"
    with inspection_session(factory) as spark:
        assert bronze_rows(spark, case.bronze_table, ("id", "label", "amount")) == [
            ("r1", "one", 10),
            ("r2", "two", 20),
            ("r3", "three", None),
            ("r4", "four", 40),
            ("r5", "five", None),
        ]
    assert CORRUPT not in _table_state(factory, case.bronze_table)[1]


def test_a_strict_batch_within_max_malformed_rows_is_written(factory, tmp_path):
    case = EnforcementCase.from_contract_file(
        "mr_jsonl_max3", HOSTILE / "base_max3.yaml", input_format="jsonl", raw_format="jsonl"
    )

    run = _run(case, tmp_path, factory, [DRIFT_PAGE], run_id="mr-jsonl-max3")

    assert run.status == "succeeded", run.failure_reason
    check = _malformed_check(run)
    assert check.outcome == "passed"
    assert check.details_as_dict()["count"] == "2"
    assert check.details_as_dict()["threshold"] == "3"
    assert CORRUPT not in _table_state(factory, case.bronze_table)[1]


# ── CSV ──────────────────────────────────────────────────────────────────────


def test_a_strict_inep_csv_counts_the_bad_integer_and_the_extra_field(factory, tmp_path):
    case = _csv_case("mr_csv_inep", INEP_CONTRACT, INEP_SOURCE_ID)
    handoff = ExtractedArtifact(path=str(MALFORMED / "line_with_drift.csv"), format="csv")

    run = _run(case, tmp_path, factory, [[]], run_id="mr-csv-inep", handoff_artifacts=(handoff,))

    samples = _assert_refused_as_malformed(run, count=2)
    assert any(sample.endswith(";abc") for sample in samples)
    assert any(sample.endswith(";EXTRA") for sample in samples)
    assert _table_state(factory, case.bronze_table) == (None, [])


def test_a_strict_headerless_cnpj_csv_counts_its_one_bad_line(factory, tmp_path):
    case = _csv_case("mr_csv_cnpj", CNPJ_MOTIVOS_CONTRACT, CNPJ_MOTIVOS_SOURCE_ID)
    crafted = tmp_path / "handoff" / "motivos.csv"
    crafted.parent.mkdir(parents=True)
    crafted.write_bytes(
        '"00";"SEM MOTIVO"\n"01";"EXTINCAO POR ENCERRAMENTO";"EXTRA"\n"02";"INCORPORAÇÃO"\n'.encode(
            "iso-8859-1"
        )
    )

    run = _run(
        case,
        tmp_path / "project",
        factory,
        [[]],
        run_id="mr-csv-cnpj",
        handoff_artifacts=(ExtractedArtifact(path=str(crafted), format="csv"),),
    )

    samples = _assert_refused_as_malformed(run, count=1)
    assert samples == ['"01";"EXTINCAO POR ENCERRAMENTO";"EXTRA"']


# ── Parquet ──────────────────────────────────────────────────────────────────


def test_a_parquet_handoff_skips_the_malformed_rows_check(factory, tmp_path):
    case = EnforcementCase.from_contract_file("mr_parquet", HOSTILE / "base.yaml")
    with inspection_session(factory) as spark:
        write_parquet(
            spark,
            tmp_path / "handoff" / "typed",
            [("p1", "one", 1, STARTED_AT), ("p2", "two", 2, STARTED_AT)],
            "id string, label string, amount bigint, when timestamp",
        )
    handoff = ExtractedArtifact(path=str(tmp_path / "handoff" / "typed"), format="parquet")

    run = _run(
        case, tmp_path / "project", factory, [[]], run_id="mr-parquet", handoff_artifacts=(handoff,)
    )

    assert run.status == "succeeded", run.failure_reason
    check = _malformed_check(run)
    assert check.outcome == "skipped"
    assert "typed by construction" in check.message
    assert CORRUPT not in _table_state(factory, case.bronze_table)[1]


# ── Spark reader invariants ───────────────────────────────────────────────────


def test_schema_flag_is_required_to_keep_the_corrupt_record(factory):
    contract = load_data_contract(HOSTILE / "base.yaml")
    path = MALFORMED / "page_with_drift.json"
    with inspection_session(factory) as spark:
        frame = SparkDatasetReader().read_paths(
            spark,
            (path,),
            format_name="json",
            schema=spark_schema_from_contract(contract),
            corrupt_record_column=CORRUPT,
        )
        assert CORRUPT not in frame.columns


def test_corrupt_only_filter_requires_a_persisted_frame(factory):
    from pyspark import StorageLevel
    from pyspark.errors import AnalysisException
    from pyspark.sql.functions import col

    contract = load_data_contract(HOSTILE / "base.yaml")
    path = MALFORMED / "page_with_drift.json"
    with inspection_session(factory) as spark:
        frame = SparkDatasetReader().read_paths(
            spark,
            (path,),
            format_name="json",
            schema=spark_schema_from_contract(contract, with_corrupt_record=True),
            corrupt_record_column=CORRUPT,
        )
        with pytest.raises(AnalysisException, match="QUERY_ONLY_CORRUPT_RECORD_COLUMN"):
            frame.where(col(CORRUPT).isNotNull()).count()

        cached = frame.persist(StorageLevel.MEMORY_AND_DISK)
        try:
            assert cached.where(col(CORRUPT).isNotNull()).count() == 5
        finally:
            cached.unpersist()
