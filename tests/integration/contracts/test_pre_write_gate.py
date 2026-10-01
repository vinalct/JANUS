"""AC-1 and AC-3: nothing reaches bronze without the structural check; appends resolve by name."""

from __future__ import annotations

import ast
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")

from janus.models import BronzeWriteIntent, ExtractedArtifact
from janus.models.data_contracts import (
    DataContract,
    load_data_contract,
    physical_type_from_spark_json,
)
from janus.readers import SparkDatasetReader
from janus.runtime import ExecutedRun
from janus.schema_contracts import spark_schema_from_contract
from tests.support.contract_enforcement import (
    EnforcementCase,
    bronze_rows,
    contract_property,
    execute_case_with_pages,
    iceberg_session_factory,
    inspection_session,
    plan_case,
    read_json,
    snapshot_count,
    table_exists,
    table_schema,
    write_case_project,
    write_frame,
    write_parquet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
BASELINE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "baseline"
STARTED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
INGESTION_DATE = date(2026, 9, 28)
TIMESTAMP_1 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
TIMESTAMP_2 = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)

ID_AMOUNT = (
    contract_property("id", "string", required=True),
    contract_property("amount", "long"),
)
GOOD_PAGES = [
    [{"id": "g1", "amount": 1}, {"id": "g2", "amount": 2}],
    [{"id": "g3", "amount": 3}],
]
NO_AMOUNT_PAGES = [[{"id": "m1"}, {"id": "m2"}], [{"id": "m3"}]]
REQUIRED_NULL_PAGES = [[{"id": None, "amount": 5}, {"id": "n2", "amount": 6}]]


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    """One warehouse per module; each run and each inspection starts and stops its own session."""
    root = tmp_path_factory.mktemp("janus-pre-write-gate")
    return iceberg_session_factory(root / "warehouse", "janus-pre-write-gate")


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
    write_case_project(project, case)
    planned = plan_case(project, case, run_id=run_id, started_at=STARTED_AT)
    return execute_case_with_pages(planned, pages, factory, **options)


def _snapshots(factory: Any, table: str) -> int | None:
    with inspection_session(factory) as spark:
        return snapshot_count(spark, table) if table_exists(spark, table) else None


def _exists(factory: Any, table: str) -> bool:
    with inspection_session(factory) as spark:
        return table_exists(spark, table)


def _columns(factory: Any, table: str) -> list[str]:
    with inspection_session(factory) as spark:
        return [name for name, _ in table_schema(spark, table)]


def _parquet_handoff(factory: Any, path: Path, rows: list[tuple], ddl: str) -> ExtractedArtifact:
    with inspection_session(factory) as spark:
        write_parquet(spark, path, rows, ddl)
    return ExtractedArtifact(path=str(path), format="parquet")


def _assert_failed_before_commit(run: ExecutedRun, *, failure_stage: str) -> None:
    assert run.status == "failed"
    assert run.failure_stage == failure_stage
    assert read_json(run.run_metadata_path)["failure_stage"] == failure_stage
    assert [result.zone for result in run.write_results if result.zone == "bronze"] == []


# ── AC-1: a structural mismatch fails before the commit ──────────────────────


def test_a_missing_column_fails_before_the_full_refresh_commits(factory, tmp_path):
    case = EnforcementCase.for_source(
        "ac1_missing_column", ID_AMOUNT, raw_format="jsonl", write_mode="overwrite", page_size=2
    )
    first = _run(case, tmp_path, factory, GOOD_PAGES, run_id="ac1-missing-1")
    assert first.status == "succeeded", first.failure_reason
    before = _snapshots(factory, case.bronze_table)

    second = _run(case, tmp_path, factory, NO_AMOUNT_PAGES, run_id="ac1-missing-2")

    _assert_failed_before_commit(second, failure_stage="contract_check")
    assert "amount" in second.failure_reason
    assert _snapshots(factory, case.bronze_table) == before
    assert "amount" in _columns(factory, case.bronze_table)


def test_a_type_mismatch_fails_before_the_commit(factory, tmp_path):
    case = EnforcementCase.for_source("ac1_type_mismatch", ID_AMOUNT, write_mode="overwrite")
    first = _run(case, tmp_path, factory, [[{"id": "t1", "amount": 1}]], run_id="ac1-type-1")
    assert first.status == "succeeded", first.failure_reason
    before = _snapshots(factory, case.bronze_table)
    handoff = _parquet_handoff(
        factory,
        tmp_path / "handoff" / "amount_as_string",
        [("t2", "12")],
        "id string, amount string",
    )

    second = _run(
        case,
        tmp_path,
        factory,
        [[{"id": "t2", "amount": 12}]],
        run_id="ac1-type-2",
        handoff_artifacts=(handoff,),
    )

    _assert_failed_before_commit(second, failure_stage="contract_check")
    assert "amount: type mismatch (contract long, frame string)" in second.failure_reason
    assert _snapshots(factory, case.bronze_table) == before


def test_a_null_in_a_required_column_of_a_strict_source_fails_before_the_append(
    factory, tmp_path
):
    case = EnforcementCase.for_source("ac1_required_null", ID_AMOUNT, enforcement="strict")
    first = _run(case, tmp_path, factory, GOOD_PAGES[:1], run_id="ac1-null-1")
    assert first.status == "succeeded", first.failure_reason
    before = _snapshots(factory, case.bronze_table)

    second = _run(case, tmp_path, factory, REQUIRED_NULL_PAGES, run_id="ac1-null-2")

    _assert_failed_before_commit(second, failure_stage="contract_check")
    assert _snapshots(factory, case.bronze_table) == before
    with inspection_session(factory) as spark:
        assert bronze_rows(spark, case.bronze_table, ("id",)) == [("g1",), ("g2",)]


@pytest.mark.parametrize("kind", ["missing_column", "type_mismatch", "required_null"])
def test_a_failing_first_batch_never_creates_the_table(factory, tmp_path, kind):
    source_id = f"ac1_first_write_{kind}"
    options: dict[str, Any] = {}
    if kind == "missing_column":
        case = EnforcementCase.for_source(
            source_id, ID_AMOUNT, raw_format="jsonl", write_mode="overwrite"
        )
        pages = NO_AMOUNT_PAGES[:1]
    elif kind == "type_mismatch":
        case = EnforcementCase.for_source(source_id, ID_AMOUNT, write_mode="overwrite")
        pages = [[{"id": "t1", "amount": 1}]]
        options["handoff_artifacts"] = (
            _parquet_handoff(
                factory, tmp_path / "handoff", [("t1", "1")], "id string, amount string"
            ),
        )
    else:
        case = EnforcementCase.for_source(source_id, ID_AMOUNT, enforcement="strict")
        pages = REQUIRED_NULL_PAGES

    run = _run(case, tmp_path, factory, pages, run_id=f"ac1-first-{kind}", **options)

    _assert_failed_before_commit(run, failure_stage="contract_check")
    assert _exists(factory, case.bronze_table) is False


# ── AC-3: appends resolve by name ────────────────────────────────────────────

BASE_PAGE_ONE = [
    {"id": "r1", "label": "one", "amount": 10, "when": "2026-09-01T10:00:00Z"},
    {"id": "r2", "label": "two", "amount": 20, "when": "2026-09-02T10:00:00Z"},
]
BASE_PAGE_TWO = [
    {"id": "r3", "label": "three", "amount": 30, "when": "2026-09-03T10:00:00Z"},
    {"id": "r4", "label": "four", "amount": 40, "when": "2026-09-04T10:00:00Z"},
]


def test_a_reordered_contract_appends_in_target_table_order(factory, tmp_path):
    first_case = EnforcementCase.from_contract_file(
        "ac3_reordered", HOSTILE / "base.yaml", enforcement="lenient"
    )
    first = _run(first_case, tmp_path, factory, [BASE_PAGE_ONE], run_id="ac3-reordered-1")
    assert first.status == "succeeded", first.failure_reason
    created_order = _columns(factory, first_case.bronze_table)

    reordered = EnforcementCase.from_contract_file(
        "ac3_reordered", HOSTILE / "base_reordered.yaml", enforcement="lenient"
    )
    second = _run(reordered, tmp_path, factory, [BASE_PAGE_TWO], run_id="ac3-reordered-2")

    assert second.status == "succeeded", second.failure_reason
    assert _columns(factory, first_case.bronze_table) == created_order
    with inspection_session(factory) as spark:
        assert bronze_rows(spark, first_case.bronze_table, ("id", "label", "amount")) == [
            ("r1", "one", 10),
            ("r2", "two", 20),
            ("r3", "three", 30),
            ("r4", "four", 40),
        ]


def test_a_same_arity_swap_of_same_typed_columns_lands_by_name(factory, tmp_path):
    case = EnforcementCase.for_source(
        "ac3_swap", (contract_property("id", "string"), contract_property("label", "string"))
    )
    project = write_case_project(tmp_path, case)
    plan = plan_case(project, case, run_id="ac3-swap", started_at=STARTED_AT).plan
    insert = BronzeWriteIntent(strategy="insert", configured_mode="append")

    with inspection_session(factory) as spark:
        write_frame(
            spark,
            plan,
            [("1", "one", INGESTION_DATE)],
            "id string, label string, ingestion_date date",
        )
        write_frame(
            spark,
            plan,
            [("two", "2", INGESTION_DATE)],
            "label string, id string, ingestion_date date",
            intent=insert,
        )
        rows = bronze_rows(spark, case.bronze_table, ("id", "label"))

    assert rows == [("1", "one"), ("2", "two")]


def test_merge_resolves_a_reordered_frame_by_name(factory, tmp_path):
    """Green on arrival: pins TASK-01 step 10's verdict (module docstring)."""
    case = EnforcementCase.for_source(
        "ac3_merge",
        (
            contract_property("id", "string", required=True, primary_key=True),
            contract_property("label", "string"),
            contract_property("amount", "long"),
        ),
        extraction_mode="incremental",
        checkpoint_field="label",
    )
    project = write_case_project(tmp_path, case)
    plan = plan_case(project, case, run_id="ac3-merge", started_at=STARTED_AT).plan
    merge = BronzeWriteIntent(
        strategy="merge_on_keys", configured_mode="append", merge_keys=("id",)
    )

    with inspection_session(factory) as spark:
        write_frame(
            spark,
            plan,
            [("1", "one", 10, TIMESTAMP_1)],
            "id string, label string, amount bigint, ingestion_timestamp timestamp",
            intent=merge,
        )
        write_frame(
            spark,
            plan,
            [(20, "uno", "1", TIMESTAMP_2), (30, "two", "2", TIMESTAMP_2)],
            "amount bigint, label string, id string, ingestion_timestamp timestamp",
            intent=merge,
        )
        rows = bronze_rows(spark, case.bronze_table, ("id", "label", "amount"))

    assert rows == [("1", "uno", 20), ("2", "two", 30)]


def test_a_first_write_creates_the_table_in_frame_order(factory, tmp_path):
    """Green on arrival: CTAS is unchanged by this order (module docstring)."""
    case = EnforcementCase.for_source(
        "ac3_ctas", (contract_property("label", "string"), contract_property("id", "string"))
    )
    project = write_case_project(tmp_path, case)
    plan = plan_case(project, case, run_id="ac3-ctas", started_at=STARTED_AT).plan

    with inspection_session(factory) as spark:
        write_frame(
            spark,
            plan,
            [("one", "1", INGESTION_DATE)],
            "label string, id string, ingestion_date date",
        )
        created = [name for name, _ in table_schema(spark, case.bronze_table)]

    assert created == ["label", "id", "ingestion_date"]


# ── D-1: the baseline contracts describe what their suites infer today ───────

BASELINE_SUITES = {
    "incremental_upsert_fixture": (
        PROJECT_ROOT
        / "tests/integration/incremental_upsert/test_incremental_upsert_idempotency.py",
        ("RUN_ONE_RECORDS", "RUN_TWO_RECORDS"),
    ),
    "multi_batch_run_keys": (
        PROJECT_ROOT / "tests/integration/incremental_upsert/test_multi_batch_run_keys.py",
        ("BATCH_KEY_ROWS",),
    ),
}


def _suite_rows(module_path: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    """The literal rows a suite declares, read from its source — never re-typed here.

    A constant is either a list of rows or a list of batches of rows; both flatten to rows.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }
    assert set(found) == set(names), f"{module_path.name} no longer declares {names}"
    rows: list[dict[str, Any]] = []
    for name in names:
        for entry in found[name]:
            rows.extend(entry if isinstance(entry, list) else [entry])
    return rows


def _vocabulary_shape(struct_type: Any) -> dict[str, tuple[str, bool]]:
    return {
        field.name: (physical_type_from_spark_json(field.dataType.jsonValue()), field.nullable)
        for field in struct_type.fields
    }


def _assert_is_a_d1_contract(contract: DataContract) -> None:
    assert (contract.version, contract.status) == ("1.0.0", "active")
    assert (contract.janus.compatibility, contract.janus.enforcement) == ("additive", "lenient")


@pytest.mark.parametrize("contract_name", sorted(BASELINE_SUITES))
def test_the_d1_baseline_contracts_reproduce_their_suites_inferred_schema(
    factory, tmp_path, contract_name
):
    module_path, names = BASELINE_SUITES[contract_name]
    rows = _suite_rows(module_path, names)
    contract = load_data_contract(BASELINE / f"{contract_name}.yaml")
    _assert_is_a_d1_contract(contract)

    with inspection_session(factory) as spark:
        if contract_name == "incremental_upsert_fixture":
            page = tmp_path / "page-0001.json"
            page.write_text(json.dumps(rows), encoding="utf-8")
            inferred = SparkDatasetReader().read_paths(spark, (page,), format_name="json").schema
        else:
            inferred = spark.createDataFrame(rows).schema
        generated = spark_schema_from_contract(contract)

    assert _vocabulary_shape(generated) == _vocabulary_shape(inferred)


def test_the_concurrency_baseline_contract_declares_its_one_column():
    """Green on arrival: that suite never materializes, so its one column is its whole shape."""
    contract = load_data_contract(BASELINE / "concurrency_contract.yaml")

    _assert_is_a_d1_contract(contract)
    assert [(p.name, p.physical_type, p.required) for p in contract.schema.properties] == [
        ("event_id", "string", True)
    ]
    assert contract.primary_key == ()
