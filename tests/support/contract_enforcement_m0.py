"""Characterise today's contract enforcement on the pinned Spark/Iceberg pair."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from janus.models import BronzeWriteIntent
from janus.models.data_contracts import load_data_contract
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.runtime import ExecutedRun
from janus.schema_contracts import spark_schema_from_contract
from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
)
from janus.writers import (
    build_create_table_as_select_sql,
    build_merge_sql,
    quote_identifier,
)
from tests.support.contract_enforcement import (
    EnforcementCase,
    bronze_rows,
    contract_property,
    current_metadata_file,
    execute_case_with_pages,
    failed_check_names,
    iceberg_session_factory,
    inspection_session,
    plan_case,
    render_contract,
    snapshot_count,
    table_exists,
    table_properties,
    table_schema,
    table_uuid,
    validation_checks,
    write_case_project,
    write_frame,
)
from tests.support.contract_enforcement_cost import cost_cnpj, cost_inep

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "malformed"
STARTED_AT = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
INGESTION_DATE = date(2026, 9, 27)
INEP_SOURCE_ID = "inep_censo_escolar_microdados"
_EXCERPT = 400

ID_AMOUNT = (
    contract_property("id", "string", required=True),
    contract_property("amount", "long"),
)
BASE_PROPERTIES = (
    contract_property("id", "string", required=True, primary_key=True),
    contract_property("label", "string"),
    contract_property("amount", "long"),
    contract_property("when", "timestamptz"),
)

PROMOTION_STATEMENTS: tuple[tuple[str, str, str], ...] = (
    ("i", "bigint", "accepted"),
    ("f", "double", "accepted"),
    ("d", "decimal(18,2)", "accepted (precision widening, same scale)"),
    ("l", "int", "refused"),
    ("s", "int", "refused"),
    ("wide", "decimal(18,2)", "refused (scale change)"),
    ("d", "decimal(8,2)", "refused (narrowing)"),
)
PROMOTION_TABLE_DDL = "i int, f float, d decimal(10,2), s string, l bigint, wide decimal(18,4)"
PROMOTION_ROW = (1, 1.5, Decimal("12.34"), "7", 9, Decimal("123.4567"))


# ── proof 1: a failed check after the commit leaves the rows ─────────────────


def proof_1(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "proof1")
    factory = iceberg_session_factory(root / "warehouse", "janus-order19-m0-proof1")
    good_pages = [
        [{"id": "g1", "amount": 1}, {"id": "g2", "amount": 2}],
        [{"id": "g3", "amount": 3}],
    ]
    no_amount = [[{"id": "m1"}, {"id": "m2"}], [{"id": "m3"}]]
    return {
        "1a_append_missing_column_schema_none_route": _two_runs(
            EnforcementCase.for_source(
                "m0_p1a_append_missing", ID_AMOUNT, raw_format="jsonl", page_size=2
            ),
            root / "p1a",
            factory,
            good_pages,
            no_amount,
            columns=("janus_run_id", "id"),
        ),
        "1b_overwrite_missing_column_schema_none_route": _two_runs(
            EnforcementCase.for_source(
                "m0_p1b_overwrite_missing",
                ID_AMOUNT,
                raw_format="jsonl",
                write_mode="overwrite",
                page_size=2,
            ),
            root / "p1b",
            factory,
            good_pages,
            no_amount,
            columns=("janus_run_id", "id"),
        ),
        "1c_append_required_null": _two_runs(
            EnforcementCase.for_source("m0_p1c_append_required_null", ID_AMOUNT, page_size=2),
            root / "p1c",
            factory,
            good_pages,
            [[{"id": None, "amount": 5}, {"id": "n2", "amount": 6}]],
            columns=("janus_run_id", "id", "amount"),
        ),
        "1d_append_undeclared_column_schema_none_route": _two_runs(
            EnforcementCase.for_source(
                "m0_p1d_append_extra", ID_AMOUNT, raw_format="jsonl", page_size=2
            ),
            root / "p1d",
            factory,
            good_pages,
            [[{"id": "x1", "amount": 7, "note": "new upstream field"}]],
            columns=("janus_run_id", "id"),
        ),
    }


def _two_runs(
    case: EnforcementCase,
    project: Path,
    factory: Callable[[], Any],
    first_pages: Sequence[Any],
    second_pages: Sequence[Any],
    *,
    columns: Sequence[str],
) -> dict[str, Any]:
    first = _run(case, project, factory, first_pages, run_id=f"{case.source_id}-run1")
    before = _table_state(factory, case.bronze_table)
    second = _run(case, project, factory, second_pages, run_id=f"{case.source_id}-run2")
    after = _table_state(factory, case.bronze_table, columns=columns)
    return {
        "table": case.bronze_table,
        "run_1": _summary(first),
        "table_after_run_1": before,
        "run_2": _summary(second),
        "table_after_run_2": after,
    }


# ── proof 2: type drift becomes nulls ────────────────────────────────────────


def proof_2(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "proof2")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-proof2")

    json_case = EnforcementCase.for_source("m0_p2_json_drift", ID_AMOUNT)
    with _recording_reads() as reads:
        json_run = _run(
            json_case,
            root / "p2json",
            factory,
            [[{"id": "a", "amount": "not-a-number"}, {"id": "b", "amount": 7}]],
            run_id="m0-p2-json",
        )
    json_state = _table_state(factory, json_case.bronze_table, columns=("id", "amount"))

    page_case = EnforcementCase.for_source("m0_p2_page_with_drift", BASE_PROPERTIES)
    page = json.loads((FIXTURES_DIR / "page_with_drift.json").read_text(encoding="utf-8"))
    with _recording_reads() as page_reads:
        page_run = _run(page_case, root / "p2page", factory, [page], run_id="m0-p2-page")
    page_state = _table_state(
        factory, page_case.bronze_table, columns=("id", "label", "amount", "when")
    )

    return {
        "2a_json_string_into_long": {
            "run": _summary(json_run),
            "checks_mentioning_amount": _checks_mentioning(json_run, "amount"),
            "reader_calls": reads,
            "table": json_state,
        },
        "2b_page_with_drift_json": {
            "fixture": "tests/fixtures/malformed/page_with_drift.json",
            "run": _summary(page_run),
            "checks_mentioning_amount": _checks_mentioning(page_run, "amount"),
            "reader_calls": page_reads,
            "table": page_state,
        },
        "2c_line_with_drift_csv": _read_inep_shaped_csv(factory),
    }


def _read_inep_shaped_csv(factory: Callable[[], Any]) -> dict[str, Any]:
    """Read the CSV fixture the way materialize reads a CSV handoff for the INEP source."""
    registry = load_registry(PROJECT_ROOT)
    source = registry.get_source(INEP_SOURCE_ID, include_disabled=True)
    contract = registry.contract_for(INEP_SOURCE_ID)
    assert contract is not None
    path = FIXTURES_DIR / "line_with_drift.csv"
    with inspection_session(factory) as spark, _recording_reads() as reads:
        schema = spark_schema_from_contract(contract)
        frame = SparkDatasetReader().read_paths(
            spark,
            (path,),
            format_name=source.spark.input_format,
            schema=schema,
            options=source.spark.read_options,
        )
        rows = [
            (row["CO_ENTIDADE"], row["TP_LOCALIZACAO"], row["QT_MAT_BAS"])
            for row in frame.collect()
        ]
    return {
        "fixture": "tests/fixtures/malformed/line_with_drift.csv",
        "contract": str(contract.contract_path.relative_to(PROJECT_ROOT)),
        "data_lines_in_file": len(path.read_text(encoding="utf-8").splitlines()) - 1,
        "rows_read": len(rows),
        "rows_as_CO_ENTIDADE_TP_LOCALIZACAO_QT_MAT_BAS": rows,
        "frame_columns": list(frame.columns),
        "reader_calls": reads,
    }


# ── proof 3: a same-arity swap lands swapped ─────────────────────────────────


def proof_3(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "proof3")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-proof3")
    case = EnforcementCase.for_source(
        "m0_p3_swap",
        (contract_property("id", "string"), contract_property("label", "string")),
    )
    project = write_case_project(root / "p3", case)
    plan = plan_case(project, case, run_id="m0-p3", started_at=STARTED_AT).plan
    insert = BronzeWriteIntent(strategy="insert", configured_mode="append")
    with inspection_session(factory) as spark:
        created = write_frame(
            spark,
            plan,
            [("1", "one", INGESTION_DATE)],
            "id string, label string, ingestion_date date",
        )
        appended = write_frame(
            spark,
            plan,
            [("two", "2", INGESTION_DATE)],
            "label string, id string, ingestion_date date",
            intent=insert,
        )
        rows = bronze_rows(spark, case.bronze_table, ("id", "label"))
        schema = table_schema(spark, case.bronze_table)
        snapshots = snapshot_count(spark, case.bronze_table)
    return {
        "3a_write_frame_swap": {
            "first_write": _write_summary(created),
            "second_write": _write_summary(appended),
            "table_schema": schema,
            "rows_id_label": rows,
            "snapshots": snapshots,
        },
        "3b_contract_reorder_changes_generated_order": _contract_reorder(root / "reorder"),
    }


def _contract_reorder(root: Path) -> dict[str, Any]:
    properties = (
        contract_property("id", "string"),
        contract_property("label", "string"),
        contract_property("amount", "long"),
    )
    reordered = (properties[2], properties[0], properties[1])
    v1 = _load_rendered_contract(root, "v1", properties, "1.0.0")
    v2 = _load_rendered_contract(root, "v2", reordered, "1.1.0")
    names_v1 = list(spark_schema_from_contract(v1).fieldNames())
    names_v2 = list(spark_schema_from_contract(v2).fieldNames())
    return {
        "v1_field_names": names_v1,
        "v2_field_names": names_v2,
        "same_set": sorted(names_v1) == sorted(names_v2),
        "same_order": names_v1 == names_v2,
    }


def _load_rendered_contract(
    root: Path, name: str, properties: Sequence[Mapping[str, Any]], version: str
) -> Any:
    import yaml

    case = EnforcementCase.for_source(f"m0_p3_{name}", properties, version=version)
    path = root / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(render_contract(case), sort_keys=False), encoding="utf-8")
    return load_data_contract(path)


# ── proof 4: a promotable retype resets history ──────────────────────────────


def proof_4(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "proof4")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-proof4")
    cells = (
        ("int_to_bigint", "integer", "int", "bigint", 10, 11),
        ("float_to_double", "float", "float", "double", 1.5, 2.5),
        (
            "decimal_10_2_to_decimal_18_2",
            "decimal(10,2)",
            "decimal(10,2)",
            "decimal(18,2)",
            Decimal("1.25"),
            Decimal("2.50"),
        ),
    )
    results: dict[str, Any] = {}
    with inspection_session(factory) as spark:
        for label, contract_type, before_type, after_type, before_value, after_value in cells:
            case = EnforcementCase.for_source(
                f"m0_p4_{label}",
                (contract_property("id", "string"), contract_property("amount", contract_type)),
                write_mode="overwrite",
            )
            project = write_case_project(root / label, case)
            plan = plan_case(project, case, run_id=f"m0-p4-{label}", started_at=STARTED_AT).plan
            table = case.bronze_table
            before_ddl = f"id string, amount {before_type}"
            first = write_frame(spark, plan, [("1", before_value)], before_ddl)
            first_snapshot = _first_snapshot_id(spark, table)
            state_before = _live_state(spark, table)
            after_ddl = f"id string, amount {after_type}"
            second = write_frame(spark, plan, [("1", after_value)], after_ddl)
            state_after = _live_state(spark, table)
            results[label] = {
                "first_write": _write_summary(first),
                "before": state_before,
                "second_write": _write_summary(second),
                "after": state_after,
                "uuid_changed": state_before["uuid"] != state_after["uuid"],
                "history_after": _history(spark, table),
                "time_travel_to_first_snapshot": _time_travel(spark, table, first_snapshot),
                "rollback_to_first_snapshot": _attempt(
                    lambda table=table, first_snapshot=first_snapshot: spark.sql(
                        f"CALL janus.system.rollback_to_snapshot('{table}', {first_snapshot})"
                    )
                ),
            }
    return results


# ── proof 5: contract-versus-table drift goes undetected ─────────────────────


def proof_5(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "proof5")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-proof5")
    merge_properties = (
        contract_property("id", "string", required=True, primary_key=True),
        contract_property("amount", "long"),
        contract_property("updated_at", "string", required=True),
    )
    cases = {
        "5a_append": EnforcementCase.for_source("m0_p5_append", ID_AMOUNT),
        "5b_overwrite": EnforcementCase.for_source(
            "m0_p5_overwrite", ID_AMOUNT, write_mode="overwrite"
        ),
        "5c_merge": EnforcementCase.for_source(
            "m0_p5_merge",
            merge_properties,
            extraction_mode="incremental",
            checkpoint_field="updated_at",
        ),
    }
    page = [{"id": "d1", "amount": 1, "updated_at": "2026-09-01"}]
    results: dict[str, Any] = {}
    for label, case in cases.items():
        project = root / label
        first = _run(case, project, factory, [page], run_id=f"{case.source_id}-run1")
        with inspection_session(factory) as spark:
            spark.sql(
                f"ALTER TABLE {quote_identifier(case.bronze_table)} "
                "ADD COLUMNS (legacy_col string)"
            )
            drifted = _live_state(spark, case.bronze_table)
        second = _run(case, project, factory, [page], run_id=f"{case.source_id}-run2")
        results[label] = {
            "run_1": _summary(first),
            "after_manual_add_column": drifted,
            "run_2": _summary(second),
            "run_2_artifacts_mentioning_legacy_col": _artifacts_mentioning(second, "legacy_col"),
            "table_after_run_2": _table_state(factory, case.bronze_table),
        }
    return results


# ── verification: merge resolves by name ─────────────────────────────────────


def verify_merge(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "merge")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-merge")
    table = "m0_merge.target"
    views = {
        "reordered": (
            "amount bigint, label string, id string",
            [(20, "uno", "1"), (30, "two", "2")],
        ),
        "missing_amount": ("label string, id string", [("tres", "3")]),
        "extra_column": (
            "id string, label string, amount bigint, extra string",
            [("4", "four", 40, "surplus")],
        ),
    }
    results: dict[str, Any] = {}
    with inspection_session(factory) as spark:
        spark.sql("CREATE NAMESPACE IF NOT EXISTS m0_merge")
        spark.createDataFrame(
            [("1", "one", 10)], "id string, label string, amount bigint"
        ).createOrReplaceTempView("m0_merge_base")
        spark.sql(
            build_create_table_as_select_sql(
                table_identifier=table, source_view="m0_merge_base", partition_columns=()
            )
        )
        results["created_schema"] = table_schema(spark, table)
        for name, (ddl, rows) in views.items():
            view = f"m0_merge_{name}"
            spark.createDataFrame(rows, ddl).createOrReplaceTempView(view)
            statement = build_merge_sql(
                table_identifier=table, source_view=view, merge_keys=("id",)
            )
            outcome = _attempt(lambda statement=statement: spark.sql(statement))
            results[name] = {
                "view_columns": ddl,
                "outcome": outcome,
                "table_schema_after": table_schema(spark, table),
                "rows_after": bronze_rows(spark, table, ("id", "label", "amount")),
            }
        results["merge_sql"] = build_merge_sql(
            table_identifier=table, source_view="<view>", merge_keys=("id",)
        )
    reordered_rows = results["reordered"]["rows_after"]
    results["by_name"] = reordered_rows == [("1", "uno", 20), ("2", "two", 30)]
    return results


# ── verification: which ALTER COLUMN … TYPE statements the pinned pair accepts ─


def verify_promotions(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "promotions")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-promotions")
    rows: list[dict[str, Any]] = []
    with inspection_session(factory) as spark:
        spark.sql("CREATE NAMESPACE IF NOT EXISTS m0_promo")
        for index, (column, target, expected) in enumerate(PROMOTION_STATEMENTS, start=1):
            table = f"m0_promo.t{index}"
            spark.createDataFrame([PROMOTION_ROW], PROMOTION_TABLE_DDL).createOrReplaceTempView(
                "m0_promo_seed"
            )
            spark.sql(
                build_create_table_as_select_sql(
                    table_identifier=table, source_view="m0_promo_seed", partition_columns=()
                )
            )
            before = _live_state(spark, table)
            statement = f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {target}"
            outcome = _attempt(lambda statement=statement: spark.sql(statement))
            after = _live_state(spark, table)
            value = spark.table(table).select(column).collect()[0][0]
            rows.append(
                {
                    "statement": statement,
                    "expected": expected,
                    "outcome": outcome,
                    "column_type_before": dict(before["schema"])[column],
                    "column_type_after": dict(after["schema"])[column],
                    "value_read_back": str(value),
                    "snapshots_before": before["snapshots"],
                    "snapshots_after": after["snapshots"],
                    "uuid_unchanged": before["uuid"] == after["uuid"],
                    "metadata_file_changed": before["metadata_file"] != after["metadata_file"],
                }
            )
        extensions = spark.conf.get("spark.sql.extensions", "")
    return {"session_extensions": extensions, "statements": rows}


# ── verification: table properties round-trip through Spark and PyIceberg ────


def verify_properties(out_dir: Path) -> dict[str, Any]:
    from tests.support.spark_sessions import sqlite_catalog_target

    root = _fresh(out_dir, "properties")
    warehouse = root / "warehouse"
    factory = iceberg_session_factory(warehouse, "janus-m0-properties")
    table = "m0_props.t"
    with inspection_session(factory) as spark:
        spark.sql("CREATE NAMESPACE IF NOT EXISTS m0_props")
        spark.createDataFrame([("1",)], "id string").createOrReplaceTempView("m0_props_seed")
        spark.sql(
            build_create_table_as_select_sql(
                table_identifier=table, source_view="m0_props_seed", partition_columns=()
            )
        )
        snapshots_before = snapshot_count(spark, table)
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES ('janus.contract_version'='1.0.0')")
        shown = spark.sql(f"SHOW TBLPROPERTIES {table} ('janus.contract_version')").collect()
        snapshots_after = snapshot_count(spark, table)

    from pyiceberg.catalog import load_catalog

    target = sqlite_catalog_target(warehouse)
    config = target.environment_config()
    catalog = load_catalog(
        derive_pyiceberg_catalog_name(config),
        **derive_pyiceberg_catalog_properties(config, target.resolved_paths),
    )
    pyiceberg_properties = dict(catalog.load_table(table).properties)
    return {
        "spark_show_tblproperties": [(row["key"], row["value"]) for row in shown],
        "pyiceberg_property": pyiceberg_properties.get("janus.contract_version"),
        "snapshots_before": snapshots_before,
        "snapshots_after": snapshots_after,
    }


# ── observations adjacent to the proofs ──────────────────────────────────────


def observations(out_dir: Path) -> dict[str, Any]:
    root = _fresh(out_dir, "observations")
    factory = iceberg_session_factory(root / "warehouse", "janus-m0-observations")
    page = [[{"id": "o1", "amount": 1}]]
    results: dict[str, Any] = {}
    for flag in (False, True):
        case = EnforcementCase.for_source(
            f"m0_o1_evolution_{str(flag).lower()}", ID_AMOUNT, allow_schema_evolution=flag
        )
        executed = _run(case, root / case.source_id, factory, page, run_id=f"{case.source_id}-1")
        check = validation_checks(executed).get("data.schema_expectations")
        results[f"allow_schema_evolution_{str(flag).lower()}"] = {
            "run": _summary(executed),
            "schema_expectations": _check_summary(check),
        }
    return {"O-1_normalization_columns_vs_names_only_check": results}


# ── shared helpers ───────────────────────────────────────────────────────────


def _fresh(out_dir: Path, name: str) -> Path:
    path = out_dir / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _run(
    case: EnforcementCase,
    project: Path,
    factory: Callable[[], Any],
    pages: Sequence[Any],
    *,
    run_id: str,
) -> ExecutedRun:
    write_case_project(project, case)
    planned = plan_case(project, case, run_id=run_id, started_at=STARTED_AT)
    return execute_case_with_pages(planned, pages, factory)


def _summary(executed: ExecutedRun) -> dict[str, Any]:
    return {
        "status": executed.status,
        "error_type": executed.error_type,
        "failure_reason": _excerpt(executed.failure_reason),
        "failed_checks": failed_check_names(executed),
        "validation_report_written": executed.validation_report is not None,
        "bronze_writes": [
            {"mode": result.mode, "metadata": result.metadata_as_dict()}
            for result in executed.write_results
            if result.zone == "bronze"
        ],
    }


def _check_summary(check: Any) -> dict[str, Any] | None:
    if check is None:
        return None
    return {
        "outcome": check.outcome,
        "message": _excerpt(check.message),
        "details": check.details_as_dict(),
    }


def _checks_mentioning(executed: ExecutedRun, needle: str) -> list[str]:
    return [
        name
        for name, check in validation_checks(executed).items()
        if needle in json.dumps(check.to_dict())
    ]


def _artifacts_mentioning(executed: ExecutedRun, needle: str) -> dict[str, bool | None]:
    paths = {
        "run_metadata": executed.run_metadata_path,
        "lineage": executed.lineage_path,
        "validation_report": (
            executed.validation_report.path if executed.validation_report else None
        ),
    }
    return {
        name: (needle in Path(path).read_text(encoding="utf-8")) if path else None
        for name, path in paths.items()
    }


def _write_summary(result: Any) -> dict[str, Any]:
    return {"mode": result.mode, "metadata": result.metadata_as_dict()}


def _table_state(
    factory: Callable[[], Any], table: str, *, columns: Sequence[str] = ()
) -> dict[str, Any]:
    with inspection_session(factory) as spark:
        if not table_exists(spark, table):
            return {"exists": False}
        state = _live_state(spark, table)
        if columns:
            state["rows"] = bronze_rows(spark, table, columns)
        return state


def _live_state(spark: Any, table: str) -> dict[str, Any]:
    return {
        "exists": True,
        "snapshots": snapshot_count(spark, table),
        "uuid": table_uuid(spark, table),
        "metadata_file": Path(current_metadata_file(spark, table)).name,
        "schema": table_schema(spark, table),
        "properties": {
            key: value
            for key, value in table_properties(spark, table).items()
            if key.startswith("janus.")
        },
    }


def _first_snapshot_id(spark: Any, table: str) -> int:
    metadata_table = quote_identifier(f"{table}.snapshots")
    return int(
        spark.sql(
            f"SELECT snapshot_id FROM {metadata_table} ORDER BY committed_at LIMIT 1"
        ).collect()[0][0]
    )


def _history(spark: Any, table: str) -> list[dict[str, Any]]:
    """``<t>.history`` joined to ``<t>.snapshots``: parent links and current ancestry."""
    history = quote_identifier(f"{table}.history")
    snapshots = quote_identifier(f"{table}.snapshots")
    rows = spark.sql(
        f"SELECT h.snapshot_id, h.parent_id, h.is_current_ancestor, s.operation "
        f"FROM {history} h JOIN {snapshots} s ON h.snapshot_id = s.snapshot_id "
        "ORDER BY h.made_current_at"
    ).collect()
    return [row.asDict() for row in rows]


def _time_travel(spark: Any, table: str, snapshot_id: int) -> dict[str, Any]:
    rows: list[Any] = []

    def read() -> None:
        rows.extend(
            spark.sql(
                f"SELECT * FROM {quote_identifier(table)} VERSION AS OF {snapshot_id}"
            ).collect()
        )

    return {"snapshot_id": snapshot_id, "outcome": _attempt(read), "rows": len(rows)}


def _attempt(action: Callable[[], Any]) -> dict[str, str]:
    try:
        action()
    except Exception as exc:
        return {"result": "refused", "error": f"{type(exc).__name__}: {_first_line(exc)}"}
    return {"result": "accepted"}


@contextmanager
def _recording_reads() -> Iterator[list[dict[str, Any]]]:
    """Record what every ``SparkDatasetReader.read_paths`` call actually sends to Spark."""
    from janus.readers import spark as reader_module

    calls: list[dict[str, Any]] = []
    original = SparkDatasetReader.read_paths

    def recording(self, spark, paths, *, format_name, schema=None, options=None): 
        calls.append(
            {
                "format_name": format_name,
                "spark_format": reader_module._spark_read_format(format_name),
                "schema": schema.simpleString() if schema is not None else None,
                "options_sent": reader_module._resolved_read_options(format_name, options),
            }
        )
        return original(
            self, spark, paths, format_name=format_name, schema=schema, options=options
        )

    SparkDatasetReader.read_paths = recording 
    try:
        yield calls
    finally:
        SparkDatasetReader.read_paths = original 


def _first_line(exc: BaseException) -> str:
    java_exception = getattr(exc, "java_exception", None)
    text = str(java_exception if java_exception is not None else exc).strip().splitlines()
    return _excerpt(text[0] if text else "") or ""


def _excerpt(text: str | None) -> str | None:
    if text is None:
        return None
    return text if len(text) <= _EXCERPT else text[:_EXCERPT] + "…"


SECTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "proof1": proof_1,
    "proof2": proof_2,
    "proof3": proof_3,
    "proof4": proof_4,
    "proof5": proof_5,
    "merge": verify_merge,
    "promotions": verify_promotions,
    "properties": verify_properties,
    "observations": observations,
    "cost-inep": cost_inep,
}
DEFAULT_SECTIONS = tuple(name for name in SECTIONS if not name.startswith("cost-"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--only", default=",".join(DEFAULT_SECTIONS))
    parser.add_argument("--cnpj-csv", type=Path)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (part.strip() for part in args.only.split(",") if part.strip()):
        ran_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
        if name == "cost-cnpj":
            if args.cnpj_csv is None:
                parser.error("cost-cnpj needs --cnpj-csv; see tests.support.cnpj_csv_generator")
            result = cost_cnpj(args.output, args.cnpj_csv)
        else:
            result = SECTIONS[name](args.output)
        record = {"section": name, "ran_at": ran_at, "result": result}
        rendered = json.dumps(record, indent=2, default=str, ensure_ascii=False)
        (args.output / f"{name}.json").write_text(rendered + "\n", encoding="utf-8")
        print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
