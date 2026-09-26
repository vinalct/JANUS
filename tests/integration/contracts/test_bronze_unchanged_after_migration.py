from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.contract_baseline import (
    BRONZE_CASES,
    EXPLICIT_ENTRIES,
    bronze_golden,
    spark_schema_golden,
)
from tests.support.spark_sessions import build_iceberg_session, require_iceberg_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "baseline"
BRONZE_IDENTITY_KEYS = ("table_identifier", "schema", "row_count", "row_digest")


@pytest.fixture(scope="module")
def bronze_captures(tmp_path_factory):
    require_iceberg_runtime()
    root = tmp_path_factory.mktemp("janus-contract-baseline")
    spark = build_iceberg_session("janus-contract-differential", root / "catalog")
    try:
        yield {
            case: bronze_golden(
                case,
                PROJECT_ROOT,
                spark=spark,
                work_root=root / "cases" / case,
            )
            for case in BRONZE_CASES
        }
    finally:
        spark.stop()


@pytest.mark.parametrize("source_id", EXPLICIT_ENTRIES)
def test_explicit_read_schema_matches_pre_contract_golden(source_id):
    require_iceberg_runtime()
    expected = _load_json(BASELINE_ROOT / "spark_schema" / f"{source_id}.json")
    actual = spark_schema_golden(source_id, PROJECT_ROOT)

    assert actual["source_id"] == expected["source_id"]
    assert actual["struct_type"] == expected["struct_type"]


@pytest.mark.parametrize("case", BRONZE_CASES)
def test_bronze_identifier_schema_and_rows_match_pre_contract_golden(case, bronze_captures):
    expected = _load_json(BASELINE_ROOT / "bronze" / f"{case}.json")
    actual = bronze_captures[case]

    assert {key: actual[key] for key in BRONZE_IDENTITY_KEYS} == {
        key: expected[key] for key in BRONZE_IDENTITY_KEYS
    }
    assert actual["read_struct_type"] == expected["read_struct_type"]
    assert set(expected["lineage_keys"]).issubset(actual["lineage_keys"])

    if expected["read_schema_source"] == "explicit":
        assert actual["quality_report"]["checks"] == expected["quality_report"]["checks"]
        assert actual["quality_report"]["schema_expectation_source"].startswith(
            "<PROJECT>/conf/contracts/"
        )
    else:
        assert expected["inferred_struct_type"] == expected["read_struct_type"]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
