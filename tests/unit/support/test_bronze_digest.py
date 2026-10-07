"""Exercise the golden contract with real Spark rows and schemas."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace

import pytest

from tests.support.bronze_digest import bronze_golden, bronze_row_digest, bronze_schema_digest
from tests.support.spark_sessions import build_iceberg_session


def test_schema_digest_needs_no_spark_runtime_and_includes_type_and_nullability():
    def frame(type_name, nullable):
        return SimpleNamespace(schema=SimpleNamespace(fields=[SimpleNamespace(
            name="id", dataType=SimpleNamespace(simpleString=lambda: type_name), nullable=nullable
        )]))

    baseline = bronze_schema_digest(frame("bigint", True))
    assert baseline != bronze_schema_digest(frame("string", True))
    assert baseline != bronze_schema_digest(frame("bigint", False))


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    session = build_iceberg_session("janus-bronze-digest", tmp_path_factory.mktemp("bronze-digest"))
    yield session
    session.stop()


@pytest.mark.parametrize("partitions", [1, 4, 200])
def test_rows_are_independent_of_input_partitions(spark, partitions):
    frame = spark.createDataFrame([(2, "São Paulo"), (1, "Brasília")], "id long, city string")
    expected = '\n'.join(
        json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)
        for row in ({"id": 1, "city": "Brasília"}, {"id": 2, "city": "São Paulo"})
    )
    assert bronze_row_digest(frame.repartition(partitions)) == sha256(expected.encode()).hexdigest()


def test_changed_value_and_provenance_change_row_digest(spark):
    frame = spark.createDataFrame([(1, "run-a"), (2, "run-a")], "id long, janus_run_id string")
    changed = spark.createDataFrame([(1, "run-a"), (3, "run-a")], frame.schema)
    provenance = spark.createDataFrame([(1, "run-b"), (2, "run-b")], frame.schema)
    baseline = bronze_row_digest(frame)
    assert baseline != bronze_row_digest(changed)
    assert baseline != bronze_row_digest(provenance)


def test_schema_order_changes_schema_digest_and_rendering_uses_names(spark):
    frame = spark.createDataFrame([(1, "a"), (2, "b")], "id long, label string")
    reordered = frame.select("label", "id")
    assert bronze_schema_digest(frame) != bronze_schema_digest(reordered)
    assert bronze_row_digest(frame) == bronze_row_digest(reordered)
    triples = [("id", "bigint", True), ("label", "string", True)]
    assert bronze_schema_digest(frame) == sha256(json.dumps(triples).encode()).hexdigest()


def test_map_requires_explicit_sortable_keys_and_ties_are_deterministic(spark):
    frame = spark.createDataFrame(
        [(1, {"z": 2, "a": 1}), (1, {"a": 3})], "id long, attributes map<string,long>"
    )
    with pytest.raises(ValueError, match="map columns"):
        bronze_row_digest(frame)
    baseline = bronze_row_digest(frame, order_by=["id"])
    assert baseline == bronze_row_digest(frame.repartition(4), order_by=["id"])
    with pytest.raises(ValueError, match="unknown ordering columns"):
        bronze_row_digest(frame, order_by=["missing"])
    with pytest.raises(ValueError, match="at least one"):
        bronze_row_digest(frame, order_by=[])


def test_nested_map_is_also_refused(spark):
    frame = spark.createDataFrame([([{"x": 1}],)], "nested array<map<string,long>>")
    with pytest.raises(ValueError, match="map columns"):
        bronze_row_digest(frame)


def test_complex_values_render_as_json_with_default_str(spark):
    values = (1, Decimal("1.20"), date(2026, 9, 16), datetime(2026, 9, 16, 12), ["ç", None])
    frame = spark.createDataFrame(
        [values],
        "id long, amount decimal(5,2), day date, instant timestamp_ntz, names array<string>",
    )
    rendered = json.dumps(
        dict(zip(frame.columns, values, strict=True)),
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    assert bronze_row_digest(frame) == sha256(rendered.encode()).hexdigest()


def test_golden_contains_all_four_fields_and_empty_rows_hash_empty_bytes(spark):
    frame = spark.createDataFrame([], "id long, name string")
    assert bronze_golden(frame, identifier="bronze.empty") == {
        "identifier": "bronze.empty",
        "schema_digest": bronze_schema_digest(frame),
        "row_digest": sha256(b"").hexdigest(),
        "row_count": "0",
    }
