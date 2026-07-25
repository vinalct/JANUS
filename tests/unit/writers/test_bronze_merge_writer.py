"""Spark integration coverage for the ``merge_on_keys`` bronze writer branch."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import BronzeWriteIntent, ExecutionPlan, RunContext
from janus.normalizers import BaseNormalizer
from janus.registry import load_registry
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)

MERGE_INTENT = BronzeWriteIntent(
    strategy="merge_on_keys",
    configured_mode="append",
    merge_keys=("id",),
)


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    pyspark_sql = pytest.importorskip("pyspark.sql")
    if not ICEBERG_RUNTIME_JAR.exists():
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    warehouse_root = tmp_path_factory.mktemp("janus-merge-writer-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-merge-writer-tests")
        .master("local[1]")
        .config("spark.jars", str(ICEBERG_RUNTIME_JAR))
        .config("spark.sql.extensions", ICEBERG_SESSION_EXTENSIONS)
        .config("spark.sql.defaultCatalog", "janus")
        .config("spark.sql.catalog.janus", ICEBERG_CATALOG_IMPL)
        .config("spark.sql.catalog.janus.type", "hadoop")
        .config("spark.sql.catalog.janus.warehouse", str(warehouse_root / "iceberg"))
        .config("spark.sql.catalog.janus.default-namespace", "bronze")
        .config("spark.sql.warehouse.dir", str(warehouse_root / "spark-warehouse"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()


def _plan(tmp_path: Path, *, run_id: str, table_name: str, allow_schema_evolution=True):
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            bronze=replace(
                source_config.outputs.bronze,
                namespace="bronze_merge_test",
                table_name=table_name,
            ),
        ),
        spark=replace(source_config.spark, partition_by=()),
        quality=replace(
            source_config.quality, allow_schema_evolution=allow_schema_evolution
        ),
    )
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "runtime",
                "raw_dir": "runtime/raw",
                "bronze_dir": "runtime/bronze",
                "metadata_dir": "runtime/metadata",
            }
        },
        tmp_path,
    )


def test_merge_first_run_degrades_to_create_and_is_key_unique(spark, tmp_path):
    plan = _plan(tmp_path, run_id="run-merge-create-001", table_name="degrade")
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    frame = normalizer.normalize(
        spark.createDataFrame([{"id": "1", "name": "a"}, {"id": "2", "name": "b"}]),
        plan,
    )
    result = writer.write(
        frame, plan, "bronze", intent=MERGE_INTENT, count_records=True
    )

    assert result.metadata_as_dict()["write_strategy"] == "create"
    assert result.metadata_as_dict()["requested_strategy"] == "merge_on_keys"
    assert result.mode == "upsert"
    persisted = spark.table(result.path)
    assert persisted.count() == 2
    assert persisted.select("id").distinct().count() == 2


def test_merge_deduplicates_repeated_key_within_one_batch(spark, tmp_path):
    plan = _plan(tmp_path, run_id="run-merge-dedup-001", table_name="dedup")
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    frame = normalizer.normalize(
        spark.createDataFrame(
            [
                {"id": "1", "name": "a"},
                {"id": "1", "name": "a-dup"},
                {"id": "2", "name": "b"},
            ]
        ),
        plan,
    )
    result = writer.write(
        frame, plan, "bronze", intent=MERGE_INTENT, count_records=True
    )

    assert result.metadata_as_dict()["in_batch_duplicates_dropped"] == "1"
    persisted = spark.table(result.path)
    assert persisted.count() == 2
    assert persisted.where("id = '1'").count() == 1


def test_merge_updates_matched_rows_last_seen_wins(spark, tmp_path):
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    plan_one = _plan(tmp_path, run_id="run-merge-d3-001", table_name="lastseen")
    frame_one = normalizer.normalize(
        spark.createDataFrame([{"id": "1", "amount": 10}, {"id": "2", "amount": 10}]),
        plan_one,
    )
    writer.write(frame_one, plan_one, "bronze", intent=MERGE_INTENT, count_records=True)

    plan_two = _plan(tmp_path, run_id="run-merge-d3-002", table_name="lastseen")
    frame_two = normalizer.normalize(
        spark.createDataFrame([{"id": "2", "amount": 11}, {"id": "3", "amount": 12}]),
        plan_two,
    )
    result = writer.write(
        frame_two, plan_two, "bronze", intent=MERGE_INTENT, count_records=True
    )

    persisted = {row["id"]: row for row in spark.table(result.path).collect()}
    assert set(persisted) == {"1", "2", "3"}
    assert persisted["2"]["amount"] == 11
    assert persisted["2"]["janus_run_id"] == "run-merge-d3-002"
    assert persisted["1"]["amount"] == 10
    assert persisted["1"]["janus_run_id"] == "run-merge-d3-001"


def test_merge_empty_batch_is_skipped_without_a_commit(spark, tmp_path):
    plan = _plan(tmp_path, run_id="run-merge-empty-001", table_name="empty")
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    frame = normalizer.normalize(
        spark.createDataFrame([{"id": "1", "name": "a"}]).where("id = 'absent'"),
        plan,
    )
    result = writer.write(
        frame, plan, "bronze", intent=MERGE_INTENT, count_records=True
    )

    assert result.records_written == 0
    assert result.metadata_as_dict()["write_skipped"] == "empty_batch"
    assert not spark.catalog.tableExists(result.path)


def test_merge_evolves_schema_when_a_new_column_appears(spark, tmp_path):
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    plan_one = _plan(tmp_path, run_id="run-merge-evo-001", table_name="evolve")
    frame_one = normalizer.normalize(
        spark.createDataFrame([{"id": "1", "name": "a"}]), plan_one
    )
    writer.write(frame_one, plan_one, "bronze", intent=MERGE_INTENT, count_records=True)

    plan_two = _plan(tmp_path, run_id="run-merge-evo-002", table_name="evolve")
    frame_two = normalizer.normalize(
        spark.createDataFrame([{"id": "2", "name": "b", "extra": "x"}]), plan_two
    )
    result = writer.write(
        frame_two, plan_two, "bronze", intent=MERGE_INTENT, count_records=True
    )

    assert result.metadata_as_dict()["schema_evolved_columns"] == "extra"
    persisted = spark.table(result.path)
    assert "extra" in persisted.columns
    assert persisted.count() == 2


def test_merge_raises_when_schema_drifts_without_evolution_allowed(spark, tmp_path):
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    plan_one = _plan(
        tmp_path,
        run_id="run-merge-noevo-001",
        table_name="noevolve",
        allow_schema_evolution=False,
    )
    frame_one = normalizer.normalize(
        spark.createDataFrame([{"id": "1", "name": "a"}]), plan_one
    )
    writer.write(frame_one, plan_one, "bronze", intent=MERGE_INTENT, count_records=True)

    plan_two = _plan(
        tmp_path,
        run_id="run-merge-noevo-002",
        table_name="noevolve",
        allow_schema_evolution=False,
    )
    frame_two = normalizer.normalize(
        spark.createDataFrame([{"id": "2", "name": "b", "extra": "x"}]), plan_two
    )
    with pytest.raises(ValueError, match="extra"):
        writer.write(
            frame_two, plan_two, "bronze", intent=MERGE_INTENT, count_records=True
        )


def test_merge_does_not_duplicate_a_null_key_across_runs(spark, tmp_path):
    writer = SparkDatasetWriter(_storage_layout(tmp_path))
    normalizer = BaseNormalizer()

    plan_one = _plan(tmp_path, run_id="run-merge-null-001", table_name="nullkey")
    frame_one = normalizer.normalize(
        spark.createDataFrame([{"id": None, "name": "a"}], "id string, name string"),
        plan_one,
    )
    writer.write(frame_one, plan_one, "bronze", intent=MERGE_INTENT, count_records=True)

    plan_two = _plan(tmp_path, run_id="run-merge-null-002", table_name="nullkey")
    frame_two = normalizer.normalize(
        spark.createDataFrame([{"id": None, "name": "b"}], "id string, name string"),
        plan_two,
    )
    result = writer.write(
        frame_two, plan_two, "bronze", intent=MERGE_INTENT, count_records=True
    )

    # <=> matches null-to-null, so the null-keyed row is updated, not duplicated.
    persisted = spark.table(result.path)
    assert persisted.where("id is null").count() == 1
