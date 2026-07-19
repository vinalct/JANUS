from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from janus.models import ExecutionPlan, RunContext
from janus.registry import load_registry
from janus.strategies.files.core import _read_checksum_sidecar
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SIDECAR_SUFFIX, RawArtifactWriter, SparkDatasetWriter
from janus.writers.spark import _rebalance_for_write

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)

NORMALIZED_COLUMNS = {
    "janus_run_id",
    "janus_source_id",
    "janus_source_name",
    "janus_environment",
    "janus_strategy_family",
    "janus_strategy_variant",
    "ingestion_timestamp",
    "ingestion_date",
}


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    pyspark_sql = pytest.importorskip("pyspark.sql")
    if not ICEBERG_RUNTIME_JAR.exists():
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    warehouse_root = tmp_path_factory.mktemp("janus-writer-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-writer-tests")
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


def test_storage_layout_resolves_zone_targets_against_runtime_storage_roots(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-storage-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)

    raw_target = storage_layout.resolve_output(plan, "raw")
    bronze_target = storage_layout.resolve_output(plan, "bronze")
    metadata_target = storage_layout.resolve_output(plan, "metadata")

    assert raw_target.resolved_path == (
        tmp_path / "runtime" / "raw" / "example" / "federal_open_data_example"
    )
    assert bronze_target.resolved_path == (
        tmp_path / "runtime" / "bronze" / "example" / "federal_open_data_example"
    )
    assert metadata_target.resolved_path == (
        tmp_path / "runtime" / "metadata" / "example" / "federal_open_data_example"
    )


def test_raw_artifact_writer_persists_json_payload_with_checksum(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-writer-raw-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    writer = RawArtifactWriter(storage_layout)

    persisted = writer.write_json(
        plan,
        "page-0001.json",
        {"page": 1, "records": [{"id": "1"}]},
        metadata={"source": "unit-test"},
    )

    persisted_path = Path(persisted.write_result.path)
    assert persisted_path == (
        tmp_path
        / "runtime"
        / "raw"
        / "example"
        / "federal_open_data_example"
        / "page-0001.json"
    )
    assert json.loads(persisted_path.read_text(encoding="utf-8")) == {
        "page": 1,
        "records": [{"id": "1"}],
    }
    assert persisted.artifact.path == str(persisted_path)
    assert persisted.artifact.checksum == persisted.write_result.metadata_as_dict()["checksum"]
    assert persisted.write_result.zone == "raw"
    assert persisted.write_result.records_written == 1
    assert persisted.write_result.metadata_as_dict()["source"] == "unit-test"


def test_raw_artifact_writer_writes_checksum_sidecar_for_every_write_method(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-writer-sidecar-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    writer = RawArtifactWriter(storage_layout)

    persisted_artifacts = [
        writer.write_bytes(plan, "bytes.bin", b"\x00\x01payload"),
        writer.write_text(plan, "text.txt", "hello sidecar"),
        writer.write_json(plan, "doc.json", {"page": 1, "records": [{"id": "1"}]}),
        writer.write_json_lines(plan, "rows.jsonl", [{"id": "1"}, {"id": "2"}]),
    ]

    for persisted in persisted_artifacts:
        persisted_path = Path(persisted.write_result.path)
        sidecar = persisted_path.with_name(persisted_path.name + SIDECAR_SUFFIX)

        on_disk_digest = sha256(persisted_path.read_bytes()).hexdigest()
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8") == f"{persisted.artifact.checksum}\n"
        assert persisted.artifact.checksum == on_disk_digest
        assert _read_checksum_sidecar(sidecar) == persisted.artifact.checksum


def test_raw_artifact_writer_refreshes_sidecar_in_ignore_mode_without_touching_payload(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-writer-sidecar-ignore-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    writer = RawArtifactWriter(storage_layout)

    first = writer.write_bytes(plan, "keep.bin", b"original bytes", mode="overwrite")
    persisted_path = Path(first.write_result.path)
    sidecar = persisted_path.with_name(persisted_path.name + SIDECAR_SUFFIX)
    sidecar.unlink()

    second = writer.write_bytes(plan, "keep.bin", b"ignored replacement", mode="ignore")

    assert persisted_path.read_bytes() == b"original bytes"
    expected_digest = sha256(b"original bytes").hexdigest()
    assert second.artifact.checksum == expected_digest
    assert sidecar.read_text(encoding="utf-8") == f"{expected_digest}\n"


def test_raw_artifact_writer_sidecar_tracks_final_bytes_after_append(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-writer-sidecar-append-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    writer = RawArtifactWriter(storage_layout)

    writer.write_bytes(plan, "log.bin", b"first-", mode="append")
    second = writer.write_bytes(plan, "log.bin", b"second", mode="append")

    persisted_path = Path(second.write_result.path)
    sidecar = persisted_path.with_name(persisted_path.name + SIDECAR_SUFFIX)

    expected_digest = sha256(b"first-second").hexdigest()
    assert persisted_path.read_bytes() == b"first-second"
    assert second.artifact.checksum == expected_digest
    assert sidecar.read_text(encoding="utf-8") == f"{expected_digest}\n"


def test_raw_artifact_writer_can_scope_raw_payloads_under_a_run_prefix(tmp_path):
    plan = _build_plan(
        tmp_path,
        run_id="run-writer-raw-002",
        started_at=datetime(2026, 4, 9, 13, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    writer = RawArtifactWriter(
        storage_layout,
        raw_path_prefix="runs/ingestion_date=2026-04-09/run_id=run-writer-raw-002",
    )

    persisted = writer.write_text(plan, "pages/page-0001.txt", "hello")

    persisted_path = Path(persisted.write_result.path)
    assert persisted_path == (
        tmp_path
        / "runtime"
        / "raw"
        / "example"
        / "federal_open_data_example"
        / "runs"
        / "ingestion_date=2026-04-09"
        / "run_id=run-writer-raw-002"
        / "pages"
        / "page-0001.txt"
    )
    assert persisted_path.read_text(encoding="utf-8") == "hello"


def test_spark_dataset_writer_persists_bronze_output_with_normalization_columns(
    spark: SparkSession,
    tmp_path,
):
    from janus.normalizers import BaseNormalizer

    plan = _build_plan(
        tmp_path,
        run_id="run-writer-bronze-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    normalizer = BaseNormalizer()
    writer = SparkDatasetWriter(storage_layout)

    dataframe = spark.createDataFrame(
        [
            {"id": "1", "name": "alpha"},
            {"id": "2", "name": "beta"},
        ]
    )
    normalized = normalizer.normalize(dataframe, plan)

    result = writer.write(normalized, plan, "bronze", count_records=True)

    assert result.zone == "bronze"
    assert result.format == "iceberg"
    assert result.records_written == 2
    assert result.partition_by == ("ingestion_date",)
    assert result.path == bronze_table_identifier(
        plan.bronze_output.path,
        fallback_name=plan.source.source_id,
        namespace=plan.bronze_output.namespace,
        table_name=plan.bronze_output.table_name,
    )

    persisted = spark.table(result.path)
    assert persisted.count() == 2
    assert NORMALIZED_COLUMNS.issubset(set(persisted.columns))

    first_row = persisted.select("janus_run_id", "janus_source_id", "ingestion_date").first()
    assert first_row.janus_run_id == "run-writer-bronze-001"
    assert first_row.janus_source_id == "federal_open_data_example"
    assert str(first_row.ingestion_date) == "2026-04-09"


def test_spark_dataset_writer_preserves_path_based_parquet_bronze_when_configured(
    spark: SparkSession,
    tmp_path,
):
    from janus.normalizers import BaseNormalizer

    base_source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        base_source_config,
        outputs=replace(
            base_source_config.outputs,
            bronze=replace(base_source_config.outputs.bronze, format="parquet"),
        ),
    )
    run_context = RunContext.create(
        run_id="run-writer-bronze-parquet-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 30, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)
    storage_layout = _build_storage_layout(tmp_path)
    normalizer = BaseNormalizer()
    writer = SparkDatasetWriter(storage_layout)

    dataframe = spark.createDataFrame([
        {"id": "1", "name": "alpha"},
        {"id": "2", "name": "beta"},
    ])
    normalized = normalizer.normalize(dataframe, plan)

    result = writer.write(normalized, plan, "bronze", count_records=True)

    assert result.zone == "bronze"
    assert result.format == "parquet"
    assert result.records_written == 2
    assert result.partition_by == ("ingestion_date",)
    assert Path(result.path) == (
        tmp_path / "runtime" / "bronze" / "example" / "federal_open_data_example"
    )

    persisted = spark.read.parquet(result.path)
    assert persisted.count() == 2
    assert NORMALIZED_COLUMNS.issubset(set(persisted.columns))


def test_spark_dataset_writer_uses_configured_bronze_iceberg_namespace_and_table(
    spark: SparkSession,
    tmp_path,
):
    from janus.normalizers import BaseNormalizer

    base_source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        base_source_config,
        outputs=replace(
            base_source_config.outputs,
            bronze=replace(
                base_source_config.outputs.bronze,
                namespace="curated",
                table_name="named_bronze_table",
            ),
        ),
    )
    run_context = RunContext.create(
        run_id="run-writer-bronze-named-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 45, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)
    storage_layout = _build_storage_layout(tmp_path)
    normalizer = BaseNormalizer()
    writer = SparkDatasetWriter(storage_layout)

    dataframe = spark.createDataFrame([
        {"id": "1", "name": "alpha"},
        {"id": "2", "name": "beta"},
    ])
    normalized = normalizer.normalize(dataframe, plan)

    result = writer.write(normalized, plan, "bronze", count_records=True)

    assert result.zone == "bronze"
    assert result.format == "iceberg"
    assert result.records_written == 2
    assert result.path == "curated.named_bronze_table"

    persisted = spark.table(result.path)
    assert persisted.count() == 2
    assert NORMALIZED_COLUMNS.issubset(set(persisted.columns))


def test_rebalance_for_write_keeps_existing_parallelism_when_target_is_lower():
    class FakeRDD:
        def __init__(self, partitions: int) -> None:
            self._partitions = partitions

        def getNumPartitions(self) -> int:
            return self._partitions

    class FakeFrame:
        def __init__(self, partitions: int) -> None:
            self.rdd = FakeRDD(partitions)
            self.calls: list[tuple[str, int]] = []

        def coalesce(self, partitions: int):
            self.calls.append(("coalesce", partitions))
            return self

        def repartition(self, partitions: int):
            self.calls.append(("repartition", partitions))
            return self

    frame = FakeFrame(partitions=12)

    result = _rebalance_for_write(
        frame,
        target_partitions=8,
        apply_repartition=True,
    )

    assert result is frame
    assert frame.calls == []


def test_rebalance_for_write_uses_repartition_when_increasing_partitions():
    class FakeRDD:
        def __init__(self, partitions: int) -> None:
            self._partitions = partitions

        def getNumPartitions(self) -> int:
            return self._partitions

    class FakeFrame:
        def __init__(self, partitions: int) -> None:
            self.rdd = FakeRDD(partitions)
            self.calls: list[tuple[str, int]] = []

        def coalesce(self, partitions: int):
            self.calls.append(("coalesce", partitions))
            return self

        def repartition(self, partitions: int):
            self.calls.append(("repartition", partitions))
            return self

    frame = FakeFrame(partitions=2)

    result = _rebalance_for_write(
        frame,
        target_partitions=8,
        apply_repartition=True,
    )

    assert result is frame
    assert frame.calls == [("repartition", 8)]


def _build_plan(tmp_path: Path, *, run_id: str, started_at: datetime) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _build_storage_layout(tmp_path: Path) -> StorageLayout:
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
