from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from janus.lineage import RunObserver
from janus.models import RunContext, SourceConfig
from janus.normalizers import BaseNormalizer
from janus.quality import QualityGate, load_expected_fields_from_schema_path
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.strategies.files import FileStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SparkDatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "inep"
SOURCE_ID = "inep_censo_escolar_microdados"
ARCHIVE_NAME = "microdados_censo_escolar_2024.zip"
ARCHIVE_MEMBER = "microdados_censo_escolar_2024/dados/microdados_ed_basica_2024.csv"
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)
SCHEMA_FIELDS = (
    "NU_ANO_CENSO",
    "CO_ENTIDADE",
    "NO_ENTIDADE",
    "CO_UF",
    "SG_UF",
    "CO_MUNICIPIO",
    "NO_MUNICIPIO",
    "TP_DEPENDENCIA",
    "TP_LOCALIZACAO",
    "TP_SITUACAO_FUNCIONAMENTO",
    "QT_MAT_BAS",
)


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    pyspark_sql = pytest.importorskip("pyspark.sql")
    if not ICEBERG_RUNTIME_JAR.exists():
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    warehouse_root = tmp_path_factory.mktemp("janus-inep-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-inep-integration")
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


def test_inep_source_contract_uses_generic_archive_file_strategy():
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    schema_path = PROJECT_ROOT / source_config.schema.path

    assert source_config.enabled is False
    assert source_config.strategy == "file"
    assert source_config.strategy_variant == "archive_package"
    assert source_config.source_hook is None
    assert source_config.access.url == (
        "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2024.zip"
    )
    assert source_config.access.format == "binary"
    assert source_config.access.file_pattern == "*/dados/microdados_ed_basica_2024.csv"
    assert source_config.access.auth.type == "none"
    assert source_config.access.pagination.type == "none"
    assert source_config.access.rate_limit.requests_per_minute == 12
    assert source_config.access.rate_limit.concurrency == 1
    assert source_config.extraction.mode == "snapshot"
    assert source_config.extraction.checkpoint_strategy == "none"
    assert source_config.spark.input_format == "csv"
    assert source_config.spark.read_options == {
        "header": "true",
        "sep": ";",
        "encoding": "ISO-8859-1",
        "inferSchema": "true",
    }
    assert source_config.outputs.raw.format == "binary"
    assert source_config.outputs.bronze.format == "iceberg"
    assert source_config.outputs.bronze.path == "data/bronze/inep/censo_escolar_microdados"
    assert load_expected_fields_from_schema_path(schema_path) == SCHEMA_FIELDS


def test_inep_archive_extracts_microdata_csv_and_materializes_bronze_and_metadata(
    spark,
    tmp_path,
):
    archive_path = _build_fixture_archive(tmp_path)
    source_config = _cloned_source_config(tmp_path, archive_path=archive_path)
    run_context = RunContext.create(
        run_id="run-inep-censo-escolar-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 13, 0, tzinfo=UTC),
    )
    storage_layout = _storage_layout(tmp_path)
    strategy = FileStrategy(
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )

    plan = strategy.plan(source_config, run_context)
    extraction_result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, extraction_result)
    metadata = extraction_result.metadata_as_dict()

    assert plan.notes == (
        "strategy_family:file",
        "strategy_variant:archive_package",
    )
    assert extraction_result.records_extracted == 1
    assert extraction_result.checkpoint_value == "2024"
    assert metadata["discovered_file_count"] == "1"
    assert metadata["selected_file_count"] == "1"
    assert metadata["persisted_file_count"] == "1"
    assert metadata["archive_member_count"] == "1"
    assert metadata["normalization_candidate_count"] == "1"
    assert metadata["resolved_versions"] == "2024"

    raw_run_dir = (
        tmp_path
        / "data"
        / "raw"
        / "inep"
        / "censo_escolar_microdados"
        / "runs"
        / "ingestion_date=2026-04-10"
        / "run_id=run-inep-censo-escolar-001"
    )
    persisted_archive = Path(extraction_result.artifacts[0].path)
    extracted_csv = Path(handoff.artifacts[0].path)
    assert persisted_archive == raw_run_dir / "downloads" / "2024" / ARCHIVE_NAME
    assert extracted_csv == (
        raw_run_dir
        / "extracted"
        / "2024"
        / "microdados_censo_escolar_2024"
        / "microdados_censo_escolar_2024"
        / "dados"
        / "microdados_ed_basica_2024.csv"
    )
    assert extracted_csv.read_text(encoding="utf-8") == _fixture_csv()

    reader = SparkDatasetReader()
    raw_dataframe = reader.read_extraction_result(
        spark,
        handoff,
        options=plan.source_config.spark.read_options,
    )
    normalized_dataframe = BaseNormalizer().normalize(raw_dataframe, plan)
    bronze_result = SparkDatasetWriter(storage_layout).write(
        normalized_dataframe,
        plan,
        "bronze",
        count_records=True,
    )

    report = QualityGate().validate(
        plan,
        dataframe=normalized_dataframe,
        write_results=(bronze_result,),
    )
    assert report.is_successful is True

    emitted_metadata = strategy.emit_metadata(plan, extraction_result, (bronze_result,))
    assert emitted_metadata["strategy_family"] == "file"
    assert emitted_metadata["strategy_variant"] == "archive_package"
    assert emitted_metadata["input_format"] == "csv"
    assert emitted_metadata["artifact_count"] == 2
    assert emitted_metadata["archive_member_count"] == "1"
    assert emitted_metadata["write_result_count"] == 1

    observer = RunObserver()
    started = observer.start_run(plan)
    persisted = observer.record_success(
        plan,
        extraction_result,
        (bronze_result,),
        finished_at=datetime(2026, 4, 10, 13, 5, tzinfo=UTC),
    )

    bronze_dataframe = spark.table(bronze_result.path)
    dependency_counts = {
        row["TP_DEPENDENCIA"]: row["count"]
        for row in bronze_dataframe.groupBy("TP_DEPENDENCIA").count().collect()
    }

    assert bronze_result.records_written == 3
    assert dependency_counts == {1: 1, 2: 1, 3: 1}
    assert bronze_result.path == bronze_table_identifier(
        plan.bronze_output.path,
        fallback_name=plan.source.source_id,
        namespace=plan.bronze_output.namespace,
        table_name=plan.bronze_output.table_name,
    )

    run_payload = json.loads(started.run_metadata_path.read_text(encoding="utf-8"))
    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))

    assert started.run_metadata_path == (
        tmp_path
        / "data"
        / "metadata"
        / "inep"
        / "censo_escolar_microdados"
        / "runs"
        / "run-inep-censo-escolar-001.json"
    )
    assert run_payload["status"] == "succeeded"
    assert run_payload["records_extracted"] == 1
    assert lineage_payload["status"] == "succeeded"
    assert lineage_payload["strategy_variant"] == "archive_package"
    assert lineage_payload["artifacts"][0]["path"].endswith(ARCHIVE_NAME)
    assert persisted.checkpoint_result is not None
    assert persisted.checkpoint_result.decision == "skipped"


def test_inep_archive_rerun_reuses_the_same_raw_paths(tmp_path):
    archive_path = _build_fixture_archive(tmp_path)
    source_config = _cloned_source_config(tmp_path, archive_path=archive_path)
    run_context = RunContext.create(
        run_id="run-inep-rerun-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 14, 0, tzinfo=UTC),
    )
    strategy = FileStrategy(
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    plan = strategy.plan(source_config, run_context)

    first_result = strategy.extract(plan)
    second_result = strategy.extract(plan)

    assert first_result.checkpoint_value == second_result.checkpoint_value == "2024"
    assert first_result.artifact_paths == second_result.artifact_paths
    assert [artifact.checksum for artifact in first_result.artifacts] == [
        artifact.checksum for artifact in second_result.artifacts
    ]
    assert first_result.metadata_as_dict()["archive_member_count"] == "1"
    assert second_result.metadata_as_dict()["archive_member_count"] == "1"


def _build_fixture_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "fixtures" / ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(ARCHIVE_MEMBER, _fixture_csv())
        archive.writestr(
            "microdados_censo_escolar_2024/leia-me.txt",
            "Reduced fixture package for JANUS integration tests.\n",
        )
    return archive_path


def _cloned_source_config(tmp_path: Path, *, archive_path: Path) -> SourceConfig:
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    config_path = PROJECT_ROOT / "conf" / "sources" / "inep" / "inep.yaml"
    schema_path = (
        PROJECT_ROOT / "conf" / "schemas" / "inep" / "censo_escolar_microdados_schema.json"
    )

    copied_config_path = tmp_path / "conf" / "sources" / "inep" / "inep.yaml"
    copied_schema_path = (
        tmp_path / "conf" / "schemas" / "inep" / "censo_escolar_microdados_schema.json"
    )
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)
    copied_schema_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["access"].pop("url", None)
    config_payload["access"]["path"] = str(archive_path)
    copied_config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    copied_schema_path.write_text(schema_path.read_text(encoding="utf-8"), encoding="utf-8")

    return replace(
        source_config,
        config_path=copied_config_path,
        schema=replace(
            source_config.schema,
            path="conf/schemas/inep/censo_escolar_microdados_schema.json",
        ),
        access=replace(
            source_config.access,
            url=None,
            path=str(archive_path),
        ),
    )


def _fixture_csv() -> str:
    return (FIXTURES_DIR / "microdados_ed_basica_2024_sample.csv").read_text(
        encoding="utf-8"
    )


def _storage_layout(project_root: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "data",
                "raw_dir": "data/raw",
                "bronze_dir": "data/bronze",
                "metadata_dir": "data/metadata",
            }
        },
        project_root,
    )
