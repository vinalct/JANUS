from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from janus.lineage import RunObserver
from janus.models import RunContext, SourceConfig
from janus.normalizers import BaseNormalizer
from janus.quality import QualityGate, load_expected_fields_from_schema_path
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SparkDatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transparencia"
SOURCE_ID = "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)


@dataclass(slots=True)
class FixtureTransport:
    fixture_paths: list[Path]
    requests: list = field(default_factory=list)
    opened: bool = False
    closed: bool = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request):
        self.requests.append(request)
        if not self.fixture_paths:
            raise AssertionError("No fixture response remains for the Transparencia transport")

        fixture_path = self.fixture_paths.pop(0)
        return ApiResponse(
            request=request,
            status_code=200,
            body=fixture_path.read_bytes(),
        )


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    pyspark_sql = pytest.importorskip("pyspark.sql")
    if not ICEBERG_RUNTIME_JAR.exists():
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    warehouse_root = tmp_path_factory.mktemp("janus-transparencia-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-transparencia-integration")
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


def test_transparencia_source_contract_uses_safe_api_settings():
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    schema_path = PROJECT_ROOT / source_config.schema.path

    assert source_config.enabled is False
    assert source_config.strategy == "api"
    assert source_config.strategy_variant == "page_number_api"
    assert source_config.source_hook is None
    assert source_config.access.base_url == "https://api.portaldatransparencia.gov.br"
    assert source_config.access.path == "/api-de-dados/servidores/por-orgao"
    assert source_config.access.auth.type == "header_token"
    assert source_config.access.auth.header_name == "chave-api-dados"
    assert source_config.access.pagination.page_size == 15
    assert source_config.access.rate_limit.requests_per_minute == 90
    assert source_config.access.rate_limit.concurrency == 10
    assert source_config.extraction.mode == "full_refresh"
    assert source_config.extraction.checkpoint_strategy == "none"
    assert source_config.outputs.bronze.format == "iceberg"
    assert load_expected_fields_from_schema_path(schema_path) == (
        "qntPessoas",
        "qntVinculos",
        "skSituacao",
        "descSituacao",
        "skTipoVinculo",
        "descTipoVinculo",
        "skTipoServidor",
        "descTipoServidor",
        "licenca",
        "codOrgaoExercicioSiape",
        "nomOrgaoExercicioSiape",
        "codOrgaoSuperiorExercicioSiape",
        "nomOrgaoSuperiorExercicioSiape",
    )


def test_transparencia_source_extracts_raw_pages_and_materializes_bronze_and_metadata(
    spark,
    tmp_path,
    monkeypatch,
):
    source_config = _cloned_source_config(tmp_path, page_size=2)
    run_context = RunContext.create(
        run_id="run-transparencia-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 15, 0, tzinfo=UTC),
    )
    transport = FixtureTransport(
        fixture_paths=[
            FIXTURES_DIR / "servidores_por_orgao_page_1.json",
            FIXTURES_DIR / "servidores_por_orgao_page_2.json",
        ]
    )
    storage_layout = _storage_layout(tmp_path)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    monkeypatch.setenv("TRANSPARENCIA_API_TOKEN", "super-secret-token")

    plan = strategy.plan(source_config, run_context)
    extraction_result = strategy.extract(plan)

    assert plan.notes == (
        "strategy_family:api",
        "strategy_variant:page_number_api",
    )
    assert transport.opened is True
    assert transport.closed is True
    assert extraction_result.records_extracted == 3
    assert extraction_result.checkpoint_value is None

    first_request = transport.requests[0]
    second_request = transport.requests[1]
    first_query = parse_qs(urlsplit(first_request.full_url()).query)
    second_query = parse_qs(urlsplit(second_request.full_url()).query)

    assert first_request.headers_as_dict()["chave-api-dados"] == "super-secret-token"
    assert "super-secret-token" not in first_request.full_url()
    assert first_query["pagina"] == ["1"]
    assert first_query["tamanhoPagina"] == ["2"]
    assert second_query["pagina"] == ["2"]
    assert second_query["tamanhoPagina"] == ["2"]

    raw_target = storage_layout.resolve_output(plan, "raw")
    raw_run_prefix = "runs/ingestion_date=2026-04-09/run_id=run-transparencia-001"
    first_artifact = Path(extraction_result.artifacts[0].path)
    second_artifact = Path(extraction_result.artifacts[1].path)
    assert first_artifact == raw_target.child(f"{raw_run_prefix}/pages/page-0001.json")
    assert second_artifact == raw_target.child(f"{raw_run_prefix}/pages/page-0002.json")
    assert json.loads(first_artifact.read_text(encoding="utf-8")) == _load_fixture(
        "servidores_por_orgao_page_1.json"
    )
    assert json.loads(second_artifact.read_text(encoding="utf-8")) == _load_fixture(
        "servidores_por_orgao_page_2.json"
    )

    reader = SparkDatasetReader()
    raw_dataframe = reader.read_extraction_result(spark, extraction_result)
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
    assert emitted_metadata["artifact_count"] == 2
    assert emitted_metadata["auth_type"] == "header_token"
    assert emitted_metadata["pagination_type"] == "page_number"
    assert emitted_metadata["request_count"] == "2"
    assert emitted_metadata["write_result_count"] == 1

    observer = RunObserver()
    started = observer.start_run(plan)
    persisted = observer.record_success(
        plan,
        extraction_result,
        (bronze_result,),
        finished_at=datetime(2026, 4, 9, 15, 5, tzinfo=UTC),
    )

    bronze_dataframe = spark.table(bronze_result.path)
    assert bronze_dataframe.count() == 3
    assert bronze_result.path == bronze_table_identifier(
        plan.bronze_output.path,
        fallback_name=plan.source.source_id,
        namespace=plan.bronze_output.namespace,
        table_name=plan.bronze_output.table_name,
    )

    run_payload = json.loads(started.run_metadata_path.read_text(encoding="utf-8"))
    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))

    metadata_target = storage_layout.resolve_output(plan, "metadata")
    assert started.run_metadata_path == metadata_target.child("runs/run-transparencia-001.json")
    assert run_payload["status"] == "succeeded"
    assert run_payload["records_extracted"] == 3
    assert lineage_payload["status"] == "succeeded"
    assert lineage_payload["strategy_variant"] == "page_number_api"
    assert lineage_payload["artifacts"][0]["path"].endswith("page-0001.json")
    assert persisted.checkpoint_result is not None
    assert persisted.checkpoint_result.decision == "skipped"


def _cloned_source_config(tmp_path: Path, *, page_size: int) -> SourceConfig:
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    config_path = PROJECT_ROOT / source_config.config_path.relative_to(PROJECT_ROOT)
    schema_path = PROJECT_ROOT / source_config.schema.path

    copied_config_path = tmp_path / source_config.config_path.relative_to(PROJECT_ROOT)
    copied_schema_path = tmp_path / source_config.schema.path
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)
    copied_schema_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "sources" in config_payload:
        for entry in config_payload["sources"]:
            if entry.get("source_id") == SOURCE_ID:
                entry["access"]["pagination"]["page_size"] = page_size
                break
        else:
            raise AssertionError(f"Source {SOURCE_ID!r} was not found in the copied config")
    else:
        config_payload["access"]["pagination"]["page_size"] = page_size

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
            path=str(source_config.schema.path),
        ),
        access=replace(
            source_config.access,
            pagination=replace(source_config.access.pagination, page_size=page_size),
        ),
    )


def _load_fixture(file_name: str):
    return json.loads((FIXTURES_DIR / file_name).read_text(encoding="utf-8"))


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
