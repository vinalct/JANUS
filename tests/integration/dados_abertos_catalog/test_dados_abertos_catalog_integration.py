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
from janus.strategies.api import ApiResponse
from janus.strategies.catalog import CatalogStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SparkDatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "dados_abertos_catalog"
SOURCE_ID = "dados_abertos_catalog__conjunto_dados__full_refresh"
SOURCE_CONFIG_PATH = (
    PROJECT_ROOT / "conf" / "sources" / "dados_abertos_catalog" / "conjunto_de_dados.yaml"
)
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)
SCHEMA_FIELDS = (
    "entity_type",
    "entity_key",
    "entity_id",
    "parent_entity_type",
    "parent_entity_key",
    "parent_entity_id",
    "catalog_collection_path",
    "catalog_record_path",
    "catalog_request_url",
    "catalog_request_index",
    "catalog_page_number",
    "catalog_offset",
    "catalog_cursor",
    "catalog_received_at",
    "catalog_raw_artifact_path",
    "payload",
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
            raise AssertionError("No fixture response remains for the dados.gov.br transport")

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
    warehouse_root = tmp_path_factory.mktemp("janus-dados-abertos-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-dados-abertos-catalog-integration")
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


def test_dados_abertos_catalog_source_contract_uses_generic_catalog_strategy():
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    schema_path = PROJECT_ROOT / source_config.schema.path

    assert source_config.enabled is False
    assert source_config.strategy == "catalog"
    assert source_config.strategy_variant == "metadata_catalog"
    assert source_config.source_hook is None
    assert source_config.access.base_url == "https://dados.gov.br"
    assert source_config.access.path == "/dados/api/publico/conjuntos-dados"
    assert source_config.access.auth.type == "header_token"
    assert source_config.access.auth.env_var == "DADOS_GOV_BR_API_TOKEN"
    assert source_config.access.auth.header_name == "chave-api-dados-abertos"
    assert source_config.access.params == {"dadosAbertos": "true", "isPrivado": "false"}
    assert source_config.access.pagination.type == "page_number"
    assert source_config.access.pagination.page_param == "pagina"
    assert source_config.access.pagination.size_param == "tamanhoPagina"
    assert source_config.access.pagination.page_size == 15
    assert source_config.extraction.mode == "full_refresh"
    assert source_config.extraction.checkpoint_field is None
    assert source_config.extraction.checkpoint_strategy == "none"
    assert source_config.spark.input_format == "jsonl"
    assert source_config.outputs.bronze.format == "iceberg"
    assert source_config.outputs.bronze.path == "data/bronze/dados_abertos/catalog"
    assert load_expected_fields_from_schema_path(schema_path) == SCHEMA_FIELDS


def test_dados_abertos_catalog_extracts_catalog_entities_and_materializes_bronze(
    spark,
    tmp_path,
    monkeypatch,
):
    source_config = _cloned_source_config(tmp_path, page_size=2)
    run_context = RunContext.create(
        run_id="run-dados-abertos-catalog-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    transport = FixtureTransport(
        fixture_paths=[
            FIXTURES_DIR / "package_search_page_1.json",
            FIXTURES_DIR / "package_search_page_2.json",
        ]
    )
    storage_layout = _storage_layout(tmp_path)
    strategy = CatalogStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    monkeypatch.setenv("DADOS_GOV_BR_API_TOKEN", "catalog-token")

    plan = strategy.plan(source_config, run_context)
    extraction_result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, extraction_result)
    metadata = extraction_result.metadata_as_dict()

    assert plan.notes == (
        "strategy_family:catalog",
        "strategy_variant:metadata_catalog",
    )
    assert transport.opened is True
    assert transport.closed is True
    assert len(transport.requests) == 2
    assert extraction_result.records_extracted == 12
    assert extraction_result.checkpoint_value is None
    assert metadata["request_count"] == "2"
    assert metadata["raw_page_count"] == "2"
    assert metadata["organizations_extracted"] == "2"
    assert metadata["groups_extracted"] == "3"
    assert metadata["datasets_extracted"] == "3"
    assert metadata["resources_extracted"] == "4"
    assert metadata["entity_types_emitted"] == "organization,group,dataset,resource"

    first_request = transport.requests[0]
    second_request = transport.requests[1]
    first_query = parse_qs(urlsplit(first_request.full_url()).query)
    second_query = parse_qs(urlsplit(second_request.full_url()).query)

    assert first_request.headers_as_dict()["chave-api-dados-abertos"] == "catalog-token"
    assert "catalog-token" not in first_request.full_url()
    assert first_query["dadosAbertos"] == ["true"]
    assert first_query["isPrivado"] == ["false"]
    assert first_query["pagina"] == ["1"]
    assert first_query["tamanhoPagina"] == ["2"]
    assert second_query["pagina"] == ["2"]
    assert second_query["tamanhoPagina"] == ["2"]

    raw_run_dir = (
        tmp_path
        / "data"
        / "raw"
        / "dados_abertos"
        / "catalog"
        / "runs"
        / "ingestion_date=2026-04-10"
        / "run_id=run-dados-abertos-catalog-001"
    )
    raw_page_one = Path(extraction_result.artifacts[0].path)
    raw_page_two = Path(extraction_result.artifacts[1].path)
    assert raw_page_one == raw_run_dir / "pages" / "page-0001.json"
    assert raw_page_two == raw_run_dir / "pages" / "page-0002.json"
    assert json.loads(raw_page_one.read_text(encoding="utf-8")) == _load_fixture(
        "package_search_page_1.json"
    )
    assert json.loads(raw_page_two.read_text(encoding="utf-8")) == _load_fixture(
        "package_search_page_2.json"
    )

    assert [Path(artifact.path).name for artifact in handoff.artifacts] == [
        "organizations.jsonl",
        "groups.jsonl",
        "datasets.jsonl",
        "resources.jsonl",
    ]

    resource_records = _read_jsonl(raw_run_dir / "normalized" / "resources.jsonl")
    dataset_records = _read_jsonl(raw_run_dir / "normalized" / "datasets.jsonl")
    organization_records = _read_jsonl(raw_run_dir / "normalized" / "organizations.jsonl")

    assert [record["entity_id"] for record in dataset_records] == [
        "dataset-orcamento-federal",
        "dataset-compras-publicas",
        "dataset-educacao-basica",
    ]
    assert [record["entity_id"] for record in organization_records] == [
        "org-planejamento",
        "org-gestao",
    ]
    assert {record["parent_entity_id"] for record in resource_records} == {
        "dataset-orcamento-federal",
        "dataset-compras-publicas",
        "dataset-educacao-basica",
    }
    assert all(record["parent_entity_type"] == "dataset" for record in resource_records)
    assert resource_records[0]["catalog_raw_artifact_path"] == str(raw_page_one)

    reader = SparkDatasetReader()
    raw_dataframe = reader.read_extraction_result(spark, handoff)
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
    assert emitted_metadata["artifact_count"] == 6
    assert emitted_metadata["strategy_family"] == "catalog"
    assert emitted_metadata["strategy_variant"] == "metadata_catalog"
    assert emitted_metadata["input_format"] == "jsonl"
    assert emitted_metadata["write_result_count"] == 1

    observer = RunObserver()
    started = observer.start_run(plan)
    persisted = observer.record_success(
        plan,
        extraction_result,
        (bronze_result,),
        finished_at=datetime(2026, 4, 10, 12, 5, tzinfo=UTC),
    )

    bronze_dataframe = spark.table(bronze_result.path)
    entity_counts = {
        row["entity_type"]: row["count"]
        for row in bronze_dataframe.groupBy("entity_type").count().collect()
    }

    assert bronze_result.records_written == 12
    assert entity_counts == {
        "dataset": 3,
        "group": 3,
        "organization": 2,
        "resource": 4,
    }
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
        / "dados_abertos"
        / "catalog"
        / "runs"
        / "run-dados-abertos-catalog-001.json"
    )
    assert run_payload["status"] == "succeeded"
    assert run_payload["records_extracted"] == 12
    assert lineage_payload["status"] == "succeeded"
    assert lineage_payload["strategy_variant"] == "metadata_catalog"
    assert lineage_payload["artifacts"][0]["path"].endswith("page-0001.json")
    assert persisted.checkpoint_result is not None
    assert persisted.checkpoint_result.decision == "skipped"


def _cloned_source_config(tmp_path: Path, *, page_size: int) -> SourceConfig:
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    schema_path = (
        PROJECT_ROOT
        / "conf"
        / "schemas"
        / "dados_abertos_catalog"
        / "catalog_metadata_schema.json"
    )

    copied_config_path = tmp_path / "conf" / "sources" / "dados_abertos_catalog" / "conjunto_de_dados.yaml"
    copied_schema_path = (
        tmp_path
        / "conf"
        / "schemas"
        / "dados_abertos_catalog"
        / "catalog_metadata_schema.json"
    )
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)
    copied_schema_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    source_entries = config_payload.get("sources") if isinstance(config_payload, dict) else None
    if not isinstance(source_entries, list):
        raise AssertionError("Expected a top-level 'sources' list in conjunto_de_dados.yaml")

    matching_source = next(
        (entry for entry in source_entries if entry.get("source_id") == SOURCE_ID),
        None,
    )
    if matching_source is None:
        raise AssertionError(f"Source {SOURCE_ID!r} not found in conjunto_de_dados.yaml")

    matching_source["access"]["pagination"]["page_size"] = page_size
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
            path="conf/schemas/dados_abertos_catalog/catalog_metadata_schema.json",
        ),
        access=replace(
            source_config.access,
            pagination=replace(source_config.access.pagination, page_size=page_size),
        ),
    )


def _load_fixture(file_name: str):
    return json.loads((FIXTURES_DIR / file_name).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
