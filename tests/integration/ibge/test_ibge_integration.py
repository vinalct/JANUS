from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from janus.checkpoints import CheckpointStore
from janus.hooks import built_in_hooks
from janus.hooks.ibge import (
    HOOK_ID,
    LEGACY_HOOK_IDS,
    IbgeSidraFlatHook,
    normalize_sidra_flat_payload,
)
from janus.lineage import RunObserver
from janus.models import RunContext, SourceConfig
from janus.normalizers import BaseNormalizer
from janus.planner import Planner, PlanningRequest
from janus.quality import QualityGate
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SparkDatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ibge"
PIB_SOURCE_ID = "ibge_pib_brasil"
PIB_CONFIG_FILE = "ibge/sidra.yaml"
AGRO_SOURCE_ID = "ibge_agro_abacaxi_pronaf"
AGRO_CONFIG_FILE = "ibge/sidra.yaml"
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
            raise AssertionError("No fixture response remains for the IBGE transport")

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
    warehouse_root = tmp_path_factory.mktemp("janus-ibge-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-ibge-integration")
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


def test_ibge_hook_catalog_registers_generic_and_legacy_sidra_ids():
    bindings = dict(built_in_hooks())

    assert HOOK_ID == "ibge.sidra_flat"
    assert LEGACY_HOOK_IDS == ("ibge.pib_brasil",)
    assert HOOK_ID in bindings
    assert LEGACY_HOOK_IDS[0] in bindings


def test_ibge_pib_source_contract_uses_generic_sidra_hook_without_explicit_schema():
    source_config = load_registry(PROJECT_ROOT).get_source(PIB_SOURCE_ID, include_disabled=True)

    _assert_generic_sidra_source_contract(
        source_config,
        expected_source_id=PIB_SOURCE_ID,
        expected_path="/api/v3/agregados/5938/periodos/2023/variaveis/37",
        expected_params={"localidades": "N1[all]", "view": "flat"},
        expected_bronze_path="data/bronze/ibge/pib_brasil",
    )

    planned_run = _planned_run_for(PIB_SOURCE_ID)
    assert planned_run.plan.notes == (
        "strategy_family:api",
        "strategy_variant:date_window_api",
        "hook:ibge.sidra_flat",
        "dispatch:api.date_window_api",
    )
    assert planned_run.pre_run_metadata_as_dict()["hook_id"] == HOOK_ID


def test_ibge_agro_source_contract_uses_generic_sidra_hook_with_more_dimensions():
    source_config = load_registry(PROJECT_ROOT).get_source(AGRO_SOURCE_ID, include_disabled=True)

    _assert_generic_sidra_source_contract(
        source_config,
        expected_source_id=AGRO_SOURCE_ID,
        expected_path="/api/v3/agregados/1712/periodos/2006/variaveis/214",
        expected_params={
            "localidades": "N1[all]",
            "classificacao": "226[4844]|218[4780]",
            "view": "flat",
        },
        expected_bronze_path="data/bronze/ibge/agro_abacaxi_pronaf",
    )

    planned_run = _planned_run_for(AGRO_SOURCE_ID)
    assert planned_run.plan.notes == (
        "strategy_family:api",
        "strategy_variant:date_window_api",
        "hook:ibge.sidra_flat",
        "dispatch:api.date_window_api",
    )
    assert planned_run.pre_run_metadata_as_dict()["hook_id"] == HOOK_ID


def test_ibge_sidra_normalization_uses_header_row_for_optional_columns():
    normalized_records = normalize_sidra_flat_payload(
        _load_fixture("sidra_flat_without_nc_nn.json"),
        aggregate_id="9999",
    )

    assert normalized_records == (
        {
            "aggregate_id": "9999",
            "sidra_attribute_count": 0,
            "sidra_attributes": (),
            "sidra_dimension_count": 4,
            "sidra_dimensions": (
                {
                    "code": "3550308",
                    "code_label": "Território (Código)",
                    "id": "D1",
                    "label": "Território",
                    "name": "São Paulo",
                    "position": 1,
                },
                {
                    "code": "2024",
                    "code_label": "Ano (Código)",
                    "id": "D2",
                    "label": "Ano",
                    "name": "2024",
                    "position": 2,
                },
                {
                    "code": "93",
                    "code_label": "Indicador (Código)",
                    "id": "D3",
                    "label": "Indicador",
                    "name": "População residente",
                    "position": 3,
                },
                {
                    "code": "1",
                    "code_label": "Sexo (Código)",
                    "id": "D4",
                    "label": "Sexo",
                    "name": "Homens",
                    "position": 4,
                },
            ),
            "sidra_period_code": "2024",
            "sidra_period_name": "2024",
            "sidra_value_label": "Valor",
            "sidra_variable_code": "93",
            "sidra_variable_name": "População residente",
            "value": "123456",
        },
    )
    assert "territorial_level_code" not in normalized_records[0]
    assert "unit_code" not in normalized_records[0]


def test_ibge_pib_source_extracts_raw_response_and_normalizes_dynamic_dimensions(
    spark,
    tmp_path,
):
    run = _extract_with_fixture(
        tmp_path,
        source_id=PIB_SOURCE_ID,
        config_file_name=PIB_CONFIG_FILE,
        fixture_name="pib_brasil_2023_flat.json",
        run_id="run-ibge-pib-001",
        started_at=datetime(2026, 4, 11, 12, 0, tzinfo=UTC),
        previous_checkpoint="2022",
    )

    assert run["plan"].notes == (
        "strategy_family:api",
        "strategy_variant:date_window_api",
        "hook:ibge.sidra_flat",
    )
    assert run["transport"].opened is True
    assert run["transport"].closed is True
    assert run["extraction_result"].records_extracted == 1
    assert run["extraction_result"].checkpoint_value == "2023"

    request = run["transport"].requests[0]
    query = parse_qs(urlsplit(request.full_url()).query)
    assert request.headers_as_dict()["Accept"] == "application/json"
    assert query == {"localidades": ["N1[all]"], "view": ["flat"]}

    raw_artifact = Path(run["extraction_result"].artifacts[0].path)
    normalized_artifact = Path(run["handoff"].artifacts[0].path)
    assert raw_artifact == (
        tmp_path
        / "data"
        / "raw"
        / "ibge"
        / "pib_brasil"
        / "runs"
        / "ingestion_date=2026-04-11"
        / "run_id=run-ibge-pib-001"
        / "pages"
        / "response-0001.json"
    )
    assert normalized_artifact == (
        tmp_path
        / "data"
        / "raw"
        / "ibge"
        / "pib_brasil"
        / "normalized"
        / "sidra_records.jsonl"
    )
    assert json.loads(raw_artifact.read_text(encoding="utf-8")) == _load_fixture(
        "pib_brasil_2023_flat.json"
    )

    normalized_records = _read_jsonl(normalized_artifact)
    assert normalized_records == [_expected_pib_normalized_record()]
    assert run["handoff"].records_extracted == 1
    assert run["handoff"].metadata_as_dict()["ibge_normalized_record_count"] == "1"
    assert run["handoff"].metadata_as_dict()["ibge_sidra_attribute_keys"] == "MC,MN,NC,NN"
    assert run["handoff"].metadata_as_dict()["ibge_sidra_dimension_ids"] == "D1,D2,D3"

    reader = SparkDatasetReader()
    raw_dataframe = reader.read_extraction_result(spark, run["handoff"])
    normalized_dataframe = BaseNormalizer().normalize(raw_dataframe, run["plan"])
    bronze_result = SparkDatasetWriter(run["storage_layout"]).write(
        normalized_dataframe,
        run["plan"],
        "bronze",
        count_records=True,
    )

    report = QualityGate().validate(
        run["plan"],
        dataframe=normalized_dataframe,
        write_results=(bronze_result,),
    )
    assert report.is_successful is True

    emitted_metadata = run["strategy"].emit_metadata(
        run["plan"],
        run["extraction_result"],
        (bronze_result,),
        hook=run["hook"],
    )
    assert emitted_metadata["strategy_family"] == "api"
    assert emitted_metadata["strategy_variant"] == "date_window_api"
    assert emitted_metadata["auth_type"] == "none"
    assert emitted_metadata["pagination_type"] == "none"
    assert emitted_metadata["ibge_hook"] == HOOK_ID
    assert emitted_metadata["ibge_aggregate_id"] == "5938"
    assert emitted_metadata["ibge_customization"] == "header_driven_sidra_projection"
    assert emitted_metadata["ibge_sidra_attribute_keys"] == "MC,MN,NC,NN"
    assert emitted_metadata["ibge_sidra_dimension_ids"] == "D1,D2,D3"
    assert emitted_metadata["write_result_count"] == 1

    observer = RunObserver()
    started = observer.start_run(run["plan"])
    persisted = observer.record_success(
        run["plan"],
        run["extraction_result"],
        (bronze_result,),
        finished_at=datetime(2026, 4, 11, 12, 5, tzinfo=UTC),
    )

    bronze_dataframe = spark.table(bronze_result.path)
    row = bronze_dataframe.select(
        "aggregate_id",
        "sidra_period_code",
        "sidra_variable_code",
        "sidra_dimension_count",
        "sidra_attribute_count",
        "value",
    ).first()

    assert bronze_result.records_written == 1
    assert row.asDict() == {
        "aggregate_id": "5938",
        "sidra_period_code": "2023",
        "sidra_variable_code": "37",
        "sidra_dimension_count": 3,
        "sidra_attribute_count": 4,
        "value": "10943345439",
    }
    assert bronze_result.path == bronze_table_identifier(
        run["plan"].bronze_output.path,
        fallback_name=run["plan"].source.source_id,
        namespace=run["plan"].bronze_output.namespace,
        table_name=run["plan"].bronze_output.table_name,
    )

    run_payload = json.loads(started.run_metadata_path.read_text(encoding="utf-8"))
    lineage_payload = json.loads(persisted.lineage_path.read_text(encoding="utf-8"))

    assert started.run_metadata_path == (
        tmp_path
        / "data"
        / "metadata"
        / "ibge"
        / "pib_brasil"
        / "runs"
        / "run-ibge-pib-001.json"
    )
    assert run_payload["status"] == "succeeded"
    assert run_payload["records_extracted"] == 1
    assert lineage_payload["status"] == "succeeded"
    assert lineage_payload["strategy_variant"] == "date_window_api"
    assert lineage_payload["source_hook"] == HOOK_ID
    assert persisted.checkpoint_result is not None
    assert persisted.checkpoint_result.decision == "advanced"
    assert persisted.checkpoint_result.state is not None
    assert persisted.checkpoint_result.state.checkpoint_value == "2023"


def test_ibge_agro_source_extracts_alternate_dimension_layout(tmp_path):
    run = _extract_with_fixture(
        tmp_path,
        source_id=AGRO_SOURCE_ID,
        config_file_name=AGRO_CONFIG_FILE,
        fixture_name="agro_abacaxi_pronaf_2006_flat.json",
        run_id="run-ibge-agro-001",
        started_at=datetime(2026, 4, 11, 13, 0, tzinfo=UTC),
        previous_checkpoint="2005",
    )

    assert run["plan"].notes == (
        "strategy_family:api",
        "strategy_variant:date_window_api",
        "hook:ibge.sidra_flat",
    )
    assert run["transport"].opened is True
    assert run["transport"].closed is True
    assert run["extraction_result"].records_extracted == 1
    assert run["extraction_result"].checkpoint_value == "2006"

    request = run["transport"].requests[0]
    query = parse_qs(urlsplit(request.full_url()).query)
    assert request.headers_as_dict()["Accept"] == "application/json"
    assert query == {
        "classificacao": ["226[4844]|218[4780]"],
        "localidades": ["N1[all]"],
        "view": ["flat"],
    }

    raw_artifact = Path(run["extraction_result"].artifacts[0].path)
    normalized_artifact = Path(run["handoff"].artifacts[0].path)
    assert raw_artifact == (
        tmp_path
        / "data"
        / "raw"
        / "ibge"
        / "agro_abacaxi_pronaf"
        / "runs"
        / "ingestion_date=2026-04-11"
        / "run_id=run-ibge-agro-001"
        / "pages"
        / "response-0001.json"
    )
    assert normalized_artifact == (
        tmp_path
        / "data"
        / "raw"
        / "ibge"
        / "agro_abacaxi_pronaf"
        / "normalized"
        / "sidra_records.jsonl"
    )
    assert json.loads(raw_artifact.read_text(encoding="utf-8")) == _load_fixture(
        "agro_abacaxi_pronaf_2006_flat.json"
    )

    normalized_records = _read_jsonl(normalized_artifact)
    assert normalized_records == [_expected_agro_normalized_record()]
    assert run["handoff"].records_extracted == 1
    assert run["handoff"].metadata_as_dict()["ibge_normalized_record_count"] == "1"
    assert run["handoff"].metadata_as_dict()["ibge_sidra_attribute_keys"] == "MC,MN,NC,NN"
    assert run["handoff"].metadata_as_dict()["ibge_sidra_dimension_ids"] == (
        "D1,D2,D3,D4,D5,D6,D7,D8,D9"
    )


def _assert_generic_sidra_source_contract(
    source_config: SourceConfig,
    *,
    expected_source_id: str,
    expected_path: str,
    expected_params: dict[str, str],
    expected_bronze_path: str,
) -> None:
    assert source_config.enabled is False
    assert source_config.source_id == expected_source_id
    assert source_config.strategy == "api"
    assert source_config.strategy_variant == "date_window_api"
    assert source_config.source_hook == HOOK_ID
    assert source_config.access.base_url == "https://servicodados.ibge.gov.br"
    assert source_config.access.path == expected_path
    assert source_config.access.auth.type == "none"
    assert source_config.access.params == expected_params
    assert source_config.access.pagination.type == "none"
    assert source_config.access.rate_limit.requests_per_minute == 30
    assert source_config.access.rate_limit.concurrency == 1
    assert source_config.extraction.mode == "incremental"
    assert source_config.extraction.checkpoint_field == "sidra_period_code"
    assert source_config.extraction.checkpoint_strategy == "max_value"
    assert source_config.schema.mode == "infer"
    assert source_config.schema.path is None
    assert source_config.spark.input_format == "jsonl"
    assert source_config.outputs.bronze.path == expected_bronze_path
    assert source_config.quality.required_fields == (
        "aggregate_id",
        "value",
        "sidra_dimension_count",
        "sidra_attribute_count",
    )


def _planned_run_for(source_id: str):
    return Planner().plan(
        PlanningRequest.create(
            source_id=source_id,
            environment="local",
            project_root=PROJECT_ROOT,
            run_id=f"run-{source_id}-plan-001",
            started_at=datetime(2026, 4, 11, 10, 0, tzinfo=UTC),
            include_disabled=True,
        )
    )


def _extract_with_fixture(
    tmp_path: Path,
    *,
    source_id: str,
    config_file_name: str,
    fixture_name: str,
    run_id: str,
    started_at: datetime,
    previous_checkpoint: str,
):
    source_config = _cloned_source_config(tmp_path, source_id, config_file_name)
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
    )
    transport = FixtureTransport(fixture_paths=[FIXTURES_DIR / fixture_name])
    storage_layout = _storage_layout(tmp_path)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    hook = IbgeSidraFlatHook()

    plan = strategy.plan(source_config, run_context, hook=hook)
    CheckpointStore().save(plan, previous_checkpoint)
    extraction_result = strategy.extract(plan, hook=hook)
    handoff = strategy.build_normalization_handoff(plan, extraction_result, hook=hook)

    return {
        "extraction_result": extraction_result,
        "handoff": handoff,
        "hook": hook,
        "plan": plan,
        "storage_layout": storage_layout,
        "strategy": strategy,
        "transport": transport,
    }


def _expected_pib_normalized_record() -> dict:
    return {
        "aggregate_id": "5938",
        "sidra_attribute_count": 4,
        "sidra_attributes": [
            {"key": "MC", "label": "Unidade de Medida (Código)", "value": "40"},
            {"key": "MN", "label": "Unidade de Medida", "value": "Mil Reais"},
            {"key": "NC", "label": "Nível Territorial (Código)", "value": "1"},
            {"key": "NN", "label": "Nível Territorial", "value": "Brasil"},
        ],
        "sidra_dimension_count": 3,
        "sidra_dimensions": [
            {
                "code": "1",
                "code_label": "Brasil (Código)",
                "id": "D1",
                "label": "Brasil",
                "name": "Brasil",
                "position": 1,
            },
            {
                "code": "2023",
                "code_label": "Ano (Código)",
                "id": "D2",
                "label": "Ano",
                "name": "2023",
                "position": 2,
            },
            {
                "code": "37",
                "code_label": "Variável (Código)",
                "id": "D3",
                "label": "Variável",
                "name": "Produto Interno Bruto a preços correntes",
                "position": 3,
            },
        ],
        "sidra_period_code": "2023",
        "sidra_period_name": "2023",
        "sidra_value_label": "Valor",
        "sidra_variable_code": "37",
        "sidra_variable_name": "Produto Interno Bruto a preços correntes",
        "value": "10943345439",
    }


def _expected_agro_normalized_record() -> dict:
    return {
        "aggregate_id": "1712",
        "sidra_attribute_count": 4,
        "sidra_attributes": [
            {"key": "MC", "label": "Unidade de Medida (Código)", "value": "31"},
            {"key": "MN", "label": "Unidade de Medida", "value": "Mil frutos"},
            {"key": "NC", "label": "Nível Territorial (Código)", "value": "1"},
            {"key": "NN", "label": "Nível Territorial", "value": "Brasil"},
        ],
        "sidra_dimension_count": 9,
        "sidra_dimensions": [
            {
                "code": "1",
                "code_label": "Brasil (Código)",
                "id": "D1",
                "label": "Brasil",
                "name": "Brasil",
                "position": 1,
            },
            {
                "code": "2006",
                "code_label": "Ano (Código)",
                "id": "D2",
                "label": "Ano",
                "name": "2006",
                "position": 2,
            },
            {
                "code": "214",
                "code_label": "Variável (Código)",
                "id": "D3",
                "label": "Variável",
                "name": "Quantidade produzida",
                "position": 3,
            },
            {
                "code": "4844",
                "code_label": "Produtos da lavoura temporária (Código)",
                "id": "D4",
                "label": "Produtos da lavoura temporária",
                "name": "Abacaxi",
                "position": 4,
            },
            {
                "code": "4780",
                "code_label": "Condição do produtor em relação às terras (Código)",
                "id": "D5",
                "label": "Condição do produtor em relação às terras",
                "name": "Proprietário",
                "position": 5,
            },
            {
                "code": "0",
                "code_label": "Grupos de área total (Código)",
                "id": "D6",
                "label": "Grupos de área total",
                "name": "Total",
                "position": 6,
            },
            {
                "code": "0",
                "code_label": "Grupos de atividade econômica (Código)",
                "id": "D7",
                "label": "Grupos de atividade econômica",
                "name": "Total",
                "position": 7,
            },
            {
                "code": "0",
                "code_label": "Grupos de área colhida (Código)",
                "id": "D8",
                "label": "Grupos de área colhida",
                "name": "Total",
                "position": 8,
            },
            {
                "code": "0",
                "code_label": "Pronafiano (Código)",
                "id": "D9",
                "label": "Pronafiano",
                "name": "Total",
                "position": 9,
            },
        ],
        "sidra_period_code": "2006",
        "sidra_period_name": "2006",
        "sidra_value_label": "Valor",
        "sidra_variable_code": "214",
        "sidra_variable_name": "Quantidade produzida",
        "value": "456692",
    }


def _cloned_source_config(tmp_path: Path, source_id: str, config_file_name: str) -> SourceConfig:
    source_config = load_registry(PROJECT_ROOT).get_source(source_id, include_disabled=True)
    config_path = PROJECT_ROOT / "conf" / "sources" / config_file_name

    copied_config_path = tmp_path / "conf" / "sources" / config_file_name
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "sources" in config_payload:
        config_payload = next(
            entry
            for entry in config_payload["sources"]
            if entry.get("source_id") == source_id
        )
    copied_config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    return replace(
        source_config,
        config_path=copied_config_path,
        schema=replace(source_config.schema, mode="infer", path=None),
        extraction=replace(source_config.extraction, checkpoint_field="sidra_period_code"),
        quality=replace(
            source_config.quality,
            required_fields=(
                "aggregate_id",
                "value",
                "sidra_dimension_count",
                "sidra_attribute_count",
            ),
        ),
    )


def _load_fixture(file_name: str):
    return json.loads((FIXTURES_DIR / file_name).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
