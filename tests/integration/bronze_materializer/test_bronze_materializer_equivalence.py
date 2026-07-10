from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from janus.models import RunContext, SourceConfig
from janus.planner import PlannedRun
from janus.registry import load_registry
from janus.runtime import SourceExecutor
from janus.scripts import ingest_raw_to_bronze
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.strategies.files import FileStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRANSPARENCIA_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transparencia"
FILE_SOURCE_ID = "inep_censo_escolar_microdados"
API_SOURCE_ID = "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)
ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}
ARCHIVE_NAME = "microdados_censo_escolar_2024.zip"
ARCHIVE_ROOT = "microdados_censo_escolar_2024"
ARCHIVE_FILE_PATTERN = "*/dados/*.csv"
ARCHIVE_MEMBER_COUNT = 6
ROWS_PER_MEMBER = 2
CSV_COLUMNS = (
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
    warehouse_root = tmp_path_factory.mktemp("janus-bronze-materializer-iceberg")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-bronze-materializer-equivalence")
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


@pytest.fixture(scope="module")
def file_family_runs(spark, tmp_path_factory):
    """Run the live executor and the raw replay once, over one shared raw zone.

    The archive yields six CSV members so the file-family batching is
    observable: batches of five produce two bronze commits, batches of one
    produce six. Both runs share the run context, so the bronze tables must
    match column-for-column.
    """
    project_root = tmp_path_factory.mktemp("janus-file-equivalence")
    archive_path = _build_multi_member_archive(project_root)
    source_config = _cloned_file_source_config(project_root, archive_path=archive_path)
    run_context = RunContext.create(
        run_id="run-bronze-equivalence-file-001",
        environment="local",
        project_root=project_root,
        started_at=datetime(2026, 7, 6, 9, 0, tzinfo=UTC),
    )
    storage_layout = _storage_layout(project_root)
    strategy = FileStrategy(
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    plan = strategy.plan(source_config, run_context)
    planned_run = PlannedRun(plan=plan, strategy=strategy)

    executed = SourceExecutor().execute(planned_run, spark, ENVIRONMENT_CONFIG)
    assert executed.status == "succeeded", executed.failure_reason
    assert executed.strategy_metadata["archive_member_count"] == str(ARCHIVE_MEMBER_COUNT)

    ingested = ingest_raw_to_bronze(
        planned_run,
        spark,
        ENVIRONMENT_CONFIG,
        bronze_table="censo_escolar_microdados_replay",
    )
    assert ingested.status == "succeeded", ingested.failure_reason
    assert len(ingested.handoff.artifacts) == ARCHIVE_MEMBER_COUNT

    return executed, ingested


def test_file_family_replay_bronze_matches_live_rows_schema_and_partitions(
    spark,
    file_family_runs,
):
    executed, ingested = file_family_runs
    live_table = _bronze_table(executed.write_results)
    replay_table = _bronze_table(ingested.write_results)
    assert live_table != replay_table

    live_dataframe = spark.table(live_table)
    replay_dataframe = spark.table(replay_table)

    assert live_dataframe.count() == ARCHIVE_MEMBER_COUNT * ROWS_PER_MEMBER
    assert replay_dataframe.count() == ARCHIVE_MEMBER_COUNT * ROWS_PER_MEMBER
    assert _collected_rows(live_dataframe) == _collected_rows(replay_dataframe)
    assert live_dataframe.schema == replay_dataframe.schema

    live_partitions = {result.partition_by for result in _bronze_results(executed.write_results)}
    replay_partitions = {result.partition_by for result in _bronze_results(ingested.write_results)}
    assert live_partitions == replay_partitions == {("ingestion_date",)}

    assert len(_bronze_results(executed.write_results)) == 2


def test_file_family_replay_bronze_commit_count_matches_live(file_family_runs):
    executed, ingested = file_family_runs

    live_commit_count = len(_bronze_results(executed.write_results))
    replay_commit_count = len(_bronze_results(ingested.write_results))
    assert replay_commit_count == live_commit_count


def test_api_family_replay_produces_identical_bronze(spark, tmp_path, monkeypatch):
    source_config = _cloned_api_source_config(tmp_path, page_size=2)
    run_context = RunContext.create(
        run_id="run-bronze-equivalence-api-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 6, 10, 0, tzinfo=UTC),
    )
    transport = FixtureTransport(
        fixture_paths=[
            TRANSPARENCIA_FIXTURES_DIR / "servidores_por_orgao_page_1.json",
            TRANSPARENCIA_FIXTURES_DIR / "servidores_por_orgao_page_2.json",
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
    planned_run = PlannedRun(plan=plan, strategy=strategy)

    executed = SourceExecutor().execute(planned_run, spark, ENVIRONMENT_CONFIG)
    assert executed.status == "succeeded", executed.failure_reason

    ingested = ingest_raw_to_bronze(
        planned_run,
        spark,
        ENVIRONMENT_CONFIG,
        bronze_table="servidores_por_orgao_replay",
    )
    assert ingested.status == "succeeded", ingested.failure_reason

    live_table = _bronze_table(executed.write_results)
    replay_table = _bronze_table(ingested.write_results)
    assert live_table != replay_table

    live_dataframe = spark.table(live_table)
    replay_dataframe = spark.table(replay_table)

    assert live_dataframe.count() == 3
    assert replay_dataframe.count() == 3
    assert _collected_rows(live_dataframe) == _collected_rows(replay_dataframe)
    assert live_dataframe.schema == replay_dataframe.schema

    live_bronze_results = _bronze_results(executed.write_results)
    replay_bronze_results = _bronze_results(ingested.write_results)
    assert len(live_bronze_results) == len(replay_bronze_results) == 1
    assert live_bronze_results[0].partition_by == replay_bronze_results[0].partition_by


def _bronze_results(write_results):
    return tuple(result for result in write_results if result.zone == "bronze")


def _bronze_table(write_results) -> str:
    tables = {result.path for result in _bronze_results(write_results)}
    assert len(tables) == 1, f"Expected one bronze table, found: {sorted(tables)}"
    return next(iter(tables))


def _collected_rows(dataframe):
    canonical = dataframe.select(*sorted(dataframe.columns))
    return sorted(canonical.collect(), key=repr)


def _build_multi_member_archive(project_root: Path) -> Path:
    archive_path = project_root / "fixtures" / ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, "w") as archive:
        for member_index in range(1, ARCHIVE_MEMBER_COUNT + 1):
            archive.writestr(
                f"{ARCHIVE_ROOT}/dados/microdados_parte_{member_index:02d}.csv",
                _member_csv(member_index),
            )
        archive.writestr(
            f"{ARCHIVE_ROOT}/leia-me.txt",
            "Reduced fixture package for JANUS integration tests.\n",
        )
    return archive_path


def _member_csv(member_index: int) -> str:
    lines = [";".join(CSV_COLUMNS)]
    for row_index in range(1, ROWS_PER_MEMBER + 1):
        lines.append(
            ";".join(
                (
                    "2024",
                    f"11{member_index:02d}{row_index:04d}",
                    f"ESCOLA MUNICIPAL {member_index:02d}-{row_index:02d}",
                    "11",
                    "RO",
                    "1100205",
                    "PORTO VELHO",
                    str((member_index + row_index) % 3 + 1),
                    "1",
                    "1",
                    str(100 * member_index + row_index),
                )
            )
        )
    return "\n".join(lines) + "\n"


def _cloned_file_source_config(project_root: Path, *, archive_path: Path) -> SourceConfig:
    source_config = load_registry(PROJECT_ROOT).get_source(FILE_SOURCE_ID, include_disabled=True)
    config_path = PROJECT_ROOT / "conf" / "sources" / "inep" / "inep.yaml"
    schema_path = (
        PROJECT_ROOT / "conf" / "schemas" / "inep" / "censo_escolar_microdados_schema.json"
    )

    copied_config_path = project_root / "conf" / "sources" / "inep" / "inep.yaml"
    copied_schema_path = (
        project_root / "conf" / "schemas" / "inep" / "censo_escolar_microdados_schema.json"
    )
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)
    copied_schema_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["access"].pop("url", None)
    config_payload["access"]["path"] = str(archive_path)
    config_payload["access"]["file_pattern"] = ARCHIVE_FILE_PATTERN
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
            file_pattern=ARCHIVE_FILE_PATTERN,
        ),
    )


def _cloned_api_source_config(project_root: Path, *, page_size: int) -> SourceConfig:
    source_config = load_registry(PROJECT_ROOT).get_source(API_SOURCE_ID, include_disabled=True)
    config_path = PROJECT_ROOT / source_config.config_path.relative_to(PROJECT_ROOT)
    schema_path = PROJECT_ROOT / source_config.schema.path

    copied_config_path = project_root / source_config.config_path.relative_to(PROJECT_ROOT)
    copied_schema_path = project_root / source_config.schema.path
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)
    copied_schema_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "sources" in config_payload:
        for entry in config_payload["sources"]:
            if entry.get("source_id") == API_SOURCE_ID:
                entry["access"]["pagination"]["page_size"] = page_size
                break
        else:
            raise AssertionError(f"Source {API_SOURCE_ID!r} was not found in the copied config")
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


def _storage_layout(project_root: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, project_root)
