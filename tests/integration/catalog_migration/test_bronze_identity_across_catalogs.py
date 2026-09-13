"""AC-4: the bronze a JDBC catalog produces is the bronze the Hadoop catalog produced."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from janus.models import RunContext, SourceConfig, WriteResult
from janus.planner import PlannedRun
from janus.registry import load_registry
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.environment import (
    HADOOP_CATALOG_TYPE,
    ICEBERG_CATALOG_IMPL,
    JDBC_SCHEMA_VERSION_OPTION,
    build_spark_options,
    load_environment_config,
    prepare_runtime,
)
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IVY_JARS_DIR = PROJECT_ROOT / "data" / "metadata" / "ivy" / "jars"
TRANSPARENCIA_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transparencia"

# The catalogs this session carries. `janus` is the profile's own name and the default;
# the other two exist only for this module.
JDBC_CATALOG = "janus"
HADOOP_BASELINE_CATALOG = "janus_hadoop_baseline"
FRESH_BOOTSTRAP_CATALOG = "janus_fresh_bootstrap"

# ── the capture BASELINE-hadoop-bronze.md recorded, reproduced verbatim ──────
API_SOURCE_ID = "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"
PAGE_SIZE = 2
FIRST_RUN_ID = "run-order13-baseline-001"
FIRST_STARTED_AT = datetime(2026, 7, 6, 10, 0, tzinfo=UTC)
SECOND_RUN_ID = "run-order13-baseline-002"
SECOND_STARTED_AT = datetime(2026, 7, 6, 11, 0, tzinfo=UTC)

BASELINE_TABLE_IDENTIFIER = (
    "bronze__transparencia.poder_executivo_federal__servidores_por_orgao"
)
BASELINE_ROW_COUNT = 3
BASELINE_PARTITION_BY = ("ingestion_date",)
BASELINE_SORTED_ROWS_SHA256 = (
    "ca7ff657f48b9a17e3301adf5089a8a6b74f1e13b7c2bbd394b170f0bbfc8387"
)
BASELINE_PAYLOAD_SHA256 = (
    "e448db3b91bee8e18fdbcda8e49bbea2130faa277f43c1c14c264330dc1929fa"
)
# §8 of the baseline: one `append`, then an `overwrite` — history grows, never resets.
BASELINE_SNAPSHOT_OPERATIONS = ("append", "overwrite")
BASELINE_SECOND_RUN_METADATA = (("overwrite_mechanism", "insert_overwrite"),)

# §7 of the baseline: the raw zone does not depend on the catalog.
BASELINE_RAW_CHECKSUMS = {
    "page-0001.json": "8b039f4eb9b78351bc9dad75b3de22b7376985f02e5ae0e87845d20275f985b5",
    "page-0002.json": "ff835a8ccca706ccf656eeddc2ca5fc03c0cd91a327acae4b474dc011ed3a7d9",
}

# Run-context stamps the normalizer adds; excluded from the cross-run payload digest.
RUN_STAMP_COLUMNS = frozenset(
    {
        "janus_run_id",
        "janus_source_id",
        "janus_source_name",
        "janus_environment",
        "janus_strategy_family",
        "janus_strategy_variant",
        "ingestion_timestamp",
        "ingestion_date",
    }
)

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}

MAX_FIXTURE_REQUESTS = 40


@dataclass
class RequestBudget:
    """A per-run request cap, so a pagination bug fails with a diagnosis, not a CI timeout."""

    limit: int = MAX_FIXTURE_REQUESTS
    lock: threading.Lock = field(default_factory=threading.Lock)
    spent: int = 0

    def spend(self) -> None:
        with self.lock:
            self.spent += 1
            if self.spent > self.limit:
                raise AssertionError(
                    f"Runaway pagination: {self.spent} requests in one run over two fixture "
                    "pages. One transport must serve the whole run — see FixtureTransport."
                )


@dataclass(slots=True)
class FixtureTransport:
    """The equivalence suite's transport: serves the checked-in pages, no network."""

    fixture_paths: list[Path]
    budget: RequestBudget = field(default_factory=RequestBudget)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def send(self, request):
        self.budget.spend()
        with self.lock:
            if not self.fixture_paths:
                raise AssertionError(
                    "No fixture response remains for the Transparencia transport"
                )
            fixture_path = self.fixture_paths.pop(0)
        return ApiResponse(
            request=request, status_code=200, body=fixture_path.read_bytes()
        )


@dataclass(frozen=True)
class BronzeEvidence:
    """Everything about one materialization that AC-4 says must not change."""

    table_identifier: str
    schema_json: dict[str, Any]
    row_count: int
    sorted_rows_sha256: str
    payload_sha256: str
    bronze_result: WriteResult
    raw_results: tuple[WriteResult, ...]
    project_root: Path


def vendored_jar(package: str) -> Path:
    """The seeded jar for a Maven coordinate, in the naming `seed-ivy` and CI use."""

    group, artifact, version = package.split(":")
    return IVY_JARS_DIR / f"{group}_{artifact}-{version}.jar"


@pytest.fixture(scope="module")
def catalogs(tmp_path_factory):
    """One session, three catalogs, built from the checked-in `local.yaml`."""

    pyspark_sql = pytest.importorskip("pyspark.sql")
    patch = pytest.MonkeyPatch()

    for name in [name for name in os.environ if name.startswith("JANUS_")]:
        patch.delenv(name, raising=False)
    patch.setenv("TRANSPARENCIA_API_TOKEN", "token")

    project_root = tmp_path_factory.mktemp("janus-catalog-migration")
    config = load_environment_config("local", PROJECT_ROOT)
    iceberg = config["spark"]["iceberg"]
    assert iceberg["catalog_type"] == "jdbc", "local.yaml must default to the safe catalog"

    iceberg_jar = vendored_jar(iceberg["runtime_package"])
    driver_jar = vendored_jar(iceberg["driver_package"])
    if not iceberg_jar.exists():
        patch.undo()
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    if not driver_jar.exists():
        patch.undo()
        pytest.skip("SQLite JDBC driver jar is not available in the local Ivy cache")

    resolved_paths = prepare_runtime(config, project_root)
    options = dict(build_spark_options(config, resolved_paths))
    options.pop("spark.jars.packages", None)
    options["spark.jars"] = f"{iceberg_jar},{driver_jar}"
    options["spark.sql.shuffle.partitions"] = "1"

    hadoop_root = project_root / "hadoop-baseline"
    fresh_root = project_root / "fresh-bootstrap"
    hadoop_root.mkdir(parents=True, exist_ok=True)
    fresh_root.mkdir(parents=True, exist_ok=True)
    options.update(_hadoop_catalog_options(HADOOP_BASELINE_CATALOG, hadoop_root))
    options.update(_jdbc_catalog_options(FRESH_BOOTSTRAP_CATALOG, fresh_root))

    builder = pyspark_sql.SparkSession.builder.appName(
        "janus-catalog-migration-differential"
    ).master("local[1]")
    for key, value in options.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")

    try:
        yield session, options, tmp_path_factory
    finally:
        session.stop()
        patch.undo()


@pytest.fixture(scope="module")
def differential(catalogs):
    """The same golden fixture, materialized once into each catalog."""

    session, _options, tmp_path_factory = catalogs
    return {
        catalog: materialize_into(
            session,
            catalog,
            tmp_path_factory.mktemp(f"janus-catalog-{catalog}"),
            run_id=FIRST_RUN_ID,
            started_at=FIRST_STARTED_AT,
        )
        for catalog in (JDBC_CATALOG, HADOOP_BASELINE_CATALOG)
    }


# ── the live differential ────────────────────────────────────────────────────


def test_both_catalogs_resolve_the_same_bronze_table_identifier(differential):
    """The writer is catalog-agnostic: the identifier it builds cannot depend on the catalog."""

    jdbc, hadoop = differential[JDBC_CATALOG], differential[HADOOP_BASELINE_CATALOG]

    assert jdbc.table_identifier == hadoop.table_identifier
    assert jdbc.table_identifier == BASELINE_TABLE_IDENTIFIER


def test_both_catalogs_produce_the_same_bronze_schema(differential):
    jdbc, hadoop = differential[JDBC_CATALOG], differential[HADOOP_BASELINE_CATALOG]

    assert jdbc.schema_json == hadoop.schema_json


def test_both_catalogs_produce_the_same_bronze_rows(differential):
    jdbc, hadoop = differential[JDBC_CATALOG], differential[HADOOP_BASELINE_CATALOG]

    assert jdbc.row_count == hadoop.row_count == BASELINE_ROW_COUNT
    assert jdbc.sorted_rows_sha256 == hadoop.sorted_rows_sha256
    assert jdbc.payload_sha256 == hadoop.payload_sha256


def test_both_catalogs_produce_the_same_write_results(differential):
    """Every `WriteResult` field, not a chosen few — the run context is identical too."""

    jdbc, hadoop = differential[JDBC_CATALOG], differential[HADOOP_BASELINE_CATALOG]

    assert jdbc.bronze_result == hadoop.bronze_result
    assert _root_relative_raw_results(jdbc) == _root_relative_raw_results(hadoop)


def test_both_catalogs_reproduce_the_recorded_baseline_raw_checksums(differential):
    """The stronger, root-independent claim: extraction is session-free, so no catalog can
    reach it."""

    for evidence in differential.values():
        checksums = {
            Path(result.path).name: dict(result.metadata).get("checksum")
            for result in evidence.raw_results
        }
        assert checksums == BASELINE_RAW_CHECKSUMS


def test_the_bronze_write_result_still_reports_what_it_reported_before_order_13(
    differential,
):
    """A first write is still a CTAS reported as `overwrite`, with no mechanism metadata."""

    bronze = differential[JDBC_CATALOG].bronze_result

    assert bronze.zone == "bronze"
    assert bronze.path == BASELINE_TABLE_IDENTIFIER
    assert bronze.format == "iceberg"
    assert bronze.mode == "overwrite"
    assert bronze.records_written == BASELINE_ROW_COUNT
    assert bronze.partition_by == BASELINE_PARTITION_BY
    assert bronze.metadata == ()


# ── the recorded baseline: both catalogs ≡ the pre state ────────────


def test_the_jdbc_bronze_reproduces_the_recorded_hadoop_baseline_digests(differential):

    jdbc = differential[JDBC_CATALOG]

    assert jdbc.sorted_rows_sha256 == BASELINE_SORTED_ROWS_SHA256
    assert jdbc.payload_sha256 == BASELINE_PAYLOAD_SHA256


def test_the_live_hadoop_catalog_still_reproduces_its_own_recorded_baseline(differential):
    """If this fails the harness drifted, not the catalog — and the diff above means nothing."""

    hadoop = differential[HADOOP_BASELINE_CATALOG]

    assert hadoop.sorted_rows_sha256 == BASELINE_SORTED_ROWS_SHA256
    assert hadoop.payload_sha256 == BASELINE_PAYLOAD_SHA256



@pytest.fixture(scope="module")
def second_full_refresh(catalogs, differential):
    """A second full refresh over the JDBC catalog's existing table (baseline §8's run 2)."""

    session, _options, tmp_path_factory = catalogs
    return materialize_into(
        session,
        JDBC_CATALOG,
        tmp_path_factory.mktemp("janus-catalog-jdbc-second"),
        run_id=SECOND_RUN_ID,
        started_at=SECOND_STARTED_AT,
    )


def test_a_second_full_refresh_grows_the_snapshot_log_under_the_jdbc_catalog(
    catalogs, second_full_refresh
):
    session, _options, _factory = catalogs
    session.conf.set("spark.sql.defaultCatalog", JDBC_CATALOG)

    operations = tuple(
        row["operation"]
        for row in session.sql(
            f"SELECT operation FROM {BASELINE_TABLE_IDENTIFIER}.snapshots "
            "ORDER BY committed_at"
        ).collect()
    )

    assert operations == BASELINE_SNAPSHOT_OPERATIONS
    assert second_full_refresh.bronze_result.mode == "overwrite"
    assert second_full_refresh.bronze_result.metadata == BASELINE_SECOND_RUN_METADATA

    assert "history_reset_reason" not in dict(second_full_refresh.bronze_result.metadata)


def test_time_travel_to_the_first_snapshot_still_reads_the_first_runs_rows(
    catalogs, differential, second_full_refresh
):
    """The commit coordinator changed; time travel — the reason order-09 exists — did not."""

    session, _options, _factory = catalogs
    session.conf.set("spark.sql.defaultCatalog", JDBC_CATALOG)

    snapshot_ids = [
        row["snapshot_id"]
        for row in session.sql(
            f"SELECT snapshot_id FROM {BASELINE_TABLE_IDENTIFIER}.snapshots "
            "ORDER BY committed_at"
        ).collect()
    ]
    assert len(snapshot_ids) == len(BASELINE_SNAPSHOT_OPERATIONS)

    first_snapshot = (
        session.read.format("iceberg")
        .option("snapshot-id", str(snapshot_ids[0]))
        .load(BASELINE_TABLE_IDENTIFIER)
    )

    count, digest = row_digest(first_snapshot)
    assert count == BASELINE_ROW_COUNT
    assert digest == differential[JDBC_CATALOG].sorted_rows_sha256
    assert digest == BASELINE_SORTED_ROWS_SHA256


# ── the namespace bootstrap a JDBC catalog needs and Hadoop never did ───────


def test_a_first_write_into_a_fresh_jdbc_warehouse_bootstraps_its_namespace(
    catalogs,
):
    """The Hadoop catalog created namespaces implicitly — they were directories. JDBC does not."""

    session, _options, tmp_path_factory = catalogs
    namespace = BASELINE_TABLE_IDENTIFIER.rsplit(".", 1)[0]

    before = _namespaces(session, FRESH_BOOTSTRAP_CATALOG)
    assert namespace not in before

    evidence = materialize_into(
        session,
        FRESH_BOOTSTRAP_CATALOG,
        tmp_path_factory.mktemp("janus-catalog-fresh"),
        run_id=FIRST_RUN_ID,
        started_at=FIRST_STARTED_AT,
    )

    assert namespace in _namespaces(session, FRESH_BOOTSTRAP_CATALOG)
    assert evidence.table_identifier == BASELINE_TABLE_IDENTIFIER
    assert evidence.row_count == BASELINE_ROW_COUNT
    assert evidence.sorted_rows_sha256 == BASELINE_SORTED_ROWS_SHA256


def test_the_local_profile_catalog_database_is_created_under_the_metadata_zone(
    catalogs, differential
):
    """The catalog is metadata, not data: it must never land in the bronze zone."""

    _session, options, _factory = catalogs
    uri = options[f"spark.sql.catalog.{JDBC_CATALOG}.uri"]
    warehouse = options[f"spark.sql.catalog.{JDBC_CATALOG}.warehouse"]

    database = Path(uri.removeprefix("jdbc:sqlite:").split("?", 1)[0])
    assert database.is_absolute()
    assert database.exists(), "the JDBC catalog never opened its database"
    assert Path(warehouse) not in database.parents
    assert "metadata" in database.parts


# ── helpers ──────────────────────────────────────────────────────────────────


def materialize_into(
    session, catalog: str, project_root: Path, *, run_id: str, started_at: datetime
) -> BronzeEvidence:
    """Run the live executor once, into ``catalog``, and collect every AC-4 observable."""

    session.conf.set("spark.sql.defaultCatalog", catalog)

    source_config = cloned_api_source_config(project_root)
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=project_root,
        started_at=started_at,
    )
    storage_layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, project_root)

    transport = FixtureTransport(
        fixture_paths=[
            TRANSPARENCIA_FIXTURES_DIR / "servidores_por_orgao_page_1.json",
            TRANSPARENCIA_FIXTURES_DIR / "servidores_por_orgao_page_2.json",
        ],
        budget=RequestBudget(),
    )
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )

    plan = strategy.plan(source_config, run_context)
    executed = SourceExecutor().execute(
        PlannedRun(plan=plan, strategy=strategy),
        SparkSessionProvider.wrapping(session),
        ENVIRONMENT_CONFIG,
    )
    assert executed.status == "succeeded", executed.failure_reason

    bronze_results = tuple(
        result for result in executed.write_results if result.zone == "bronze"
    )
    assert len(bronze_results) == 1, bronze_results
    bronze_result = bronze_results[0]

    dataframe = session.table(bronze_result.path)
    row_count, sorted_rows = row_digest(dataframe)
    _payload_count, payload = payload_digest(dataframe)

    return BronzeEvidence(
        table_identifier=bronze_result.path,
        schema_json=json.loads(dataframe.schema.json()),
        row_count=row_count,
        sorted_rows_sha256=sorted_rows,
        payload_sha256=payload,
        bronze_result=bronze_result,
        raw_results=tuple(
            result for result in executed.write_results if result.zone == "raw"
        ),
        project_root=project_root,
    )


def _root_relative_raw_results(evidence: BronzeEvidence) -> tuple[WriteResult, ...]:
    """Raw results with their absolute scratch prefix removed, so two runs are comparable."""

    return tuple(
        replace(result, path=str(Path(result.path).relative_to(evidence.project_root)))
        for result in evidence.raw_results
    )


def row_digest(dataframe) -> tuple[int, str]:
    """The equivalence suite's digest: sorted columns, rows sorted by `repr`, sha256."""

    canonical = dataframe.select(*sorted(dataframe.columns))
    rows = sorted(canonical.collect(), key=repr)
    digest = hashlib.sha256("\n".join(repr(row) for row in rows).encode("utf-8"))
    return len(rows), digest.hexdigest()


def payload_digest(dataframe) -> tuple[int, str]:
    payload_columns = sorted(set(dataframe.columns) - RUN_STAMP_COLUMNS)
    return row_digest(dataframe.select(*payload_columns))


def cloned_api_source_config(project_root: Path) -> SourceConfig:
    """The baseline capture's clone: the same source, at the same page size, under scratch."""

    source_config = load_registry(PROJECT_ROOT).get_source(API_SOURCE_ID, include_disabled=True)
    config_path = PROJECT_ROOT / source_config.config_path.relative_to(PROJECT_ROOT)
    schema_path = PROJECT_ROOT / source_config.schema.path

    copied_config_path = project_root / source_config.config_path.relative_to(PROJECT_ROOT)
    copied_schema_path = project_root / source_config.schema.path
    copied_config_path.parent.mkdir(parents=True, exist_ok=True)
    copied_schema_path.parent.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "sources" in payload:
        for entry in payload["sources"]:
            if entry.get("source_id") == API_SOURCE_ID:
                entry["access"]["pagination"]["page_size"] = PAGE_SIZE
                break
        else:
            raise AssertionError(f"Source {API_SOURCE_ID!r} was not found in the copied config")
    else:
        payload["access"]["pagination"]["page_size"] = PAGE_SIZE

    copied_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    copied_schema_path.write_text(schema_path.read_text(encoding="utf-8"), encoding="utf-8")

    return replace(
        source_config,
        config_path=copied_config_path,
        access=replace(
            source_config.access,
            pagination=replace(source_config.access.pagination, page_size=PAGE_SIZE),
        ),
    )


def _hadoop_catalog_options(catalog: str, root: Path) -> dict[str, str]:
    """The catalog this order is migrating *off*, kept alive to be diffed against."""

    return {
        f"spark.sql.catalog.{catalog}": ICEBERG_CATALOG_IMPL,
        f"spark.sql.catalog.{catalog}.type": HADOOP_CATALOG_TYPE,
        f"spark.sql.catalog.{catalog}.warehouse": str(root / "iceberg"),
        f"spark.sql.catalog.{catalog}.default-namespace": "bronze",
    }


def _jdbc_catalog_options(catalog: str, root: Path) -> dict[str, str]:
    """A second JDBC catalog over a database file that does not exist yet."""

    return {
        f"spark.sql.catalog.{catalog}": ICEBERG_CATALOG_IMPL,
        f"spark.sql.catalog.{catalog}.type": "jdbc",
        f"spark.sql.catalog.{catalog}.uri": f"jdbc:sqlite:{root / 'catalog.sqlite'}",
        f"spark.sql.catalog.{catalog}.{JDBC_SCHEMA_VERSION_OPTION[0]}": (
            JDBC_SCHEMA_VERSION_OPTION[1]
        ),
        f"spark.sql.catalog.{catalog}.warehouse": str(root / "iceberg"),
        f"spark.sql.catalog.{catalog}.default-namespace": "bronze",
    }


def _namespaces(session, catalog: str) -> set[str]:
    return {
        row[0] for row in session.sql(f"SHOW NAMESPACES IN {catalog}").collect()
    }
