from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyspark.sql")
from pyspark.sql import SparkSession

from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, RunContext
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.schema_contracts import load_spark_schema_from_schema_path
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("janus-reader-tests")
        .master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_spark_dataset_reader_loads_json_artifacts_from_extraction_result(
    spark: SparkSession,
    tmp_path,
):
    plan = _build_plan(
        tmp_path,
        run_id="run-reader-001",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    storage_layout = _build_storage_layout(tmp_path)
    raw_writer = RawArtifactWriter(storage_layout)
    reader = SparkDatasetReader()

    first_artifact = raw_writer.write_json(plan, "page-0001.json", {"id": "1", "page": 1})
    second_artifact = raw_writer.write_json(plan, "page-0002.json", {"id": "2", "page": 2})
    extraction_result = ExtractionResult.from_plan(
        plan,
        artifacts=(first_artifact.artifact, second_artifact.artifact),
        records_extracted=2,
    )

    dataframe = reader.read_extraction_result(
        spark,
        extraction_result,
        format_name=plan.source_config.spark.input_format,
    )
    rows = {(row.id, row.page) for row in dataframe.collect()}

    assert rows == {("1", 1), ("2", 2)}


def test_spark_dataset_reader_requires_explicit_format_for_mixed_artifacts(
    spark: SparkSession,
    tmp_path,
):
    plan = _build_plan(
        tmp_path,
        run_id="run-reader-002",
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    reader = SparkDatasetReader()
    extraction_result = ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(path=str(tmp_path / "artifact-1.json"), format="json"),
            ExtractedArtifact(path=str(tmp_path / "artifact-2.csv"), format="csv"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Artifacts contain multiple formats; pass format_name explicitly to read them",
    ):
        reader.read_extraction_result(spark, extraction_result)


def test_spark_dataset_reader_applies_explicit_schema_to_headerless_cnpj_csv(
    spark: SparkSession,
    tmp_path,
):
    with pytest.warns(DeprecationWarning, match="schema.contract"):
        source_config = load_registry(PROJECT_ROOT).get_source(
            "receita_federal__cnpj__empresas_full_refresh",
            include_disabled=True,
        )
    source_config = replace(
        source_config,
        schema=replace(
            source_config.schema,
            path=str(PROJECT_ROOT / source_config.schema.path),
        ),
    )
    run_context = RunContext.create(
        run_id="run-reader-cnpj-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)
    raw_path = tmp_path / "K3241.K03200Y0.D30513.EMPRECSV"
    raw_path.write_text(
        "12345678;ACME LTDA;2062;49;1000,00;01;\n",
        encoding="ISO-8859-1",
    )
    extraction_result = ExtractionResult.from_plan(
        plan,
        artifacts=(ExtractedArtifact(path=str(raw_path), format="csv"),),
        records_extracted=1,
    )

    dataframe = SparkDatasetReader().read_extraction_result(
        spark,
        extraction_result,
        format_name="csv",
        schema=load_spark_schema_from_schema_path(Path(source_config.schema.path)),
        options=source_config.spark.read_options,
    )

    assert dataframe.columns == [
        "cnpj_basico",
        "razao_social",
        "natureza_juridica",
        "qualificacao_do_responsavel",
        "capital_social",
        "porte_empresa",
        "ente_federativo_responsavel",
    ]
    row = dataframe.first()
    assert row.cnpj_basico == "12345678"
    assert row.razao_social == "ACME LTDA"
    assert row.natureza_juridica == "2062"


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
