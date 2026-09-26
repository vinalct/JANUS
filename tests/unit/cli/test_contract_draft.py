"""Host-side behavior and Spark-gated coverage for ``janus contract draft``."""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import janus.cli.contract as contract_cli
import janus.runtime.spark_lifecycle as spark_lifecycle
from janus.main import main
from janus.models import load_data_contract
from janus.planner import Planner, PlanningRequest
from tests.support.observability_baseline import capture_case

PROJECT_ROOT = Path(__file__).resolve().parents[3]
def _write_project(root: Path, *, declares_contract: bool = False) -> Path:
    sources = root / "conf" / "sources"
    sources.mkdir(parents=True)
    (root / "conf" / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: '*.yaml'\n",
        encoding="utf-8",
    )
    (root / "conf" / "environments").mkdir(parents=True)
    (root / "conf" / "environments" / "local.yaml").write_text(
        """
name: local
storage:
  root_dir: data
  raw_dir: data/raw
  bronze_dir: data/bronze
  metadata_dir: data/metadata
spark:
  app_name: janus-contract-draft-tests
  master: local[1]
  warehouse_dir: data/metadata/warehouse
  config: {}
""".lstrip(),
        encoding="utf-8",
    )
    schema = (
        "schema:\n  contract: conf/contracts/test/active.yaml\n"
        if declares_contract
        else "schema:\n  mode: infer\n"
    )
    source = f"""
source_id: draft_source
name: Draft source
owner: janus-tests
enabled: true
source_type: api
strategy: api
strategy_variant: page_number_api
federation_level: federal
domain: test
public_access: true
tags: [fixture]
access:
  base_url: https://fixtures.invalid
  path: /draft-source
  method: GET
  format: json
  timeout_seconds: 30
  auth:
    type: none
  pagination:
    type: page_number
    page_param: page
    size_param: size
    page_size: 10
  rate_limit:
    concurrency: 1
    backoff_seconds: 1
extraction:
  mode: full_refresh
  retry:
    max_attempts: 1
    backoff_seconds: 1
{schema}spark:
  input_format: json
  write_mode: overwrite
outputs:
  raw:
    path: data/raw/test/draft_source
    format: json
  bronze:
    path: data/bronze/test/draft_source
    format: iceberg
    namespace: bronze_test
    table_name: draft_table
  metadata:
    path: data/metadata/test/draft_source
    format: json
quality:
  allow_schema_evolution: true
""".lstrip()
    (sources / "draft.yaml").write_text(source, encoding="utf-8")
    if declares_contract:
        contract = root / "conf" / "contracts" / "test" / "active.yaml"
        contract.parent.mkdir(parents=True)
        shutil.copyfile(
            PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "minimal_contract.yaml",
            contract,
        )
    fixture = root / "fixtures" / "records.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text('[{"id": "1", "name": "Ada"}]\n', encoding="utf-8")
    return fixture


def _fixture_arguments(root: Path, fixture: Path, *extra: str) -> list[str]:
    return [
        "contract",
        "draft",
        "--source-id",
        "draft_source",
        "--from-fixture",
        str(fixture),
        "--project-root",
        str(root),
        *extra,
    ]


class _FakeSchema:
    def jsonValue(self) -> dict[str, Any]:
        return {
            "type": "struct",
            "fields": [
                {"name": "id", "type": "string", "nullable": True, "metadata": {}},
                {"name": "name", "type": "string", "nullable": True, "metadata": {}},
            ],
        }


class _StubSession:
    @property
    def sparkContext(self) -> SimpleNamespace:
        return SimpleNamespace(appName="janus-contract-test", master="local[1]")

    def stop(self) -> None:
        pass


class _SpyProvider(spark_lifecycle.SparkSessionProvider):
    def __init__(self, events: list[str]) -> None:
        super().__init__({}, {}, session_factory=_StubSession)
        self.events = events

    def get(self):
        self.events.append("get")
        return super().get()

    def stop(self) -> None:
        super().stop()
        self.events.append("stop")


class _FakeReader:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def read_extraction_result(self, spark, extraction_result, **kwargs):
        assert len(extraction_result.artifacts) == 1
        assert kwargs == {
            "format_name": "json",
            "schema": None,
            "options": None,
        }
        return SimpleNamespace(schema=_FakeSchema())


def test_help_returns_zero_and_warns_about_data_dependent_inference(capsys):
    result = main(["contract", "draft", "--help"])

    assert result == 0
    help_text = capsys.readouterr().out
    assert "--from-fixture PATH" in help_text
    assert "All-null fields" in help_text
    assert "review a draft before activating it" in " ".join(help_text.split())


def test_draft_arguments_require_one_input_mode(capsys, tmp_path):
    fixture = _write_project(tmp_path)
    common = ["contract", "draft", "--source-id", "draft_source", "--project-root", str(tmp_path)]

    missing = main(common)
    both = main([*common, "--from-raw", "run-1", "--from-fixture", str(fixture)])

    assert missing == 2
    assert both == 2
    assert capsys.readouterr().err.count("usage:") == 2


def test_outside_project_output_is_refused_before_planning(capsys, tmp_path):
    fixture = _write_project(tmp_path)
    outside = tmp_path.parent / "draft-outside.yaml"

    result = main(_fixture_arguments(tmp_path, fixture, "--out", str(outside)))

    assert result == 2
    assert "--out must resolve inside the project root" in capsys.readouterr().err
    assert not outside.exists()


@pytest.mark.parametrize("zone", ("raw", "bronze"))
def test_explicit_output_cannot_write_into_raw_or_bronze(capsys, tmp_path, zone):
    fixture = _write_project(tmp_path)
    output = tmp_path / "data" / zone / "draft.yaml"

    result = main(_fixture_arguments(tmp_path, fixture, "--out", str(output)))

    assert result == 2
    assert f"--out must not write into the {zone} zone" in capsys.readouterr().err
    assert not output.exists()


def test_source_with_contract_requires_an_explicit_different_output(capsys, tmp_path):
    fixture = _write_project(tmp_path, declares_contract=True)

    result = main(_fixture_arguments(tmp_path, fixture))

    assert result == 2
    error = capsys.readouterr().err
    assert "already declares schema.contract" in error
    assert "conf/contracts/test/active.yaml" in error


def test_source_with_contract_allows_a_different_explicit_output(
    monkeypatch, capsys, tmp_path
):
    fixture = _write_project(tmp_path, declares_contract=True)
    output = tmp_path / "drafts" / "second-version.yaml"
    drafted = contract_cli.DraftedContract(
        source_id="draft_source",
        output_path=output,
        rows=1,
        columns=("id",),
        required=("id",),
        drafted_from="fixture records.json, 1 rows",
    )
    monkeypatch.setattr(contract_cli, "draft_contract", lambda *_args, **_kwargs: drafted)
    monkeypatch.setattr(spark_lifecycle, "SparkSessionProvider", lambda *_args: object())

    result = main(_fixture_arguments(tmp_path, fixture, "--out", str(output)))

    assert result == 0
    assert json.loads(capsys.readouterr().out)["out"] == str(output)


def test_default_contract_path_is_not_overwritten_without_out(capsys, tmp_path):
    fixture = _write_project(tmp_path)
    existing = tmp_path / "conf" / "contracts" / "test" / "draft_table.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing: contract\n", encoding="utf-8")

    result = main(_fixture_arguments(tmp_path, fixture))

    assert result == 2
    assert "Refusing to overwrite existing contract file" in capsys.readouterr().err
    assert existing.read_text(encoding="utf-8") == "existing: contract\n"


def test_draft_uses_one_session_and_releases_it_before_writing(
    monkeypatch, capsys, tmp_path
):
    fixture = _write_project(tmp_path)
    output = tmp_path / "drafts" / "draft.yaml"
    events: list[str] = []
    monkeypatch.setattr(contract_cli, "SparkDatasetReader", lambda: _FakeReader(events))
    monkeypatch.setattr(
        contract_cli,
        "_profile_dataframe",
        lambda dataframe: contract_cli._DataProfile(
            schema=dataframe.schema,
            row_count=3,
            null_counts=(0, 1),
        ),
    )
    monkeypatch.setattr(
        spark_lifecycle,
        "SparkSessionProvider",
        lambda config, paths: _SpyProvider(events),
    )
    write_text = Path.write_text

    def observe_write(path: Path, *args, **kwargs):
        if path.resolve() == output.resolve():
            events.append("write")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", observe_write)

    result = main(_fixture_arguments(tmp_path, fixture, "--out", str(output)))

    assert result == 0
    assert events == ["get", "stop", "write"]
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "source_id": "draft_source",
        "out": str(output),
        "rows": 3,
        "columns": ["id", "name"],
        "required": ["id"],
        "drafted_from": f"fixture {fixture}, 3 rows",
    }
    contract = load_data_contract(output)
    assert contract.status == "draft"
    assert contract.version == "0.1.0"
    assert contract.column_names == ("id", "name")
    assert contract.schema.properties[0].primary_key is False

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (PROJECT_ROOT / "docs" / "schemas" / "odcs" / "odcs-json-schema-v3.2.0.json")
        .read_text(encoding="utf-8")
    )
    validator_class = jsonschema.validators.validator_for(schema)
    assert list(validator_class(schema).iter_errors(yaml.safe_load(output.read_text()))) == []


def test_main_dispatches_contract_lazily(monkeypatch):
    import janus.cli.contract as cli_contract

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(cli_contract, "main", lambda argv: calls.append(tuple(argv)) or 7)

    assert main(["contract", "draft", "--help"]) == 7
    assert calls == [("draft", "--help")]


def test_run_paths_do_not_import_the_contract_cli():
    for relative in (
        "src/janus/cli/run_all.py",
        "src/janus/runtime/executor.py",
        "src/janus/runtime/batch.py",
    ):
        tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert "janus.cli.contract" not in imports


def test_raw_run_pin_selects_the_requested_run(tmp_path):
    _write_project(tmp_path)
    planned = Planner().plan(
        PlanningRequest.create(
            source_id="draft_source",
            environment="local",
            project_root=tmp_path,
        )
    )
    raw_root = (tmp_path / planned.plan.raw_output.path).resolve()
    resolved_plan = replace(
        planned.plan,
        raw_output=replace(planned.plan.raw_output, path=str(raw_root)),
    )
    selected = raw_root / "runs" / "ingestion_date=2026-09-18" / "run_id=selected"
    latest = raw_root / "runs" / "ingestion_date=2026-09-22" / "run_id=latest"
    selected.mkdir(parents=True)
    latest.mkdir(parents=True)

    from janus.scripts.replay_plan import _plan_with_active_raw_root

    replay = _plan_with_active_raw_root(resolved_plan, run_id="selected")

    assert Path(replay.raw_output.path) == selected
    with pytest.raises(FileNotFoundError, match="missing"):
        _plan_with_active_raw_root(resolved_plan, run_id="missing")


@pytest.fixture(scope="module")
def spark():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-contract-draft-tests")
        .master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _production_planned_run(
    source_id: str,
    *,
    mode: str,
    output_path: Path,
    fixture_paths: tuple[Path, ...] = (),
    raw_run_id: str | None = None,
) -> Any:
    payload = {
        "mode": mode,
        "raw_run_id": raw_run_id,
        "fixture_paths": [str(path) for path in fixture_paths],
        "output_path": str(output_path),
    }
    return Planner().plan(
        PlanningRequest.create(
            source_id=source_id,
            environment="local",
            project_root=PROJECT_ROOT,
            include_disabled=True,
            attributes={contract_cli._DRAFT_REQUEST_ATTRIBUTE: json.dumps(payload)},
        )
    )


@pytest.mark.parametrize(
    ("source_id", "fixture_name", "golden_name"),
    (
        (
            "ibge_pib_brasil",
            "pib_brasil_2023_flat.json",
            "ibge_pib_brasil.json",
        ),
        (
            "ibge_agro_abacaxi_pronaf",
            "agro_abacaxi_pronaf_2006_flat.json",
            "ibge_agro_abacaxi_pronaf.json",
        ),
    ),
)
def test_ibge_fixture_draft_reproduces_the_inferred_bronze_golden(
    spark, tmp_path, source_id, fixture_name, golden_name
):
    from janus.runtime import SparkSessionProvider
    from janus.schema_contracts import spark_schema_from_contract

    fixture = PROJECT_ROOT / "tests" / "fixtures" / "ibge" / fixture_name
    output = tmp_path / f"{source_id}.yaml"
    planned = _production_planned_run(
        source_id,
        mode="fixture",
        output_path=output,
        fixture_paths=(fixture,),
    )
    environment = yaml.safe_load(
        (PROJECT_ROOT / "conf" / "environments" / "local.yaml").read_text(encoding="utf-8")
    )
    provider = SparkSessionProvider.wrapping(spark)

    drafted = contract_cli.draft_contract(
        planned,
        provider,
        source=planned.plan.source_config,
        environment_config=environment,
    )

    assert "aggregate_id" in drafted.columns
    draft_schema = spark_schema_from_contract(load_data_contract(output))
    baseline = json.loads(
        (PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "baseline" / "bronze" / golden_name)
        .read_text(encoding="utf-8")
    )
    assert draft_schema.jsonValue() == baseline["inferred_struct_type"]


def test_transparencia_fixture_draft_matches_legacy_physical_types(spark, tmp_path):
    from janus.models.data_contracts import contract_from_legacy_schema_file
    from janus.runtime import SparkSessionProvider

    source_id = "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"
    fixtures = (
        PROJECT_ROOT / "tests" / "fixtures" / "transparencia" / "servidores_por_orgao_page_1.json",
        PROJECT_ROOT / "tests" / "fixtures" / "transparencia" / "servidores_por_orgao_page_2.json",
    )
    output = tmp_path / "servidores_por_orgao.yaml"
    planned = _production_planned_run(
        source_id,
        mode="fixture",
        output_path=output,
        fixture_paths=fixtures,
    )
    environment = yaml.safe_load(
        (PROJECT_ROOT / "conf" / "environments" / "local.yaml").read_text(encoding="utf-8")
    )
    drafted = contract_cli.draft_contract(
        planned,
        SparkSessionProvider.wrapping(spark),
        source=planned.plan.source_config,
        environment_config=environment,
    )
    expected = contract_from_legacy_schema_file(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "contracts"
        / "legacy_schemas"
        / "transparencia"
        / "servidores_por_orgao_schema.json",
        source_id=source_id,
        bronze_table="poder_executivo_federal__servidores_por_orgao",
        domain="transparencia",
        project_root=PROJECT_ROOT,
    )

    actual_types = {
        prop.name: prop.physical_type for prop in load_data_contract(output).schema.properties
    }
    expected_types = {prop.name: prop.physical_type for prop in expected.schema.properties}
    assert drafted.columns == tuple(expected_types)
    assert actual_types == expected_types
    assert set(actual_types.values()) <= {"integer", "string"}


def test_from_raw_draft_reads_the_requested_baseline_execute_run(spark, tmp_path):
    from janus.runtime import SparkSessionProvider

    root = tmp_path / "baseline"
    capture_case(root, "api_success")
    output = root / "drafts" / "baseline_api.yaml"
    run_id = "order15-api_success"
    planned = _production_planned_run_for_root(root, output, run_id)
    environment = {
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        }
    }

    drafted = contract_cli.draft_contract(
        planned,
        SparkSessionProvider.wrapping(spark),
        source=planned.plan.source_config,
        environment_config=environment,
    )

    assert drafted.source_id == "baseline_api"
    assert drafted.rows == 2
    assert drafted.columns == ("id", "title", "updated_at")
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["customProperties"][-1]["value"].startswith(f"raw run {run_id} on ")


def _production_planned_run_for_root(root: Path, output: Path, raw_run_id: str) -> Any:
    payload = {
        "mode": "raw",
        "raw_run_id": raw_run_id,
        "fixture_paths": [],
        "output_path": str(output),
    }
    return Planner().plan(
        PlanningRequest.create(
            source_id="baseline_api",
            environment="local",
            project_root=root,
            attributes={contract_cli._DRAFT_REQUEST_ATTRIBUTE: json.dumps(payload)},
        )
    )
