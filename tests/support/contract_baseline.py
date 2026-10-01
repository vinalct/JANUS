"""Deterministic pre-contract Bronze captures.

The module deliberately imports no Spark package at module scope. Run the capture with a
Spark-capable interpreter::

    python -m tests.support.contract_baseline tests/fixtures/contracts/baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any
from unittest import mock
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

import yaml

from janus.checkpoints import CheckpointStore
from janus.planner import HookCatalog, Planner, PlanningRequest, StrategyBinding, StrategyCatalog
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.schema_contracts import resolve_contract_path_for_plan, resolve_spark_schema_for_plan
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.strategies.catalog import CatalogStrategy
from janus.strategies.files import FileStrategy

STARTED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
PROJECT_PLACEHOLDER = "<PROJECT>"

EXPLICIT_ENTRIES: tuple[str, ...] = (
    "dados_abertos_catalog__conjunto_dados__full_refresh",
    "dados_abertos_catalog__conjunto_dados_details__full_refresh",
    "federal_open_data_example",
    "inep_censo_escolar_microdados",
    "receita_federal__cnpj__cnaes_full_refresh",
    "receita_federal__cnpj__empresas_full_refresh",
    "receita_federal__cnpj__estabelecimentos_full_refresh",
    "receita_federal__cnpj__motivos_full_refresh",
    "receita_federal__cnpj__municipios_full_refresh",
    "receita_federal__cnpj__naturezas_full_refresh",
    "receita_federal__cnpj__paises_full_refresh",
    "receita_federal__cnpj__qualificacoes_full_refresh",
    "receita_federal__cnpj__simples_full_refresh",
    "receita_federal__cnpj__socios_full_refresh",
    "transparencia__gastos_cartoes__cartoes__full_refresh",
    "transparencia__gastos_cartoes__cartoes__incremental",
    "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh",
)

BRONZE_CASES: tuple[str, ...] = (
    "bronze_materializer_inep",
    "dados_abertos_catalog_conjunto_dados",
    "ibge_agro_abacaxi_pronaf",
    "ibge_pib_brasil",
    "inep_censo_escolar_microdados",
    "transparencia_servidores_por_orgao",
)

_NORMALIZATION_COLUMNS = frozenset(
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
_ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}
_INEP_ARCHIVE_NAME = "microdados_censo_escolar_2024.zip"
_INEP_ARCHIVE_MEMBER = (
    "microdados_censo_escolar_2024/dados/microdados_ed_basica_2024.csv"
)
_MULTI_MEMBER_COLUMNS = (
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


@dataclass(frozen=True, slots=True)
class BronzeCase:
    name: str
    source_id: str
    config_path: str
    family: str
    variant: str
    fixture_paths: tuple[str, ...] = ()
    token: tuple[str, str] | None = None
    checkpoint: str | None = None
    page_size: int | None = None
    archive_kind: str | None = None


_CASE_BY_NAME = {
    case.name: case
    for case in (
        BronzeCase(
            name="bronze_materializer_inep",
            source_id="inep_censo_escolar_microdados",
            config_path="inep/inep.yaml",
            family="file",
            variant="archive_package",
            archive_kind="multi_member",
        ),
        BronzeCase(
            name="dados_abertos_catalog_conjunto_dados",
            source_id="dados_abertos_catalog__conjunto_dados__full_refresh",
            config_path="dados_abertos_catalog/conjunto_de_dados.yaml",
            family="catalog",
            variant="metadata_catalog",
            fixture_paths=(
                "dados_abertos_catalog/package_search_page_1.json",
                "dados_abertos_catalog/package_search_page_2.json",
            ),
            token=("DADOS_GOV_BR_API_TOKEN", "catalog-token"),
            page_size=2,
        ),
        BronzeCase(
            name="ibge_agro_abacaxi_pronaf",
            source_id="ibge_agro_abacaxi_pronaf",
            config_path="ibge/sidra.yaml",
            family="api",
            variant="date_window_api",
            fixture_paths=("ibge/agro_abacaxi_pronaf_2006_flat.json",),
            checkpoint="2005",
        ),
        BronzeCase(
            name="ibge_pib_brasil",
            source_id="ibge_pib_brasil",
            config_path="ibge/sidra.yaml",
            family="api",
            variant="date_window_api",
            fixture_paths=("ibge/pib_brasil_2023_flat.json",),
            checkpoint="2022",
        ),
        BronzeCase(
            name="inep_censo_escolar_microdados",
            source_id="inep_censo_escolar_microdados",
            config_path="inep/inep.yaml",
            family="file",
            variant="archive_package",
            archive_kind="sample",
        ),
        BronzeCase(
            name="transparencia_servidores_por_orgao",
            source_id=(
                "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"
            ),
            config_path="transparencia/servidores.yaml",
            family="api",
            variant="page_number_api",
            fixture_paths=(
                "transparencia/servidores_por_orgao_page_1.json",
                "transparencia/servidores_por_orgao_page_2.json",
            ),
            token=("TRANSPARENCIA_API_TOKEN", "transparencia-token"),
            page_size=2,
        ),
    )
}


@dataclass(slots=True)
class FixtureTransport:
    fixture_paths: tuple[Path, ...]
    received_at: datetime
    _next_index: int = 0
    _lock: Lock = field(default_factory=Lock)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def send(self, request):
        with self._lock:
            index = self._response_index(request)
            if index >= len(self.fixture_paths):
                raise AssertionError(f"No fixture response remains for {request.redacted_url()}")
            self._next_index = max(self._next_index, index + 1)
            fixture_path = self.fixture_paths[index]
        return ApiResponse(
            request=request,
            status_code=200,
            body=fixture_path.read_bytes(),
            received_at=self.received_at,
        )

    def _response_index(self, request) -> int:
        query = parse_qs(urlsplit(request.full_url()).query)
        for page_key in ("pagina", "page"):
            if page_key in query:
                return int(query[page_key][0]) - 1
        return self._next_index


def spark_schema_golden(source_id: str, project_root: Path) -> dict[str, Any]:
    """Capture the exact Spark schema handed to one explicitly declared source."""

    project_root = project_root.resolve()
    planned = Planner().plan(
        PlanningRequest.create(
            source_id=source_id,
            environment="local",
            project_root=project_root,
            include_disabled=True,
            run_id="m0",
            started_at=STARTED_AT,
        )
    )
    schema = resolve_spark_schema_for_plan(planned.plan)
    if schema is None or not hasattr(schema, "jsonValue"):
        raise RuntimeError(f"Spark schema is unavailable for explicit source {source_id!r}")

    schema_path = _declared_schema_path(planned.plan, project_root)
    if schema_path is None:
        raise RuntimeError(f"Schema declaration path is unavailable for source {source_id!r}")
    return {
        "source_id": source_id,
        "schema_path": _portable_path(schema_path, project_root),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "struct_type": schema.jsonValue(),
    }


def bronze_golden(
    case: str,
    project_root: Path,
    *,
    spark: Any | None = None,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Run one offline source through ``SourceExecutor`` and capture its Bronze shape."""

    if case not in _CASE_BY_NAME:
        raise ValueError(f"Unknown Bronze baseline case {case!r}")

    if spark is not None and work_root is not None:
        return _capture_bronze_case(_CASE_BY_NAME[case], project_root, spark, work_root)

    if spark is not None or work_root is not None:
        raise ValueError("spark and work_root must be provided together")

    from tests.support.spark_sessions import build_iceberg_session

    with TemporaryDirectory(prefix="janus-bronze-") as temporary:
        temporary_root = Path(temporary)
        session = build_iceberg_session(
            f"janus{case}", temporary_root / "catalog"
        )
        try:
            return _capture_bronze_case(
                _CASE_BY_NAME[case], project_root, session, temporary_root / case
            )
        finally:
            session.stop()


def capture_baseline(project_root: Path, output_dir: Path) -> None:
    """Write all schema and Bronze snapshots to ``output_dir``."""

    from tests.support.spark_sessions import build_iceberg_session

    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    schema_dir = output_dir / "spark_schema"
    bronze_dir = output_dir / "bronze"
    schema_dir.mkdir(parents=True, exist_ok=True)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    for source_id in EXPLICIT_ENTRIES:
        _write_json(schema_dir / f"{source_id}.json", spark_schema_golden(source_id, project_root))

    with TemporaryDirectory(prefix="janus-capture-") as temporary:
        work_root = Path(temporary)
        session = build_iceberg_session("m0", work_root / "catalog")
        try:
            for case in BRONZE_CASES:
                captured = bronze_golden(
                    case,
                    project_root,
                    spark=session,
                    work_root=work_root / "cases" / case,
                )
                _write_json(bronze_dir / f"{case}.json", captured)
        finally:
            session.stop()


def _capture_bronze_case(
    case: BronzeCase,
    project_root: Path,
    spark: Any,
    work_root: Path,
) -> dict[str, Any]:
    case_root = work_root.resolve()
    _prepare_case_project(case, project_root.resolve(), case_root)
    strategy = _strategy_for(case, project_root.resolve(), case_root)
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            (StrategyBinding(case.family, case.variant, strategy),)
        ),
        hook_catalog=HookCatalog.with_defaults(),
    )
    planned = planner.plan(
        PlanningRequest.create(
            source_id=case.source_id,
            environment="local",
            project_root=case_root,
            include_disabled=True,
            run_id=f"m0-{case.name}",
            started_at=STARTED_AT,
        )
    )
    declared_schema = resolve_spark_schema_for_plan(planned.plan)
    read_schema_source = "explicit" if declared_schema is not None else "inferred"
    if case.checkpoint is not None:
        CheckpointStore().save(planned.plan, case.checkpoint)

    token_environment = dict([case.token] if case.token is not None else [])
    with mock.patch.dict(os.environ, token_environment, clear=False):
        executed = SourceExecutor().execute(
            planned,
            SparkSessionProvider.wrapping(spark),
            _ENVIRONMENT_CONFIG,
        )
    if executed.status != "succeeded":
        raise AssertionError(
            f"Bronze baseline case {case.name!r} failed: {executed.failure_reason}"
        )

    bronze_paths = {result.path for result in executed.write_results if result.zone == "bronze"}
    if len(bronze_paths) != 1:
        raise AssertionError(f"Expected one Bronze table for {case.name!r}: {bronze_paths}")
    table_identifier = next(iter(bronze_paths))
    dataframe = spark.table(table_identifier)
    rows = [_portable_row(row.asDict(recursive=True), case_root) for row in dataframe.collect()]
    canonical_rows = sorted(_canonical_json(row) for row in rows)
    digest_payload = f"[{','.join(canonical_rows)}]".encode()
    table_schema = dataframe.schema
    read_struct_type = (
        declared_schema.jsonValue()
        if declared_schema is not None
        else _data_columns_struct_type(table_schema)
    )

    if executed.validation_report is None or executed.lineage_path is None:
        raise AssertionError(f"Case {case.name!r} did not persist quality and lineage evidence")
    validation_payload = json.loads(
        executed.validation_report.path.read_text(encoding="utf-8")
    )
    lineage_payload = json.loads(executed.lineage_path.read_text(encoding="utf-8"))

    captured = {
        "case": case.name,
        "source_id": case.source_id,
        "table_identifier": table_identifier,
        "schema": [
            {
                "name": field.name,
                "type": field.dataType.simpleString(),
                "nullable": field.nullable,
            }
            for field in table_schema.fields
        ],
        "row_count": len(rows),
        "row_digest": hashlib.sha256(digest_payload).hexdigest(),
        "row_digest_algorithm": "sha256(canonical-json-array(sorted(canonical-json-row)))",
        "read_schema_source": read_schema_source,
        "read_struct_type": read_struct_type,
        "quality_report": {
            "schema_expectation_source": _portable(
                validation_payload["metadata"].get("schema_expectation_source", ""),
                case_root,
            ),
            "checks": [
                {"name": check["name"], "outcome": check["outcome"]}
                for check in validation_payload["checks"]
            ],
        },
        "lineage_keys": sorted(lineage_payload),
        "spark_session_timezone": spark.conf.get("spark.sql.session.timeZone"),
    }
    if read_schema_source == "inferred":
        captured["inferred_struct_type"] = read_struct_type
    return captured


def _prepare_case_project(case: BronzeCase, project_root: Path, case_root: Path) -> None:
    shutil.copytree(project_root / "conf", case_root / "conf")
    source_path = case_root / "conf" / "sources" / case.config_path
    document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source = _source_entry(document, case.source_id)

    if case.page_size is not None:
        source["access"]["pagination"]["page_size"] = case.page_size
        source["access"]["rate_limit"]["concurrency"] = 1

    if case.archive_kind is not None:
        archive_path = _build_inep_archive(case, project_root, case_root)
        source["access"].pop("url", None)
        source["access"]["path"] = str(archive_path)
        if case.archive_kind == "multi_member":
            source["access"]["file_pattern"] = "*/dados/*.csv"

    source_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    for zone in ("raw", "bronze", "metadata"):
        (case_root / "data" / zone).mkdir(parents=True, exist_ok=True)


def _source_entry(document: Any, source_id: str) -> dict[str, Any]:
    candidates = document.get("sources") if isinstance(document, Mapping) else None
    entries: Sequence[Any] = candidates if isinstance(candidates, list) else (document,)
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source_id") == source_id:
            return entry
    raise AssertionError(f"Source {source_id!r} was not found in its copied config")


def _build_inep_archive(case: BronzeCase, project_root: Path, case_root: Path) -> Path:
    archive_path = case_root / "fixtures" / _INEP_ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "w") as archive:
        if case.archive_kind == "sample":
            sample = (
                project_root
                / "tests"
                / "fixtures"
                / "inep"
                / "microdados_ed_basica_2024_sample.csv"
            ).read_text(encoding="utf-8")
            archive.writestr(_INEP_ARCHIVE_MEMBER, sample)
        else:
            for member_index in range(1, 7):
                archive.writestr(
                    "microdados_censo_escolar_2024/dados/"
                    f"microdados_parte_{member_index:02d}.csv",
                    _multi_member_csv(member_index),
                )
        archive.writestr(
            "microdados_censo_escolar_2024/leia-me.txt",
            "Reduced fixture package for JANUS integration tests.\n",
        )
    return archive_path


def _multi_member_csv(member_index: int) -> str:
    lines = [";".join(_MULTI_MEMBER_COLUMNS)]
    for row_index in range(1, 3):
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


def _strategy_for(case: BronzeCase, project_root: Path, case_root: Path):
    from janus.utils.storage import StorageLayout

    storage_layout = StorageLayout.from_environment_config(_ENVIRONMENT_CONFIG, case_root)
    common = {
        "storage_layout_factory": lambda _plan: storage_layout,
        "sleeper": lambda _seconds: None,
        "clock": lambda: 0.0,
    }
    if case.family == "file":
        return FileStrategy(**common)

    fixtures = tuple(project_root / "tests" / "fixtures" / path for path in case.fixture_paths)
    transport = FixtureTransport(fixtures, STARTED_AT)
    strategy_type = CatalogStrategy if case.family == "catalog" else ApiStrategy
    return strategy_type(transport_factory=lambda: transport, **common)


def _declared_schema_path(plan: Any, project_root: Path) -> Path | None:
    del project_root
    return resolve_contract_path_for_plan(plan)


def _data_columns_struct_type(table_schema: Any) -> dict[str, Any]:
    return {
        "type": "struct",
        "fields": [
            field.jsonValue()
            for field in table_schema.fields
            if field.name not in _NORMALIZATION_COLUMNS
        ],
    }


def _portable_row(row: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Undo PySpark driver-local timestamp conversion before canonical serialization."""

    timestamp = row.get("ingestion_timestamp")
    if isinstance(timestamp, datetime):
        # TimestampType.fromInternal returns a naive datetime in the Python process timezone,
        # even when spark.sql.session.timeZone is UTC. Convert that representation back to the
        # UTC wall time so a host in America/Sao_Paulo and a UTC CI container hash the same row.
        row["ingestion_timestamp"] = timestamp.astimezone(UTC).replace(tzinfo=None)
    return _portable(row, project_root)


def _portable(value: Any, project_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(project_root), PROJECT_PLACEHOLDER)
    if isinstance(value, dict):
        return {key: _portable(item, project_root) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_portable(item, project_root) for item in value]
    return value


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    capture_baseline(args.project_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
