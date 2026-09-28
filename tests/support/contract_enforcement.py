"""Synthetic contract-backed api sources, run end to end."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from janus.models import BronzeWriteIntent, WriteResult, resolve_bronze_write_intent
from janus.models.data_contracts import odcs_logical_type_for
from janus.planner import PlannedRun, Planner, PlanningRequest
from janus.runtime import ExecutedRun, SourceExecutor, SparkSessionProvider
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers import SparkDatasetWriter, quote_identifier

ENVIRONMENT_CONFIG: Mapping[str, Any] = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}

CONTRACTS_DIR = Path("conf/contracts/test")
SOURCES_DIR = Path("conf/sources")
DEFAULT_NAMESPACE = "bronze_order19"
ENVIRONMENT = "local"


def default_contract_path(source_id: str) -> Path:
    """The project-relative path a case's contract is written to."""
    return CONTRACTS_DIR / f"{source_id}.yaml"


def contract_property(
    name: str,
    physical_type: str,
    *,
    required: bool = False,
    primary_key: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """One ODCS property dict, its ``logicalType`` derived through the vocabulary."""
    prop: dict[str, Any] = {
        "name": name,
        "logicalType": odcs_logical_type_for(physical_type),
        "physicalType": physical_type,
    }
    if required:
        prop["required"] = True
    if primary_key:
        prop["primaryKey"] = True
    prop.update(extra)
    return prop


@dataclass(frozen=True)
class EnforcementCase:
    """One synthetic api source + contract pair, writable through SourceExecutor or the writer."""

    source_id: str
    contract_path: Path  
    properties: tuple[Mapping[str, Any], ...]
    compatibility: str = "additive"
    enforcement: str = "lenient"
    write_mode: str = "append"  
    extraction_mode: str = "full_refresh"
    version: str = "1.0.0"
    status: str = "active"
    input_format: str = "json"
    raw_format: str = "json"
    partition_by: tuple[str, ...] = ()
    page_size: int = 100
    namespace: str = DEFAULT_NAMESPACE
    table_name: str | None = None
    checkpoint_field: str | None = None
    allow_schema_evolution: bool = True

    @classmethod
    def for_source(
        cls, source_id: str, properties: Sequence[Mapping[str, Any]], **options: Any
    ) -> EnforcementCase:
        """Build a case whose contract lands at :func:`default_contract_path`."""
        return cls(
            source_id=source_id,
            contract_path=default_contract_path(source_id),
            properties=tuple(properties),
            **options,
        )

    @property
    def contract_id(self) -> str:
        return f"test.{self.source_id}"

    @property
    def bronze_path(self) -> str:
        return f"data/bronze/test/{self.source_id}"

    @property
    def bronze_table(self) -> str:
        """The writer's own identity for this case's table, never a re-derivation."""
        return bronze_table_identifier(
            self.bronze_path,
            fallback_name=self.source_id,
            namespace=self.namespace,
            table_name=self.table_name or self.source_id,
        )

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(prop["name"] for prop in self.properties if prop.get("required"))

    @property
    def unique_fields(self) -> tuple[str, ...]:
        return tuple(prop["name"] for prop in self.properties if prop.get("primaryKey"))


def render_contract(case: EnforcementCase) -> dict[str, Any]:
    """The ODCS document for one case, in the shape order-18's loader accepts."""
    return {
        "apiVersion": "v3.2.0",
        "kind": "DataContract",
        "id": case.contract_id,
        "name": case.source_id,
        "version": case.version,
        "status": case.status,
        "domain": "test",
        "description": {"purpose": f"enforcement fixture {case.source_id}."},
        "team": [{"username": "janus-tests", "role": "owner"}],
        "schema": [
            {
                "name": case.bronze_table,
                "physicalType": "table",
                "properties": [dict(prop) for prop in case.properties],
            }
        ],
        "customProperties": [
            {"property": "janus.compatibility", "value": case.compatibility},
            {"property": "janus.enforcement", "value": case.enforcement},
        ],
    }


def render_source_config(case: EnforcementCase) -> dict[str, Any]:
    """The source YAML mapping for one case: a page-number api over ``example.invalid``."""
    extraction: dict[str, Any] = {
        "mode": case.extraction_mode,
        "dead_letter_max_items": 0,
        "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
    }
    if case.checkpoint_field is not None:
        extraction["checkpoint_field"] = case.checkpoint_field
        extraction["checkpoint_strategy"] = "max_value"

    quality: dict[str, Any] = {"allow_schema_evolution": case.allow_schema_evolution}
    if case.required_fields:
        quality["required_fields"] = list(case.required_fields)
    if case.unique_fields:
        quality["unique_fields"] = list(case.unique_fields)

    return {
        "source_id": case.source_id,
        "name": case.source_id,
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "test",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/records",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": case.page_size,
            },
            "rate_limit": {"requests_per_minute": None, "concurrency": 1},
        },
        "extraction": extraction,
        "schema": {"contract": case.contract_path.as_posix()},
        "spark": {
            "input_format": case.input_format,
            "write_mode": case.write_mode,
            "repartition": 1,
            "partition_by": list(case.partition_by),
        },
        "outputs": {
            "raw": {"path": f"data/raw/test/{case.source_id}", "format": case.raw_format},
            "bronze": {
                "path": case.bronze_path,
                "format": "iceberg",
                "namespace": case.namespace,
                "table_name": case.table_name or case.source_id,
            },
            "metadata": {"path": f"data/metadata/test/{case.source_id}", "format": "json"},
        },
        "quality": quality,
    }


def write_case_project(tmp_path: Path, case: EnforcementCase) -> Path:
    """Write ``conf/app.yaml``, the source YAML and the contract; return the project root.

    Idempotent and overwriting, so a test can rewrite the contract between two runs (a
    reorder, a version bump) and re-plan against the same warehouse.
    """
    project_root = tmp_path
    _write_yaml(
        project_root / "conf" / "app.yaml",
        {"registry": {"sources_dir": SOURCES_DIR.as_posix(), "file_pattern": "*.yaml"}},
    )
    _write_yaml(project_root / SOURCES_DIR / f"{case.source_id}.yaml", render_source_config(case))
    _write_yaml(project_root / case.contract_path, render_contract(case))
    return project_root


def plan_case(
    project_root: Path,
    case: EnforcementCase,
    *,
    run_id: str,
    started_at: datetime,
) -> PlannedRun:
    """Plan one case through the production registry snapshot and ``Planner``."""
    request = PlanningRequest.create(
        source_id=case.source_id,
        environment=ENVIRONMENT,
        project_root=project_root,
        run_id=run_id,
        started_at=started_at,
    )
    return Planner().plan(request)


@dataclass(slots=True)
class FixtureTransport:
    """Serves the canned pages in order, then an empty page, and keeps every request."""

    payloads: list[Any] = field(default_factory=list)
    requests: list[Any] = field(default_factory=list)
    opened: bool = False
    closed: bool = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request: Any) -> ApiResponse:
        self.requests.append(request)
        payload = self.payloads.pop(0) if self.payloads else []
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        return ApiResponse(request=request, status_code=200, body=body)


def execute_case_with_pages(
    planned_run: PlannedRun,
    pages: Sequence[Any],
    session_factory: Callable[[], Any],
    environment_config: Mapping[str, Any] = ENVIRONMENT_CONFIG,
    *,
    transport: FixtureTransport | None = None,
) -> ExecutedRun:
    """Run one planned case through ``SourceExecutor`` with a private provider."""
    served = transport if transport is not None else FixtureTransport()
    served.payloads.extend(pages)
    storage_layout = StorageLayout.from_environment_config(
        environment_config, planned_run.plan.run_context.project_root
    )
    strategy = ApiStrategy(
        transport_factory=lambda: served,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    provider = SparkSessionProvider({}, {}, session_factory=session_factory)
    return SourceExecutor().execute(
        replace(planned_run, strategy=strategy), provider, environment_config
    )


def iceberg_session_factory(root: Path, app_name: str = "janus-order19") -> Callable[[], Any]:
    """A factory building a fresh Iceberg session over one warehouse, per call."""
    from tests.support.spark_sessions import build_iceberg_session, require_iceberg_runtime

    require_iceberg_runtime()

    def build() -> Any:
        return build_iceberg_session(app_name, root)

    return build


@contextmanager
def inspection_session(session_factory: Callable[[], Any]) -> Iterator[Any]:
    """A session the runs do not hold, stopped on exit so the next run starts its own."""
    session = session_factory()
    try:
        yield session
    finally:
        session.stop()


def snapshot_count(spark: Any, identifier: str) -> int:
    """``SELECT COUNT(*) FROM <t>.snapshots`` — the number of committed Iceberg snapshots."""
    metadata_table = quote_identifier(f"{identifier}.snapshots")
    return int(spark.sql(f"SELECT COUNT(*) AS n FROM {metadata_table}").collect()[0]["n"])


def table_exists(spark: Any, identifier: str) -> bool:
    return bool(spark.catalog.tableExists(identifier))


def current_metadata_file(spark: Any, identifier: str) -> str:
    """The table's current metadata file, the newest ``metadata_log_entries`` row."""
    metadata_table = quote_identifier(f"{identifier}.metadata_log_entries")
    row = spark.sql(
        f"SELECT file FROM {metadata_table} ORDER BY timestamp DESC LIMIT 1"
    ).collect()[0]
    return str(row["file"])


def table_uuid(spark: Any, identifier: str) -> str:
    """The Iceberg ``table-uuid`` read from the current metadata file (local warehouses)."""
    location = urlsplit(current_metadata_file(spark, identifier))
    path = Path(location.path) if location.scheme in {"", "file"} else None
    if path is None:
        raise NotImplementedError(f"table_uuid reads local metadata only, not {location.scheme}")
    return str(json.loads(path.read_text(encoding="utf-8"))["table-uuid"])


def table_schema(spark: Any, identifier: str) -> tuple[tuple[str, str], ...]:
    """``(name, Spark simpleString type)`` for every column, in table order."""
    return tuple(
        (column.name, column.dataType.simpleString())
        for column in spark.table(identifier).schema.fields
    )


def table_properties(spark: Any, identifier: str) -> dict[str, str]:
    rows = spark.sql(f"SHOW TBLPROPERTIES {quote_identifier(identifier)}").collect()
    return {str(row["key"]): str(row["value"]) for row in rows}


def bronze_rows(spark: Any, identifier: str, columns: Sequence[str]) -> list[tuple[Any, ...]]:
    """The named columns of every row, ordered by those columns (nulls first)."""
    frame = spark.table(identifier).select(*columns).orderBy(*columns)
    return [tuple(row) for row in frame.collect()]


def write_frame(
    spark: Any,
    plan: Any,
    rows: Sequence[tuple[Any, ...]],
    ddl_schema: str,
    *,
    intent: BronzeWriteIntent | None = None,
) -> WriteResult:
    """``SparkDatasetWriter.write`` over ``createDataFrame(rows, ddl_schema)`` — no materializer."""
    storage_layout = StorageLayout.from_environment_config(
        ENVIRONMENT_CONFIG, plan.run_context.project_root
    )
    dataframe = spark.createDataFrame(list(rows), ddl_schema)
    return SparkDatasetWriter(storage_layout).write(
        dataframe,
        plan,
        "bronze",
        intent=intent or resolve_bronze_write_intent(plan),
        apply_repartition=False,
    )


def validation_checks(executed: ExecutedRun) -> dict[str, Any]:
    """The run's validation checks keyed ``"<phase>.<name>"``; empty when none were run."""
    if executed.validation_report is None:
        return {}
    return {
        f"{check.phase}.{check.name}": check
        for check in executed.validation_report.report.checks
    }


def failed_check_names(executed: ExecutedRun) -> list[str]:
    if executed.validation_report is None:
        return []
    return [
        f"{check.phase}.{check.name}"
        for check in executed.validation_report.report.failed_checks
    ]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


__all__ = [
    "CONTRACTS_DIR",
    "DEFAULT_NAMESPACE",
    "ENVIRONMENT_CONFIG",
    "EnforcementCase",
    "FixtureTransport",
    "bronze_rows",
    "contract_property",
    "current_metadata_file",
    "default_contract_path",
    "execute_case_with_pages",
    "failed_check_names",
    "iceberg_session_factory",
    "inspection_session",
    "plan_case",
    "read_json",
    "render_contract",
    "render_source_config",
    "snapshot_count",
    "table_exists",
    "table_properties",
    "table_schema",
    "table_uuid",
    "validation_checks",
    "write_case_project",
    "write_frame",
]
