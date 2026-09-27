from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    SourceConfig,
    WriteResult,
)
from janus.models.data_contracts import DataContract, contract_from_legacy_schema_file
from janus.planner import PlannedRun
from janus.runtime.materialize import BronzeMaterializer
from janus.schema_contracts import resolve_spark_schema_for_plan
from janus.utils.storage import StorageLayout

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}
COLUMNS = ("event_id", "event_date", "payload")


# ── the resolution itself ─────────────────────────────────────────────────────


def test_the_read_schema_is_generated_from_the_contract_the_plan_carries(tmp_path: Path):
    plan = _plan(tmp_path, contract=_contract(tmp_path))

    schema = resolve_spark_schema_for_plan(plan)

    assert schema is not None
    assert tuple(schema.fieldNames()) == COLUMNS


def test_resolving_the_read_schema_opens_no_file(tmp_path: Path, monkeypatch):
    """The declaration was read once, at registry load; a read here would be a second one."""
    plan = _plan(tmp_path, contract=_contract(tmp_path))
    reads: list[str] = []

    for method in ("read_text", "read_bytes"):
        original = getattr(Path, method)

        def recording(self, *args, _original=original, _method=method, **kwargs):
            reads.append(f"{_method}:{self}")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method, recording)

    resolve_spark_schema_for_plan(plan)

    assert reads == []


def test_a_source_that_declares_nothing_still_hands_the_reader_no_schema(tmp_path: Path):
    """``infer`` is unchanged: Spark decides the types, exactly as it did before."""
    assert resolve_spark_schema_for_plan(_plan(tmp_path)) is None


# ── the materializer's call site ──────────────────────────────────────────────


def test_the_materializer_reads_with_the_generated_schema(tmp_path: Path):
    plan = _plan(tmp_path, contract=_contract(tmp_path))
    reader = SchemaRecordingReader()

    _materialize(plan, reader, tmp_path, artifact_format="json")

    assert tuple(reader.schemas[0].fieldNames()) == COLUMNS
    assert reader.options == [plan.source_config.spark.read_options]


def test_a_handoff_in_another_format_is_still_read_without_a_schema(tmp_path: Path):
    """The format condition is the one that decides, and this order did not touch it."""
    plan = _plan(tmp_path, contract=_contract(tmp_path))
    reader = SchemaRecordingReader()

    _materialize(plan, reader, tmp_path, artifact_format="csv")

    assert reader.schemas == [None]
    assert reader.options == [None]


def test_the_read_event_names_the_contract_it_was_shaped_by(tmp_path: Path):
    contract = _contract(tmp_path)
    plan = _plan(tmp_path, contract=contract)
    logger = RecordingLogger()

    _materialize(
        plan, SchemaRecordingReader(), tmp_path, artifact_format="json", logger=logger
    )

    fields = logger.fields_for("spark_read_started")
    assert fields["contract_id"] == contract.id
    assert fields["schema_version"] == contract.schema_version


def test_the_read_event_of_an_inferred_source_names_no_contract(tmp_path: Path):
    logger = RecordingLogger()

    _materialize(
        _plan(tmp_path), SchemaRecordingReader(), tmp_path, artifact_format="json", logger=logger
    )

    fields = logger.fields_for("spark_read_started")
    assert "contract_id" not in fields
    assert "schema_version" not in fields


# ── fakes ─────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SchemaRecordingReader:
    """Captures the schema and options the materializer hands one read."""

    schemas: list[Any] = field(default_factory=list)
    options: list[Any] = field(default_factory=list)

    def read_extraction_result(
        self, spark, extraction_result, format_name=None, schema=None, options=None
    ):
        del spark, extraction_result, format_name
        self.schemas.append(schema)
        self.options.append(options)
        return object()


@dataclass(slots=True)
class FakeNormalizer:
    def normalize(self, dataframe, plan):
        del dataframe, plan
        return object()


@dataclass(slots=True)
class FakeWriter:
    def write(self, dataframe, plan, zone, *, intent=None, count_records=False, **kwargs):
        del dataframe, count_records, kwargs
        return WriteResult.from_plan(
            plan,
            zone,
            path=plan.bronze_output.path,
            format_name="iceberg",
            mode=intent.reported_mode,
            records_written=None,
            partition_by=intent.partition_columns,
        )


@dataclass(slots=True)
class RecordingLogger:
    """Only the events and fields — enough to assert what a read announced."""

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def info(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def fields_for(self, event: str) -> dict[str, Any]:
        for name, fields in self.events:
            if name == event:
                return fields
        raise AssertionError(f"{event!r} was never logged; saw {[n for n, _ in self.events]}")


# ── builders ──────────────────────────────────────────────────────────────────


def _materialize(
    plan: ExecutionPlan,
    reader: SchemaRecordingReader,
    tmp_path: Path,
    *,
    artifact_format: str,
    logger: RecordingLogger | None = None,
) -> None:
    handoff = ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(
                path=str(tmp_path / "data" / "raw" / f"page-0001.{artifact_format}"),
                format=artifact_format,
            ),
        ),
        records_extracted=1,
    )
    BronzeMaterializer(
        reader=reader,
        normalizer=FakeNormalizer(),
        writer_factory=lambda storage_layout: FakeWriter(),
    ).materialize(
        PlannedRun(plan=plan, strategy=SimpleNamespace(strategy_family="api"), hook=None),
        plan,
        SimpleNamespace(),  # spark — untouched by the fakes
        handoff,
        StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path),
        logger,
    )


def _contract(tmp_path: Path) -> DataContract:
    """A contract converted from a legacy columns-only file, as the registry converts it."""
    schema_path = tmp_path / "conf" / "schemas" / "generated_read.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps({"columns": list(COLUMNS)}), encoding="utf-8")
    return contract_from_legacy_schema_file(
        schema_path,
        source_id="generated_read_fixture",
        bronze_table="bronze_test.generated_read_fixture",
        domain="example",
        project_root=tmp_path,
    )


def _plan(tmp_path: Path, *, contract: DataContract | None = None) -> ExecutionPlan:
    run_context = RunContext.create(
        run_id="run-generated-read-schema-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(
        _source_config(tmp_path), run_context, data_contract=contract
    )


def _source_config(tmp_path: Path) -> SourceConfig:
    source_id = "generated_read_fixture"
    payload: dict[str, Any] = {
        "source_id": source_id,
        "name": source_id,
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/events",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 5},
        },
        "extraction": {
            "mode": "full_refresh",
            "dead_letter_max_items": 0,
            "retry": {"max_attempts": 3, "backoff_strategy": "fixed", "backoff_seconds": 1},
        },
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "json",
            "write_mode": "overwrite",
            "repartition": 1,
            "partition_by": [],
            "read_options": {"multiLine": "true"},
        },
        "outputs": {
            "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
            "bronze": {
                "path": f"data/bronze/example/{source_id}",
                "format": "iceberg",
                "namespace": "bronze_test",
                "table_name": source_id,
            },
            "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
        },
        "quality": {
            "required_fields": [],
            "unique_fields": [],
            "allow_schema_evolution": True,
        },
    }
    config_path = tmp_path / "conf" / "sources" / f"{source_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return SourceConfig.from_mapping(payload, config_path)
