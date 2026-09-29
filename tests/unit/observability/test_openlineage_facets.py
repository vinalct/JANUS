"""OpenLineage 2-0-2 mapping, schema, identity and architecture contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from janus.checkpoints import CheckpointWriteResult
from janus.lineage import (
    ArtifactSnapshot,
    ConfiguredOutput,
    LineageRecord,
    MaterializedOutput,
    RunMetadata,
)
from janus.models import SourceDependencyEdge, SourceDependencyGraph, SourceDependencyNode
from janus.models.data_contracts import (
    ContractProperty,
    ContractSchema,
    DataContract,
    JanusContractOptions,
)
from janus.observability import RunEvidencePaths, RunRecord
from janus.observability.openlineage import (
    CUSTOM_ONLY_LINEAGE_FIELDS,
    DELIBERATELY_DROPPED_LINEAGE_FIELDS,
    LINEAGE_FIELD_MAPPING,
    OPENLINEAGE_SCHEMA_URL,
    OPENLINEAGE_SPEC_VERSION,
    SCHEMA_DATASET_FACET_SCHEMA_URL,
    OpenLineageDatasetContext,
    build_openlineage_run_event,
    openlineage_run_id,
)
from janus.quality import ValidationCheck, ValidationReport

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "openlineage"
OPENLINEAGE_SCHEMA = FIXTURES / "OpenLineage-2-0-2.json"
JANUS_FACET_SCHEMA = (
    PROJECT_ROOT / "docs" / "schemas" / "openlineage" / "JanusRunFacet.json"
)
SCHEMA_DATASET_FACET_SCHEMA = FIXTURES / "SchemaDatasetFacet-1-1-1.json"

STARTED_AT = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 4, 12, 0, 5, tzinfo=UTC)
SOURCE_ID = "consumer"
RAW_PATH = "/data/raw/consumer/runs/run/page-0001.json"
BRONZE_TABLE = "bronze.consumer"
DATASETS = OpenLineageDatasetContext(
    catalog_name="janus",
    warehouse="s3://janus-bronze/warehouse",
)


def _data_contract(
    properties: tuple[ContractProperty, ...],
    *,
    purpose: str = (
        "Contract example that exercises the api family against example.invalid; "
        "not a live source."
    ),
) -> DataContract:
    return DataContract(
        contract_path=Path("conf/contracts/example/federal_open_data_example.yaml"),
        api_version="v3.2.0",
        id="example.consumer",
        name="Example consumer contract",
        version="1.0.0",
        status="active",
        domain="example",
        purpose=purpose,
        owners=("janus",),
        tags=("example",),
        schema=ContractSchema(
            name="consumer",
            physical_type="table",
            properties=properties,
        ),
        janus=JanusContractOptions(compatibility="additive", enforcement="lenient"),
        schema_version="a" * 64,
    )


EXAMPLE_CONTRACT = _data_contract(
    (
        ContractProperty(
            name="id",
            physical_type="string",
            logical_type="string",
            description="Upstream record identifier, verbatim.",
        ),
        ContractProperty(
            name="updated_at",
            physical_type="string",
            logical_type="string",
            description="Upstream update timestamp, kept as the string the API sent.",
        ),
    )
)
GRAPH = SourceDependencyGraph(
    nodes=(
        SourceDependencyNode(SOURCE_ID, enabled=True, bronze_table=BRONZE_TABLE),
        SourceDependencyNode("producer", enabled=True, bronze_table="bronze.producer"),
    ),
    edges=(
        SourceDependencyEdge(
            producer_id="producer",
            consumer_id=SOURCE_ID,
            table="bronze.producer",
            input_paths=("access.request_inputs",),
        ),
    ),
)


def _configured_outputs() -> tuple[ConfiguredOutput, ...]:
    return (
        ConfiguredOutput("raw", "/data/raw/consumer", "json"),
        ConfiguredOutput("bronze", "/data/bronze/consumer", "iceberg"),
        ConfiguredOutput("metadata", "/data/metadata/consumer", "json"),
    )


def _materialized_outputs(*, bronze: bool = True) -> tuple[MaterializedOutput, ...]:
    raw = MaterializedOutput(
        zone="raw",
        path=RAW_PATH,
        format="json",
        mode="overwrite",
        records_written=1,
        partition_by=("ingestion_date",),
        metadata=(("checksum", "abc123"),),
    )
    if not bronze:
        return (raw,)
    return (
        raw,
        MaterializedOutput(
            zone="bronze",
            path=BRONZE_TABLE,
            format="iceberg",
            mode="overwrite",
            records_written=2,
            partition_by=("ingestion_date",),
            metadata=(("writer", "spark"),),
        ),
    )


def _run_metadata(
    *,
    run_id: str,
    status: str,
    source_id: str = SOURCE_ID,
    environment: str = "local",
    outputs: tuple[MaterializedOutput, ...] = (),
    records_extracted: int | None = None,
    failure_reason: str | None = None,
    error_type: str | None = None,
    attributes: tuple[tuple[str, str], ...] = (),
) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        source_id=source_id,
        source_name=f"Source {source_id}",
        environment=environment,
        strategy_family="api",
        strategy_variant="page_number_api",
        extraction_mode="incremental",
        checkpoint_strategy="max_value",
        checkpoint_field="updated_at",
        status=status,
        started_at=STARTED_AT,
        ended_at=FINISHED_AT if status != "running" else None,
        duration_seconds=5.0 if status != "running" else None,
        source_config_path="conf/sources/consumer.yaml",
        configured_outputs=_configured_outputs(),
        materialized_outputs=outputs,
        records_extracted=records_extracted,
        checkpoint_value="2026-07-02" if records_extracted is not None else None,
        failure_reason=failure_reason,
        error_type=error_type,
        run_attributes=attributes,
        plan_notes=("dispatch:api.page_number_api",),
        metadata=(("strategy.request_count", "1"),),
        schema_version="e" * 64,
        contract_id="example.consumer",
        contract_version="1.0.0",
    )


def _lineage(metadata: RunMetadata, *, replay: bool = False) -> LineageRecord:
    return LineageRecord(
        run_id=metadata.run_id,
        source_id=metadata.source_id,
        source_name=metadata.source_name,
        environment=metadata.environment,
        strategy_family=metadata.strategy_family,
        strategy_variant=metadata.strategy_variant,
        extraction_mode=metadata.extraction_mode,
        checkpoint_strategy=metadata.checkpoint_strategy,
        checkpoint_field=metadata.checkpoint_field,
        source_hook="janus.hooks.example",
        status=metadata.status,
        emitted_at=FINISHED_AT,
        source_config_path=metadata.source_config_path,
        config_version="f" * 64,
        configured_outputs=metadata.configured_outputs,
        materialized_outputs=metadata.materialized_outputs,
        artifacts=(ArtifactSnapshot(RAW_PATH, "json", "abc123"),)
        if metadata.materialized_outputs
        else (),
        records_extracted=metadata.records_extracted,
        checkpoint_value=metadata.checkpoint_value,
        failure_reason=metadata.failure_reason,
        error_type=metadata.error_type,
        run_attributes=metadata.run_attributes,
        plan_notes=metadata.plan_notes,
        extraction_metadata=(
            (("raw_to_bronze", "true"),)
            if replay
            else (("request_count", "1"),)
        ),
        metadata=metadata.metadata,
        schema_version=metadata.schema_version,
        contract_id=metadata.contract_id,
        contract_version=metadata.contract_version,
    )


def _validation_report(metadata: RunMetadata, *, failed: bool) -> ValidationReport:
    check = (
        ValidationCheck.failed("data", "required_fields", "updated_at is null")
        if failed
        else ValidationCheck.passed("data", "required_fields", "all present")
    )
    return ValidationReport(
        run_id=metadata.run_id,
        source_id=metadata.source_id,
        source_name=metadata.source_name,
        environment=metadata.environment,
        strategy_family=metadata.strategy_family,
        strategy_variant=metadata.strategy_variant,
        emitted_at=FINISHED_AT,
        checks=(check,),
    )


def _terminal_records(
    shape: str,
    *,
    run_id: str | None = None,
    source_id: str = SOURCE_ID,
    environment: str = "local",
) -> tuple[RunMetadata, LineageRecord, RunRecord]:
    failed = shape in {"extraction_failure", "quality_failure"}
    extraction_failed = shape == "extraction_failure"
    empty_handoff = shape == "empty_handoff"
    replay = shape == "replay"
    outputs = () if extraction_failed else _materialized_outputs(bronze=not empty_handoff)
    records_extracted = None if extraction_failed or replay else 2
    attributes = (
        (
            ("pipeline_attempt", "2"),
            ("pipeline_run_id", "pipeline-20260704"),
            ("trigger", "run-all"),
        )
        if shape == "batch_attempt"
        else ()
    )
    reason = (
        "scripted extraction failure"
        if extraction_failed
        else "Quality validation failed: data.required_fields"
        if shape == "quality_failure"
        else None
    )
    metadata = _run_metadata(
        run_id=run_id or f"order15-{shape}",
        status="failed" if failed else "succeeded",
        source_id=source_id,
        environment=environment,
        outputs=outputs,
        records_extracted=records_extracted,
        failure_reason=reason,
        error_type="RuntimeError" if failed else None,
        attributes=attributes,
    )
    lineage = _lineage(metadata, replay=replay)
    validation = (
        None
        if extraction_failed
        else _validation_report(metadata, failed=shape == "quality_failure")
    )
    checkpoint = (
        CheckpointWriteResult(
            state=None,
            decision="skipped" if replay else "advanced",
            advanced=not replay,
            current_path=Path("metadata/checkpoints/current.json"),
            history_path=Path("metadata/checkpoints/history/run.json"),
        )
        if not failed
        else None
    )
    run_record = RunRecord.from_run(
        metadata,
        lineage,
        emitted_at=FINISHED_AT,
        checkpoint_result=checkpoint,
        validation_report=validation,
        evidence=RunEvidencePaths(
            run_metadata_path="metadata/runs/run.json",
            lineage_path="metadata/lineage/run.json",
            validation_report_path=(
                "metadata/validations/run.json" if validation is not None else None
            ),
        ),
    )
    return metadata, lineage, run_record


def _terminal_event(
    shape: str,
    *,
    data_contract: DataContract | None = EXAMPLE_CONTRACT,
    **kwargs: str,
) -> dict:
    metadata, lineage, run_record = _terminal_records(shape, **kwargs)
    graph = GRAPH if metadata.source_id == SOURCE_ID else None
    return build_openlineage_run_event(
        metadata,
        DATASETS,
        lineage_record=lineage,
        run_record=run_record,
        graph=graph,
        data_contract=data_contract,
    )


def _validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(
        json.loads(path.read_text(encoding="utf-8")),
        format_checker=FormatChecker(),
    )


def _schema_dataset_facet_validator() -> Draft202012Validator:
    facet_schema = json.loads(SCHEMA_DATASET_FACET_SCHEMA.read_text(encoding="utf-8"))
    core_schema = json.loads(OPENLINEAGE_SCHEMA.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        "https://openlineage.io/spec/2-0-2/OpenLineage.json",
        Resource.from_contents(core_schema),
    )
    return Draft202012Validator(
        facet_schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


@pytest.mark.parametrize(
    "shape",
    [
        "success",
        "extraction_failure",
        "quality_failure",
        "empty_handoff",
        "replay",
        "batch_attempt",
    ],
)
def test_every_terminal_shape_validates_against_the_pinned_schema(shape):
    event = _terminal_event(shape)

    _validator(OPENLINEAGE_SCHEMA).validate(event)
    _validator(JANUS_FACET_SCHEMA).validate(event["run"]["facets"]["janusRun"])
    for output in event["outputs"]:
        schema_facet = output.get("facets", {}).get("schema")
        if schema_facet is not None:
            _schema_dataset_facet_validator().validate({"schema": schema_facet})


def test_bronze_output_carries_the_schema_facet_rendered_from_the_contract():
    metadata, lineage, run_record = _terminal_records("success")
    contract = _data_contract(
        (
            ContractProperty(
                name="estabelecimento",
                physical_type="struct",
                logical_type="object",
                description="Merchant establishment.",
                properties=(
                    ContractProperty(
                        name="nome",
                        physical_type="string",
                        logical_type="string",
                        description="Merchant name.",
                    ),
                    ContractProperty(
                        name="identificacao",
                        physical_type="struct",
                        logical_type="object",
                        description="Merchant identifiers.",
                        properties=(
                            ContractProperty(
                                name="cnpjFormatado",
                                physical_type="string",
                                logical_type="string",
                                description="Formatted CNPJ.",
                            ),
                        ),
                    ),
                ),
            ),
            ContractProperty(
                name="valorTransacao",
                physical_type="decimal(18,2)",
                logical_type="number",
                description="Transaction amount.",
            ),
            ContractProperty(
                name="dataTransacao",
                physical_type="timestamptz",
                logical_type="date-time",
                description="Transaction timestamp.",
            ),
        ),
        purpose="Card expenses by transaction date.",
    )
    event = build_openlineage_run_event(
        metadata,
        DATASETS,
        lineage_record=lineage,
        run_record=run_record,
        data_contract=contract,
    )
    bronze = next(output for output in event["outputs"] if output["name"] == BRONZE_TABLE)
    schema_facet = bronze["facets"]["schema"]

    assert schema_facet["_schemaURL"] == SCHEMA_DATASET_FACET_SCHEMA_URL
    assert schema_facet["fields"] == [
        {
            "name": "estabelecimento",
            "type": "struct",
            "description": "Merchant establishment.",
            "fields": [
                {"name": "nome", "type": "string", "description": "Merchant name."},
                {
                    "name": "identificacao",
                    "type": "struct",
                    "description": "Merchant identifiers.",
                    "fields": [
                        {
                            "name": "cnpjFormatado",
                            "type": "string",
                            "description": "Formatted CNPJ.",
                        }
                    ],
                },
            ],
        },
        {
            "name": "valorTransacao",
            "type": "decimal(18,2)",
            "description": "Transaction amount.",
        },
        {
            "name": "dataTransacao",
            "type": "timestamptz",
            "description": "Transaction timestamp.",
        },
    ]


def test_raw_artifact_outputs_carry_no_schema_facet():
    event = _terminal_event("success", data_contract=EXAMPLE_CONTRACT)
    raw = next(output for output in event["outputs"] if output["name"] == RAW_PATH)
    bronze = next(output for output in event["outputs"] if output["name"] == BRONZE_TABLE)

    assert "facets" not in raw
    assert "schema" in bronze["facets"]


def test_no_contract_means_no_schema_facet_and_source_name_documentation():
    event = _terminal_event("success", data_contract=None)
    expected = json.loads((FIXTURES / "success.json").read_text(encoding="utf-8"))
    expected["job"]["facets"]["documentation"]["description"] = "Source consumer"
    bronze = next(output for output in expected["outputs"] if output["name"] == BRONZE_TABLE)
    bronze.pop("facets")

    assert event == expected


def test_documentation_facet_uses_contract_purpose():
    event = _terminal_event("success", data_contract=EXAMPLE_CONTRACT)

    assert (
        event["job"]["facets"]["documentation"]["description"]
        == EXAMPLE_CONTRACT.purpose
    )


def test_start_complete_and_fail_lifecycle_mapping_uses_recorded_timestamps():
    started = _run_metadata(run_id="paired-run", status="running")
    start = build_openlineage_run_event(
        started, DATASETS, graph=GRAPH, data_contract=EXAMPLE_CONTRACT
    )
    _validator(OPENLINEAGE_SCHEMA).validate(start)
    _validator(JANUS_FACET_SCHEMA).validate(start["run"]["facets"]["janusRun"])
    complete = _terminal_event("success", run_id="paired-run")
    failed = _terminal_event("extraction_failure")

    assert (start["eventType"], complete["eventType"], failed["eventType"]) == (
        "START",
        "COMPLETE",
        "FAIL",
    )
    assert start["eventTime"] == STARTED_AT.isoformat()
    assert complete["eventTime"] == FINISHED_AT.isoformat()
    assert start["run"]["runId"] == complete["run"]["runId"]


def test_run_uuid_is_valid_deterministic_cross_process_and_collision_resistant():
    run_id = "pipeline-consumer-a2-cafe1234"
    expected = openlineage_run_id(run_id)
    command = (
        "from janus.observability.openlineage import openlineage_run_id; "
        f"print(openlineage_run_id({run_id!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )

    assert UUID(expected).version == 5
    assert completed.stdout.strip() == expected
    assert openlineage_run_id("another-run") != expected


def test_job_identity_is_source_scoped_and_environment_scoped():
    first = _terminal_event("success", run_id="run-1")
    second = _terminal_event("success", run_id="run-2")
    other_source = _terminal_event("success", source_id="other-source")
    other_environment = _terminal_event("success", environment="prod")

    assert first["job"] == second["job"]
    assert first["job"]["name"] != other_source["job"]["name"]
    assert first["job"]["namespace"] != other_environment["job"]["namespace"]


def test_declared_graph_edge_becomes_the_iceberg_input_dataset():
    event = _terminal_event("success")

    assert event["inputs"] == [
        {
            "namespace": "iceberg://janus/s3%3A%2F%2Fjanus-bronze%2Fwarehouse",
            "name": "bronze.producer",
        }
    ]
    assert event["run"]["facets"]["janusRun"]["declared_inputs"] == [
        {
            "producer_id": "producer",
            "table": "bronze.producer",
            "input_paths": ["access.request_inputs"],
        }
    ]


def test_absent_contract_identity_is_omitted_from_the_janus_run_facet():
    metadata, lineage, run_record = _terminal_records("success")
    metadata = replace(
        metadata, schema_version=None, contract_id=None, contract_version=None
    )
    lineage = replace(
        lineage, schema_version=None, contract_id=None, contract_version=None
    )
    event = build_openlineage_run_event(
        metadata,
        DATASETS,
        lineage_record=lineage,
        run_record=run_record,
        graph=GRAPH,
    )

    facet = event["run"]["facets"]["janusRun"]
    assert not {"schema_version", "contract_id", "contract_version"} & facet.keys()


def test_every_lineage_field_has_a_mapping_decision_and_custom_only_fields_survive():
    event = _terminal_event("quality_failure")
    facet = event["run"]["facets"]["janusRun"]

    assert set(LINEAGE_FIELD_MAPPING) == {field.name for field in fields(LineageRecord)}
    assert not DELIBERATELY_DROPPED_LINEAGE_FIELDS
    assert set(facet) >= CUSTOM_ONLY_LINEAGE_FIELDS
    assert facet["run_id"] == "order15-quality_failure"
    assert facet["error_type"] == "RuntimeError"
    assert facet["quality"] == {
        "outcome": "failed",
        "checks_passed": 0,
        "checks_failed": 1,
        "checks_skipped": 0,
        "failed_checks": ["data.required_fields"],
    }


@pytest.mark.parametrize(
    ("shape", "fixture_name"),
    [("success", "success.json"), ("quality_failure", "quality_failure.json")],
)
def test_success_and_quality_failure_payloads_match_goldens(shape, fixture_name):
    expected = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    assert _terminal_event(shape) == expected


def test_schema_version_and_vendored_bytes_are_pinned():
    expected = (FIXTURES / "OpenLineage-2-0-2.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(OPENLINEAGE_SCHEMA.read_bytes()).hexdigest()
    facet_expected = (
        FIXTURES / "SchemaDatasetFacet-1-1-1.sha256"
    ).read_text(encoding="utf-8").split()[0]
    facet_actual = hashlib.sha256(SCHEMA_DATASET_FACET_SCHEMA.read_bytes()).hexdigest()

    assert OPENLINEAGE_SPEC_VERSION == "2-0-2"
    assert OPENLINEAGE_SCHEMA_URL == (
        "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
    )
    assert actual == expected
    assert SCHEMA_DATASET_FACET_SCHEMA_URL == (
        "https://openlineage.io/spec/facets/1-1-1/SchemaDatasetFacet.json"
        "#/$defs/SchemaDatasetFacet"
    )
    assert facet_actual == facet_expected


def test_mapping_imports_no_compute_or_catalog_engine():
    command = """
import json
import sys
import janus.observability.openlineage
forbidden = sorted(
    name for name in sys.modules
    if name.split('.', 1)[0] in {'pyarrow', 'pyiceberg', 'pyspark'}
)
print(json.dumps(forbidden))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(completed.stdout) == []


def test_same_inputs_produce_byte_identical_payloads():
    first = json.dumps(_terminal_event("batch_attempt"), separators=(",", ":"))
    second = json.dumps(_terminal_event("batch_attempt"), separators=(",", ":"))

    assert first == second


def test_terminal_records_must_describe_one_run():
    metadata, lineage, run_record = _terminal_records("success")

    with pytest.raises(ValueError, match="lineage_record must describe run"):
        build_openlineage_run_event(
            metadata,
            DATASETS,
            lineage_record=replace(lineage, run_id="another-run"),
            run_record=run_record,
        )


# --------------------------------------------------------------------------------------
# the janusRun facet carries what the contract decided (FR-8, D-12)
# --------------------------------------------------------------------------------------

RED_TASK = pytest.mark.xfail(strict=True, reason="red until implementation finishes")
ENFORCEMENT_FACET_FIELDS = frozenset(
    {"contract_preflight_outcome", "schema_evolution", "malformed_rows", "failure_stage"}
)


def _enforcement_event(metadata: RunMetadata, lineage: LineageRecord, report: ValidationReport):
    run_record = RunRecord.from_run(
        metadata,
        lineage,
        emitted_at=FINISHED_AT,
        validation_report=report,
        evidence=RunEvidencePaths(
            run_metadata_path="metadata/runs/run.json",
            lineage_path="metadata/lineage/run.json",
            validation_report_path="metadata/validations/run.json",
        ),
    )
    return build_openlineage_run_event(
        metadata,
        DATASETS,
        lineage_record=lineage,
        run_record=run_record,
        graph=GRAPH,
        data_contract=EXAMPLE_CONTRACT,
    )


def test_an_enforcement_failure_names_its_stage():
    metadata = replace(
        _run_metadata(
            run_id="enforcement",
            status="failed",
            outputs=(raw,),
            records_extracted=2,
            failure_reason="example.consumer v1.0.0: frame does not match the contract",
            error_type="ContractViolationError",
        ),
        failure_stage="contract_check",
    )
    lineage = replace(_lineage(metadata), failure_stage="contract_check")

    event = _enforcement_event(metadata, lineage, _validation_report(metadata, failed=True))
    facet = event["run"]["facets"]["janusRun"]

    assert facet["failure_stage"] == "contract_check"
    assert facet["error_type"] == "ContractViolationError"
    _validator(JANUS_FACET_SCHEMA).validate(facet)


class TestOrder19JanusRunFacet:

    @RED_TASK
    def test_a_run_carries_its_preflight_evolution_and_malformed_count(self):
        raw, bronze = _materialized_outputs()
        bronze = replace(bronze, metadata=(("schema_evolution", "added:note"),))
        metadata = _run_metadata(
            run_id="enforcement",
            status="succeeded",
            outputs=(raw, bronze),
            records_extracted=2,
            attributes=(("contract_preflight_outcome", "will_evolve"),),
        )
        lineage = _lineage(metadata)
        report = replace(
            _validation_report(metadata, failed=False),
            checks=(
                ValidationCheck.passed("data", "required_fields", "all present"),
                ValidationCheck.passed(
                    "data", "malformed_rows", "no malformed rows", details={"count": 0}
                ),
            ),
        )

        event = _enforcement_event(metadata, lineage, report)
        facet = event["run"]["facets"]["janusRun"]

        assert facet["contract_preflight_outcome"] == "will_evolve"
        assert facet["schema_evolution"] == "added:note"
        assert facet["malformed_rows"] == 0
        assert facet["failure_stage"] is None
        _validator(OPENLINEAGE_SCHEMA).validate(event)
        _validator(JANUS_FACET_SCHEMA).validate(facet)

    @RED_TASK
    def test_the_facet_schema_declares_the_four_fields_as_optional(self):
        facet_schema = json.loads(JANUS_FACET_SCHEMA.read_text(encoding="utf-8"))
        declared = facet_schema["$defs"]["JanusRunFacet"]

        assert set(declared["properties"]) >= ENFORCEMENT_FACET_FIELDS
        assert not ENFORCEMENT_FACET_FIELDS & set(declared["required"])
        assert declared["properties"]["malformed_rows"] == {"type": ["integer", "null"]}
