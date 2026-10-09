"""Offline live/replay cells sharing the M0 fixtures and all bronze provenance."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from janus.checkpoints import CheckpointStore
from janus.models import RunContext, SourceConfig
from janus.planner import HookCatalog, PlannedRun
from janus.readers import SparkDatasetReader
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.utils.storage import StorageLayout, bronze_table_identifier
from tests.fixtures.handoff.generate_pages import generate_pages, synthetic_source_mapping
from tests.support.bronze_digest import bronze_golden
from tests.support.contract_baseline import (
    _CASE_BY_NAME,
    _ENVIRONMENT_CONFIG,
    STARTED_AT,
    FixtureTransport,
    _prepare_case_project,
    _source_entry,
    _strategy_for,
)
from tests.support.contracts import with_registry_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_DIR = PROJECT_ROOT / "tests/integration/handoff/goldens"
SYNTHETIC_CASE = "synthetic_pages_500"


@dataclass
class MatrixReader(SparkDatasetReader):
    """Observe actual materializer reads; explicitly exercise inference when requested."""

    infer_schema: bool = False
    reads: list[dict[str, Any]] = field(default_factory=list)

    def read_extraction_result(self, spark, extraction_result, **kwargs):
        declared_schema = kwargs.get("schema")
        if self.infer_schema:
            kwargs["schema"] = None
        dataframe = super().read_extraction_result(spark, extraction_result, **kwargs)
        self.reads.append(
            {
                "artifact_count": len(extraction_result.artifacts),
                "partition_count": dataframe.rdd.getNumPartitions(),
                "schema_applied": declared_schema is not None and not self.infer_schema,
                "schema": dataframe.schema.jsonValue(),
                "handoff_metadata": extraction_result.metadata_as_dict(),
            }
        )
        corrupt_column = kwargs.get("corrupt_record_column")
        if self.infer_schema and corrupt_column and corrupt_column not in dataframe.columns:
            from pyspark.sql import functions as sf

            dataframe = dataframe.withColumn(corrupt_column, sf.lit(None).cast("string"))
        return dataframe


def prepare_fixture(case_name: str, project_root: Path):
    """Reuse the M0 source YAML, transport payloads and six-member archive."""
    if case_name == SYNTHETIC_CASE:
        return _prepare_synthetic(project_root)
    case = _CASE_BY_NAME[case_name]
    _prepare_case_project(case, PROJECT_ROOT, project_root)
    config_path = project_root / "conf/sources" / case.config_path
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = _source_entry(document, case.source_id)
    if case.family == "catalog":
        baseline = json.loads((GOLDENS_DIR / "baseline_goldens.json").read_text())
        mapping["outputs"]["raw"]["path"] = str(
            Path(baseline["metadata"]["work_root"])
            / "cases"
            / case_name
            / mapping["outputs"]["raw"]["path"]
        )
    return mapping, config_path, lambda: _strategy_for(case, PROJECT_ROOT, project_root)


def _prepare_synthetic(project_root: Path):
    from janus.strategies.api import ApiStrategy

    fixture_root = project_root / "responses"
    artifacts = generate_pages(fixture_root, page_count=500)
    terminal = project_root / "empty.json"
    terminal.write_text("[]\n", encoding="utf-8")
    paths = (*(Path(artifact.path) for artifact in artifacts), terminal)
    mapping = synthetic_source_mapping(project_root / "data/raw/synthetic")
    mapping["spark"]["handoff"] = {"mode": "compact", "max_artifact_mb": 1}
    config_path = project_root / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    layout = StorageLayout.from_environment_config(_ENVIRONMENT_CONFIG, project_root)

    def strategy_factory():
        transport = FixtureTransport(paths, STARTED_AT)
        return ApiStrategy(
            transport_factory=lambda: transport,
            storage_layout_factory=lambda _plan: layout,
            sleeper=lambda _seconds: None,
            clock=lambda: 0.0,
        )

    return mapping, config_path, strategy_factory


def identifier_for(mapping: dict) -> str:
    target = mapping["outputs"]["bronze"]
    return bronze_table_identifier(
        target["path"],
        fallback_name=mapping["source_id"],
        namespace=target.get("namespace"),
        table_name=target.get("table_name"),
    )


def plan_cell(mapping, config_path, strategy, *, mode, case_name, inferred=False):
    """Parse each frozen config variant from one YAML mapping through from_mapping."""
    variant = copy.deepcopy(mapping)
    handoff = variant["spark"].get("handoff", {})
    if not isinstance(handoff, dict):
        handoff = {}
    variant["spark"]["handoff"] = {**handoff, "mode": mode}
    identifier = identifier_for(mapping)
    namespace, table = identifier.rsplit(".", 1)
    suffix = f"__ac1_{case_name}_{'inferred' if inferred else 'explicit'}_{mode}_live"
    variant["outputs"]["bronze"].update(namespace=namespace, table_name=table + suffix)
    if inferred:
        project_root = config_path.parents[3]
        contract_path = project_root / variant["schema"]["contract"]
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        for prop in contract["schema"][0]["properties"]:
            if prop.get("physicalType") == "integer":
                prop["physicalType"] = "long"
        inferred_path = contract_path.with_name(contract_path.stem + "_inferred.yaml")
        inferred_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
        variant["schema"]["contract"] = str(inferred_path)
    config = SourceConfig.from_mapping(variant, config_path=config_path)
    context = RunContext.create(
        run_id="baseline" if case_name == SYNTHETIC_CASE else f"m0-{case_name}",
        environment="local",
        project_root=config_path.parents[3] if case_name != SYNTHETIC_CASE else config_path.parent,
        started_at=STARTED_AT,
    )
    plan = with_registry_contract(strategy.plan(config, context))
    hook = HookCatalog.with_defaults().resolve(config.source_hook, source_id=config.source_id)
    if case_name != SYNTHETIC_CASE and _CASE_BY_NAME[case_name].checkpoint is not None:
        CheckpointStore().save(plan, _CASE_BY_NAME[case_name].checkpoint)
    return PlannedRun(plan=plan, strategy=strategy, hook=hook)


def _provider(spark, target):
    return SparkSessionProvider.wrapping(spark, resolved_paths=target.resolved_paths)


def run_mode(spark, target, planned, *, case_name, canonical_identifier, inferred=False):
    """Run live then the CLI loader against its raw zone, each into a fresh table."""
    environment = {**_ENVIRONMENT_CONFIG, **target.environment_config()}
    live_identifier = bronze_table_identifier(
        planned.plan.bronze_output.path,
        fallback_name=planned.plan.source.source_id,
        namespace=planned.plan.bronze_output.namespace,
        table_name=planned.plan.bronze_output.table_name,
    )
    assert not spark.catalog.tableExists(live_identifier), live_identifier
    live_reader = MatrixReader(infer_schema=inferred)
    live = SourceExecutor(reader=live_reader).execute(
        planned, _provider(spark, target), environment
    )
    cells = {"live": capture_cell(spark, live, live_reader, canonical_identifier, live_identifier)}
    if planned.strategy.strategy_family == "catalog" and live.extraction_result is not None:
        for artifact in live.extraction_result.artifacts:
            if artifact.format == "json":
                os.utime(artifact.path, (STARTED_AT.timestamp(), STARTED_AT.timestamp()))
    replay_identifier = live_identifier.removesuffix("_live") + "_replay"
    assert not spark.catalog.tableExists(replay_identifier), replay_identifier
    replay_reader = MatrixReader(infer_schema=inferred)
    with patch.object(
        type(planned.strategy), "extract", side_effect=AssertionError("re-extraction")
    ):
        replay = RawToBronzeLoader(reader=replay_reader).ingest(
            planned,
            _provider(spark, target),
            environment,
            bronze_table=replay_identifier,
            verify_checksums=True,
        )
    cells["replay"] = capture_cell(
        spark, replay, replay_reader, canonical_identifier, replay_identifier
    )
    return cells


def capture_cell(spark, result, reader, canonical_identifier, expected_identifier):
    extraction = result.extraction_result
    extraction_metadata = extraction.metadata_as_dict() if extraction is not None else {}
    cell = {
        "status": result.status,
        "failure_reason": result.failure_reason,
        "expected_identifier": expected_identifier,
        "extraction_metadata": extraction_metadata,
        "handoff_compaction": result.strategy_metadata.get("handoff_compaction"),
        "handoff_artifact_count": sum(read["artifact_count"] for read in reader.reads),
        "spark_read_partition_count": sum(read["partition_count"] for read in reader.reads),
        "batch_artifact_counts": [read["artifact_count"] for read in reader.reads],
        "reads": reader.reads,
    }
    if result.status != "succeeded":
        return cell
    tables = {write.path for write in result.write_results if write.zone == "bronze"}
    cell["actual_identifiers"] = sorted(tables)
    if tables != {expected_identifier}:
        return cell
    cell["golden"] = bronze_golden(
        spark.table(expected_identifier), identifier=canonical_identifier
    )
    cell["bronze_data_file_count"] = spark.table(f"{expected_identifier}.files").count()
    cell["bronze_commit_count"] = len(
        [write for write in result.write_results if write.zone == "bronze"]
    )
    cell["schema"] = spark.table(expected_identifier).schema.jsonValue()
    return cell


def capture_additional_pages_goldens(output: Path):
    """Capture only the new inference and reduced-size cases, never refresh M0."""
    import tempfile

    from tests.support.spark_sessions import sqlite_catalog_target, start_session

    captures = {}
    with tempfile.TemporaryDirectory(prefix="ac1-pages-capture-") as temporary:
        root = Path(temporary)
        target = sqlite_catalog_target(root / "catalog")
        target.prepare()
        spark = start_session("ac1-additional-pages-baselines", target.session_options())
        try:
            for case_name, inferred in (
                ("transparencia_servidores_por_orgao", True),
                (SYNTHETIC_CASE, False),
            ):
                project = root / case_name
                mapping, config_path, strategy_factory = prepare_fixture(case_name, project)
                planned = plan_cell(
                    mapping,
                    config_path,
                    strategy_factory(),
                    mode="pages",
                    case_name=case_name,
                    inferred=inferred,
                )
                reader = MatrixReader(infer_schema=inferred)
                with patch.dict("os.environ", {"TRANSPARENCIA_API_TOKEN": "fixture-token"}):
                    executed = SourceExecutor(reader=reader).execute(
                        planned,
                        _provider(spark, target),
                        {**_ENVIRONMENT_CONFIG, **target.environment_config()},
                    )
                identifier = identifier_for(mapping)
                expected = identifier_for(
                    {
                        **mapping,
                        "outputs": {
                            **mapping["outputs"],
                            "bronze": {
                                "path": planned.plan.bronze_output.path,
                                "namespace": planned.plan.bronze_output.namespace,
                                "table_name": planned.plan.bronze_output.table_name,
                            },
                        },
                    }
                )
                cell = capture_cell(spark, executed, reader, identifier, expected)
                assert executed.status == "succeeded", json.dumps(cell, default=str)
                name = f"{case_name}_inferred" if inferred else case_name
                captures[name] = cell["golden"]
        finally:
            spark.stop()
    output.write_text(
        json.dumps(
            {
                "_comment": "Capture pages/live only: "
                "python -m tests.support.handoff_matrix OUTPUT.json. "
                "M0 baseline_goldens.json is never regenerated by this command.",
                "metadata": {
                    "synthetic_page_count": 500,
                    "seed": 22,
                    "records_per_page": 15,
                    "provenance": "All schema and row columns hashed verbatim.",
                },
                "baseline_goldens": captures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    capture_additional_pages_goldens(parser.parse_args().output)
