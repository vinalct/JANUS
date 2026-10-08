import io
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from janus.models import ExecutionPlan, ExtractedArtifact, SourceConfig, WriteResult
from janus.planner import PlannedRun
from janus.quality import CORRUPT_RECORD_COLUMN
from janus.runtime.materialize import (
    BronzeMaterializer,
    _read_batch,
    _tracks_corrupt_records,
    handoff_format_satisfies,
)
from janus.schema_contracts import spark_schema_from_contract
from janus.utils.logging import build_structured_logger
from tests.fixtures.handoff.cases import build_raw_zone, source_mapping
from tests.support.contract_frames import read_frame
from tests.support.contracts import with_registry_contract, write_minimal_contract


@dataclass
class SpyReader:
    supports_corrupt_record_read = True
    captured: dict[str, Any] | None = None

    def read_extraction_result(self, spark, handoff, **kwargs):
        self.captured = kwargs
        return read_frame(kwargs["schema"])


def _read(tmp_path, declared):
    zone = build_raw_zone(
        tmp_path,
        spark={
            "input_format": declared,
            "read_options": {"mode": "FAILFAST", "samplingRatio": "1.0"},
        },
    )
    write_minimal_contract(tmp_path)
    plan = with_registry_contract(zone.plan)
    handoff = replace(
        zone.extraction,
        artifacts=(
            ExtractedArtifact(
                path=str(tmp_path / "records-0001.jsonl"),
                format="jsonl",
            ),
        ),
    )
    reader = SpyReader()
    _read_batch(reader, object(), handoff, plan, plan.data_contract, None, {})
    return plan, reader


def test_jsonl_handoff_keeps_resolved_json_contract_schema(tmp_path):
    plan, reader = _read(tmp_path, "json")
    assert reader.captured["schema"] is not None
    assert reader.captured["schema"].fieldNames() == ["id", CORRUPT_RECORD_COLUMN]
    assert plan.data_contract is not None
    assert reader.captured["schema"].jsonValue() == spark_schema_from_contract(
        plan.data_contract, with_corrupt_record=True
    ).jsonValue()
    # The spy asserts on the argument, before any inference can conceal a missing schema.
    try:
        from pyspark.sql.types import StructType
    except ImportError:
        return
    assert isinstance(reader.captured["schema"], StructType)


def test_jsonl_handoff_keeps_declared_json_read_options(tmp_path):
    plan, reader = _read(tmp_path, "json")
    assert reader.captured["options"] == plan.source_config.spark.read_options
    assert reader.captured["options"] is plan.source_config.spark.read_options


def test_jsonl_handoff_does_not_satisfy_a_csv_declaration(tmp_path):
    _, reader = _read(tmp_path, "csv")
    assert reader.captured["schema"] is None
    assert reader.captured["options"] is None
    assert "corrupt_record_column" not in reader.captured


def test_jsonl_handoff_keeps_corrupt_record_tracking_for_json(tmp_path):
    plan, reader = _read(tmp_path, "json")
    assert _tracks_corrupt_records(reader, "jsonl", plan)
    assert reader.captured["corrupt_record_column"] == CORRUPT_RECORD_COLUMN


@pytest.mark.parametrize(
    ("declared", "actual", "expected"),
    [
        ("json", "json", True),
        ("json", "jsonl", True),
        ("jsonl", "json", True),
        ("jsonl", "jsonl", True),
        ("csv", "jsonl", False),
        ("json", "csv", False),
    ],
)
def test_handoff_format_satisfies_reader_family(declared, actual, expected):
    assert handoff_format_satisfies(declared, actual) is expected
    assert handoff_format_satisfies(actual, declared) is expected
    assert handoff_format_satisfies(f" {declared.upper()} ", f"\t{actual.upper()}\n") is expected


@pytest.mark.parametrize("format_name", ["csv", "parquet", "text", "binary"])
def test_identical_non_json_formats_still_satisfy_the_declaration(format_name):
    assert handoff_format_satisfies(f" {format_name.upper()} ", format_name)


@pytest.mark.parametrize(("declared", "actual"), [("csv", "jsonl"), ("json", "csv")])
def test_file_family_parser_mismatch_gets_no_schema_or_options(tmp_path, declared, actual):
    zone = build_raw_zone(tmp_path)
    mapping = source_mapping(spark={"input_format": declared, "read_options": {"header": "true"}})
    mapping.update(source_type="file", strategy="file", strategy_variant="static_file")
    mapping["access"]["pagination"] = {"type": "none"}
    config = SourceConfig.from_mapping(mapping, tmp_path / "file.yaml")
    write_minimal_contract(tmp_path)
    plan = with_registry_contract(ExecutionPlan.from_source_config(config, zone.plan.run_context))
    handoff = replace(
        zone.extraction,
        source=plan.source,
        artifacts=(ExtractedArtifact(path=str(tmp_path / f"member.{actual}"), format=actual),),
    )
    reader = SpyReader()

    _read_batch(reader, object(), handoff, plan, plan.data_contract, None, {})

    assert plan.source.strategy == "file"
    assert reader.captured["schema"] is None
    assert reader.captured["options"] is None
    assert "corrupt_record_column" not in reader.captured


@pytest.mark.parametrize(
    ("compaction", "options", "expected_conflicts"),
    [
        ("jsonl", {"multiLine": "true"}, 1),
        ("jsonl", {"MULTILINE": "TRUE"}, 1),
        ("jsonl", {"multiLine": "TrUe"}, 1),
        ("jsonl", {"multiLine": "false"}, 0),
        ("jsonl", {"multiLine": "FALSE"}, 0),
        ("jsonl", {}, 0),
        ("jsonl", None, 0),
        ("pages", {"multiLine": "true"}, 0),
        ("hook", {"multiLine": "true"}, 0),
        ("none", {"multiLine": "true"}, 0),
    ],
)
def test_multiline_conflict_is_logged_once_and_options_are_preserved(
    tmp_path, compaction, options, expected_conflicts
):
    zone = build_raw_zone(tmp_path, spark={"read_options": options})
    write_minimal_contract(tmp_path)
    plan = with_registry_contract(zone.plan)
    handoff = replace(
        zone.extraction,
        artifacts=tuple(
            ExtractedArtifact(path=str(tmp_path / f"records-{index}.jsonl"), format="jsonl")
            for index in range(6)
        ),
    ).with_metadata("handoff_compaction", compaction)
    # Exercise two actual materializer batches with a schema-only reader.
    planned_run = PlannedRun(plan, SimpleNamespace(strategy_family="file"))
    reader = SpyReader()
    reader.supports_corrupt_record_read = False
    writer = Mock()
    writer.write.return_value = WriteResult.from_plan(plan, "bronze")
    stream = io.StringIO()
    logger = build_structured_logger("test-handoff-format-seam", stream=stream)

    BronzeMaterializer(
        reader, SimpleNamespace(normalize=lambda dataframe, plan: dataframe), lambda layout: writer
    ).materialize(planned_run, plan, object(), handoff, zone.layout, logger)

    conflicts = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if json.loads(line)["event"] == "handoff_compaction_read_option_conflict"
    ]
    assert writer.write.call_count == 2
    assert len(conflicts) == expected_conflicts
    if conflicts:
        assert conflicts[0]["level"] == "WARNING"
        assert conflicts[0]["fields"]["source_id"] == plan.source.source_id
        assert conflicts[0]["fields"]["option"] == "multiLine"
    assert reader.captured["options"] is plan.source_config.spark.read_options
    assert reader.captured["options"] == options
