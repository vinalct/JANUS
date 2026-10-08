from dataclasses import dataclass, replace
from typing import Any

import pytest

from janus.models import ExtractedArtifact
from janus.quality import CORRUPT_RECORD_COLUMN
from janus.runtime.materialize import _read_batch, _tracks_corrupt_records
from tests.fixtures.handoff.cases import build_raw_zone
from tests.support.contracts import with_registry_contract, write_minimal_contract


@dataclass
class SpyReader:
    supports_corrupt_record_read = True
    captured: dict[str, Any] | None = None

    def read_extraction_result(self, spark, handoff, **kwargs):
        self.captured = kwargs
        return object()


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


@pytest.mark.xfail(strict=True, reason="green")
def test_jsonl_handoff_keeps_resolved_json_contract_schema(tmp_path):
    plan, reader = _read(tmp_path, "json")
    assert reader.captured["schema"] is not None
    assert reader.captured["schema"].fieldNames() == ["id", CORRUPT_RECORD_COLUMN]
    assert plan.data_contract is not None


@pytest.mark.xfail(strict=True, reason="green")
def test_jsonl_handoff_keeps_declared_json_read_options(tmp_path):
    plan, reader = _read(tmp_path, "json")
    assert reader.captured["options"] == plan.source_config.spark.read_options


def test_genuinely_different_csv_handoff_gets_no_schema_or_options(tmp_path):
    _, reader = _read(tmp_path, "csv")
    assert reader.captured["schema"] is None
    assert reader.captured["options"] is None


@pytest.mark.xfail(strict=True, reason="green")
def test_jsonl_handoff_keeps_corrupt_record_tracking_for_json(tmp_path):
    plan, reader = _read(tmp_path, "json")
    assert _tracks_corrupt_records(reader, "jsonl", plan)
    assert reader.captured["corrupt_record_column"] == CORRUPT_RECORD_COLUMN


@pytest.mark.xfail(strict=True, reason="green")
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
    from janus.runtime.materialize import handoff_format_satisfies

    assert handoff_format_satisfies(declared, actual) is expected
