import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from janus.models import ExtractionResult
from janus.strategies.api import ApiHook
from janus.strategies.api.records import _default_records_from_payload
from janus.writers import RawArtifactWriter
from tests.fixtures.handoff.cases import (
    HandoffHook,
    RecordsHook,
    build_raw_zone,
    page_snapshot,
    read_records,
)


def _compact(zone, extraction=None):
    from janus.strategies.api.handoff import compact_api_handoff

    return compact_api_handoff(
        zone.plan,
        zone.extraction if extraction is None else extraction,
        raw_writer=RawArtifactWriter(zone.layout),
    )


def _bytes_and_sidecars(artifacts):
    return [
        (Path(artifact.path).read_bytes(), Path(artifact.path + ".sha256").read_bytes())
        for artifact in artifacts
    ]


def _expected_records(zone):
    if zone.extraction.single_artifact_format() == "jsonl":
        return read_records(zone.extraction.artifacts)
    return [
        record
        for artifact in zone.extraction.artifacts
        for record in _default_records_from_payload(json.loads(Path(artifact.path).read_bytes()))
    ]


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize("case", ["single_input", "two_inputs"])
def test_same_page_bytes_produce_identical_jsonl_and_sidecar_digests(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    before = page_snapshot(zone)
    first = _compact(zone)
    first_bytes = _bytes_and_sidecars(first.artifacts)
    second = _compact(zone)
    assert first.artifacts and first.artifacts == second.artifacts
    assert first_bytes == _bytes_and_sidecars(second.artifacts)
    for artifact, (body, sidecar) in zip(first.artifacts, first_bytes, strict=True):
        assert artifact.format == "jsonl"
        assert sha256(body).hexdigest() == artifact.checksum == sidecar.decode().strip()
    assert page_snapshot(zone) == before


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("single_input", ["handoff/pages/records-0001.jsonl"]),
        (
            "two_inputs",
            [
                "handoff/request-input-000001/records-0001.jsonl",
                "handoff/request-input-000002/records-0001.jsonl",
            ],
        ),
    ],
)
def test_grouping_mirrors_request_input_directories(tmp_path, case, expected):
    zone = build_raw_zone(tmp_path, case)
    outcome = _compact(
        zone, replace(zone.extraction, artifacts=tuple(reversed(zone.extraction.artifacts)))
    )
    assert [
        Path(artifact.path).relative_to(zone.run_root).as_posix() for artifact in outcome.artifacts
    ] == expected


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize("raw_format", ["json", "jsonl"])
def test_concatenated_records_match_default_extraction_in_page_order(tmp_path, raw_format):
    zone = build_raw_zone(tmp_path, raw_format=raw_format)
    outcome = _compact(zone)
    assert read_records(outcome.artifacts) == _expected_records(zone)
    assert len(read_records(outcome.artifacts)) == zone.extraction.records_extracted


@pytest.mark.xfail(strict=True, reason="green")
def test_page_order_is_numeric_even_when_artifact_order_is_reversed(tmp_path):
    zone = build_raw_zone(tmp_path)
    artifacts = tuple(
        zone.writer.write_json(zone.plan, f"pages/page-{index}.json", [{"id": index}]).artifact
        for index in range(1, 13)
    )
    extraction = replace(zone.extraction, artifacts=tuple(reversed(artifacts)))
    outcome = _compact(zone, extraction)
    assert [record["id"] for record in read_records(outcome.artifacts)] == list(range(1, 13))


@pytest.mark.xfail(strict=True, reason="green")
def test_size_split_keeps_every_record_once_in_page_order(tmp_path):
    zone = build_raw_zone(tmp_path, spark={"handoff": {"max_artifact_mb": 1}})
    records = [{"id": index, "payload": "ação" * 60000} for index in range(9)]
    artifacts = tuple(
        zone.writer.write_json(zone.plan, f"pages/page-{index + 1:04d}.json", [record]).artifact
        for index, record in enumerate(records)
    )
    outcome = _compact(zone, replace(zone.extraction, artifacts=artifacts))
    assert len(outcome.artifacts) > 1
    assert [Path(artifact.path).name for artifact in outcome.artifacts] == [
        f"records-{index:04d}.jsonl" for index in range(1, len(outcome.artifacts) + 1)
    ]
    assert read_records(outcome.artifacts) == records
    largest_line = max(
        len(json.dumps(record, ensure_ascii=False).encode()) + 1 for record in records
    )
    for artifact in outcome.artifacts:
        body = Path(artifact.path).read_bytes()
        assert body.endswith(b"\n")
        # A split may roll after the record that crosses the cap; never split a record.
        assert len(body) <= 1024 * 1024 + largest_line


@pytest.mark.xfail(strict=True, reason="green")
def test_empty_compaction_outcome_has_no_artifacts_or_bytes(tmp_path):
    zone = build_raw_zone(tmp_path)
    outcome = _compact(zone, replace(zone.extraction, artifacts=(), records_extracted=0))
    assert outcome.artifacts == ()
    assert outcome.compaction == "empty"
    assert outcome.source_page_count == 0
    assert outcome.handoff_bytes == 0


@pytest.mark.xfail(strict=True, reason="green")
def test_strategy_compacts_and_records_all_four_metadata_keys(tmp_path):
    zone = build_raw_zone(tmp_path)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    assert handoff.artifacts and all(artifact.format == "jsonl" for artifact in handoff.artifacts)
    expected = {
        "fixture": "single_input",
        "handoff_compaction": "jsonl",
        "handoff_source_page_count": "3",
        "handoff_artifact_count": "1",
        "handoff_bytes": str(
            sum(Path(artifact.path).stat().st_size for artifact in handoff.artifacts)
        ),
    }
    assert {key: handoff.metadata_as_dict()[key] for key in expected} == expected
    assert handoff.records_extracted == zone.extraction.records_extracted
    assert handoff.checkpoint_value == zone.extraction.checkpoint_value


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("raw_format", "spark", "reason"),
    [("text", {}, "none"), ("binary", {}, "none"), ("json", {"handoff": "pages"}, "pages")],
)
def test_declared_bypasses_keep_original_artifacts_and_record_reason(
    tmp_path, raw_format, spark, reason
):
    zone = build_raw_zone(
        tmp_path,
        "text_input" if raw_format != "json" else "single_input",
        spark=spark,
        raw_format=raw_format,
    )
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    assert handoff.artifacts == zone.extraction.artifacts
    metadata = handoff.metadata_as_dict()
    assert metadata["handoff_compaction"] == reason
    assert metadata["handoff_source_page_count"] == str(len(zone.extraction.artifacts))
    assert metadata["handoff_artifact_count"] == str(len(handoff.artifacts))
    assert metadata["handoff_bytes"] == str(
        sum(Path(artifact.path).stat().st_size for artifact in handoff.artifacts)
    )
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.xfail(strict=True, reason="green")
def test_custom_record_hook_disables_compaction_without_reextracting_records(tmp_path):
    zone = build_raw_zone(tmp_path)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction, RecordsHook())
    assert handoff.artifacts == zone.extraction.artifacts
    assert handoff.metadata_as_dict()["handoff_compaction"] == "hook_records"
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.xfail(strict=True, reason="green")
def test_handoff_override_receives_original_json_pages_and_owns_the_result(tmp_path):
    zone = build_raw_zone(tmp_path)
    hook = HandoffHook(zone)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction, hook)
    assert hook.received is zone.extraction
    assert read_records(handoff.artifacts) == [{"id": "hook-output"}]
    assert handoff.metadata_as_dict()["handoff_compaction"] == "hook"
    assert handoff.metadata_as_dict()["fixture_hook"] == "true"
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.xfail(strict=True, reason="green")
def test_inherited_noop_hook_does_not_disable_compaction(tmp_path):
    zone = build_raw_zone(tmp_path)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction, ApiHook())
    assert handoff.single_artifact_format() == "jsonl"
    assert handoff.metadata_as_dict()["handoff_compaction"] == "jsonl"


def test_empty_strategy_handoff_stays_empty(tmp_path):
    zone = build_raw_zone(tmp_path)
    empty = ExtractionResult.from_plan(zone.plan, artifacts=(), records_extracted=0)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, empty)
    assert handoff.artifacts == ()
    assert handoff.is_empty is True


@pytest.mark.skipif(
    not hasattr(RawArtifactWriter, "write_stream"),
    reason="order-17 TASK-10 dependency: RawArtifactWriter.write_stream is not available",
)
@pytest.mark.xfail(strict=True, reason="green")
def test_compactor_uses_raw_writer_streaming_for_each_derived_artifact(tmp_path):
    zone = build_raw_zone(tmp_path)
    original = RawArtifactWriter.write_stream
    calls = []

    def spy(writer, plan, relative_path, stream, **kwargs):
        calls.append(str(relative_path))
        return original(writer, plan, relative_path, stream, **kwargs)

    with patch.object(RawArtifactWriter, "write_stream", spy):
        outcome = _compact(zone)
    assert outcome.artifacts
    assert len(calls) == len(outcome.artifacts)
    assert all("handoff" in Path(path).parts for path in calls)


@pytest.mark.parametrize("case", ["single_input", "two_inputs", "text_input"])
def test_shared_zones_have_real_checksums_and_named_input_directories(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    assert zone.extraction.artifacts
    for artifact in zone.extraction.artifacts:
        assert sha256(Path(artifact.path).read_bytes()).hexdigest() == artifact.checksum
        assert Path(artifact.path + ".sha256").read_text().strip() == artifact.checksum
