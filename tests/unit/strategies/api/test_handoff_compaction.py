import json
import logging
import tracemalloc
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from janus.models import ExtractedArtifact, ExtractionResult
from janus.strategies.api import ApiHook
from janus.strategies.api.handoff import (
    COMPACTION_EMPTY,
    COMPACTION_HOOK_RECORDS,
    COMPACTION_JSONL,
    COMPACTION_NONE,
    COMPACTION_PAGES,
)
from janus.strategies.api.records import _default_records_from_payload
from janus.writers import RawArtifactWriter
from janus.writers.raw import iter_json_lines
from tests.fixtures.handoff.cases import (
    HandoffHook,
    RecordsHook,
    build_raw_zone,
    page_snapshot,
    read_records,
)


def _compact(zone, extraction=None, *, hook=None, raw_writer=None):
    from janus.strategies.api.handoff import compact_api_handoff

    return compact_api_handoff(
        zone.plan,
        zone.extraction if extraction is None else extraction,
        raw_writer=raw_writer or RawArtifactWriter(zone.layout),
        hook=hook,
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


@pytest.mark.parametrize("raw_format", ["json", "jsonl"])
def test_concatenated_records_match_default_extraction_in_page_order(tmp_path, raw_format):
    zone = build_raw_zone(tmp_path, raw_format=raw_format)
    outcome = _compact(zone)
    assert read_records(outcome.artifacts) == _expected_records(zone)
    assert len(read_records(outcome.artifacts)) == zone.extraction.records_extracted


def test_page_order_is_numeric_even_when_artifact_order_is_reversed(tmp_path):
    zone = build_raw_zone(tmp_path)
    artifacts = tuple(
        zone.writer.write_json(zone.plan, f"pages/page-{index}.json", [{"id": index}]).artifact
        for index in range(1, 13)
    )
    extraction = replace(zone.extraction, artifacts=tuple(reversed(artifacts)))
    outcome = _compact(zone, extraction)
    assert [record["id"] for record in read_records(outcome.artifacts)] == list(range(1, 13))


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


def test_empty_compaction_outcome_has_no_artifacts_or_bytes(tmp_path):
    zone = build_raw_zone(tmp_path)
    outcome = _compact(zone, replace(zone.extraction, artifacts=(), records_extracted=0))
    assert outcome.artifacts == ()
    assert outcome.compaction == "empty"
    assert outcome.source_page_count == 0
    assert outcome.handoff_bytes == 0


@pytest.mark.xfail(strict=True, reason="order-22 TASK-05: strategy wiring and hook metadata")
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


@pytest.mark.xfail(strict=True, reason="order-22 TASK-05: strategy wiring and hook metadata")
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


@pytest.mark.xfail(strict=True, reason="order-22 TASK-05: strategy wiring and hook metadata")
def test_custom_record_hook_disables_compaction_without_reextracting_records(tmp_path):
    zone = build_raw_zone(tmp_path)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction, RecordsHook())
    assert handoff.artifacts == zone.extraction.artifacts
    assert handoff.metadata_as_dict()["handoff_compaction"] == "hook_records"
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.xfail(strict=True, reason="order-22 TASK-05: strategy wiring and hook metadata")
def test_handoff_override_receives_original_json_pages_and_owns_the_result(tmp_path):
    zone = build_raw_zone(tmp_path)
    hook = HandoffHook(zone)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction, hook)
    assert hook.received is zone.extraction
    assert read_records(handoff.artifacts) == [{"id": "hook-output"}]
    assert handoff.metadata_as_dict()["handoff_compaction"] == "hook"
    assert handoff.metadata_as_dict()["fixture_hook"] == "true"
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.xfail(strict=True, reason="order-22 TASK-05: strategy wiring and hook metadata")
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


@pytest.mark.parametrize(
    ("raw_format", "spark", "hook", "reason"),
    [
        ("json", {"handoff": "pages"}, None, COMPACTION_PAGES),
        ("text", {}, None, COMPACTION_NONE),
        ("binary", {}, None, COMPACTION_NONE),
        ("json", {}, RecordsHook(), COMPACTION_HOOK_RECORDS),
    ],
)
def test_compactor_refusals_preserve_original_artifacts_and_bytes(
    tmp_path, raw_format, spark, hook, reason
):
    zone = build_raw_zone(
        tmp_path,
        "text_input" if raw_format in {"text", "binary"} else "single_input",
        spark=spark,
        raw_format=raw_format,
    )
    before = page_snapshot(zone)
    outcome = _compact(zone, hook=hook)
    assert outcome.artifacts is zone.extraction.artifacts
    assert outcome.compaction == reason
    assert outcome.metadata == {
        "handoff_compaction": reason,
        "handoff_source_page_count": str(len(zone.extraction.artifacts)),
        "handoff_artifact_count": str(len(zone.extraction.artifacts)),
        "handoff_bytes": str(sum(Path(a.path).stat().st_size for a in outcome.artifacts)),
    }
    assert page_snapshot(zone) == before
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.parametrize(
    ("spark", "raw_format", "empty", "reason"),
    [
        ({"handoff": "pages"}, "text", True, COMPACTION_EMPTY),
        ({"handoff": "pages"}, "text", False, COMPACTION_PAGES),
        ({}, "text", False, COMPACTION_NONE),
        ({}, "json", False, COMPACTION_HOOK_RECORDS),
    ],
)
def test_refusals_take_precedence_in_the_documented_order(
    tmp_path, spark, raw_format, empty, reason
):
    zone = build_raw_zone(tmp_path, spark=spark, raw_format=raw_format)
    unexpected = ExtractedArtifact(path=str(tmp_path / "absent.json"), format="other")
    extraction = replace(zone.extraction, artifacts=() if empty else (unexpected,))
    outcome = _compact(zone, extraction, hook=RecordsHook())
    assert outcome.compaction == reason
    assert outcome.artifacts is extraction.artifacts
    assert not (zone.run_root / "handoff").exists()


@pytest.mark.parametrize(
    ("relative_path", "format_name"),
    [
        ("pages/page-0001.json", "jsonl"),
        ("pages/page-0001.jsonl", "json"),
        ("pages/unknown.json", "json"),
        ("pages/page-bad.json", "json"),
        ("pages/nested/page-0001.json", "json"),
        ("handoff/pages/page-0001.json", "json"),
        ("normalized/page-0001.json", "json"),
        ("request-input-invalid/page-0001.json", "json"),
    ],
)
def test_unexpected_artifact_sets_bypass_without_writing(tmp_path, relative_path, format_name):
    zone = build_raw_zone(tmp_path)
    unexpected = replace(
        zone.writer.write_json(zone.plan, relative_path, [{"id": "unexpected"}]).artifact,
        format=format_name,
    )
    extraction = replace(zone.extraction, artifacts=(*zone.extraction.artifacts, unexpected))
    before = page_snapshot(replace(zone, extraction=extraction))
    outcome = _compact(zone, extraction)
    assert outcome.compaction == COMPACTION_NONE
    assert outcome.artifacts is extraction.artifacts
    assert page_snapshot(replace(zone, extraction=extraction)) == before
    assert not list(zone.run_root.rglob("records-*.jsonl"))


@pytest.mark.parametrize("symlink", [False, True])
def test_pages_from_another_run_or_an_external_symlink_are_refused(tmp_path, symlink):
    zone = build_raw_zone(tmp_path)
    outside = tmp_path / "another-run" / "pages" / "page-0001.json"
    outside.parent.mkdir(parents=True)
    outside.write_text('[{"id": "other-run"}]', encoding="utf-8")
    path = outside
    if symlink:
        path = zone.run_root / "pages" / "page-9999.json"
        path.symlink_to(outside)
    extraction = replace(zone.extraction, artifacts=(ExtractedArtifact(str(path), "json"),))
    outcome = _compact(zone, extraction)
    assert outcome.compaction == COMPACTION_NONE
    assert outcome.artifacts is extraction.artifacts
    assert not (zone.run_root / "handoff").exists()


def test_record_hook_capability_is_inherited_and_noop_hook_can_compact(tmp_path):
    class InheritedRecordsHook(RecordsHook):
        pass

    class NoopHook(ApiHook):
        pass

    zone = build_raw_zone(tmp_path)
    assert _compact(zone, hook=InheritedRecordsHook()).compaction == COMPACTION_HOOK_RECORDS
    assert _compact(zone, hook=NoopHook()).compaction == COMPACTION_JSONL


@pytest.mark.parametrize("prefix", ["page-", "offset-", "cursor-", "response-"])
def test_all_pagination_filename_families_use_numeric_order(tmp_path, prefix):
    zone = build_raw_zone(tmp_path)
    artifacts = tuple(
        zone.writer.write_json(zone.plan, f"pages/{prefix}{index}.json", [{"id": index}]).artifact
        for index in (10, 2, 1)
    )
    outcome = _compact(zone, replace(zone.extraction, artifacts=artifacts, records_extracted=3))
    assert read_records(outcome.artifacts) == [{"id": 1}, {"id": 2}, {"id": 10}]


def test_scoped_writer_does_not_duplicate_the_run_prefix(tmp_path):
    zone = build_raw_zone(tmp_path)
    outcome = _compact(zone, raw_writer=zone.writer)
    assert Path(outcome.artifacts[0].path).relative_to(zone.run_root).as_posix() == (
        "handoff/pages/records-0001.jsonl"
    )


@pytest.mark.parametrize("raw_format", ["json", "jsonl"])
def test_empty_record_groups_write_no_artifact(tmp_path, raw_format):
    zone = build_raw_zone(tmp_path, raw_format=raw_format)
    if raw_format == "json":
        artifact = zone.writer.write_json(zone.plan, "pages/page-0001.json", []).artifact
    else:
        artifact = zone.writer.write_json_lines(zone.plan, "pages/page-0001.jsonl", []).artifact
    extraction = replace(zone.extraction, artifacts=(artifact,), records_extracted=0)
    outcome = _compact(zone, extraction)
    assert outcome.compaction == COMPACTION_JSONL
    assert outcome.artifacts == ()
    assert outcome.handoff_bytes == 0
    assert outcome.source_page_count == 1
    assert not (zone.run_root / "handoff").exists()


def test_bare_mappings_and_jsonl_envelopes_use_default_record_extraction(tmp_path):
    zone = build_raw_zone(tmp_path, raw_format="jsonl")
    artifact = zone.writer.write_text(
        zone.plan,
        "pages/page-0001.jsonl",
        '\n{"value": [{"id": 1}, {"id": 2}]}\n\n{"id": 3, "nullable": null}\n',
    ).artifact
    extraction = replace(
        zone.extraction,
        artifacts=(replace(artifact, format="jsonl"),),
        records_extracted=3,
    )
    outcome = _compact(zone, extraction)
    assert read_records(outcome.artifacts) == [
        {"id": 1},
        {"id": 2},
        {"id": 3, "nullable": None},
    ]


@pytest.mark.parametrize("oversized", [False, True])
def test_exact_size_boundary_and_oversized_record_do_not_create_empty_splits(tmp_path, oversized):
    zone = build_raw_zone(tmp_path, spark={"handoff": {"max_artifact_mb": 1}})
    limit = zone.plan.source_config.spark.handoff.max_artifact_bytes
    overhead = len(next(iter_json_lines([{"payload": ""}])))
    record = {"payload": "x" * (limit - overhead + (100 if oversized else 0))}
    records = [record, record, {"tail": True}]
    artifact = zone.writer.write_json(zone.plan, "pages/page-0001.json", records).artifact
    outcome = _compact(zone, replace(zone.extraction, artifacts=(artifact,), records_extracted=3))
    assert len(outcome.artifacts) == 3
    assert read_records(outcome.artifacts) == records
    assert [Path(a.path).stat().st_size for a in outcome.artifacts[:2]] == [
        limit + (100 if oversized else 0),
        limit + (100 if oversized else 0),
    ]


def test_mismatch_is_logged_with_both_counts_without_changing_the_extraction(tmp_path, caplog):
    zone = build_raw_zone(tmp_path)
    extraction = replace(zone.extraction, records_extracted=999)
    with caplog.at_level(logging.WARNING, logger="janus.strategies.api.handoff"):
        outcome = _compact(zone, extraction)
    mismatch = next(
        record for record in caplog.records if "handoff_record_count_mismatch" in record.msg
    )
    assert mismatch.records_extracted == 999
    assert mismatch.handoff_records == len(read_records(outcome.artifacts))
    assert extraction.records_extracted == 999
    assert extraction.checkpoint_value == zone.extraction.checkpoint_value


def test_500_page_compaction_keeps_peak_allocation_below_one_tenth_of_the_input(tmp_path):
    zone = build_raw_zone(tmp_path, spark={"handoff": {"max_artifact_mb": 1}})
    artifacts = tuple(
        zone.writer.write_json(
            zone.plan,
            f"pages/page-{index:04d}.json",
            [{"id": index, "payload": "ação" * 16384}],
        ).artifact
        for index in range(1, 501)
    )
    extraction = replace(zone.extraction, artifacts=artifacts, records_extracted=500)
    input_bytes = sum(Path(a.path).stat().st_size for a in artifacts)
    tracemalloc.start()
    try:
        outcome = _compact(zone, extraction)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < input_bytes / 10
    assert peak < 8 * 1024 * 1024
    assert len(read_records(outcome.artifacts)) == 500
    assert len(outcome.artifacts) > 1
    print(f"compaction memory: pages=500 input_bytes={input_bytes} peak_bytes={peak}")


def test_compactor_keeps_serialization_and_source_identity_out_of_the_module():
    import janus.strategies.api.handoff as handoff

    source = Path(handoff.__file__).read_text(encoding="utf-8")
    assert "json.dumps" not in source
    assert "source_id" not in source


@pytest.mark.parametrize("case", ["single_input", "two_inputs", "text_input"])
def test_shared_zones_have_real_checksums_and_named_input_directories(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    assert zone.extraction.artifacts
    for artifact in zone.extraction.artifacts:
        assert sha256(Path(artifact.path).read_bytes()).hexdigest() == artifact.checksum
        assert Path(artifact.path + ".sha256").read_text().strip() == artifact.checksum
