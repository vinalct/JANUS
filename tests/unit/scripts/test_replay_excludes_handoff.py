import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from janus.models import ExtractionResult
from janus.planner import PlannedRun
from janus.runtime import SparkSessionProvider
from janus.scripts.rehydrate import (
    _build_extraction_result_from_raw,
    _rediscover_raw_artifacts,
)
from janus.strategies.api.handoff import HANDOFF_DIRECTORY_NAME
from janus.writers import SIDECAR_SUFFIX
from tests.fixtures.handoff.cases import build_raw_zone, page_snapshot, read_records


def _zone_with_handoff(tmp_path, *, directory=HANDOFF_DIRECTORY_NAME):
    zone = build_raw_zone(tmp_path)
    zone.writer.write_json_lines(
        zone.plan, Path(directory) / "pages/records-0001.jsonl", [{"id": "one"}]
    )
    return zone


def _handoff_tree_digest(zone):
    root = zone.run_root / HANDOFF_DIRECTORY_NAME
    files = sorted(path for path in root.rglob("*") if path.is_file())
    assert files, "the digest must cover a real handoff and its sidecars"
    entries = [
        f"{path.relative_to(root).as_posix()}:{sha256(path.read_bytes()).hexdigest()}"
        for path in files
    ]
    return sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _replay_handoff(zone):
    strategy = zone.strategy()
    replay_plan = replace(
        zone.plan,
        raw_output=replace(
            zone.plan.raw_output,
            path=str(zone.layout.resolve_output(zone.plan, "raw").resolved_path),
        ),
    )
    provider = SparkSessionProvider(
        {},
        {},
        session_factory=lambda: pytest.fail("rehydration and compaction must be session-free"),
    )
    extraction = _build_extraction_result_from_raw(
        PlannedRun(plan=replay_plan, strategy=strategy),
        replay_plan,
        provider,
        zone.layout,
        verify_checksums=True,
    )
    return extraction, strategy.build_normalization_handoff(zone.plan, extraction)


def test_jsonl_replay_does_not_duplicate_records(tmp_path):
    zone = build_raw_zone(tmp_path, raw_format="jsonl", spark={"input_format": "jsonl"})
    live = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)

    extraction, replay = _replay_handoff(zone)

    assert extraction.artifacts == zone.extraction.artifacts
    assert extraction.single_artifact_format() == "jsonl"
    assert read_records(replay.artifacts) == read_records(live.artifacts)
    assert len(read_records(replay.artifacts)) == zone.extraction.records_extracted
    assert replay.metadata_as_dict()["handoff_compaction"] == "jsonl"


@pytest.mark.parametrize("directory", [HANDOFF_DIRECTORY_NAME, "nested/handoff"])
def test_replay_rehydrates_only_pages_when_handoff_exists(tmp_path, directory):
    zone = _zone_with_handoff(tmp_path, directory=directory)
    artifacts = _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True)
    assert artifacts == zone.extraction.artifacts
    assert all(
        HANDOFF_DIRECTORY_NAME not in Path(artifact.path).relative_to(zone.run_root).parts
        for artifact in artifacts
    )


def test_rehydrated_result_keeps_a_single_raw_format(tmp_path):
    zone = _zone_with_handoff(tmp_path)
    result = ExtractionResult.from_plan(zone.plan, _rediscover_raw_artifacts(zone.replay_plan()))
    assert result.single_artifact_format() == "json"


def test_checksum_sidecars_remain_excluded_even_with_a_handoff(tmp_path):
    zone = _zone_with_handoff(tmp_path)
    artifacts = _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True)
    assert artifacts
    assert all(not artifact.path.endswith(SIDECAR_SUFFIX) for artifact in artifacts)
    assert all(artifact.checksum for artifact in artifacts)


@pytest.mark.parametrize("case", ["single_input", "two_inputs", "text_input"])
def test_legacy_raw_zone_without_handoff_rehydrates_unchanged(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    assert _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True) == (
        zone.extraction.artifacts
    )


@pytest.mark.parametrize("raw_format", ["json", "jsonl"])
@pytest.mark.parametrize("case", ["single_input", "two_inputs"])
def test_replay_rebuilds_a_byte_identical_handoff_tree(tmp_path, case, raw_format):
    zone = build_raw_zone(tmp_path, case, raw_format=raw_format)
    pages_before = page_snapshot(zone)
    live = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    digest_before = _handoff_tree_digest(zone)

    extraction, replay = _replay_handoff(zone)

    assert extraction.artifacts == zone.extraction.artifacts
    assert replay.artifacts == live.artifacts
    assert _handoff_tree_digest(zone) == digest_before
    assert page_snapshot(zone) == pages_before
    assert extraction.metadata_as_dict()["rediscovered_raw_artifact_count"] == str(
        len(zone.extraction.artifacts)
    )


@pytest.mark.parametrize("raw_format", ["json", "jsonl"])
@pytest.mark.parametrize("case", ["single_input", "two_inputs"])
def test_legacy_replay_creates_a_handoff_from_verified_pages(tmp_path, case, raw_format):
    zone = build_raw_zone(tmp_path, case, raw_format=raw_format)
    pages_before = page_snapshot(zone)
    assert not (zone.run_root / HANDOFF_DIRECTORY_NAME).exists()

    extraction, replay = _replay_handoff(zone)

    assert extraction.artifacts == zone.extraction.artifacts
    assert extraction.single_artifact_format() == raw_format
    assert replay.single_artifact_format() == "jsonl"
    assert replay.metadata_as_dict()["handoff_compaction"] == "jsonl"
    expected_rows = []
    for artifact in zone.extraction.artifacts:
        payloads = (
            [json.loads(Path(artifact.path).read_text())]
            if raw_format == "json"
            else read_records((artifact,))
        )
        for payload in payloads:
            expected_rows.extend(payload if isinstance(payload, list) else [payload])
    assert read_records(replay.artifacts) == expected_rows
    assert all(
        Path(artifact.path + SIDECAR_SUFFIX).read_text().strip() == artifact.checksum
        for artifact in replay.artifacts
    )
    assert (zone.run_root / HANDOFF_DIRECTORY_NAME).is_dir()
    assert page_snapshot(zone) == pages_before


def test_replay_rebuilds_a_stale_handoff_instead_of_verifying_or_reading_it(tmp_path):
    zone = build_raw_zone(tmp_path)
    live = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    digest_before = _handoff_tree_digest(zone)
    Path(live.artifacts[0].path).write_bytes(b"stale, invalid JSONL\n")

    extraction, replay = _replay_handoff(zone)

    assert extraction.artifacts == zone.extraction.artifacts
    assert replay.artifacts == live.artifacts
    assert _handoff_tree_digest(zone) == digest_before


@pytest.mark.parametrize("directory", ["handoff_notes", "handoff-old", "old-handoff"])
def test_replay_preserves_directories_with_handoff_as_a_substring(tmp_path, directory):
    zone = _zone_with_handoff(tmp_path)
    extra = zone.writer.write_json(
        zone.plan, Path(directory) / "page-0001.json", [{"id": "notes"}]
    ).artifact

    artifacts = _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True)

    assert artifacts == tuple(sorted((*zone.extraction.artifacts, extra), key=lambda a: a.path))


def test_handoff_in_an_ancestor_of_the_run_root_does_not_exclude_pages(tmp_path):
    zone = _zone_with_handoff(tmp_path / HANDOFF_DIRECTORY_NAME)
    assert _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True) == (
        zone.extraction.artifacts
    )


def test_a_handoff_without_raw_pages_is_not_a_replayable_raw_zone(tmp_path):
    zone = _zone_with_handoff(tmp_path)
    for artifact in zone.extraction.artifacts:
        Path(artifact.path).unlink()
        Path(artifact.path + SIDECAR_SUFFIX).unlink()

    with pytest.raises(FileNotFoundError, match="No raw artifacts were found"):
        _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True)
