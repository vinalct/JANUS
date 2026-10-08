import pytest

from janus.models import ExtractionResult
from janus.scripts.rehydrate import _rediscover_raw_artifacts
from tests.fixtures.handoff.cases import build_raw_zone


def _zone_with_handoff(tmp_path):
    zone = build_raw_zone(tmp_path)
    zone.writer.write_json_lines(zone.plan, "handoff/pages/records-0001.jsonl", [{"id": "one"}])
    return zone


@pytest.mark.xfail(strict=True, reason="green")
def test_replay_rehydrates_only_pages_when_handoff_exists(tmp_path):
    zone = _zone_with_handoff(tmp_path)
    artifacts = _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True)
    assert artifacts == zone.extraction.artifacts
    assert all("handoff" not in artifact.path.split("/") for artifact in artifacts)


@pytest.mark.xfail(strict=True, reason="green")
def test_rehydrated_result_keeps_a_single_raw_format(tmp_path):
    zone = _zone_with_handoff(tmp_path)
    result = ExtractionResult.from_plan(zone.plan, _rediscover_raw_artifacts(zone.replay_plan()))
    assert result.single_artifact_format() == "json"


def test_checksum_sidecars_remain_excluded_even_with_a_handoff(tmp_path):
    zone = _zone_with_handoff(tmp_path)
    artifacts = _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True)
    assert artifacts
    assert all(not artifact.path.endswith(".sha256") for artifact in artifacts)
    assert all(artifact.checksum for artifact in artifacts)


@pytest.mark.parametrize("case", ["single_input", "two_inputs", "text_input"])
def test_legacy_raw_zone_without_handoff_rehydrates_unchanged(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    assert _rediscover_raw_artifacts(zone.replay_plan(), verify_checksums=True) == (
        zone.extraction.artifacts
    )
