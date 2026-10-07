"""Check seed determinism, real sidecars, late nulls and production rediscovery."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from janus.strategies.api.records import _default_records_from_payload
from tests.fixtures.handoff.generate_pages import generate_pages, synthetic_plan


def test_seed_produces_identical_pages_and_real_checksums(tmp_path):
    first = generate_pages(tmp_path / "first", page_count=3)
    second = generate_pages(tmp_path / "second", page_count=3)
    different = generate_pages(tmp_path / "different", page_count=3, seed=23)
    assert [artifact.checksum for artifact in first] == [artifact.checksum for artifact in second]
    assert first[0].checksum != different[0].checksum
    for artifact in (*first, *second):
        path = Path(artifact.path)
        assert artifact.format == "json"
        assert artifact.checksum == sha256(path.read_bytes()).hexdigest()
        assert path.with_name(path.name + ".sha256").read_text() == artifact.checksum + "\n"
        assert len(_default_records_from_payload(json.loads(path.read_bytes()))) == 15


def test_early_nulls_and_later_values_are_deliberate(tmp_path):
    artifacts = generate_pages(tmp_path, page_count=101, records_per_page=2)
    pages = [json.loads(Path(artifact.path).read_bytes()) for artifact in artifacts]
    assert all(
        record["observacaoTardia"] is None for page in pages[:100] for record in page["data"]
    )
    assert all(record["observacaoTardia"] == "Disponível" for record in pages[100]["data"])
    ids = [record["id"] for page in pages for record in page["data"]]
    assert ids == list(range(1, 203))
    assert all(len(record) == 12 for page in pages for record in page["data"])


def test_generated_zone_is_accepted_by_real_replay_discovery(tmp_path):
    from janus.scripts.rehydrate import _rediscover_raw_artifacts
    expected = generate_pages(tmp_path, page_count=3)
    plan = synthetic_plan(tmp_path)
    discovered = _rediscover_raw_artifacts(plan, verify_checksums=True)
    assert discovered == expected


@pytest.mark.parametrize("page_count,records_per_page", [(-1, 15), (1, 0)])
def test_invalid_sizes_are_refused(tmp_path, page_count, records_per_page):
    with pytest.raises(ValueError, match="page_count"):
        generate_pages(tmp_path, page_count=page_count, records_per_page=records_per_page)


def test_zero_pages_stays_empty(tmp_path):
    assert generate_pages(tmp_path, page_count=0) == ()
