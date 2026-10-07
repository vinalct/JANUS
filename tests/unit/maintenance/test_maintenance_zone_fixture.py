"""Green fixture checks: reds must not conceal a malformed test harness."""

import json
from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from tests.support.maintenance_zone import FAMILIES, SOURCES, build_maintenance_zone, zone_plans


def test_builder_plants_every_declared_path_and_family(planted):
    assert len(planted.protected_paths | planted.candidate_paths) == 225
    assert len(planted.candidate_paths) == 41
    assert len(planted.protected_paths) == 184
    assert set(planted.run_ids_by_source) == set(SOURCES)
    for source, ids in planted.run_ids_by_source.items():
        assert len(ids) == 25
        for family in FAMILIES:
            assert all(
                (planted.root / source / family / f"{run_id}.json").exists() for run_id in ids
            )
        assert all(
            planted.root / source / family / f"{run_id}.json" in planted.protected_paths
            for run_id in ids[-20:]
            for family in FAMILIES
        )
    assert len(list((planted.root / "pipelines").glob("*/summary.json"))) == 3
    assert len(list(planted.root.rglob("*.tmp"))) == 2
    assert json.loads((planted.root / SOURCES[0] / "extraction_progress.json").read_text())[
        "raw_path_prefix"
    ]
    assert "raw_path_prefix" not in json.loads(
        (planted.root / SOURCES[1] / "extraction_progress.json").read_text()
    )
    assert len(zone_plans(planted)) == 2


def test_builder_is_frozen_and_uses_only_supplied_clock(planted, tmp_path):
    other = build_maintenance_zone(tmp_path / "other", planted.now + timedelta(days=1))
    assert len(other.candidate_paths) == len(planted.candidate_paths)
    assert len(other.protected_paths) == len(planted.protected_paths)
    with pytest.raises(FrozenInstanceError):
        planted.now = other.now
    with pytest.raises(TypeError):
        planted.run_ids_by_source["third"] = ()
