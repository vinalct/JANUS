"""The status fixture must represent the real registry and authoritative writer."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from janus.registry import load_registry
from janus.utils.environment import resolve_project_path
from tests.support.metadata_zone_fixture import build_metadata_zone

NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)
EXPECTED_OUTCOMES = {
    "fixture.fresh": "fresh",
    "fixture.stale": "stale",
    "fixture.never": "never_ran",
    "fixture.undeclared": "undeclared",
    "fixture.unreadable": "fresh",
    "fixture.empty": "never_ran",
}
EXPECTED_SUCCESS_AGES = {
    "fixture.fresh": 2,
    "fixture.stale": 50,
    "fixture.never": None,
    "fixture.undeclared": 200,
    "fixture.unreadable": 3,
    "fixture.empty": None,
}


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_same_seed_and_instant_produce_identical_trees_under_different_roots(tmp_path):
    first = build_metadata_zone(tmp_path / "first", now=NOW, seed=23)
    second = build_metadata_zone(tmp_path / "second", now=NOW, seed=23)

    assert _tree(first.root) == _tree(second.root)
    assert first.expected_outcomes == second.expected_outcomes == EXPECTED_OUTCOMES
    assert not list(first.root.rglob("*.tmp"))


def test_six_sources_load_through_real_registry_and_resolve_separate_metadata_paths(tmp_path):
    fixture = build_metadata_zone(tmp_path / "project", now=NOW)
    registry = load_registry(fixture.root)

    assert {source.source_id for source in registry.sources} == set(EXPECTED_OUTCOMES)
    assert len({source.metadata_dir for source in fixture.sources}) == 6
    for source in fixture.sources:
        config = registry.get_source(source.source_id)
        resolved = resolve_project_path(fixture.root, config.outputs.metadata.path)
        assert resolved == source.metadata_dir
        assert source.metadata_dir == fixture.root / "data" / "metadata" / "fixture" / (
            source.source_id.removeprefix("fixture.")
        )
        assert config.config_path == source.config_path
        document = yaml.safe_load(source.config_path.read_text(encoding="utf-8"))
        if source.max_age is None:
            assert "operations" not in document
        else:
            assert document["operations"] == {"freshness": {"max_age": source.max_age}}


def test_histories_use_authoritative_payloads_and_cover_latest_success_edge_cases(tmp_path):
    fixture = build_metadata_zone(tmp_path / "project", now=NOW)
    expected_run_counts = {
        "fresh": 3, "stale": 2, "never": 2, "undeclared": 1, "unreadable": 1, "empty": 0
    }
    run_ids = set()
    for source in fixture.sources:
        name = source.source_id.removeprefix("fixture.")
        assert len(source.runs) == expected_run_counts[name]
        files = list((source.metadata_dir / "runs").glob("*.json"))
        assert len(files) == len(source.runs) + source.unreadable_run_files
        for run in source.runs:
            path = source.metadata_dir / "runs" / f"{run.run_id}.json"
            expected_bytes = (json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n").encode()
            assert path.read_bytes() == expected_bytes
            assert json.loads(path.read_bytes())["source_id"] == source.source_id
            assert run.ended_at - run.started_at == timedelta(minutes=5)
            assert run.run_id not in run_ids
            run_ids.add(run.run_id)
        successes = [run.ended_at for run in source.runs if run.status == "succeeded"]
        expected_age = EXPECTED_SUCCESS_AGES[source.source_id]
        if expected_age is None:
            assert successes == []
        else:
            assert max(successes) == NOW - timedelta(hours=expected_age)

    stale = next(source for source in fixture.sources if source.source_id == "fixture.stale")
    latest = max(stale.runs, key=lambda run: run.ended_at)
    assert latest.status == "failed"
    assert latest.ended_at == NOW - timedelta(hours=1)
    unreadable = next(
        source for source in fixture.sources if source.source_id == "fixture.unreadable"
    )
    assert unreadable.unreadable_run_files == 1
    with pytest.raises(json.JSONDecodeError):
        json.loads((unreadable.metadata_dir / "runs" / "truncated.json").read_bytes())
    empty = next(source for source in fixture.sources if source.source_id == "fixture.empty")
    assert empty.metadata_dir.is_dir()
    assert not (empty.metadata_dir / "runs").exists()


def test_seed_changes_run_ids_and_injected_now_changes_all_run_times(tmp_path):
    first = build_metadata_zone(tmp_path / "first", now=NOW, seed=0)
    shifted = build_metadata_zone(tmp_path / "shifted", now=NOW + timedelta(days=1), seed=1)
    assert first.expected_outcomes == shifted.expected_outcomes
    for original_source, shifted_source in zip(first.sources, shifted.sources, strict=True):
        for original, new in zip(original_source.runs, shifted_source.runs, strict=True):
            assert original.run_id != new.run_id
            assert new.ended_at - original.ended_at == timedelta(days=1)
            assert new.started_at - original.started_at == timedelta(days=1)


def test_timezone_is_normalized_to_utc_and_naive_now_is_refused(tmp_path):
    local_now = NOW.astimezone(timezone(timedelta(hours=-3)))
    utc = build_metadata_zone(tmp_path / "utc", now=NOW)
    local = build_metadata_zone(tmp_path / "local", now=local_now)
    assert local.now == utc.now == NOW
    assert _tree(local.root) == _tree(utc.root)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_metadata_zone(tmp_path / "naive", now=NOW.replace(tzinfo=None))


def test_existing_project_is_refused_without_changing_history(tmp_path):
    fixture = build_metadata_zone(tmp_path / "project", now=NOW)
    original = _tree(fixture.root)
    with pytest.raises(FileExistsError, match="new or empty"):
        build_metadata_zone(fixture.root, now=NOW, seed=99)
    assert _tree(fixture.root) == original
