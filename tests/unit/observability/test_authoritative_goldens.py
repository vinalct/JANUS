from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.observability_baseline import BASELINE_CASES
from tests.support.observability_goldens import capture_authoritative_goldens


def test_capture_copies_every_authoritative_byte_and_preserves_original_paths(tmp_path):
    project_root = tmp_path / "capture"
    destination = tmp_path / "goldens"
    captured = capture_authoritative_goldens(destination, project_root=project_root)

    assert tuple(captured) == BASELINE_CASES
    for case, files in captured.items():
        metadata_root = project_root / case / "data" / "metadata"
        source_id = "baseline_catalog" if case == "catalog_success" else "baseline_api"
        expected = {
            f"metadata/{source_id}/runs/{case}.json",
            f"metadata/{source_id}/lineage/{case}.json",
        }
        if case != "extraction_failure":
            expected.add(f"metadata/{source_id}/validations/{case}.json")
        if case in {"api_success", "catalog_success", "empty_handoff"}:
            expected.update({
                f"metadata/{source_id}/checkpoints/current.json",
                f"metadata/{source_id}/checkpoints/history/{case}.json",
            })
        assert set(files) == expected
        for relative in files:
            actual = destination / case / relative
            original = metadata_root / Path(relative).relative_to("metadata")
            assert actual.read_bytes() == original.read_bytes()
        run_path = destination / case / f"metadata/{source_id}/runs/{case}.json"
        run = json.loads(run_path.read_bytes())
        assert run["source_config_path"].startswith(str(project_root / case))
        assert not list(metadata_root.rglob("*.ndjson"))


def test_capture_refuses_to_replace_existing_goldens(tmp_path):
    destination = tmp_path / "goldens"
    destination.mkdir()
    sentinel = destination / "original.json"
    sentinel.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="both be new"):
        capture_authoritative_goldens(destination, project_root=tmp_path / "capture")
    assert sentinel.read_bytes() == b"original"


@pytest.mark.parametrize("nested", ["project_inside_goldens", "goldens_inside_project"])
def test_capture_directories_must_be_independent(tmp_path, nested):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    destination, project_root = (
        (outer, inner) if nested == "project_inside_goldens" else (inner, outer)
    )
    with pytest.raises(ValueError, match="outside"):
        capture_authoritative_goldens(destination, project_root=project_root)
    assert not outer.exists()
