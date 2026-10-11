from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support.parallel_baseline import (
    CAPTURE_ROOT,
    CASES,
    EXPECTED_EXITS,
    VOLATILE_FILES,
    capture_run_all,
    directory_collisions,
    metadata_digests,
)

GOLDENS = Path(__file__).resolve().parents[2] / "fixtures/orchestration/parallel_baseline"


@pytest.mark.parametrize("case", CASES)
def test_sequential_run_all_matches_verbatim_golden(case):

    root = CAPTURE_ROOT / case
    assert not root.exists(), f"Remove the stale capture project before testing: {root}"
    try:
        actual = capture_run_all(case, root)
        golden = GOLDENS / case
        expected = json.loads((golden / "capture.json").read_text(encoding="utf-8"))
        expected["stdout"] = (golden / "stdout.txt").read_text(encoding="utf-8")
        expected["summary_json"] = (golden / "summary.json").read_text(encoding="utf-8")
        assert actual == expected
        assert actual["exit_code"] == EXPECTED_EXITS[case]
        sources = {row["source_id"]: row for row in json.loads(actual["summary_json"])["sources"]}
        if case in {"upstream_fails", "upstream_raises", "planning_failure"}:
            assert sources["A"]["status"] == "failed"
            assert sources["B"]["status"] == "skipped"
            assert sources["C"]["status"] == "succeeded"
            assert not any(path.startswith("B/") for path in actual["metadata_zone"])
        if case == "diamond_one_arm_fails":
            assert sources["C"]["status"] == "succeeded"
            assert sources["D"]["skip"]["direct_blocking_upstream_ids"] == ["B"]
            assert sources["D"]["skip"]["root_failed_source_ids"] == ["B"]
            assert not any(path.startswith("D/") for path in actual["metadata_zone"])
        if case == "planning_failure":
            assert sources["A"]["failure"]["phase"] == "planning"
            assert sources["A"]["attempts"] == []
            assert not any(path.startswith("A/") for path in actual["metadata_zone"])
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_volatile_files_are_whole_file_exclusions_with_reasons(tmp_path):
    assert VOLATILE_FILES and all(reason.strip() for reason in VOLATILE_FILES.values())
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/run.log").write_text("volatile timestamp")
    (tmp_path / "lineage/openlineage").mkdir(parents=True)
    (tmp_path / "lineage/openlineage/2026-01-01.ndjson").write_text("volatile eventTime")
    (tmp_path / "lineage/run.json").write_text('{"immutable":true}')
    (tmp_path / "new.json").write_text('{"unexpected":true}')
    assert set(metadata_digests(tmp_path)) == {"lineage/run.json", "new.json"}
    before = metadata_digests(tmp_path)
    (tmp_path / "new.json").write_text('{"unexpected":false}')
    assert metadata_digests(tmp_path)["new.json"] != before["new.json"]


def test_directory_inventory_distinguishes_shared_nested_and_similar_prefixes():
    rows = [
        {"source_id": source, "executor": directory}
        for source, directory in (
            ("A", "data/metadata/shared"),
            ("B", "data/metadata/shared"),
            ("C", "data/metadata/shared/child"),
            ("D", "data/metadata/shared-other"),
        )
    ]
    result = directory_collisions(rows)
    assert result["shared"] == [{"directory": "data/metadata/shared", "sources": ["A", "B"]}]
    assert result["nested"] == [
        {
            "ancestor": "data/metadata/shared",
            "ancestor_sources": ["A", "B"],
            "descendant": "data/metadata/shared/child",
            "descendant_sources": ["C"],
        }
    ]
