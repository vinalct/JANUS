import copy
import json

import pytest

from tests.integration.handoff.test_bronze_unchanged_under_compaction import assert_matrix

BASELINE = {
    "identifier": "bronze.fixture",
    "schema_digest": "schema",
    "row_digest": "rows",
    "row_count": "3",
}


def _matrix():
    return {
        label: {
            "status": "succeeded",
            "expected_identifier": "bronze.fixture_" + label.replace("/", "_"),
            "actual_identifiers": ["bronze.fixture_" + label.replace("/", "_")],
            "golden": dict(BASELINE),
            "extraction_metadata": {
                "raw_to_bronze": "true",
                "rediscovered_raw_artifact_count": "2",
                "checksums_verified": "true",
            }
            if label.endswith("/replay")
            else {},
        }
        for label in ("pages/live", "compact/live", "pages/replay", "compact/replay")
    }


def _refused(matrix, baseline=BASELINE):
    diagnostic = json.dumps({"matrix": matrix, "baseline": baseline}, sort_keys=True)
    with pytest.raises(AssertionError) as refused:
        assert_matrix(matrix, baseline, diagnostic)
    assert diagnostic in str(refused.value)
    assert all(label in diagnostic for label in matrix)


def test_equal_four_cell_matrix_matches_baseline():
    assert_matrix(_matrix(), BASELINE, "full matrix")


@pytest.mark.parametrize("key", list(BASELINE))
def test_each_golden_field_is_binding_with_full_matrix_diagnostics(key):
    matrix = _matrix()
    matrix["compact/replay"]["golden"][key] = "changed"
    _refused(matrix)


def test_equal_matrix_cannot_hide_baseline_drift():
    changed = {**BASELINE, "row_digest": "drifted before compaction"}
    _refused(_matrix(), changed)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("raw_to_bronze", "false"),
        ("rediscovered_raw_artifact_count", "0"),
        ("checksums_verified", "false"),
    ],
)
def test_replay_proof_is_binding_even_when_goldens_match(key, value):
    matrix = _matrix()
    matrix["pages/replay"]["extraction_metadata"][key] = value
    _refused(matrix)


def test_missing_cell_cannot_pass_as_a_three_way_differential():
    matrix = _matrix()
    del matrix["compact/replay"]
    _refused(matrix)


def test_failed_materialization_reports_every_cell():
    matrix = _matrix()
    matrix["compact/live"]["status"] = "failed"
    del matrix["compact/live"]["golden"]
    _refused(matrix)


def test_actual_identifier_derivation_cannot_be_hidden_by_canonical_golden():
    matrix = _matrix()
    matrix["compact/live"]["actual_identifiers"] = ["bronze.wrong_table"]
    _refused(matrix)


def test_same_table_cannot_be_reused_for_two_cells():
    matrix = _matrix()
    matrix["compact/replay"] = copy.deepcopy(matrix["pages/replay"])
    _refused(matrix)
