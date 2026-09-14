"""One bronze table, two pipelines: the declaration that makes sharing deliberate.

A full-refresh rebuild and an incremental delta job over the same dataset are two
operations, and both write the dataset's table. The type layer decides only the shape of
that declaration; whether the named sources exist, write the same table and name this one
back is a whole-registry question answered in ``tests/unit/orchestration``.
"""

from __future__ import annotations

import pytest

from janus.models import OutputTarget
from janus.models.config.issues import ValidationIssue
from janus.models.config.outputs import _build_output_target


def test_a_bronze_target_declares_no_co_writers_by_default():
    """The common case is one writer, so sharing is opt-in and visible in the diff."""
    target = OutputTarget(path="data/bronze/example/dataset", format="iceberg")

    assert target.shared_with == ()


def test_a_directly_constructed_target_rejects_a_co_writer_that_names_nobody():
    """The parser collects this as an issue; the invariant stops a fixture skipping it."""
    with pytest.raises(ValueError, match="non-empty source ids"):
        OutputTarget(
            path="data/bronze/example/dataset",
            format="iceberg",
            shared_with=("rebuild_source", "   "),
        )


def test_a_directly_constructed_target_rejects_a_repeated_co_writer():
    with pytest.raises(ValueError, match="must not repeat a source id"):
        OutputTarget(
            path="data/bronze/example/dataset",
            format="iceberg",
            shared_with=("delta_source", "delta_source"),
        )


def test_the_parser_keeps_declaration_order():
    """Order is the author's; the registry compares sets, so nothing re-sorts it here."""
    issues: list[ValidationIssue] = []

    target = _build_output_target(
        {
            "path": "data/bronze/example/dataset",
            "format": "iceberg",
            "namespace": "bronze_example",
            "table_name": "dataset",
            "shared_with": ["delta_source", "backfill_source"],
        },
        "outputs.bronze",
        issues,
    )

    assert issues == []
    assert target.shared_with == ("delta_source", "backfill_source")


@pytest.mark.parametrize(
    ("field_path", "target_format", "expected"),
    [
        ("outputs.raw", "json", "is only supported for outputs.bronze"),
        ("outputs.bronze", "parquet", "requires format='iceberg'"),
    ],
)
def test_sharing_is_only_meaningful_for_a_bronze_iceberg_table(
    field_path, target_format, expected
):
    """A directory of files has no table identity for a second writer to share."""
    issues: list[ValidationIssue] = []

    _build_output_target(
        {
            "path": "data/example/dataset",
            "format": target_format,
            "shared_with": ["somebody_else"],
        },
        field_path,
        issues,
    )

    assert ValidationIssue(f"{field_path}.shared_with", expected) in issues


def test_a_malformed_declaration_collects_an_issue_without_raising():
    """Builders never raise: one load must still report every other problem in the file."""
    issues: list[ValidationIssue] = []

    target = _build_output_target(
        {
            "path": "data/bronze/example/dataset",
            "format": "iceberg",
            "shared_with": "delta_source",
        },
        "outputs.bronze",
        issues,
    )

    assert [issue.path for issue in issues] == ["outputs.bronze.shared_with"]
    assert target.shared_with == ()
