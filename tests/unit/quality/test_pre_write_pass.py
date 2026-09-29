"""The one pre-write data pass: required nulls and malformed rows, counted before the write."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")

from janus.models.data_contracts import DataContract, load_data_contract
from janus.quality import ContractViolationError, run_pre_write_pass
from tests.support.spark_sessions import build_iceberg_session

RED_TASK_2 = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
CORRUPT = "_janus_corrupt_record"
BASE_DDL = "id string, label string, amount bigint, when timestamp"
TRACKED_DDL = f"{BASE_DDL}, {CORRUPT} string"
WHEN = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
SAMPLE_LIMIT = 5
SAMPLE_CHARACTER_LIMIT = 500

JOB_TRIGGERING_METHODS = (
    "collect",
    "count",
    "first",
    "foreach",
    "foreachPartition",
    "head",
    "show",
    "take",
    "toLocalIterator",
    "toPandas",
)


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    session = build_iceberg_session(
        "janus-pre-write-pass", tmp_path_factory.mktemp("janus-pre-write-pass")
    )
    yield session
    session.stop()


def _contract(name: str) -> DataContract:
    return load_data_contract(HOSTILE / f"{name}.yaml")


def _rows(*ids: str | None) -> list[tuple[Any, ...]]:
    return [(value, f"label-{index}", index, WHEN) for index, value in enumerate(ids)]


def _tracked_rows(*corrupt: str | None) -> list[tuple[Any, ...]]:
    """One clean-looking row per entry; a non-``None`` entry is that row's raw malformed text."""
    return [
        (f"r{index}", f"label-{index}", None if text else index, WHEN, text)
        for index, text in enumerate(corrupt)
    ]


def _run(frame: Any, contract: DataContract, **options: Any) -> tuple[Any, Any]:
    options.setdefault("batch_index", 1)
    options.setdefault("batch_count", 1)
    return run_pre_write_pass(frame, contract, **options)


def _check(evidence: Any, name: str, *, enforcement: str) -> Any:
    return next(
        check
        for check in evidence.checks(enforcement=enforcement)
        if (check.phase, check.name) == ("data", name)
    )


class _ActionCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.depth = 0


@contextmanager
def _counting_actions(monkeypatch: pytest.MonkeyPatch, frame: Any) -> Iterator[_ActionCounter]:
    """Count job-triggering calls on the frame's concrete DataFrame class, outermost only."""
    counter = _ActionCounter()
    frame_class = type(frame)
    for name in JOB_TRIGGERING_METHODS:
        original = getattr(frame_class, name, None)
        if original is None:
            continue

        def counted(self, *args, _original=original, _name=name, **kwargs):
            counter.depth += 1
            try:
                if counter.depth == 1:
                    counter.calls.append(_name)
                return _original(self, *args, **kwargs)
            finally:
                counter.depth -= 1

        monkeypatch.setattr(frame_class, name, counted)
    yield counter


# ── the structural gate and the required-null half ──────────────────


def test_strict_counts_required_nulls_in_exactly_one_action(spark, monkeypatch):
    frame = spark.createDataFrame(_rows("a", "b", "c"), BASE_DDL)

    with _counting_actions(monkeypatch, frame) as counter:
        checked, evidence = _run(frame, _contract("base"), enforcement="strict")

    assert len(counter.calls) == 1, counter.calls
    assert dict(evidence.required_null_counts) == {"id": 0}
    assert (evidence.batch_index, evidence.batch_count) == (1, 1)
    assert checked.columns == frame.columns


def test_blank_strings_count_as_null_for_a_required_string_column(spark):
    frame = spark.createDataFrame(_rows("a", None, "", "   "), BASE_DDL)

    with pytest.raises(ContractViolationError) as raised:
        _run(frame, _contract("base"), enforcement="strict")

    assert raised.value.failure_stage == "contract_check"
    required = [
        mismatch
        for mismatch in raised.value.check.mismatches
        if mismatch.kind == "required_null"
    ]
    assert [(mismatch.column, mismatch.observed) for mismatch in required] == [
        ("id", "3 null/blank")
    ]


def test_a_structural_mismatch_raises_before_any_action(spark, monkeypatch):
    frame = spark.createDataFrame(
        [("a", "one", WHEN)], "id string, label string, when timestamp"
    )

    with (
        _counting_actions(monkeypatch, frame) as counter,
        pytest.raises(ContractViolationError) as raised,
    ):
        _run(frame, _contract("base"), enforcement="strict")

    assert counter.calls == []
    assert [(m.kind, m.column) for m in raised.value.check.mismatches] == [
        ("missing_column", "amount")
    ]


def test_lenient_runs_no_action_and_leaves_required_nulls_to_the_gate(spark, monkeypatch):
    frame = spark.createDataFrame(_rows("a", None), BASE_DDL)

    with _counting_actions(monkeypatch, frame) as counter:
        _, evidence = _run(frame, _contract("base_lenient"), enforcement="lenient")

    assert counter.calls == []
    assert dict(evidence.required_null_counts) == {}


def test_the_evidence_reports_the_structural_check_as_schema_expectations(spark):
    frame = spark.createDataFrame(_rows("a"), BASE_DDL)

    _, evidence = _run(frame, _contract("base"), enforcement="strict", batch_index=2, batch_count=3)

    report = _check(evidence, "schema_expectations", enforcement="strict")
    assert report.outcome == "passed"
    assert evidence.contract_check.ok is True
    assert (evidence.batch_index, evidence.batch_count) == (2, 3)


# ── malformed rows ──────────────────────────────────────────────────


@RED_TASK_2
def test_malformed_rows_share_the_one_action_and_samples_cost_a_second(spark, monkeypatch):
    clean = spark.createDataFrame(_tracked_rows(None, None, None), TRACKED_DDL)
    drifted = spark.createDataFrame(
        _tracked_rows(None, '{"amount": "abc"}', None, '{"amount": 12.5}'), TRACKED_DDL
    )

    with _counting_actions(monkeypatch, clean) as clean_counter:
        _, clean_evidence = _run(
            clean, _contract("base"), enforcement="strict", tracks_corrupt=True
        )
    clean_calls = list(clean_counter.calls)
    with _counting_actions(monkeypatch, drifted) as drifted_counter:
        _, drifted_evidence = _run(
            drifted,
            _contract("base"),
            enforcement="strict",
            max_malformed_rows=3,
            tracks_corrupt=True,
        )

    assert len(clean_calls) == 1, clean_calls
    assert clean_evidence.malformed_count == 0
    assert clean_evidence.malformed_samples == ()
    assert len(drifted_counter.calls) == 2, drifted_counter.calls
    assert drifted_evidence.malformed_count == 2


@RED_TASK_2
def test_the_frame_handed_back_never_carries_the_corrupt_column(spark):
    frame = spark.createDataFrame(_tracked_rows(None, '{"amount": "abc"}'), TRACKED_DDL)

    checked, _ = _run(
        frame, _contract("base_lenient"), enforcement="lenient", tracks_corrupt=True
    )

    assert CORRUPT not in checked.columns
    assert checked.columns == ["id", "label", "amount", "when"]


@RED_TASK_2
def test_samples_are_at_most_five_bounded_and_flag_truncation(spark):
    long_record = '{"label": "' + "x" * 1200 + '", "amount": "abc"}'
    frame = spark.createDataFrame(
        _tracked_rows(long_record, *(f'{{"amount": "bad-{index}"}}' for index in range(6))),
        TRACKED_DDL,
    )

    _, evidence = _run(
        frame,
        _contract("base"),
        enforcement="strict",
        max_malformed_rows=10,
        tracks_corrupt=True,
    )

    assert evidence.malformed_count == 7
    samples = evidence.malformed_samples
    assert len(samples) == SAMPLE_LIMIT
    assert all(len(sample) <= SAMPLE_CHARACTER_LIMIT for sample in samples)
    assert any(sample.endswith("…") for sample in samples)
    details = _check(evidence, "malformed_rows", enforcement="strict").details_as_dict()
    assert details["truncated"].lower() == "true"
    assert len(json.loads(details["samples"])) == SAMPLE_LIMIT


@RED_TASK_2
def test_samples_are_scrubbed_before_they_reach_the_validation_json(spark):
    echoed = '{"next": "https://example.invalid/r?token=sk-live-123456&page=2", "amount": "x"}'
    frame = spark.createDataFrame(_tracked_rows(echoed), TRACKED_DDL)

    _, evidence = _run(
        frame,
        _contract("base"),
        enforcement="strict",
        max_malformed_rows=1,
        tracks_corrupt=True,
    )

    assert evidence.malformed_count == 1
    assert not any("sk-live-123456" in sample for sample in evidence.malformed_samples)
    details = _check(evidence, "malformed_rows", enforcement="strict").details_as_dict()
    assert "sk-live-123456" not in details["samples"]


@RED_TASK_2
def test_strict_over_the_threshold_raises_malformed_rows_error(spark):
    from janus.quality.malformed_rows import MalformedRowsError

    frame = spark.createDataFrame(
        _tracked_rows(None, '{"amount": "abc"}', '{"amount": 12.5}'), TRACKED_DDL
    )

    with pytest.raises(MalformedRowsError) as raised:
        _run(frame, _contract("base"), enforcement="strict", tracks_corrupt=True)

    error = raised.value
    assert error.failure_stage == "malformed_rows"
    assert error.count == 2
    assert len(error.samples) == 2
    assert "abc" in error.samples[0] or "abc" in error.samples[1]


@RED_TASK_2
def test_strict_within_the_threshold_writes_and_still_reports_the_count(spark):
    frame = spark.createDataFrame(
        _tracked_rows(None, '{"amount": "abc"}', '{"amount": 12.5}'), TRACKED_DDL
    )

    _, evidence = _run(
        frame,
        _contract("base"),
        enforcement="strict",
        max_malformed_rows=3,
        tracks_corrupt=True,
    )

    check = _check(evidence, "malformed_rows", enforcement="strict")
    assert check.outcome == "passed"
    assert check.details_as_dict()["count"] == "2"
    assert check.details_as_dict()["threshold"] == "3"


@RED_TASK_2
def test_lenient_never_raises_and_reports_a_warning(spark):
    frame = spark.createDataFrame(
        _tracked_rows(None, '{"amount": "abc"}', '{"amount": 12.5}'), TRACKED_DDL
    )

    _, evidence = _run(
        frame, _contract("base_lenient"), enforcement="lenient", tracks_corrupt=True
    )

    check = _check(evidence, "malformed_rows", enforcement="lenient")
    assert check.outcome == "passed"
    assert check.message.startswith("WARNING:")
    assert check.details_as_dict()["severity"] == "warning"
    assert check.details_as_dict()["count"] == "2"


@RED_TASK_2
def test_a_parquet_handoff_skips_malformed_rows_but_strict_still_counts_nulls(spark):
    frame = spark.createDataFrame(_rows("a", "b"), BASE_DDL)

    _, evidence = _run(frame, _contract("base"), enforcement="strict", tracks_corrupt=False)

    check = _check(evidence, "malformed_rows", enforcement="strict")
    assert check.outcome == "skipped"
    assert "typed by construction" in check.message
    assert evidence.malformed_count is None
    assert dict(evidence.required_null_counts) == {"id": 0}
