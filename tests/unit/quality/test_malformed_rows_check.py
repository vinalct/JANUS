"""Host-side checks for malformed-row evidence and report formatting."""

from __future__ import annotations

import json

import pytest

from janus.quality.contract_checks import ContractCheck
from janus.quality.malformed_rows import (
    MalformedRowsError,
    bounded_samples,
    malformed_rows_check,
)
from janus.quality.pre_write import PreWriteEvidence, summarize_pre_write_evidence


def test_samples_are_scrubbed_bounded_and_truncation_is_reported():
    rows = [
        ('{"token": "sk-live-123456", "label": "' + "x" * 600 + '"}',),
        ("https://example.invalid/?token=sk-live-123456",),
    ]

    samples, truncated = bounded_samples(rows)

    assert truncated
    assert len(samples) == 2
    assert all(len(sample) <= 500 and "sk-live-123456" not in sample for sample in samples)
    assert "REDACTED" in samples[0]
    assert samples[0].endswith("…")


@pytest.mark.parametrize(
    ("enforcement", "count", "threshold", "outcome"),
    [
        ("strict", 2, 0, "failed"),
        ("strict", 2, 3, "passed"),
        ("lenient", 2, 0, "passed"),
    ],
)
def test_check_reports_count_threshold_and_samples(enforcement, count, threshold, outcome):
    check = malformed_rows_check(
        count,
        enforcement=enforcement,
        threshold=threshold,
        samples=('bad "value"',),
        batch_index=2,
    )

    details = check.details_as_dict()
    assert check.outcome == outcome
    assert details["count"] == "2"
    assert details["threshold"] == str(threshold)
    assert details["batch_index"] == "2"
    assert json.loads(details["samples"]) == ['bad "value"']
    if enforcement == "lenient":
        assert details["severity"] == "warning"
        assert check.message.startswith("WARNING:")


def test_typed_handoff_skips_the_check():
    check = malformed_rows_check(None, enforcement="strict", threshold=0)
    assert check.outcome == "skipped"
    assert "typed by construction" in check.message


def test_malformed_rows_error_carries_stage_without_echoing_samples():
    error = MalformedRowsError(2, ("secret-record",), 0, 1, 3)
    assert error.failure_stage == "malformed_rows"
    assert (error.count, error.threshold, error.batch_index) == (2, 0, 1)
    assert "secret-record" not in str(error)


def test_batch_summary_adds_malformed_counts_and_bounded_samples():
    contract_check = ContractCheck("test", "1.0.0", "a" * 64, (), (), 1)
    evidence = (
        PreWriteEvidence(1, 2, contract_check, malformed_count=2, malformed_samples=("one",)),
        PreWriteEvidence(2, 2, contract_check, malformed_count=3, malformed_samples=("two",)),
    )

    check = summarize_pre_write_evidence(evidence, enforcement="lenient")["malformed_rows"]

    assert check.outcome == "passed"
    assert check.details_as_dict()["count"] == "5"
    assert json.loads(check.details_as_dict()["samples"]) == ["one", "two"]
    assert check.message.startswith("WARNING:")
