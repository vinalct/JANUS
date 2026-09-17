"""Regression coverage for the non-vacuous pytest report gate used by CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.required_test_gate import assert_required_tests_ran


def _report(tmp_path: Path, cases: str) -> Path:
    report = tmp_path / "pytest.xml"
    report.write_text(
        f'<testsuites><testsuite tests="1">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return report


def test_clean_matching_cases_satisfy_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="first" />'
        '<testcase classname="other.suite" name="irrelevant"><skipped /></testcase>',
    )

    assert_required_tests_ran(report, minimum_passed=1, class_name="required.suite")


def test_a_skipped_required_case_fails_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="hidden"><skipped /></testcase>',
    )

    with pytest.raises(AssertionError, match="did not execute successfully"):
        assert_required_tests_ran(report, minimum_passed=1)


def test_missing_or_under_collected_cases_fail_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="only" />',
    )

    with pytest.raises(AssertionError, match="expected at least 2"):
        assert_required_tests_ran(report, minimum_passed=2)
