"""Fail a CI gate when its required pytest cases were skipped or not collected."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree


def _outcome(case: ElementTree.Element) -> str:
    for outcome in ("failure", "error", "skipped"):
        if case.find(outcome) is not None:
            return outcome
    return "passed"


def _matches_test_name(reported_name: str | None, required_name: str) -> bool:
    """Match a pytest function name with or without its parameter-set suffix."""
    return reported_name == required_name or (
        reported_name is not None and reported_name.startswith(f"{required_name}[")
    )


def assert_required_tests_ran(
    report: Path,
    *,
    minimum_passed: int,
    class_name: str | None = None,
    test_name: str | None = None,
    additional_reports: Sequence[Path] = (),
) -> None:
    cases = [
        case
        for path in (report, *additional_reports)
        for case in ElementTree.parse(path).getroot().iter("testcase")
        if class_name is None or case.get("classname") == class_name
        if test_name is None or _matches_test_name(case.get("name"), test_name)
    ]
    outcomes = {
        f"{case.get('classname')}::{case.get('name')}": _outcome(case) for case in cases
    }
    not_passed = {
        f"{case.get('classname')}::{case.get('name')}": _outcome(case)
        for case in cases if _outcome(case) != "passed"
    }

    if len(outcomes) < minimum_passed:
        raise AssertionError(
            f"required pytest gate collected {len(outcomes)} matching cases; "
            f"expected at least {minimum_passed}"
        )
    if not_passed:
        raise AssertionError(f"required pytest cases did not execute successfully: {not_passed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--additional-report", type=Path, action="append", default=[])
    parser.add_argument("--minimum-passed", type=int, required=True)
    parser.add_argument("--class-name")
    parser.add_argument("--test-name")
    args = parser.parse_args(argv)
    assert_required_tests_ran(
        args.report,
        minimum_passed=args.minimum_passed,
        class_name=args.class_name,
        test_name=args.test_name,
        additional_reports=args.additional_report,
    )
    reports = ", ".join(str(path) for path in (args.report, *args.additional_report))
    print(f"OK: required pytest cases executed ({reports})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
