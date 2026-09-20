"""Fail a CI gate when its required pytest cases were skipped or not collected."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree


def _outcome(case: ElementTree.Element) -> str:
    for outcome in ("failure", "error", "skipped"):
        if case.find(outcome) is not None:
            return outcome
    return "passed"


def assert_required_tests_ran(
    report: Path,
    *,
    minimum_passed: int,
    class_name: str | None = None,
) -> None:
    root = ElementTree.parse(report).getroot()
    cases = [
        case
        for case in root.iter("testcase")
        if class_name is None or case.get("classname") == class_name
    ]
    outcomes = {
        f"{case.get('classname')}::{case.get('name')}": _outcome(case) for case in cases
    }
    not_passed = {case: outcome for case, outcome in outcomes.items() if outcome != "passed"}

    if len(cases) < minimum_passed:
        raise AssertionError(
            f"required pytest gate collected {len(cases)} matching cases; "
            f"expected at least {minimum_passed}"
        )
    if not_passed:
        raise AssertionError(f"required pytest cases did not execute successfully: {not_passed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-passed", type=int, required=True)
    parser.add_argument("--class-name")
    args = parser.parse_args(argv)
    assert_required_tests_ran(
        args.report,
        minimum_passed=args.minimum_passed,
        class_name=args.class_name,
    )
    print(f"OK: required pytest cases executed ({args.report})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
