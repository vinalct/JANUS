"""Fail a CI gate when its required pytest cases were skipped or not collected."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

REQUIRED_SUITES = {
    "status_catalog": (
        (
            "tests.unit.cli.test_status.TestRunsTable",
            "test_runs_table_reader_matches_zone_using_one_fake_catalog_scan",
        ),
        (
            "tests.unit.cli.test_status.TestRunsTable",
            "test_status_paths_import_no_spark_in_a_fresh_process",
        ),
        *(
            (
                "tests.unit.cli.test_status.TestRunsTable",
                f"test_unreachable_runs_table_exits_two_with_the_reason[{stage}]",
            )
            for stage in ("catalog", "table", "scan")
        ),
        (
            "tests.unit.cli.test_status.TestRunsTable",
            "test_table_selection_filters_one_scan_and_uses_the_configured_catalog",
        ),
    ),
    "metrics": (
        (
            "tests.integration.catalog_commits.test_runs_table_append_sink.TestOrder23Evolution",
            "test_order23_existing_v3_table_adds_seven_columns_and_preserves_old_rows",
        ),
        (
            "tests.integration.catalog_commits.test_queryable_observability.TestOrder23Queries",
            "test_order23_stale_query_and_both_status_readers_return_the_same_breaches",
        ),
    ),
}


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


def assert_required_suite_ran(
    report: Path, *, suite: str, additional_reports: Sequence[Path] = ()
) -> None:
    """Require each named acceptance case; neighbouring tests cannot replace it."""
    for class_name, test_name in REQUIRED_SUITES[suite]:
        assert_required_tests_ran(
            report,
            minimum_passed=1,
            class_name=class_name,
            test_name=test_name,
            additional_reports=additional_reports,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--additional-report", type=Path, action="append", default=[])
    parser.add_argument("--minimum-passed", type=int)
    parser.add_argument("--class-name")
    parser.add_argument("--test-name")
    parser.add_argument("--suite", choices=sorted(REQUIRED_SUITES))
    args = parser.parse_args(argv)
    if args.suite:
        if any(value is not None for value in (
            args.minimum_passed, args.class_name, args.test_name
        )):
            parser.error("--suite cannot be combined with individual case selectors")
        assert_required_suite_ran(
            args.report, suite=args.suite, additional_reports=args.additional_report
        )
    else:
        if args.minimum_passed is None:
            parser.error("--minimum-passed is required unless --suite is selected")
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
