"""Regression coverage for the non-vacuous pytest report gate used by CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.required_test_gate import (
    REQUIRED_SUITES,
    assert_required_suite_ran,
    assert_required_tests_ran,
    main,
)


def _report(tmp_path: Path, cases: str, *, name: str = "pytest.xml") -> Path:
    report = tmp_path / name
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


def test_one_named_case_can_be_required_without_accepting_a_neighbour(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="acceptance" />'
        '<testcase classname="required.suite" name="neighbour"><skipped /></testcase>',
    )

    assert_required_tests_ran(
        report,
        minimum_passed=1,
        class_name="required.suite",
        test_name="acceptance",
    )


def test_a_parametrized_named_case_satisfies_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="acceptance[sqlite]" />',
    )

    assert_required_tests_ran(
        report,
        minimum_passed=1,
        class_name="required.suite",
        test_name="acceptance",
    )


def test_a_longer_unparametrized_name_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="acceptance_extended" />',
    )

    with pytest.raises(AssertionError, match="collected 0 matching cases"):
        assert_required_tests_ran(
            report,
            minimum_passed=1,
            class_name="required.suite",
            test_name="acceptance",
        )


def test_a_missing_named_case_fails_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="different" />',
    )

    with pytest.raises(AssertionError, match="collected 0 matching cases"):
        assert_required_tests_ran(
            report,
            minimum_passed=1,
            class_name="required.suite",
            test_name="acceptance",
        )


def test_missing_or_under_collected_cases_fail_the_gate(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testcase classname="required.suite" name="only" />',
    )

    with pytest.raises(AssertionError, match="expected at least 2"):
        assert_required_tests_ran(report, minimum_passed=2)


def test_cli_combines_required_cases_from_separate_processes(tmp_path: Path) -> None:
    report = _report(tmp_path, '<testcase classname="required.suite" name="small" />')
    large = _report(
        tmp_path, '<testcase classname="required.suite" name="large" />', name="large.xml"
    )

    assert main([
        str(report), "--additional-report", str(large), "--class-name", "required.suite",
        "--minimum-passed", "2",
    ]) == 0


def test_a_skip_in_an_additional_report_is_never_accepted(tmp_path: Path) -> None:
    report = _report(tmp_path, '<testcase classname="required.suite" name="small" />')
    large = _report(
        tmp_path,
        '<testcase classname="required.suite" name="large"><skipped /></testcase>',
        name="large.xml",
    )

    with pytest.raises(AssertionError, match="did not execute successfully"):
        assert_required_tests_ran(
            report, additional_reports=(large,), minimum_passed=2, class_name="required.suite"
        )


def test_a_missing_additional_report_fails_closed(tmp_path: Path) -> None:
    report = _report(tmp_path, '<testcase classname="required.suite" name="small" />')

    with pytest.raises(FileNotFoundError):
        assert_required_tests_ran(
            report, additional_reports=(tmp_path / "missing.xml",), minimum_passed=1
        )


def test_duplicate_reports_cannot_inflate_the_required_count(tmp_path: Path) -> None:
    report = _report(tmp_path, '<testcase classname="required.suite" name="same" />')

    with pytest.raises(AssertionError, match="collected 1 matching cases; expected at least 2"):
        assert_required_tests_ran(report, additional_reports=(report,), minimum_passed=2)


def test_a_later_pass_cannot_hide_an_earlier_skip(tmp_path: Path) -> None:
    skipped = _report(
        tmp_path,
        '<testcase classname="required.suite" name="same"><skipped /></testcase>',
    )
    passed = _report(
        tmp_path, '<testcase classname="required.suite" name="same" />', name="passed.xml"
    )

    with pytest.raises(AssertionError, match="did not execute successfully"):
        assert_required_tests_ran(skipped, additional_reports=(passed,), minimum_passed=1)


def _metrics_cases(*, missing: int | None = None, skipped: int | None = None) -> str:
    cases = []
    for index, (class_name, test_name) in enumerate(REQUIRED_SUITES["metrics"]):
        # A green neighbouring test must not replace the required acceptance case.
        reported_name = test_name + "_neighbour" if index == missing else test_name
        outcome = "<skipped />" if index == skipped else ""
        cases.append(
            f'<testcase classname="{class_name}" name="{reported_name}">{outcome}</testcase>'
        )
    return "".join(cases)


@pytest.mark.parametrize("missing", range(len(REQUIRED_SUITES["metrics"])))
def test_each_deselected_metrics_acceptance_case_fails_the_suite(tmp_path, missing):
    assert len(REQUIRED_SUITES["metrics"]) == 2
    report = _report(tmp_path, _metrics_cases(missing=missing))
    with pytest.raises(AssertionError, match="collected 0 matching cases"):
        assert_required_suite_ran(report, suite="metrics")


@pytest.mark.parametrize("skipped", range(len(REQUIRED_SUITES["metrics"])))
def test_each_skipped_metrics_acceptance_case_fails_the_suite(tmp_path, skipped):
    report = _report(tmp_path, _metrics_cases(skipped=skipped))
    with pytest.raises(AssertionError, match="did not execute successfully"):
        assert_required_suite_ran(report, suite="metrics")


def test_metrics_suite_cli_requires_both_cases(tmp_path):
    report = _report(tmp_path, _metrics_cases())
    assert main([str(report), "--suite", "metrics"]) == 0


@pytest.mark.parametrize("selectors", [[], ["--suite", "metrics", "--minimum-passed", "0"]])
def test_suite_cli_cannot_disable_the_required_count(tmp_path, selectors):
    report = _report(tmp_path, "")
    with pytest.raises(SystemExit) as result:
        main([str(report), *selectors])
    assert result.value.code == 2


def test_metrics_acceptance_suite_is_required_by_local_and_hosted_spark_gates():
    import yaml

    root = Path(__file__).resolve().parents[3]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    ci_recipe = makefile.split("\nci: ensure-up\n", 1)[1].split("\n# Registry", 1)[0]
    assert "tests.support.required_test_gate $(CI_GATE_REPORTS) --suite metrics" in ci_recipe
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    commands = [step.get("run", "") for step in workflow["jobs"]["spark"]["steps"]]
    assert any(
        "tests.support.required_test_gate" in command and "--suite metrics" in command
        for command in commands
    )


@pytest.mark.parametrize("missing", range(len(REQUIRED_SUITES["status_catalog"])))
def test_each_missing_catalog_status_case_fails_the_suite(tmp_path, missing):
    cases = "".join(
        f'<testcase classname="{class_name}" name="{test_name}" />'
        for index, (class_name, test_name) in enumerate(REQUIRED_SUITES["status_catalog"])
        if index != missing
    )
    with pytest.raises(AssertionError, match="collected 0 matching cases"):
        assert_required_suite_ran(_report(tmp_path, cases), suite="status_catalog")


@pytest.mark.parametrize("skipped", range(len(REQUIRED_SUITES["status_catalog"])))
def test_each_skipped_catalog_status_case_fails_the_suite(tmp_path, skipped):
    cases = "".join(
        f'<testcase classname="{class_name}" name="{test_name}">'
        f'{"<skipped />" if index == skipped else ""}</testcase>'
        for index, (class_name, test_name) in enumerate(REQUIRED_SUITES["status_catalog"])
    )
    with pytest.raises(AssertionError, match="did not execute successfully"):
        assert_required_suite_ran(_report(tmp_path, cases), suite="status_catalog")


def test_all_catalog_status_cases_satisfy_the_suite(tmp_path):
    cases = "".join(
        f'<testcase classname="{class_name}" name="{test_name}" />'
        for class_name, test_name in REQUIRED_SUITES["status_catalog"]
    )
    assert main([str(_report(tmp_path, cases)), "--suite", "status_catalog"]) == 0


def test_catalog_status_cases_are_required_by_local_and_hosted_spark_gates():
    import yaml

    root = Path(__file__).resolve().parents[3]
    makefile = (root / "Makefile").read_text()
    recipe = makefile.split("\nci: ensure-up\n", 1)[1].split("\n\n", 1)[0]
    assert "required_test_gate $(CI_UNIT_TEST_REPORT) --suite status_catalog" in recipe
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    assert any(
        "tests.support.required_test_gate" in step.get("run", "")
        and "--suite status_catalog" in step["run"]
        for step in workflow["jobs"]["spark"]["steps"]
    )
