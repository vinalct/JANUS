"""Regression coverage for the non-vacuous pytest report gate used by CI."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

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


LOCKING_PARALLEL_CLASSES = (
    "tests.unit.runtime.test_source_state_lock",
    "tests.unit.runtime.test_lock_refusal_evidence",
    "tests.unit.cli.test_operator_verbs_hold_the_lock",
    "tests.unit.toolchain.test_lock_acquisition_sites",
    "tests.unit.orchestration.test_ready_queue_scheduler",
    "tests.unit.orchestration.test_attempt_wire_format",
    "tests.unit.runtime.test_parallel_runner",
    "tests.unit.toolchain.test_no_threaded_source_execution",
    "tests.unit.orchestration.test_orchestration_settings",
)
ROOT = Path(__file__).resolve().parents[3]


def _gate_configuration():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    variables = dict(re.findall(
        r"^(\w+) := (.*)$", makefile.replace("\\\n", " "), re.MULTILINE
    ))
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    return makefile, variables, workflow["jobs"]


def _recipe(makefile, target):
    return makefile.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]


def _report_gates(commands, variables):
    """Compare selectors, counts, and report order after expanding Make's gate variables."""
    gates = {}
    for reported in re.findall(
        r"tests\.support\.required_test_gate ([^\n]+)", commands.replace("\\\n", " ")
    ):
        command = reported
        while re.search(r"\$\((\w+)\)", command):
            command = re.sub(r"\$\((\w+)\)", lambda match: variables[match[1]], command)
        tokens = shlex.split(command.removesuffix(")"))
        reports, options = [tokens[0]], {}
        for flag, value in zip(tokens[1::2], tokens[2::2], strict=True):
            if flag == "--additional-report":
                reports.append(value)
            else:
                options[flag] = value
        key = tuple(options.get(flag) for flag in ("--class-name", "--test-name", "--suite"))
        assert key not in gates, f"duplicate required gate: {key}"
        gates[key] = (int(options.get("--minimum-passed", "0")), tuple(reports))
    assert gates, "no required report gates found"
    return gates


def test_fast_requires_all_nine_locking_and_parallel_suites():
    makefile, variables, jobs = _gate_configuration()
    assert tuple(variables["LOCKING_PARALLEL_REQUIRED_CLASSES"].split()) == LOCKING_PARALLEL_CLASSES
    recipe = _recipe(makefile, "test-fast")
    assert "$(LOCKING_PARALLEL_REQUIRED_CLASSES)" in recipe
    assert '--class-name "$$class_name" --minimum-passed 1 || exit $$?' in recipe
    assert any(step.get("run") == "make test-fast PYTHON=python" for step in jobs["fast"]["steps"])


@pytest.mark.parametrize(("class_name", "count"), [
    ("tests.unit.toolchain.test_lock_acquisition_sites", 39),
    ("tests.unit.runtime.test_source_state_lock", 34),
])
def test_fast_requires_every_lock_sweep_and_kernel_lock_case(class_name, count):
    makefile, variables, _jobs = _gate_configuration()
    # Counted commands follow the loops; the loop's shell variable is not a Make variable.
    counted = _recipe(makefile, "test-fast").split("\tdone\n")[-1]
    gates = _report_gates(counted, variables)
    assert gates[(class_name, None, None)] == (count, (variables["FAST_TEST_REPORT"],))


def test_local_and_hosted_spark_require_the_same_suites_counts_and_reports():
    makefile, variables, jobs = _gate_configuration()
    recipe = makefile.split("\nci: ensure-up\n", 1)[1].split("\n\n", 1)[0]
    local = _report_gates(recipe, variables)
    steps = jobs["spark"]["steps"]
    first = next(index for index, step in enumerate(steps) if "--cov=janus" in step.get("run", ""))
    hosted = _report_gates("\n".join(step.get("run", "") for step in steps[first:]), {})
    assert local == hosted
    reports = tuple(variables[name] for name in (
        "CI_TEST_REPORT", "CI_UNIT_TEST_REPORT", "CI_LARGE_TEST_REPORT"
    ))
    for class_name, count in (
        ("tests.integration.orchestration.test_dependency_execution", 8),
        ("tests.integration.orchestration.test_parallel_execution", 18),
        ("tests.integration.dagster.test_cli_adapter_parity", 15),
    ):
        assert local[(class_name, None, None)] == (count, reports)
    classes = [key[0] for key in local]
    orchestration = classes.index("tests.integration.orchestration.test_dependency_execution")
    assert classes[orchestration + 1:orchestration + 3] == [
        "tests.integration.orchestration.test_parallel_execution",
        "tests.integration.dagster.test_cli_adapter_parity",
    ]


def test_adapter_requires_the_full_green_selection_in_both_python_jobs():
    makefile, variables, jobs = _gate_configuration()
    gates = _report_gates(_recipe(makefile, "test-adapter"), variables)
    assert gates[(None, None, None)] == (40, (variables["DAGSTER_TEST_REPORT"],))
    assert jobs["adapter"]["strategy"]["matrix"]["python-version"] == ["3.13", "3.14"]
    assert any(
        step.get("run") == "make test-adapter PYTHON=python" for step in jobs["adapter"]["steps"]
    )


def test_ci_executes_complete_spark_adapter_parity_in_the_optional_extra_venv():
    makefile, _variables, jobs = _gate_configuration()
    recipe = makefile.split("\nci: ensure-up\n", 1)[1].split("\n\n", 1)[0]
    assert "$(JANUS_ENTRYPOINT) $(MAKE) test-ci-integration" in recipe
    assert any(
        "make test-ci-integration" in step.get("run", "") for step in jobs["spark"]["steps"]
    )
    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "test-ci-integration"],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    commands = result.stdout.splitlines()
    assert '--system-site-packages "/tmp/janus-ci-dagster"' in commands[0]
    assert 'pip install "dagster==1.13.22"' in commands[1]
    assert "/tmp/janus-ci-dagster/bin/python -m pytest" in commands[2]
    assert "tests/integration" in commands[2]
    assert "--ignore=tests/integration/dagster" not in commands[2]
    assert "not spark" not in commands[2]
    assert "--junitxml=data/metadata/test-reports/ci.xml" in commands[2]
