from __future__ import annotations

import ast
import errno
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from janus.cli.run_all import main as run_all_main
from janus.models.config.issues import SourceConfigValidationError
from janus.planner import Planner, PlanningRequest
from janus.registry import SourceLocation, build_source_dependency_graph, load_registry
from janus.registry.loader import _load_declared_contracts
from tests.support.operator_cli import run_janus
from tests.support.semantics_fixtures import (
    CLEAN,
    CLEAN_CONSUMER,
    CLEAN_PRODUCER,
    LOADS_CLEAN_TODAY,
    MULTI_ISSUE,
    MULTI_ISSUE_EXPECTED,
    REPO_ROOT,
    RULE_CASES,
    fixture_root,
    install_profile,
    load_semantic_inputs,
    materialize,
)

SEMANTICS_MODULE = REPO_ROOT / "src" / "janus" / "registry" / "semantics.py"
CHECKED_IN_ENTRIES = 31

RED_WIRING = pytest.mark.xfail(
    strict=True,
    reason="load_registry does not run the semantic pass, and "
    "janus.registry.SourceSemanticsValidationError does not exist yet",
)
RED_VALIDATE = pytest.mark.xfail(
    strict=True,
    reason="janus.cli.dispatch registers no `validate` verb yet",
)
RED_F1 = pytest.mark.xfail(
    strict=True,
    reason="conf/sources still maps `catalog_payload.id` from a producer "
    "whose contract has no such column (evidence F-1).",
)


def _semantic_issues(
    root: Path | str,
    *,
    contracts: dict | None = None,
    hook_ids: frozenset[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Run the engine over one registry tree and return ``(source_id, path, message)``."""
    from janus.registry.semantics import collect_semantic_issues

    sources, snapshot = load_semantic_inputs(root if isinstance(root, Path) else fixture_root(root))
    options = {} if hook_ids is None else {"hook_ids": hook_ids}
    found = collect_semantic_issues(
        sources, contracts=snapshot if contracts is None else contracts, **options
    )
    return [(source_id, issue.path, issue.message) for source_id, issue in found]


# ---------------------------------------------------------------------------------------
# The fixtures themselves (green): a rule test can only fail for its rule


@pytest.mark.parametrize("name", LOADS_CLEAN_TODAY)
def test_every_rule_fixture_is_otherwise_valid(name: str) -> None:
    root = fixture_root(name).resolve()
    sources_dir = root / "conf" / "sources"
    sources, contracts = load_semantic_inputs(root)
    locations = [SourceLocation(source.source_id, source.config_path) for source in sources]

    graph = build_source_dependency_graph(sources, locations=locations, sources_dir=sources_dir)
    agreed = _load_declared_contracts(
        sources, locations=locations, project_root=root, sources_dir=sources_dir
    )

    assert len(graph.nodes) == len(sources) >= 1
    assert set(agreed) == set(contracts) == {source.source_id for source in sources}


@pytest.mark.parametrize(
    ("name", "path_suffix", "detail"),
    [
        ("rule_a_required_not_in_schema", "quality.required_fields", "not_in_contract"),
        ("rule_d_schema_path_missing", "schema.contract", "does_not_exist.yaml"),
    ],
    ids=["rule_a", "rule_d"],
)
def test_rules_a_and_d_are_already_refused_at_load_exactly_once(
    name: str, path_suffix: str, detail: str
) -> None:
    with pytest.raises(SourceConfigValidationError) as caught:
        load_registry(fixture_root(name))

    issues = caught.value.issues
    assert len(issues) == 1, [issue.render() for issue in issues]
    assert issues[0].path.endswith(path_suffix)
    assert detail in issues[0].message


def test_rules_a_and_d_do_not_fire_on_the_clean_registry() -> None:
    registry = load_registry(fixture_root(CLEAN))

    assert {source.source_id for source in registry.sources} == {CLEAN_PRODUCER, CLEAN_CONSUMER}
    assert set(registry.contracts) == {CLEAN_PRODUCER, CLEAN_CONSUMER}


def _remove(path: Path) -> None:
    path.unlink()


def _replace_with_a_directory(path: Path) -> None:
    path.unlink()
    path.mkdir()


def _empty(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _not_yaml(path: Path) -> None:
    path.write_text("{ not: [yaml", encoding="utf-8")


MISSING_CONTRACT = "the file does not exist"


@pytest.mark.parametrize(
    ("damage", "says"),
    [
        (_remove, MISSING_CONTRACT),
        (_replace_with_a_directory, os.strerror(errno.EISDIR)),
        (_empty, "is required"),
        (_not_yaml, "must be valid YAML"),
    ],
    ids=["missing", "directory", "empty", "not_yaml"],
)
def test_rule_d_tells_a_missing_contract_from_an_unreadable_one(
    tmp_path: Path, damage: Callable[[Path], None], says: str
) -> None:
    """Rule (d) belongs to the contract snapshot (TD-1): one collected issue per damaged file.

    Only a missing file says it does not exist, so "fix the path" and "fix the file" read
    differently. The consumer is disabled, so no active-contract issue joins this one.
    """
    root = materialize(CLEAN, tmp_path / "project")
    damage(root / "conf" / "contracts" / "semantics" / "clean_consumer.yaml")

    with pytest.raises(SourceConfigValidationError) as caught:
        load_registry(root)

    (issue,) = caught.value.issues
    assert issue.path == f"{CLEAN_CONSUMER}.schema.contract"
    assert says in issue.message
    assert (MISSING_CONTRACT in issue.message) is (says == MISSING_CONTRACT)


# ---------------------------------------------------------------------------------------
# The engine: each rule fires on its tree and is silent on the clean control


@pytest.mark.parametrize("rule", sorted(RULE_CASES), ids=lambda rule: f"rule_{rule}")
def test_rule_fires_on_its_violating_fixture(rule: str) -> None:
    """Exactly one issue, on the right source, with the contracted path and message."""
    case = RULE_CASES[rule]

    assert _semantic_issues(case.fixture) == [case.triple]


@pytest.mark.parametrize("rule", sorted(RULE_CASES), ids=lambda rule: f"rule_{rule}")
def test_rule_does_not_fire_on_the_clean_registry(rule: str) -> None:
    case = RULE_CASES[rule]

    assert [issue for issue in _semantic_issues(CLEAN) if issue[1] == case.path] == []


def test_the_clean_registry_reports_nothing_at_all() -> None:
    assert _semantic_issues(CLEAN) == []


def test_rule_c_checks_only_the_top_level_segment_of_a_mapped_column() -> None:
    assert [issue for issue in _semantic_issues(CLEAN) if issue[0] == CLEAN_CONSUMER] == []


def test_rule_c_is_silent_when_the_producer_declares_no_contract() -> None:
    case = RULE_CASES["c"]
    _sources, contracts = load_semantic_inputs(fixture_root(case.fixture))
    contracts.pop("semantics_rule_c_producer")

    assert _semantic_issues(case.fixture, contracts=contracts) == []


def test_rule_e_honours_an_injected_hook_catalog() -> None:
    case = RULE_CASES["e"]

    assert _semantic_issues(case.fixture, hook_ids=frozenset({"semantics.unregistered_hook"})) == []


def test_rules_b_c_and_f_are_silent_for_a_source_without_a_contract() -> None:
    """With no declared schema there is nothing to check (b), (c) and (f) against; only the
    hook rule, which needs no contract, still reports."""
    assert _semantic_issues(MULTI_ISSUE, contracts={}) == [MULTI_ISSUE_EXPECTED[-1]]


def test_issues_come_back_in_rule_order_within_a_source_and_by_source_id_across_sources() -> None:
    """`01_zulu.yaml` is discovered first and sorts last: discovery order is never the order."""
    assert _semantic_issues(MULTI_ISSUE) == list(MULTI_ISSUE_EXPECTED)


def test_a_required_fields_declaration_without_a_contract_is_reported_not_failed() -> None:
    """D-3: no issue — an unverifiable declaration is a `validate` note, never an error."""
    from janus.registry.semantics import unverified_required_fields

    sources, _contracts = load_semantic_inputs(fixture_root(CLEAN))

    assert _semantic_issues(CLEAN, contracts={}) == []
    assert unverified_required_fields(sources, contracts={}) == (
        (CLEAN_CONSUMER, ("detail_id", "code")),
    )


def test_no_checked_in_source_declares_required_fields_without_a_contract() -> None:
    """README §5.3, number 2: D-3's population is empty since every entry has a contract."""
    from janus.registry.semantics import unverified_required_fields

    sources, contracts = load_semantic_inputs(REPO_ROOT)

    assert unverified_required_fields(sources, contracts=contracts) == ()


def test_expected_fields_reads_the_contract_for_every_checked_in_source() -> None:
    from janus.registry.semantics import expected_fields

    sources, contracts = load_semantic_inputs(REPO_ROOT)
    declared = {
        source.source_id: expected_fields(source, contracts=contracts) for source in sources
    }

    assert len(declared) == CHECKED_IN_ENTRIES
    assert all(declared.values()), "every checked-in contract declares at least one column"
    assert declared == {
        source_id: contract.column_names for source_id, contract in contracts.items()
    }


def test_expected_fields_is_none_without_a_contract() -> None:
    from janus.registry.semantics import expected_fields

    sources, _contracts = load_semantic_inputs(fixture_root(CLEAN))

    assert [expected_fields(source, contracts={}) for source in sources] == [None, None]


def test_rule_ids_name_the_engine_rules_in_evaluation_order() -> None:
    from janus.registry import RULE_IDS

    assert RULE_IDS == (
        "primary_key_in_required",
        "iceberg_columns_in_producer_contract",
        "source_hook_resolves",
        "partition_columns_known",
    )
    assert len(RULE_IDS) == len(RULE_CASES)


def test_rule_b_says_what_the_run_time_quality_check_says() -> None:
    from janus.quality.validators import validate_quality_contract

    case = RULE_CASES["b"]
    _sources, contracts = load_semantic_inputs(fixture_root(case.fixture))
    run_time = validate_quality_contract(contracts[case.source_id])

    assert run_time.outcome == "failed"
    assert _semantic_issues(case.fixture) == [(case.source_id, case.path, run_time.message)]


def test_rule_c_is_silent_when_the_producer_is_missing_from_the_registry() -> None:
    """`registry/dependencies.py` refuses a missing producer, with the AC-2 text."""
    from janus.registry.semantics import collect_semantic_issues

    case = RULE_CASES["c"]
    sources, contracts = load_semantic_inputs(fixture_root(case.fixture))
    orphaned = tuple(
        source for source in sources if source.source_id != "semantics_rule_c_producer"
    )

    assert collect_semantic_issues(orphaned, contracts=contracts) == []


def test_rule_c_quotes_a_dotted_column_as_the_consumer_maps_it() -> None:
    """The verdict is the top-level segment's; the message names the mapping as written."""
    _sources, contracts = load_semantic_inputs(fixture_root(CLEAN))
    contracts[CLEAN_PRODUCER] = contracts[CLEAN_CONSUMER] 

    assert _semantic_issues(CLEAN, contracts=contracts) == [
        (
            CLEAN_CONSUMER,
            "access.request_inputs.columns",
            f"reads column(s) the producer {CLEAN_PRODUCER} does not declare: payload.label",
        )
    ]


@RED_F1
def test_the_checked_in_registry_reports_no_semantic_issue() -> None:
    assert _semantic_issues(REPO_ROOT) == []


# ---------------------------------------------------------------------------------------
# Purity: the pass is host-testable

_FORBIDDEN_PASS_IMPORTS = (
    "pyspark",
    "pyiceberg",
    "pyarrow",
    "dagster",
    "janus.planner",
    "janus.runtime",
    "janus.strategies",
    "janus.writers",
)


def _forbidden_imports(source: str, roots: tuple[str, ...] = _FORBIDDEN_PASS_IMPORTS) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        found.extend(
            module
            for module in modules
            if any(module == root or module.startswith(f"{root}.") for root in roots)
        )
    return found


def test_the_semantic_pass_imports_no_engine_and_no_upper_layer() -> None:
    """`janus.planner` imports `janus.registry`: importing it back would be an immediate cycle."""
    assert _forbidden_imports(SEMANTICS_MODULE.read_text(encoding="utf-8")) == []

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, janus.registry.semantics; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & "
            "{'pyspark', 'pyiceberg', 'pyarrow', 'dagster'}))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"


def test_the_import_detector_flags_a_deliberate_violation() -> None:
    violating = "from janus.planner import Planner\nimport pyspark.sql as sql\n"

    assert _forbidden_imports(violating) == ["janus.planner", "pyspark.sql"]


def test_the_import_detector_does_not_flag_clean_code() -> None:
    clean = (
        "from janus.models.config.issues import ValidationIssue\n"
        "from janus.hooks import built_in_hooks\n"
        "from janus.planners_notes import nothing\n"
        "from . import sibling\n"
    )

    assert _forbidden_imports(clean) == []


# ---------------------------------------------------------------------------------------
# One raise site, every issue at once


@RED_WIRING
def test_every_issue_is_collected_before_the_single_raise() -> None:
    from janus.registry import SourceSemanticsValidationError

    with pytest.raises(SourceSemanticsValidationError) as caught:
        load_registry(fixture_root(MULTI_ISSUE))

    assert [(issue.path, issue.message) for issue in caught.value.issues] == [
        (f"{source_id}: {path}", message) for source_id, path, message in MULTI_ISSUE_EXPECTED
    ]


@RED_WIRING
def test_issue_paths_are_prefixed_with_the_source_id() -> None:
    """`- <source_id>: <field path>: <message>` under `Invalid source registry: <sources_dir>`."""
    from janus.registry import SourceSemanticsValidationError

    case = RULE_CASES["b"]
    root = fixture_root(case.fixture).resolve()

    with pytest.raises(SourceSemanticsValidationError) as caught:
        load_registry(root)

    assert str(caught.value).splitlines() == [
        f"Invalid source registry: {root / 'conf' / 'sources'}",
        case.rendered,
    ]


# ---------------------------------------------------------------------------------------
# AC-3: one rule fixture, four call sites, one assertion each



PARITY_CASE = RULE_CASES["c"]


def _issue_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


def _run_all(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], root: Path):
    for key in [key for key in os.environ if key.startswith("JANUS_")]:
        monkeypatch.delenv(key)
    exit_code = run_all_main(("--project-root", str(root), "--environment", "local"))
    return exit_code, capsys.readouterr()


@RED_WIRING
def test_ac3_load_registry_rejects_it() -> None:
    """The registry sweep test's call site; still a `SourceConfigValidationError` (D-6)."""
    from janus.registry import SourceSemanticsValidationError

    with pytest.raises(SourceSemanticsValidationError) as caught:
        load_registry(fixture_root(PARITY_CASE.fixture))

    assert isinstance(caught.value, SourceConfigValidationError)
    assert _issue_lines(str(caught.value)) == [PARITY_CASE.rendered]


@RED_WIRING
def test_ac3_planner_plan_rejects_it() -> None:
    """`Planner.plan` loads the registry itself when no snapshot is injected."""
    from janus.registry import SourceSemanticsValidationError

    request = PlanningRequest.create(
        source_id="semantics_rule_c_unrelated",
        environment="local",
        project_root=fixture_root(PARITY_CASE.fixture),
        include_disabled=True,
    )

    with pytest.raises(SourceSemanticsValidationError) as caught:
        Planner().plan(request)

    assert _issue_lines(str(caught.value)) == [PARITY_CASE.rendered]


@RED_WIRING
def test_ac3_run_all_rejects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 — nothing executed — and not one runtime directory created."""
    root = materialize(PARITY_CASE.fixture, tmp_path / "project")
    install_profile(root, "local")

    exit_code, captured = _run_all(monkeypatch, capsys, root)

    assert exit_code == 2
    assert _issue_lines(captured.err) == [PARITY_CASE.rendered]
    assert captured.out == ""
    assert not (root / "data").exists()


@RED_VALIDATE
def test_ac3_validate_rejects_it(tmp_path: Path) -> None:
    root = materialize(PARITY_CASE.fixture, tmp_path / "project")

    result = run_janus(("validate", "--project-root", str(root)))

    assert result.exit_code == 2
    assert _issue_lines(result.stderr) == [PARITY_CASE.rendered]


@RED_VALIDATE
def test_ac3_all_four_render_the_same_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One pass and one renderer, so the four call sites cannot word the issue differently."""
    root = materialize(PARITY_CASE.fixture, tmp_path / "project")
    install_profile(root, "local")
    rendered: dict[str, list[str]] = {}

    with pytest.raises(SourceConfigValidationError) as loaded:
        load_registry(root)
    rendered["load_registry"] = _issue_lines(str(loaded.value))

    with pytest.raises(SourceConfigValidationError) as planned:
        Planner().plan(
            PlanningRequest.create(
                source_id="semantics_rule_c_unrelated",
                environment="local",
                project_root=root,
                include_disabled=True,
            )
        )
    rendered["planner"] = _issue_lines(str(planned.value))

    _exit_code, captured = _run_all(monkeypatch, capsys, root)
    rendered["run-all"] = _issue_lines(captured.err)
    rendered["validate"] = _issue_lines(run_janus(("validate", "--project-root", str(root))).stderr)

    assert rendered == {call_site: [PARITY_CASE.rendered] for call_site in rendered}
