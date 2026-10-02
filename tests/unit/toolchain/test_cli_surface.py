from __future__ import annotations

import ast
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import janus
from tests.support.cli_golden import materialize_fixture
from tests.support.operator_cli import arm_spark_tripwire, engine_modules_loaded, run_janus
from tests.support.semantics_fixtures import CLEAN, CLEAN_PRODUCER, materialize

PACKAGE_ROOT = Path(janus.__file__).resolve().parent
CLI_ROOT = PACKAGE_ROOT / "cli"
MAIN_MODULE = PACKAGE_ROOT / "main.py"

EXPECTED_CLI_MODULES = frozenset(
    {
        "common",
        "dispatch",
        "run",
        "run_all",
        "contract",
        "validate",
        "list_sources",
        "dead_letters",
        "checkpoint",
        "operator",
    }
)

EXPECTED_VERBS = frozenset(
    {"run", "run-all", "contract", "validate", "list", "dead-letters", "checkpoint"}
)

FORBIDDEN_MODULE_SCOPE_IMPORTS = ("pyspark", "pyiceberg", "pyarrow", "dagster")

CLI_TIME_BUDGET_SECONDS = 2.0

RED_SURFACE = pytest.mark.xfail(
    strict=True,
    reason="the surface is complete only once the dispatcher and "
    "every verb module exist; whichever lands last lifts this",
)


def _red(what: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"{what} does not exist yet")


def _cli_modules() -> dict[str, Path]:
    return {path.stem: path for path in sorted(CLI_ROOT.glob("*.py")) if path.stem != "__init__"}


def _swept_files() -> dict[str, Path]:
    return {**{f"cli/{name}": path for name, path in _cli_modules().items()}, "main": MAIN_MODULE}


# ---------------------------------------------------------------------------------------
# Detectors


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_scope_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement that runs at import: the module body and the blocks nested in it,
    minus function bodies (a lazy import is allowed) and ``if TYPE_CHECKING:`` blocks."""
    for node in body:
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            yield from _module_scope_statements(node.orelse)
            continue
        for field in ("body", "handlers", "orelse", "finalbody"):
            for child in getattr(node, field, None) or ():
                nested = child.body if isinstance(child, ast.ExceptHandler) else [child]
                yield from _module_scope_statements(nested)


def _module_scope_engine_imports(source: str) -> list[str]:
    found: list[str] = []
    for node in _module_scope_statements(ast.parse(source).body):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        found.extend(
            module
            for module in modules
            if module.split(".")[0] in FORBIDDEN_MODULE_SCOPE_IMPORTS
        )
    return found


def _main_guards(source: str) -> int:
    return sum(
        1
        for node in ast.parse(source).body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def _argv_verb_dispatch(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        subscripts = [
            operand
            for operand in operands
            if isinstance(operand, ast.Subscript)
            and isinstance(operand.value, ast.Name)
            and "argv" in operand.value.id
        ]
        literals = [
            operand.value
            for operand in operands
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str)
        ]
        if subscripts and literals:
            found.extend(literals)
    return found


# ---------------------------------------------------------------------------------------
# 1. Coverage — first, because every sweep below is vacuous without it


def test_the_sweep_finds_the_cli_package_and_its_entry() -> None:
    """Green from day one: the package and today's modules exist, so a glob that matched
    nothing cannot pass the sweeps below."""
    modules = _cli_modules()

    assert CLI_ROOT.is_dir() and MAIN_MODULE.is_file()
    assert {"common", "run_all", "contract"} <= modules.keys()


@RED_SURFACE
def test_the_sweep_covers_every_module_of_the_operator_surface() -> None:
    assert _cli_modules().keys() >= EXPECTED_CLI_MODULES


# ---------------------------------------------------------------------------------------
# 2. One registration path


@RED_SURFACE
def test_every_verb_is_registered_in_the_dispatcher() -> None:
    from janus.cli.dispatch import verbs

    assert {verb.name for verb in verbs()} == EXPECTED_VERBS


def test_every_registered_verb_answers_help_under_its_own_name() -> None:
    """`run` is skipped: its usage line is the implicit form, pinned by the `run_verb_help`
    golden."""
    from janus.cli.dispatch import verbs

    for verb in sorted(verbs(), key=lambda candidate: candidate.name):
        if verb.name == "run":
            continue
        result = run_janus((verb.name, "--help"))
        assert result.exit_code == 0, (verb.name, result.output)
        assert result.stdout.startswith(f"usage: janus {verb.name}"), verb.name


def test_an_unknown_verb_exits_2_and_lists_the_verbs() -> None:
    from janus.cli.dispatch import verbs

    result = run_janus(("frobnicate",))

    assert result.exit_code == 2
    assert "frobnicate" in result.stderr
    assert all(verb.name in result.stderr for verb in verbs())


def test_main_delegates_to_the_dispatcher_and_holds_no_parser() -> None:
    import janus.cli.dispatch as dispatch
    import janus.main as main

    tree = ast.parse(MAIN_MODULE.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    defined = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]

    assert main.main is dispatch.main
    assert "argparse" not in imported
    assert defined == []


def test_no_module_but_the_dispatcher_decides_a_verb_from_argv() -> None:
    violations = {
        name: literals
        for name, path in _swept_files().items()
        if name != "cli/dispatch"
        and (literals := _argv_verb_dispatch(path.read_text(encoding="utf-8")))
    }

    assert violations == {}


def test_no_cli_module_carries_a_second_process_entry_point() -> None:
    """Green guard: only `janus.main` (the console-script target) and, if it wants one, the
    dispatcher may run as `__main__`; a verb module with its own guard is a second entry."""
    guarded = {
        name
        for name, path in _cli_modules().items()
        if name != "dispatch" and _main_guards(path.read_text(encoding="utf-8"))
    }

    assert guarded == set()


def test_the_dispatch_detector_flags_a_deliberate_violation() -> None:
    violating = (
        "def main(argv):\n"
        "    if argv and argv[0] == 'validate':\n"
        "        return 0\n"
        "    if 'list' == resolved_argv[0]:\n"
        "        return 0\n"
    )

    assert sorted(_argv_verb_dispatch(violating)) == ["list", "validate"]


def test_the_dispatch_detector_does_not_flag_ordinary_comparisons() -> None:
    clean = (
        "def render(args, parser):\n"
        "    if args.format == 'json':\n"
        "        return 1\n"
        "    parser.add_parser('validate')\n"
        "    return len(argv) == 0\n"
    )

    assert _argv_verb_dispatch(clean) == []


def test_the_entry_point_detector_flags_a_guard_and_ignores_a_function() -> None:
    assert _main_guards("if __name__ == '__main__':\n    raise SystemExit(main())\n") == 1
    assert _main_guards("def main():\n    if __name__ == 'x':\n        pass\n") == 0


# ---------------------------------------------------------------------------------------
# 3. No engine at module scope (green guard over today's modules, and every new one)


def test_no_cli_module_imports_an_engine_at_module_scope() -> None:
    violations = {
        name: found
        for name, path in _swept_files().items()
        if (found := _module_scope_engine_imports(path.read_text(encoding="utf-8")))
    }

    assert len(_swept_files()) >= 5, "the sweep must find the package before it can clear it"
    assert violations == {}


def test_the_engine_detector_flags_a_deliberate_violation() -> None:
    violating = (
        "import pyspark\n"
        "from pyiceberg.catalog import load_catalog\n"
        "try:\n"
        "    import pyarrow as pa\n"
        "except ImportError:\n"
        "    import dagster\n"
        "class Holder:\n"
        "    from pyspark.sql import SparkSession\n"
    )

    assert _module_scope_engine_imports(violating) == [
        "pyspark",
        "pyiceberg.catalog",
        "pyarrow",
        "dagster",
        "pyspark.sql",
    ]


def test_the_engine_detector_does_not_flag_a_lazy_or_type_only_import() -> None:
    clean = (
        "from typing import TYPE_CHECKING\n"
        "import janus.runtime\n"
        "if TYPE_CHECKING:\n"
        "    from pyspark.sql import SparkSession\n"
        "def build():\n"
        "    from pyspark.sql import SparkSession\n"
        "    return SparkSession\n"
    )

    assert _module_scope_engine_imports(clean) == []


# ---------------------------------------------------------------------------------------
# 4. Session-free verbs (NFR-2): `dead-letters replay --execute` and `--verify-checksums`
#    are absent on purpose — they inherit the executor's lifecycle unchanged.

_VALIDATE = _red("the `validate` verb")
_ENVIRONMENT = _red("`validate --environment`")
_LIST = _red("the `list` verb")
_DEAD_LETTERS = _red("the `dead-letters` verb")
_CHECKPOINT = _red("the `checkpoint` verb")

SESSION_FREE_FORMS = (
    pytest.param(("validate",), marks=_VALIDATE, id="validate"),
    pytest.param(("list", "--graph"), marks=_LIST, id="list"),
    pytest.param(
        ("dead-letters", "list", "--source-id", CLEAN_PRODUCER),
        marks=_DEAD_LETTERS,
        id="dead-letters list",
    ),
    pytest.param(
        ("dead-letters", "release", "--source-id", CLEAN_PRODUCER, "--all", "--reason", "sweep"),
        marks=_DEAD_LETTERS,
        id="dead-letters release",
    ),
    pytest.param(
        ("checkpoint", "show", "--source-id", CLEAN_PRODUCER),
        marks=_CHECKPOINT,
        id="checkpoint show",
    ),
    pytest.param(
        ("checkpoint", "set", "--source-id", CLEAN_PRODUCER, "--to", "2026-09-01T00:00:00Z",
         "--reason", "sweep"),
        marks=_CHECKPOINT,
        id="checkpoint set",
    ),
    pytest.param(
        ("checkpoint", "clear", "--source-id", CLEAN_PRODUCER, "--reason", "sweep"),
        marks=_CHECKPOINT,
        id="checkpoint clear",
    ),
)


@pytest.mark.parametrize("argv", SESSION_FREE_FORMS)
def test_the_verb_never_acquires_a_spark_session(
    argv: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from janus.checkpoints import CheckpointStore, DeadLetterStore
    from janus.planner import Planner, PlanningRequest

    root = materialize(CLEAN, tmp_path / "project")
    plan = (
        Planner()
        .plan(
            PlanningRequest.create(
                source_id=CLEAN_PRODUCER,
                environment="local",
                project_root=root,
                include_disabled=True,
            )
        )
        .plan
    )
    DeadLetterStore().record(
        plan, item_key="window_start=2026-09-01", item_type="request_input",
        error=RuntimeError("status 400"),
    )
    CheckpointStore().save(plan, "2026-09-03T00:00:00Z")
    arm_spark_tripwire(monkeypatch)
    before = engine_modules_loaded()

    verb, *rest = argv
    result = run_janus((verb, *rest[:1], "--project-root", str(root), *rest[1:]))

    assert result.exit_code == 0, result.output
    assert engine_modules_loaded() == before


# ---------------------------------------------------------------------------------------
# 5. The two-second ceiling (AC-9) on the checked-in registry


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(("validate",), marks=_VALIDATE, id="validate"),
        pytest.param(
            ("validate", "--environment", "local"), marks=_ENVIRONMENT, id="validate --environment"
        ),
        pytest.param(("list",), marks=_LIST, id="list"),
        pytest.param(
            ("dead-letters", "list", "--source-id", "ibge_pib_brasil"),
            marks=_DEAD_LETTERS,
            id="dead-letters list",
        ),
        pytest.param(
            ("checkpoint", "show", "--source-id", "ibge_pib_brasil"),
            marks=_CHECKPOINT,
            id="checkpoint show",
        ),
    ],
)
def test_the_verb_completes_within_the_budget(argv: tuple[str, ...], tmp_path: Path) -> None:
    """A ceiling, not a benchmark: the measured figures belong in the evidence file."""
    root = materialize_fixture("repository", tmp_path / "repository")
    verb, *rest = argv
    split = 1 if verb in {"dead-letters", "checkpoint"} else 0
    command = (verb, *rest[:split], "--project-root", str(root), *rest[split:])

    started = time.perf_counter()
    result = run_janus(command)
    elapsed = time.perf_counter() - started

    assert result.exit_code == 0, result.output
    assert elapsed < CLI_TIME_BUDGET_SECONDS, f"{' '.join(argv)} took {elapsed:.2f}s"
