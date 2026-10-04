from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
import textwrap
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import pytest

import janus
from tests.support.cli_golden import REPO_ROOT, materialize_fixture
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.support.semantics_fixtures import CLEAN, CLEAN_PRODUCER, materialize

PACKAGE_ROOT = Path(janus.__file__).resolve().parent
CLI_ROOT = PACKAGE_ROOT / "cli"
MAIN_MODULE = PACKAGE_ROOT / "main.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


EXPECTED_CLI_MODULES = frozenset(
    {
        "__init__",
        "common",
        "dispatch",
        "run",
        "run_all",
        "contract",
        "contract_drafting",
        "validate",
        "list_sources",
        "dead_letters",
        "checkpoint",
        "operator",
        "maintain",
    }
)

# The modules that define a `main` today. The registration sweep must find all three, so a
# detector that stopped seeing definitions cannot pass by finding none.
EXPECTED_MAIN_MODULES = frozenset({"janus.cli.dispatch", "janus.cli.run_all", "janus.cli.contract"})

EXPECTED_VERBS = frozenset(
    {"run", "run-all", "contract", "validate", "list", "dead-letters", "checkpoint", "maintain"}
)

FORBIDDEN_MODULE_SCOPE_IMPORTS = ("pyspark", "pyiceberg", "pyarrow", "dagster")

CLI_TIME_BUDGET_SECONDS = 2.0


def _cli_modules() -> dict[str, Path]:
    """Every module of the package, nested ones included: a verb moved into a subpackage
    stays in scope."""
    return {
        path.relative_to(CLI_ROOT).with_suffix("").as_posix(): path
        for path in sorted(CLI_ROOT.rglob("*.py"))
    }


def _swept_files() -> dict[str, Path]:
    return {**{f"cli/{name}": path for name, path in _cli_modules().items()}, "main": MAIN_MODULE}


def _module_name(key: str) -> str:
    parts = ("janus", "cli", *key.split("/"))
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _loaded_engines() -> frozenset[str]:
    return frozenset(name for name in FORBIDDEN_MODULE_SCOPE_IMPORTS if name in sys.modules)


def _builder_bindings(builder: object) -> frozenset[str]:
    """The loaded `janus` modules whose own namespace binds ``builder``."""
    return frozenset(
        name
        for name, module in tuple(sys.modules.items())
        if name.partition(".")[0] == "janus"
        and module is not None
        and vars(module).get("build_spark_session") is builder
    )


# ---------------------------------------------------------------------------------------
# Detectors


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_scope_statements(
    body: list[ast.stmt], *, into_classes: bool = True
) -> Iterator[ast.stmt]:
    """Every statement that runs at import: the module body and the blocks nested in it,
    minus function bodies (a lazy import is allowed) and ``if TYPE_CHECKING:`` blocks.
    ``into_classes=False`` also leaves class bodies out, for questions about module names."""
    for node in body:
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.ClassDef) and not into_classes:
            continue
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            yield from _module_scope_statements(node.orelse, into_classes=into_classes)
            continue
        for field in ("body", "handlers", "cases", "orelse", "finalbody"):
            for child in getattr(node, field, None) or ():
                nested = (
                    child.body if isinstance(child, ast.ExceptHandler | ast.match_case) else [child]
                )
                yield from _module_scope_statements(nested, into_classes=into_classes)


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


def _is_main_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    operands = (test.left, *test.comparators)
    return any(
        isinstance(operand, ast.Name) and operand.id == "__name__" for operand in operands
    ) and any(
        isinstance(operand, ast.Constant) and operand.value == "__main__" for operand in operands
    )


def _main_guards(source: str) -> int:
    return sum(
        1
        for node in _module_scope_statements(ast.parse(source).body, into_classes=False)
        if isinstance(node, ast.If) and _is_main_guard(node.test)
    )


def _defines_main(source: str) -> bool:
    """A module-scope `def main` or `main = …`: the shape of a process entry. A method or a
    nested function named `main` is not a module name, so it does not count."""
    for node in _module_scope_statements(ast.parse(source).body, into_classes=False):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "main":
            return True
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "main" for target in targets):
            return True
    return False


def _entry_modules_named_by(handler: Callable[..., int]) -> frozenset[str]:
    """The modules whose `main` a verb's handler is, or imports when called: the dispatcher's
    `contract` wrapper keeps that import lazy, so identity alone cannot see it."""
    if handler.__name__ == "main":
        return frozenset({handler.__module__})
    package = handler.__module__.rpartition(".")[0]
    return frozenset(
        importlib.util.resolve_name(f"{'.' * node.level}{node.module}", package)
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(handler))))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and any(alias.name == "main" for alias in node.names)
    )


def _unnamed_mains(sources: Mapping[str, str], handlers: Iterable[Callable[..., int]]) -> set[str]:
    """Modules that define a `main` no verb handler is or imports. The dispatcher's own `main`
    is the process entry itself."""
    named = {"janus.cli.dispatch"}.union(
        *(_entry_modules_named_by(handler) for handler in handlers)
    )
    return {module for module, source in sources.items() if _defines_main(source)} - named


def _resolve_entry_point(target: str) -> object:
    module_name, _, attribute = target.partition(":")
    resolved: object = importlib.import_module(module_name.strip())
    for part in attribute.strip().split("."):
        resolved = getattr(resolved, part)
    return resolved


def _reads_argv(operand: ast.expr) -> bool:
    """`argv`, `resolved_argv`, `sys.argv`, or any index or slice of them."""
    while isinstance(operand, ast.Subscript):
        operand = operand.value
    return (isinstance(operand, ast.Name) and "argv" in operand.id) or (
        isinstance(operand, ast.Attribute) and operand.attr == "argv"
    )


def _argv_verb_dispatch(source: str) -> list[str]:
    """String literals compared with argv: `argv[0] == "x"`, `sys.argv[1] == "x"`, and
    `"x" in sys.argv` all decide a verb from the command line."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        literals = [
            operand.value
            for operand in operands
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str)
        ]
        if literals and any(_reads_argv(operand) for operand in operands):
            found.extend(literals)
    return found


def _lazy_contract_handler(argv: Sequence[str]) -> int:
    # Detector fixture, never called: the shape of the dispatcher's `contract` wrapper.
    from janus.cli.contract import main as contract_main

    return contract_main(argv)


# ---------------------------------------------------------------------------------------
# 1. Coverage — first, because every sweep below is vacuous without it


def test_the_sweep_actually_covers_the_cli_package() -> None:
    """A glob that matched nothing would pass every assertion below it. The walk is recursive
    and includes `__init__`, like the sibling package sweeps."""
    modules = _cli_modules()

    assert CLI_ROOT.is_dir() and MAIN_MODULE.is_file()
    assert modules.keys() >= EXPECTED_CLI_MODULES, sorted(EXPECTED_CLI_MODULES - modules.keys())
    assert _swept_files().keys() >= {f"cli/{name}" for name in EXPECTED_CLI_MODULES} | {"main"}


# ---------------------------------------------------------------------------------------
# 2. One registration path


def test_every_verb_is_registered_in_the_dispatcher() -> None:
    """`verbs()` is the whole surface. Set equality, so a missing verb and an accidental extra
    one both fail; `contract` is order-18's verb, registered by TASK-03 (TD-7)."""
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
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    defined = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]

    assert main.main is dispatch.main
    assert not [name for name in imported if name.split(".")[0] == "argparse"]
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
    """Only `janus.main` (the console-script target) and, if it wants one, the dispatcher may
    run as `__main__`. A verb module with its own guard, or a `cli/__main__.py` that makes
    `python -m janus.cli` work, is a second way in."""
    guarded = {
        name
        for name, path in _cli_modules().items()
        if name != "dispatch" and _main_guards(path.read_text(encoding="utf-8"))
    }

    assert _main_guards(MAIN_MODULE.read_text(encoding="utf-8")) == 1, (
        "the detector must see the one guard that exists"
    )
    assert guarded == set()
    assert [name for name in _cli_modules() if name.rpartition("/")[2] == "__main__"] == []


def test_a_cli_module_defines_main_only_if_a_verb_names_it() -> None:
    """What stops verb number seven from being wired with an `if` in `main.py` again: a `main`
    under `janus/cli` that no `Verb` handler is, or imports, is an entry the dispatcher cannot
    see."""
    from janus.cli.dispatch import verbs

    sources = {
        _module_name(name): path.read_text(encoding="utf-8")
        for name, path in _cli_modules().items()
    }
    defining = {module for module, source in sources.items() if _defines_main(source)}

    assert defining >= EXPECTED_MAIN_MODULES, "the detector must find the mains that exist"
    assert _unnamed_mains(sources, [verb.handler for verb in verbs()]) == set()


def test_every_console_script_resolves_to_the_dispatcher() -> None:
    """A packaging-level second entry (`janus-validate = "janus.cli.validate:…"`) would bypass
    the verb table as surely as an `if` in `main.py`."""
    from janus.cli.dispatch import main as dispatch_main

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    entry_points = project.get("entry-points", {})
    scripts = {
        **project.get("scripts", {}),
        **project.get("gui-scripts", {}),
        **entry_points.get("console_scripts", {}),
        **entry_points.get("gui_scripts", {}),
    }

    assert scripts, "pyproject.toml declares no console script: the sweep compared nothing"
    assert [
        name
        for name, target in scripts.items()
        if _resolve_entry_point(target) is not dispatch_main
    ] == []


def test_the_dispatch_detector_flags_a_deliberate_violation() -> None:
    violating = (
        "def main(argv):\n"
        "    if argv and argv[0] == 'validate':\n"
        "        return 0\n"
        "    if 'list' == resolved_argv[0]:\n"
        "        return 0\n"
        "if sys.argv[1:] and sys.argv[1:][0] == 'checkpoint':\n"
        "    pass\n"
        "if 'dead-letters' in sys.argv:\n"
        "    pass\n"
    )

    assert sorted(_argv_verb_dispatch(violating)) == [
        "checkpoint",
        "dead-letters",
        "list",
        "validate",
    ]


def test_the_dispatch_detector_does_not_flag_ordinary_comparisons() -> None:
    clean = (
        "def render(args, parser):\n"
        "    if args.format == 'json':\n"
        "        return 1\n"
        "    parser.add_parser('validate')\n"
        "    if argv is None or args.argv_file == 'x':\n"
        "        return 2\n"
        "    return len(argv) == 0\n"
    )

    assert _argv_verb_dispatch(clean) == []


def test_the_entry_point_detectors_flag_a_deliberate_violation() -> None:
    from janus.cli import run_all

    planted = {
        "janus.cli.dispatch": "def main(argv=None):\n    return 0\n",
        "janus.cli.run_all": "def main(argv=None):\n    return 0\n",
        "janus.cli.contract": "def main(argv=None):\n    return 0\n",
        "janus.cli.planted": "def main(argv):\n    return 0\n",
        "janus.cli.aliased": "import functools\nmain = functools.partial(run_command, ())\n",
        "janus.cli.guarded": (
            "try:\n    import yaml\nexcept ImportError:\n    async def main():\n        pass\n"
        ),
    }

    guards = (
        "if __name__ == '__main__':\n    raise SystemExit(main())\n"
        "try:\n    pass\nfinally:\n    if '__main__' == __name__:\n        main()\n"
    )

    assert _main_guards(guards) == 2
    assert _entry_modules_named_by(run_all.main) == {"janus.cli.run_all"}
    assert _entry_modules_named_by(_lazy_contract_handler) == {"janus.cli.contract"}
    assert _unnamed_mains(planted, [run_all.main, _lazy_contract_handler]) == {
        "janus.cli.planted",
        "janus.cli.aliased",
        "janus.cli.guarded",
    }


def test_the_entry_point_detectors_do_not_flag_clean_code() -> None:
    from janus.cli import run_all, validate

    clean = {
        "janus.cli.dispatch": "def main(argv=None):\n    return 0\n",
        "janus.cli.run_all": "def main(argv=None):\n    return 0\n",
        "janus.cli.contract": "def main(argv=None):\n    return 0\n",
        "janus.cli.menu": (
            "def main_menu():\n    pass\nclass Verb:\n    def main(self):\n        pass\n"
        ),
        "janus.cli.factory": "def build():\n    def main():\n        pass\n    return main\n",
    }

    assert _main_guards("def main():\n    if __name__ == '__main__':\n        pass\n") == 0
    assert _main_guards("if __name__ == 'janus.cli':\n    pass\n") == 0
    assert _entry_modules_named_by(validate.validate_command) == frozenset()
    assert _unnamed_mains(clean, [run_all.main, _lazy_contract_handler]) == set()


# ---------------------------------------------------------------------------------------
# 3. No engine at module scope (green guard over today's modules, and every new one)


def test_no_cli_module_imports_an_engine_at_module_scope() -> None:
    swept = _swept_files()
    violations = {
        name: found
        for name, path in swept.items()
        if (found := _module_scope_engine_imports(path.read_text(encoding="utf-8")))
    }

    assert swept.keys() >= {f"cli/{name}" for name in EXPECTED_CLI_MODULES} | {"main"}, (
        "the sweep must find the package before it can clear it"
    )
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
        "with suppress(ImportError):\n"
        "    import pyarrow.parquet\n"
        "if sys.version_info >= (3, 13):\n"
        "    import pyiceberg\n"
    )

    assert _module_scope_engine_imports(violating) == [
        "pyspark",
        "pyiceberg.catalog",
        "pyarrow",
        "dagster",
        "pyspark.sql",
        "pyarrow.parquet",
        "pyiceberg",
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
        "class Reader:\n"
        "    def read(self):\n"
        "        import pyarrow\n"
    )

    assert _module_scope_engine_imports(clean) == []


# ---------------------------------------------------------------------------------------
# 4. Session-free verbs (NFR-2): `dead-letters replay --execute` and `--verify-checksums`
#    are absent on purpose — they inherit the executor's lifecycle unchanged.

SESSION_FREE_FORMS = (
    pytest.param(("validate",), id="validate"),
    pytest.param(("list", "--graph"), id="list"),
    pytest.param(
        ("dead-letters", "list", "--source-id", CLEAN_PRODUCER),
        id="dead-letters list",
    ),
    pytest.param(
        ("dead-letters", "release", "--source-id", CLEAN_PRODUCER, "--all", "--reason", "sweep"),
        id="dead-letters release",
    ),
    pytest.param(
        ("checkpoint", "show", "--source-id", CLEAN_PRODUCER),
        id="checkpoint show",
    ),
    pytest.param(
        ("checkpoint", "set", "--source-id", CLEAN_PRODUCER, "--to", "2026-09-01T00:00:00Z",
         "--reason", "sweep"),
        id="checkpoint set",
    ),
    pytest.param(
        ("checkpoint", "clear", "--source-id", CLEAN_PRODUCER, "--reason", "sweep"),
        id="checkpoint clear",
    ),
)


def test_the_tripwire_trips_wherever_a_session_can_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof below is only as strong as its tripwire. Each module that imported the
    builder by name holds its own binding, and `cli/run.py` builds the `--with-spark` session
    with its copy, so a tripwire that patched only the defining module would miss it."""
    import janus.cli.run as run
    import janus.utils.environment as environment
    from janus.runtime.spark_lifecycle import SparkSessionProvider

    importlib.import_module("janus.utils.spark")
    builder = environment.build_spark_session
    bound = _builder_bindings(builder)

    arm_spark_tripwire(monkeypatch)

    assert bound >= {
        "janus.cli.run",
        "janus.runtime.spark_lifecycle",
        "janus.utils.environment",
        "janus.utils.spark",
    }
    assert _builder_bindings(builder) == frozenset()
    with pytest.raises(pytest.fail.Exception, match="session-free verb"):
        SparkSessionProvider({}, {}).get()
    with pytest.raises(pytest.fail.Exception, match="session-free verb"):
        run.build_spark_session({}, {})


@pytest.mark.parametrize("argv", SESSION_FREE_FORMS)
def test_the_verb_never_acquires_a_spark_session(
    argv: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order-06 spy, armed wherever a session can be built, and an engine-module check.

    The check is in process, so it can speak only for engines the process had not already
    loaded. In the fast job none is installed, so nothing is loaded before the call and the
    check reads `"pyspark" not in sys.modules` after it: the strongest form. In the container
    an earlier test may have imported pyspark, and then the tripwire is the proof for it.
    """
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
    loaded_before = _loaded_engines()

    verb, *rest = argv
    result = run_janus((verb, *rest[:1], "--project-root", str(root), *rest[1:]))

    assert result.exit_code == 0, result.output
    newly_loaded = _loaded_engines() - loaded_before
    assert newly_loaded == frozenset(), f"{' '.join(argv)} loaded {sorted(newly_loaded)}"


# ---------------------------------------------------------------------------------------
# 5. The two-second ceiling (AC-9) on the checked-in registry


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(("validate",), id="validate"),
        pytest.param(("validate", "--environment", "local"), id="validate --environment"),
        pytest.param(("list",), id="list"),
        pytest.param(
            ("dead-letters", "list", "--source-id", "ibge_pib_brasil"),
            id="dead-letters list",
        ),
        pytest.param(
            ("checkpoint", "show", "--source-id", "ibge_pib_brasil"),
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
    if "--environment" in argv:
        # Every verb parses --environment through the parent parser; a form that times a
        # profile has to have read one, or it timed the registry half twice.
        profile = argv[argv.index("--environment") + 1]
        assert f"conf/environments/{profile}.yaml" in result.stdout
    assert elapsed < CLI_TIME_BUDGET_SECONDS, f"{' '.join(argv)} took {elapsed:.2f}s"
