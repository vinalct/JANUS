"""Forward/backward lock coverage and one-inode locking, over the entire package."""

import ast
import inspect
import textwrap
from importlib import import_module

import pytest

from tests.support.parallel_execution import REPO_ROOT
from tests.unit.toolchain.test_no_threaded_source_execution import names, qualified

PACKAGE = REPO_ROOT / "src/janus"
PRIMITIVES = {
    "start_run",
    "record_success",
    "record_failure",
    "reset",
    "clear_state",
    "execute_retention",
    "save_between_inputs",
    "release",
}
HELPERS = {"hold_source_state_lock", "hold_operator_source_lock", "source_lock"}
needs_lock_sites = pytest.mark.xfail(
    strict=True,
    reason="locked entry-point registry and acquisition sites absent",
)


def scope_calls(node):
    calls = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Call(self, node):
            calls.append(node)
            self.generic_visit(node)

    visitor = Visitor()
    for statement in node.body:
        visitor.visit(statement)
    return calls


def acquisition_violation(source, reached=()):
    tree = ast.parse(textwrap.dedent(source))
    aliases = names(tree)
    targets = PRIMITIVES | {name.rsplit(".", 1)[-1] for name in reached}
    violations = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        calls = scope_calls(function)
        mutations = [
            call for call in calls if qualified(call.func, aliases).split(".")[-1] in targets
        ]
        contexts = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and qualified(item.context_expr.func, aliases).split(".")[-1] in HELPERS
                for item in node.items
            )
        ]
        entered = [
            call
            for call in calls
            if qualified(call.func, aliases).endswith(".enter_context")
            and call.args
            and isinstance(call.args[0], ast.Call)
            and qualified(call.args[0].func, aliases).split(".")[-1] in HELPERS
        ]
        guarded = {
            id(call)
            for context in contexts
            for statement in context.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        }
        if not (contexts or entered) or any(
            id(call) not in guarded and not any(entry.lineno < call.lineno for entry in entered)
            for call in mutations
        ):
            violations.append(function.name)
    return violations


def unregistered_mutations(source, registered):
    tree = ast.parse(textwrap.dedent(source))
    aliases = names(tree)
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name not in registered
        and any(
            qualified(call.func, aliases).split(".")[-1] in PRIMITIVES for call in scope_calls(node)
        )
    ]


def second_lock(source):
    tree = ast.parse(textwrap.dedent(source))
    aliases = names(tree)
    calls = [qualified(node.func, aliases) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return (
        any(name in {"fcntl.flock", "fcntl.lockf"} for name in calls)
        or any(
            value == "os.O_EXCL" or value.startswith(("fcntl.F_SETLK", "fcntl.F_OFD_"))
            for value in aliases.values()
        )
        or any(
            (isinstance(node, ast.Attribute) and node.attr.startswith(("F_SETLK", "F_OFD_")))
            or (isinstance(node, ast.Attribute) and node.attr == "O_EXCL")
            or (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.endswith((".lock", ".pid"))
            )
            for node in ast.walk(tree)
        )
    )


def replaces_lock(source):
    tree = ast.parse(textwrap.dedent(source))
    aliases = names(tree)
    return any(
        qualified(node.func, aliases) in {"os.replace", "os.rename", "os.remove", "os.unlink"}
        or qualified(node.func, aliases).split(".")[-1] in {"unlink", "write_json_atomic"}
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"replace", "rename"}
            and qualified(node.func, aliases) != "dataclasses.replace"
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def registry():
    from janus.runtime.locking import LOCKED_ENTRY_POINTS, REACHED_THROUGH

    assert len(LOCKED_ENTRY_POINTS) >= 7
    assert PRIMITIVES
    return LOCKED_ENTRY_POINTS, REACHED_THROUGH


def resolve_function(reference):
    module_name, object_path = reference.split(":")
    value = import_module(module_name)
    for part in object_path.split("."):
        value = getattr(value, part)
    assert inspect.isfunction(value), reference
    return value


@needs_lock_sites
def test_registered_entry_points_resolve_and_acquire_before_mutation():
    entries, reached = registry()
    assert entries
    for entry in entries:
        function = resolve_function(entry.qualified_name)
        assert not acquisition_violation(inspect.getsource(function), reached), entry.qualified_name
    for reference, (target, reason) in reached.items():
        assert reason.strip() and target in {entry.qualified_name for entry in entries}
        assert resolve_function(reference) and resolve_function(target)
        function = ast.parse(textwrap.dedent(inspect.getsource(resolve_function(reference)))).body[
            0
        ]
        assert any(
            qualified(call.func, {}).split(".")[-1] in PRIMITIVES
            or qualified(call.func, {}).split(".")[-1]
            in {name.rsplit(".", 1)[-1] for name in reached}
            for call in scope_calls(function)
        ), f"stale REACHED_THROUGH entry: {reference}"


def functions_with_scopes(tree, prefix=""):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield from functions_with_scopes(node, f"{prefix}{node.name}.")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield f"{prefix}{node.name}", node


@needs_lock_sites
def test_package_primitive_callers_are_registered_or_reached_through():
    entries, reached = registry()
    allowed = {entry.qualified_name for entry in entries} | reached.keys()
    paths = sorted(PACKAGE.rglob("*.py"))
    assert paths and PRIMITIVES
    defined = {
        node.name
        for path in paths
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert defined and defined >= PRIMITIVES
    sites = []
    for path in paths:
        module = "janus." + ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
        tree = ast.parse(path.read_text())
        aliases = names(tree)
        for scope, node in functions_with_scopes(tree):
            for call in scope_calls(node):
                if qualified(call.func, aliases).split(".")[-1] in PRIMITIVES:
                    sites.append((f"{module}:{scope}", call.lineno))
    assert len(sites) >= 15
    assert [(name, line) for name, line in sites if name not in allowed] == []


@needs_lock_sites
def test_exactly_one_lock_implementation_and_no_replaced_inode():
    registry()
    paths = sorted(PACKAGE.rglob("*.py"))
    assert paths
    implementations = [
        path.relative_to(PACKAGE).as_posix() for path in paths if second_lock(path.read_text())
    ]
    assert implementations and implementations == ["runtime/locking.py"]
    assert replaces_lock((PACKAGE / "runtime/locking.py").read_text()) is False


@needs_lock_sites
def test_read_only_operator_entry_points_do_not_acquire_locks():
    registry()
    references = (
        "janus.cli.checkpoint:_show",
        "janus.cli.dead_letters:_list",
        "janus.cli.dead_letters:_replay",
    )
    assert references
    for reference in references:
        source = textwrap.dedent(inspect.getsource(resolve_function(reference)))
        tree = ast.parse(source)
        assert not any(
            qualified(call.func, names(tree)).split(".")[-1] in HELPERS
            for call in scope_calls(tree.body[0])
        )


@needs_lock_sites
@pytest.mark.parametrize(
    "detector,bad,clean",
    [
        (
            acquisition_violation,
            "def f():\n observer.start_run(plan)\n with hold_source_state_lock(): pass",
            "def f():\n with hold_source_state_lock(): observer.start_run(plan)",
        ),
        (
            acquisition_violation,
            "def f():\n store.reset(plan)",
            "def f():\n stack.enter_context(source_lock(lock, source_id))\n store.reset(plan)",
        ),
        (
            acquisition_violation,
            "def f():\n hold_source_state_lock()\n store.reset(plan)",
            "def f():\n with hold_source_state_lock(): store.reset(plan)",
        ),
        (
            acquisition_violation,
            "def f():\n with hold_source_state_lock(): pass\n store.reset(plan)",
            "def f():\n with hold_source_state_lock(): store.reset(plan)",
        ),
        (second_lock, "import fcntl as f\nf.flock(fd, 2)", "hold_source_state_lock(plan, layout)"),
        (
            second_lock,
            "from fcntl import lockf as lock\nlock(fd, 2)",
            "hold_source_state_lock(plan, layout)",
        ),
        (second_lock, "from fcntl import F_SETLKW as FLAG\nfcntl.fcntl(fd, FLAG)", "os.fsync(fd)"),
        (second_lock, "path = base / 'worker.pid'", "path = base / 'current.json'"),
        (
            second_lock,
            "import os\nos.open(path, os.O_EXCL)",
            "import os\nos.open(path, os.O_CREAT)",
        ),
        (second_lock, "path = base / '.lock'", "path = base / 'current.json'"),
        (replaces_lock, "os.replace(tmp, lock_path)", "os.pwrite(fd, record, 0)"),
        (replaces_lock, "lock_path.unlink()", "os.ftruncate(fd, 0)"),
        (
            replaces_lock,
            "os.rename(tmp, lock_path)",
            "from dataclasses import replace\nreplace(holder, pid=2)",
        ),
        (replaces_lock, "write_json_atomic(lock_path, record)", "os.pwrite(fd, record, 0)"),
    ],
)
def test_detector_flags_violation_and_accepts_clean_form(detector, bad, clean):
    assert detector(bad)
    assert not detector(clean)
    registry()


@needs_lock_sites
def test_backward_detector_flags_new_caller_and_accepts_registered_caller():
    snippet = "def release_source():\n DeadLetterStore().release(plan)"
    assert unregistered_mutations(snippet, set()) == ["release_source"]
    assert unregistered_mutations(snippet, {"release_source"}) == []
    registry()


@pytest.mark.parametrize("profile", ["local", "cluster"])
def test_default_lock_path_agrees_for_every_checked_in_source(profile, monkeypatch):
    import os

    from janus.lineage import MetadataZonePaths
    from janus.models import ExecutionPlan, RunContext
    from janus.runtime.locking import source_state_lock_path
    from janus.utils.environment import load_environment_config
    from janus.utils.storage import StorageLayout
    from tests.support.parallel_baseline import STARTED_AT, _sources

    for key in tuple(os.environ):
        if key.startswith("JANUS_"):
            monkeypatch.delenv(key)
    sources = _sources(REPO_ROOT)
    assert len(sources) >= 31
    layout = StorageLayout.from_environment_config(
        load_environment_config(profile, REPO_ROOT), REPO_ROOT
    )
    for source in sources:
        context = RunContext.create("paths", profile, REPO_ROOT, started_at=STARTED_AT)
        plan = ExecutionPlan.from_source_config(source, context)
        assert (
            source_state_lock_path(plan, layout).parent
            == MetadataZonePaths.from_plan(plan).base_dir
        )
