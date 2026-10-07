"""Whole-package deletion boundary, including the write-path surface."""

from __future__ import annotations

import ast
import inspect
import re
from collections import Counter
from pathlib import Path

import pytest

import janus

PACKAGE_ROOT = Path(inspect.getfile(janus)).parent
PROCEDURE_NAMES = frozenset(
    {"expire_snapshots", "remove_orphan_files", "rewrite_data_files", "rewrite_manifests"}
)
DELETION_CALLS = frozenset({"unlink", "remove", "rmtree", "rmdir"})
# TASK-09 explicitly reserves this procedure; it has no executable call to allow.
RESERVED_PROCEDURES = {
    "rewrite_manifests": "Reserved by the boundary, documented but not implemented."
}
PROCEDURE_ALLOWANCES = {
    ("maintenance/execute.py", name): (
        "The declared-policy executor is the only module authorized to invoke this procedure."
    )
    for name in PROCEDURE_NAMES - RESERVED_PROCEDURES.keys()
}
POLICY_IDENTIFIER_ALLOWANCES = {
    ("maintenance/settings.py", "remove_orphan_files"): (
        "The policy's boolean field controls opt-in orphan removal; it invokes no procedure."
    ),
    ("maintenance/planning.py", "remove_orphan_files"): (
        "The pure planner reads that boolean field when proposing an orphan-removal item."
    ),
}
ALLOWANCES = {
    (
        "checkpoints/dead_letters.py",
        "DeadLetterStore.clear",
        "unlink",
    ): "Clears this source's resolved dead-letter state after the last entry is released.",
    (
        "checkpoints/progress.py",
        "ExtractionProgressStore.clear",
        "unlink",
    ): "Clears this source's resume progress after successful extraction.",
    (
        "checkpoints/store.py",
        "CheckpointStore.clear_state",
        "unlink",
    ): "Explicit operator reset saves history before removing current checkpoint state.",
    (
        "orchestration/persistence.py",
        "PipelineSummaryStore.persist",
        "unlink",
    ): "Removes only its own temporary file after failed atomic summary persistence.",
    (
        "writers/raw.py",
        "RawArtifactWriter.write_stream",
        "unlink",
    ): "Removes its own partial raw stream after a failed write.",
    (
        "writers/raw.py",
        "StagedWrite._remove_staged_path",
        "unlink",
    ): "Discards only the writer's owned staged partial file.",
    (
        "writers/raw.py",
        "_remove_empty_staging_dir",
        "rmdir",
    ): "Removes only an empty staging directory; committed raw files are untouched.",
    (
        "maintenance/execute.py",
        "execute_metadata_item",
        "unlink",
    ): "Applies declared metadata/lineage retention with protected-state checks and evidence.",
    (
        "maintenance/execute.py",
        "delete_prefix",
        "rmtree",
    ): "Applies opt-in raw retention to one contained, unprotected run prefix with evidence.",
}
NON_DELETION_ALLOWANCES = {
    (
        "lineage/persistence.py",
        "write_json_atomic",
        "replace",
    ): "Atomic publication of the new JSON record; replace is not a retention deletion."
}


def _literal_parts(node: ast.AST) -> str:
    """Keep SQL context across literal concatenation and formatted-string pieces."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_literal_parts(value) for value in node.values)
    if isinstance(node, ast.FormattedValue):
        return _literal_parts(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_parts(node.left) + _literal_parts(node.right)
    return ""


def _procedures(tree: ast.Module, *, identifiers: bool = True) -> frozenset[str]:
    """Procedure identifiers and rendered SQL, excluding every scope's docstring."""
    docstrings = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
        ):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstrings.add(id(value))
    found = set()
    for node in ast.walk(tree):
        if identifiers and isinstance(node, ast.Name):
            found.update({node.id} & PROCEDURE_NAMES)
        elif identifiers and isinstance(node, ast.Attribute):
            found.update({node.attr} & PROCEDURE_NAMES)
        elif identifiers and isinstance(node, ast.alias):
            found.update({node.name.rsplit(".", 1)[-1]} & PROCEDURE_NAMES)
        elif (
            isinstance(node, ast.Constant | ast.JoinedStr | ast.BinOp)
            and id(node) not in docstrings
            and re.search(r"\bCALL\b|system\.", literal := _literal_parts(node), re.I)
        ):
            found.update(name for name in PROCEDURE_NAMES if name in literal.lower())
    return frozenset(found)


class _CallSites(ast.NodeVisitor):
    """Attribute calls and imported os/shutil aliases, with exact lexical scopes."""

    def __init__(self, operations: frozenset[str]) -> None:
        self.operations = operations
        self.scopes: list[str] = []
        self.aliases: dict[str, str] = {}
        self.found: list[tuple[str, str]] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"os", "shutil"}:
            for alias in node.names:
                if alias.name in self.operations:
                    self.aliases[alias.asname or alias.name] = alias.name

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope(node)

    def _scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        operation = None
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.operations:
            operation = node.func.attr
        elif isinstance(node.func, ast.Name):
            operation = self.aliases.get(node.func.id)
        if operation is not None:
            self.found.append((".".join(self.scopes), operation))
        self.generic_visit(node)


def _call_sites(tree: ast.Module, operations: frozenset[str]) -> tuple[tuple[str, str], ...]:
    visitor = _CallSites(operations)
    visitor.visit(tree)
    return tuple(visitor.found)


def _deletions(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    return _call_sites(tree, DELETION_CALLS)


def _modules() -> dict[str, ast.Module]:
    return {
        path.relative_to(PACKAGE_ROOT).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }


def test_package_sweep_covers_all_source_modules_and_anchors():
    modules = _modules()
    assert len(modules) >= 174
    assert {
        "main.py",
        "runtime/materialize.py",
        "writers/spark.py",
        "lineage/store.py",
        "observability/iceberg_sink.py",
        "observability/openlineage/transport.py",
        "maintenance/execute.py",
    } <= modules.keys()


def test_procedures_are_confined_to_the_declared_executor():
    observed = {(path, name) for path, tree in _modules().items() for name in _procedures(tree)}
    allowed = PROCEDURE_ALLOWANCES.keys() | POLICY_IDENTIFIER_ALLOWANCES.keys()
    assert observed == allowed, (
        f"unallowed procedures: {observed - allowed}; "
        f"stale procedure allowances: {allowed - observed}"
    )


def test_policy_identifier_allowances_never_authorize_procedure_calls():
    modules = _modules()
    for path, _name in POLICY_IDENTIFIER_ALLOWANCES:
        tree = modules[path]
        assert _procedures(tree, identifiers=False) == frozenset()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else (function.attr if isinstance(function, ast.Attribute) else "")
                )
                assert name not in PROCEDURE_NAMES
            elif isinstance(node, ast.alias):
                assert node.name.rsplit(".", 1)[-1] not in PROCEDURE_NAMES


def test_deletions_are_confined_to_maintenance_or_exact_existing_lifecycle_calls():
    violations = [
        (path, scope, operation)
        for path, source in _modules().items()
        for scope, operation in _deletions(source)
        if (path, scope, operation) not in ALLOWANCES
    ]
    assert violations == []


def test_allowances_have_reasons_and_are_not_stale():
    observed = {
        (path, scope, operation)
        for path, source in _modules().items()
        for scope, operation in _deletions(source)
    }
    for allowances in (
        ALLOWANCES,
        PROCEDURE_ALLOWANCES,
        POLICY_IDENTIFIER_ALLOWANCES,
        NON_DELETION_ALLOWANCES,
        RESERVED_PROCEDURES,
    ):
        assert all(reason.strip() for reason in allowances.values())
    assert set(ALLOWANCES) <= observed, "stale deletion allowance"
    assert len(ALLOWANCES) == 9


def test_allowlisted_deletion_scopes_do_not_gain_additional_calls():
    counts = Counter(
        (path, scope, operation)
        for path, tree in _modules().items()
        for scope, operation in _deletions(tree)
    )
    assert {site: counts[site] for site in ALLOWANCES} == dict.fromkeys(ALLOWANCES, 1)


def test_atomic_publication_allowance_is_live_and_is_not_a_deletion():
    modules = _modules()
    publications = {
        (path, scope, operation)
        for path, tree in modules.items()
        for scope, operation in _call_sites(tree, frozenset({"replace"}))
    }
    assert NON_DELETION_ALLOWANCES.keys() <= publications, "stale atomic publication allowance"
    assert "replace" not in DELETION_CALLS
    assert _deletions(modules["lineage/persistence.py"]) == ()


def test_recursive_removal_is_only_the_guarded_raw_prefix_executor():
    # FR-8 permits recursion only for one contained, unprotected run directory,
    # including its data/checksum companions. Other maintenance paths remove files.
    observed = {
        (path, scope, operation)
        for path, source in _modules().items()
        for scope, operation in _deletions(source)
        if operation == "rmtree"
    }
    assert observed == {("maintenance/execute.py", "delete_prefix", "rmtree")}


def test_maintenance_procedure_sweep_is_not_vacuous():
    tree = _modules()["maintenance/execute.py"]
    found = _procedures(tree)
    assert found == PROCEDURE_NAMES - RESERVED_PROCEDURES.keys()
    # A fourth executable procedure would be fiction: TASK-09 reserves it in prose.
    # Its positive meta-tests still prove the executable-code detector flags it.
    docstring = ast.get_docstring(tree) or ""
    assert all(name in docstring for name in RESERVED_PROCEDURES)
    assert found | RESERVED_PROCEDURES.keys() == PROCEDURE_NAMES


def test_deletion_sweep_is_not_vacuous():
    sites = {
        (path, scope, operation)
        for path, tree in _modules().items()
        for scope, operation in _deletions(tree)
    }
    assert len(sites) >= 9, "deletion detector missed the existing lifecycle/executor calls"
    assert ALLOWANCES.keys() <= sites


@pytest.mark.parametrize("name", sorted(PROCEDURE_NAMES))
@pytest.mark.parametrize(
    "template",
    [
        "{name}(table)",
        "actions.{name}(table)",
        "spark.sql(\"CALL janus.system.{name}(table => 't')\")",
        'spark.sql(f"CALL {{catalog}}.system.{name}()").collect()',
        "from engine import {name} as hidden",
        'spark.sql("CALL cat.system." + "{name}(table)")',
        "spark.sql(f\"CALL cat.system.{{'{name}'}}(table)\")",
    ],
)
def test_procedure_detector_positive(name, template):
    assert _procedures(ast.parse(template.format(name=name))) == {name}


@pytest.mark.parametrize(
    "source",
    [
        "# CALL janus.system.expire_snapshots()\npass",
        '"""CALL janus.system.expire_snapshots() is described here."""\npass',
        'def write():\n    """See maintain for expire_snapshots."""\n    return 1',
        'class Writer:\n    """CALL cat.system.rewrite_manifests() is documentation."""',
        'async def write():\n    """CALL cat.system.remove_orphan_files()."""\n    pass',
        '"""See janus maintain for expire_snapshots."""\n# CALL cat.system.rewrite_data_files()',
        'procedure = "expire_snapshots"',
        'spark.sql("SELECT * FROM table.snapshots")',
        "path.replace(final_path)",
    ],
)
def test_procedure_detector_negative(source):
    assert _procedures(ast.parse(source)) == frozenset()


@pytest.mark.parametrize(
    "source",
    [
        "def sweep(path):\n    path.unlink()",
        "def sweep(path):\n    Path(path).unlink()",
        "def sweep(path):\n    os.remove(path)",
        "def sweep(path):\n    os.unlink(path)",
        "def sweep(path):\n    shutil.rmtree(path)",
        "def sweep(path):\n    path.rmdir()",
        "from os import unlink as drop\ndef sweep(path):\n    drop(path)",
        "from shutil import rmtree as drop\ndef sweep(path):\n    drop(path)",
    ],
)
def test_deletion_detector_positive(source):
    found = _deletions(ast.parse(source))
    assert len(found) == 1
    assert found[0][0] == "sweep"
    assert found[0][1] in DELETION_CALLS


@pytest.mark.parametrize(
    "source",
    [
        "path.replace(final)",
        "shutil.copy(source, target)",
        "path.exists()",
        "# path.unlink()\npass",
        'text = "path.unlink()"',
    ],
)
def test_deletion_detector_negative(source):
    assert _deletions(ast.parse(source)) == ()


def test_allowance_does_not_authorize_other_scopes_in_same_module():
    found = _deletions(
        ast.parse("class DeadLetterStore:\n    def sweep(self, path):\n        path.unlink()")
    )
    assert found == (("DeadLetterStore.sweep", "unlink"),)
    assert ("checkpoints/dead_letters.py", *found[0]) not in ALLOWANCES


def test_allowance_does_not_authorize_another_maintenance_module():
    found = _deletions(ast.parse("def execute_metadata_item(path):\n    path.unlink()"))
    assert ("maintenance/inventory.py", *found[0]) not in ALLOWANCES


def test_deletion_detector_preserves_nested_async_scopes():
    tree = ast.parse("class Store:\n    async def clear(self):\n        self.path.unlink()")
    assert _deletions(tree) == (("Store.clear", "unlink"),)
