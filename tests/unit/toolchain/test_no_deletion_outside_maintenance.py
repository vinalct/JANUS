"""Whole-package deletion boundary, with independently tested AST detectors."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import janus
from tests.support.retention_baseline import DeletionCalls

PACKAGE_ROOT = Path(inspect.getfile(janus)).parent
PROCEDURE_NAMES = frozenset(
    {"expire_snapshots", "remove_orphan_files", "rewrite_data_files", "rewrite_manifests"}
)
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
}


def _procedures(source: str) -> frozenset[str]:
    tree = ast.parse(source)
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
        if isinstance(node, ast.Name):
            found.update({node.id} & PROCEDURE_NAMES)
        elif isinstance(node, ast.Attribute):
            found.update({node.attr} & PROCEDURE_NAMES)
        elif isinstance(node, ast.alias):
            found.update({node.name.rsplit(".", 1)[-1]} & PROCEDURE_NAMES)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and re.search(r"\bCALL\b|system\.", node.value, re.I)
        ):
            found.update(name for name in PROCEDURE_NAMES if name in node.value.lower())
    return frozenset(found)


def _deletions(source: str) -> tuple[tuple[str, str], ...]:
    visitor = DeletionCalls()
    visitor.visit(ast.parse(source))
    found = []
    for call in visitor.calls:
        parsed = ast.parse(str(call["call"]), mode="eval").body
        assert isinstance(parsed, ast.Call)
        func = parsed.func
        if isinstance(func, ast.Attribute):
            operation = func.attr
        else:
            assert isinstance(func, ast.Name)
            operation = visitor.aliases[func.id].rsplit(".", 1)[-1]
        found.append((str(call["scope"]), operation))
    return tuple(found)


def _modules() -> dict[str, str]:
    return {
        path.relative_to(PACKAGE_ROOT).as_posix(): path.read_text(encoding="utf-8")
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
    } <= modules.keys()


def test_procedures_are_confined_to_maintenance_package():
    violations = {
        path: _procedures(source)
        for path, source in _modules().items()
        if not path.startswith("maintenance/") and _procedures(source)
    }
    assert violations == {}


def test_deletions_are_confined_to_maintenance_or_exact_existing_lifecycle_calls():
    violations = [
        (path, scope, operation)
        for path, source in _modules().items()
        if not path.startswith("maintenance/")
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
    assert all(reason.strip() for reason in ALLOWANCES.values())
    assert set(ALLOWANCES) <= observed, "stale deletion allowance"
    assert len(ALLOWANCES) == 7


@pytest.mark.xfail(strict=True, reason="maintenance procedure sweep is empty")
def test_maintenance_procedure_sweep_is_not_vacuous():
    modules = {
        path: source for path, source in _modules().items() if path.startswith("maintenance/")
    }
    assert modules, "maintenance package contains no modules"
    found = set().union(*(_procedures(source) for source in modules.values()))
    assert {"expire_snapshots", "remove_orphan_files", "rewrite_data_files"} <= found
    assert any(_deletions(source) for source in modules.values())


@pytest.mark.parametrize("name", sorted(PROCEDURE_NAMES))
@pytest.mark.parametrize(
    "template",
    [
        "{name}(table)",
        "actions.{name}(table)",
        "spark.sql(\"CALL janus.system.{name}(table => 't')\")",
        'spark.sql(f"CALL {{catalog}}.system.{name}()").collect()',
        "from engine import {name} as hidden",
    ],
)
def test_procedure_detector_positive(name, template):
    assert _procedures(template.format(name=name)) == {name}


@pytest.mark.parametrize(
    "source",
    [
        "# CALL janus.system.expire_snapshots()\npass",
        '"""CALL janus.system.expire_snapshots() is described here."""\npass',
        'def write():\n    """See maintain for expire_snapshots."""\n    return 1',
        'spark.sql("SELECT * FROM table.snapshots")',
        "path.replace(final_path)",
    ],
)
def test_procedure_detector_negative(source):
    assert _procedures(source) == frozenset()


@pytest.mark.parametrize(
    "source",
    [
        "def sweep(path):\n    path.unlink()",
        "def sweep(path):\n    os.remove(path)",
        "def sweep(path):\n    shutil.rmtree(path)",
        "def sweep(path):\n    path.rmdir()",
        "from os import unlink as drop\ndef sweep(path):\n    drop(path)",
        "from shutil import rmtree as drop\ndef sweep(path):\n    drop(path)",
    ],
)
def test_deletion_detector_positive(source):
    assert len(_deletions(source)) == 1
    assert _deletions(source)[0][0] == "sweep"


@pytest.mark.parametrize(
    "source",
    ["path.replace(final)", "path.exists()", "# path.unlink()\npass", 'text = "path.unlink()"'],
)
def test_deletion_detector_negative(source):
    assert _deletions(source) == ()


def test_allowance_does_not_authorize_other_scopes_in_same_module():
    found = _deletions("class DeadLetterStore:\n    def sweep(self, path):\n        path.unlink()")
    assert found == (("DeadLetterStore.sweep", "unlink"),)
    assert ("checkpoints/dead_letters.py", *found[0]) not in ALLOWANCES
