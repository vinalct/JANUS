"""Executable architecture boundaries for queryable observability."""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

import janus
import janus.lineage as lineage_package
import janus.observability as observability_package

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(inspect.getfile(janus)).parent
OBSERVABILITY_ROOT = Path(inspect.getfile(observability_package)).parent
LINEAGE_ROOT = Path(inspect.getfile(lineage_package)).parent
WORKFLOW_ROOT_CANDIDATES = (
    PROJECT_ROOT / ".github" / "workflows",
    Path.cwd() / ".github" / "workflows",
)

OBSERVABILITY_ANCHORS = {
    "__init__.py",
    "emission.py",
    "iceberg_sink.py",
    "openlineage/facets.py",
    "openlineage/sink.py",
    "openlineage/transport.py",
    "records.py",
    "runs_table.py",
}
LINEAGE_ANCHORS = {"__init__.py", "models.py", "persistence.py", "store.py"}
ENGINE_ROOTS = ("pyspark", "pyiceberg", "pyarrow")

SPARK_IMPORT_ROOTS = ("pyspark",)
SPARK_SYMBOLS = frozenset(
    {
        "SparkContext",
        "SparkSession",
        "SparkSessionProvider",
        "build_iceberg_session",
        "build_spark_session",
        "getOrCreate",
    }
)
EMISSION_ENTRYPOINTS = frozenset(
    {"append_run_record", "emit", "emit_failed", "emit_started", "emit_succeeded", "send"}
)

JSON_WRITE_ALLOWANCES = {
    "openlineage/transport.py": (
        "The file transport appends OpenLineage events as NDJSON under its configured "
        "metadata-zone path. It does not write authoritative run, lineage, quality, or "
        "checkpoint JSON and may never call write_json_atomic."
    ),
}

CATALOG_PROPERTY_MODULE = "janus.utils.catalog_properties"
ALLOWED_CATALOG_PROPERTY_IMPORTS = frozenset(
    {
        "derive_pyiceberg_catalog_name",
        "derive_pyiceberg_catalog_properties",
    }
)
BANNED_CATALOG_HELPERS = frozenset(
    {"build_spark_options", "required_catalog_value", "resolve_catalog_type"}
)
CATALOG_CONNECTION_PREFIXES = (
    "jdbc:",
    "postgresql:",
    "s3://",
    "sqlite:",
    "JANUS_ICEBERG_CATALOG_",
)
CATALOG_CONNECTION_KEYS = frozenset(
    {
        "access-key-id",
        "credential",
        "credentials",
        "jdbc.password",
        "jdbc.user",
        "secret-access-key",
        "uri",
    }
)

REQUIRED_CI_JOBS = frozenset({"fast", "adapter", "spark"})
LINEAGE_ENDPOINT_ENV_KEYS = frozenset(
    {"JANUS_OPENLINEAGE_ENDPOINT", "JANUS_OPENLINEAGE_URL"}
)
LINEAGE_TRANSPORT_ENV_KEY = "JANUS_OPENLINEAGE_TRANSPORT"
ENDPOINT_ASSIGNMENT = re.compile(
    r"\bJANUS_OPENLINEAGE_(?:ENDPOINT|URL)\s*="
)
HTTP_TRANSPORT_ASSIGNMENT = re.compile(
    r"\bJANUS_OPENLINEAGE_TRANSPORT\s*=\s*['\"]?http\b",
    re.IGNORECASE,
)


def _workflow_root() -> Path:
    roots = tuple(dict.fromkeys(WORKFLOW_ROOT_CANDIDATES))
    for root in roots:
        if (root / "ci.yml").is_file():
            return root
    return roots[0]


def _modules(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
    }


def _assert_surface(modules: Mapping[str, str], anchors: set[str], package: str) -> None:
    assert modules, f"the {package} sweep matched no Python modules"
    assert anchors <= set(modules), (
        f"the {package} sweep missed intended modules: {sorted(anchors - set(modules))}"
    )


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


def _import_targets(source: str) -> tuple[tuple[int, str], ...]:
    targets: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                targets.append((node.lineno, module))
            targets.extend(
                (
                    node.lineno,
                    f"{module}.{alias.name}" if module else alias.name,
                )
                for alias in node.names
            )
    return tuple(targets)


def _matches_root(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def _spark_references(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    findings = {
        f"line {line}: import {target}"
        for line, target in _import_targets(source)
        if any(_matches_root(target, root) for root in SPARK_IMPORT_ROOTS)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in SPARK_SYMBOLS:
            findings.add(f"line {node.lineno}: symbol {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in SPARK_SYMBOLS:
            findings.add(f"line {node.lineno}: symbol {node.attr}")
        elif isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] in SPARK_SYMBOLS:
            findings.add(f"line {node.lineno}: symbol {node.name}")
    return tuple(sorted(findings))


class _RaiseVisitor(ast.NodeVisitor):
    def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.root = root
        self.guarded = 0
        self.findings: list[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        broad_handler = any(_catches_emission_exception(handler) for handler in node.handlers)
        if broad_handler:
            self.guarded += 1
        for statement in node.body:
            self.visit(statement)
        if broad_handler:
            self.guarded -= 1
        for statement in (*node.handlers, *node.orelse, *node.finalbody):
            self.visit(statement)

    def visit_Raise(self, node: ast.Raise) -> None:
        if self.guarded == 0:
            self.findings.append(node.lineno)


def _catches_emission_exception(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    name = _qualified_name(handler.type)
    return name in {"BaseException", "Exception", "builtins.BaseException", "builtins.Exception"}


def _entrypoint_raises(source: str) -> tuple[str, ...]:
    findings: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name not in EMISSION_ENTRYPOINTS:
            continue
        visitor = _RaiseVisitor(node)
        visitor.visit(node)
        findings.extend(f"{node.name}:line {line}" for line in visitor.findings)
    return tuple(sorted(findings))


def _entrypoint_names(source: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in EMISSION_ENTRYPOINTS
    }


def _imports_root(source: str, root: str) -> tuple[str, ...]:
    return tuple(
        f"line {line}: {target}"
        for line, target in _import_targets(source)
        if _matches_root(target, root)
    )


def _call_names(source: str) -> tuple[tuple[int, str], ...]:
    calls: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            calls.append((node.lineno, _qualified_name(node.func) or "<dynamic>"))
    return tuple(calls)


def _json_write_findings(source: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (line, name)
        for line, name in _call_names(source)
        if name.rsplit(".", 1)[-1] == "write_json_atomic" or name == "os.write"
    )


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _catalog_import_findings(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, ast.ImportFrom):
        return ()
    findings: set[str] = set()
    if node.module == CATALOG_PROPERTY_MODULE:
        findings.update(
            f"line {node.lineno}: catalog import {alias.name}"
            for alias in node.names
            if (
                alias.name not in ALLOWED_CATALOG_PROPERTY_IMPORTS
                and not alias.name.endswith("CatalogUnrepresentableError")
            )
        )
    findings.update(
        f"line {node.lineno}: catalog helper {alias.name}"
        for alias in node.names
        if alias.name in BANNED_CATALOG_HELPERS
    )
    return tuple(sorted(findings))


def _catalog_node_findings(node: ast.AST) -> tuple[str, ...]:
    imported = _catalog_import_findings(node)
    if imported:
        return imported

    if isinstance(node, ast.Call) and node.args:
        name = _qualified_name(node.func) or ""
        key = _string_constant(node.args[0])
        if name.endswith(".get") and key in {"spark", "iceberg"}:
            return (f"line {node.lineno}: direct profile read {key!r}",)

    if isinstance(node, ast.Subscript):
        key = _string_constant(node.slice)
        if key in {"spark", "iceberg"}:
            return (f"line {node.lineno}: direct profile read {key!r}",)

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
        if value in CATALOG_CONNECTION_KEYS or value.startswith(CATALOG_CONNECTION_PREFIXES):
            return (f"line {node.lineno}: catalog connection literal {value!r}",)
    return ()


def _catalog_boundary_findings(source: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            finding
            for node in ast.walk(ast.parse(source))
            for finding in _catalog_node_findings(node)
        )
    )


def _workflow_lineage_findings(value: Any, path: str = "workflow") -> tuple[str, ...]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            current = f"{path}.{key_text}"
            normalized = key_text.upper()
            if normalized in LINEAGE_ENDPOINT_ENV_KEYS:
                findings.append(f"{current}: endpoint environment configured")
            if (
                normalized == LINEAGE_TRANSPORT_ENV_KEY
                and isinstance(nested, str)
                and nested.strip().lower() == "http"
            ):
                findings.append(f"{current}: HTTP transport configured")
            findings.extend(_workflow_lineage_findings(nested, current))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            findings.extend(_workflow_lineage_findings(nested, f"{path}[{index}]"))
    elif isinstance(value, str):
        if ENDPOINT_ASSIGNMENT.search(value):
            findings.append(f"{path}: endpoint assignment in command")
        if HTTP_TRANSPORT_ASSIGNMENT.search(value):
            findings.append(f"{path}: HTTP transport assignment in command")
    return tuple(findings)


def test_no_observability_module_names_a_spark_symbol() -> None:
    modules = _modules(OBSERVABILITY_ROOT)
    _assert_surface(modules, OBSERVABILITY_ANCHORS, "janus.observability")
    violations = {
        module: findings
        for module, source in modules.items()
        if (findings := _spark_references(source))
    }
    assert not violations, (
        "observability must never acquire or name Spark compute; append through the "
        f"engine-neutral sink instead: {violations}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "from pyspark.sql import SparkSession\n",
        "from janus.runtime import SparkSessionProvider\n",
        "def emit():\n    return getOrCreate()\n",
    ],
)
def test_spark_detector_rejects_deliberate_violations(source: str) -> None:
    assert _spark_references(source)


def test_spark_detector_accepts_profile_data_and_engine_neutral_code() -> None:
    source = (
        "def read(config):\n"
        "    spark = config.get('spark', {})\n"
        "    return append_run_record(spark)\n"
    )
    assert _spark_references(source) == ()


def test_emission_entrypoints_contain_no_unguarded_raise() -> None:
    modules = _modules(OBSERVABILITY_ROOT)
    _assert_surface(modules, OBSERVABILITY_ANCHORS, "janus.observability")
    entrypoints = set().union(*(_entrypoint_names(source) for source in modules.values()))
    assert {"append_run_record", "emit", "send"} <= entrypoints, (
        f"the raise sweep found only {sorted(entrypoints)}; it no longer covers emission"
    )
    violations = {
        module: findings
        for module, source in modules.items()
        if (findings := _entrypoint_raises(source))
    }
    assert not violations, (
        "an emission entry point contains an explicit unguarded raise; turn ordinary "
        f"failure into an emission result: {violations}"
    )


def test_raise_detector_rejects_a_deliberate_entrypoint_raise() -> None:
    assert _entrypoint_raises("def emit(event):\n    raise RuntimeError('boom')\n")


def test_raise_detector_accepts_a_locally_guarded_raise_and_non_entrypoint_validation() -> None:
    clean = (
        "def emit(event):\n"
        "    try:\n"
        "        raise RuntimeError('boom')\n"
        "    except Exception:\n"
        "        return None\n"
        "\n"
        "def validate(event):\n"
        "    raise ValueError('bad profile')\n"
    )
    assert _entrypoint_raises(clean) == ()


def test_lineage_never_imports_observability_and_the_graph_stays_one_way() -> None:
    lineage_modules = _modules(LINEAGE_ROOT)
    observability_modules = _modules(OBSERVABILITY_ROOT)
    _assert_surface(lineage_modules, LINEAGE_ANCHORS, "janus.lineage")
    _assert_surface(observability_modules, OBSERVABILITY_ANCHORS, "janus.observability")

    reverse_edges = {
        module: imports
        for module, source in lineage_modules.items()
        if (imports := _imports_root(source, "janus.observability"))
    }
    forward_edges = {
        module: imports
        for module, source in observability_modules.items()
        if (imports := _imports_root(source, "janus.lineage"))
    }
    assert forward_edges, "observability no longer imports lineage; the graph check is vacuous"
    assert not reverse_edges, (
        "the package graph must remain runtime -> observability -> lineage; "
        f"lineage imported observability at {reverse_edges}"
    )


def test_import_detector_rejects_a_reverse_edge_and_accepts_the_forward_edge() -> None:
    assert _imports_root("from janus import observability\n", "janus.observability")
    assert _imports_root(
        "from janus.observability import RunRecord\n", "janus.observability"
    )
    assert _imports_root(
        "from janus.lineage.models import RunMetadata\n", "janus.observability"
    ) == ()


@pytest.mark.parametrize("module", ["janus.lineage", "janus.observability"])
def test_public_package_imports_load_no_compute_or_catalog_engine(module: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PACKAGE_ROOT.parent), env.get("PYTHONPATH")) if part
    )
    program = (
        f"import {module}, json, sys\n"
        f"roots = {ENGINE_ROOTS!r}\n"
        "loaded = sorted(name for name in sys.modules "
        "if name.split('.', 1)[0] in roots)\n"
        "print(json.dumps(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_observability_never_writes_authoritative_json() -> None:
    modules = _modules(OBSERVABILITY_ROOT)
    _assert_surface(modules, OBSERVABILITY_ANCHORS, "janus.observability")
    writes = {
        module: _json_write_findings(source)
        for module, source in modules.items()
        if _json_write_findings(source)
    }
    authoritative = {
        module: findings
        for module, findings in writes.items()
        if any(name.rsplit(".", 1)[-1] == "write_json_atomic" for _, name in findings)
    }
    raw_writers = {
        module: findings
        for module, findings in writes.items()
        if any(name == "os.write" for _, name in findings)
    }

    assert not authoritative, (
        "metadata-zone JSON remains owned by lineage, quality, and checkpoints: "
        f"{authoritative}"
    )
    assert set(raw_writers) == set(JSON_WRITE_ALLOWANCES), (
        "only the named OpenLineage NDJSON transport may perform a raw append; "
        f"found {raw_writers}"
    )
    assert all(reason.strip() for reason in JSON_WRITE_ALLOWANCES.values())


def test_json_write_detector_rejects_authoritative_json_and_accepts_serialization() -> None:
    violation = "def emit(path, payload):\n    write_json_atomic(path, payload)\n"
    clean = "def encode(payload):\n    return json.dumps(payload)\n"
    assert _json_write_findings(violation)
    assert _json_write_findings(clean) == ()


def test_observability_uses_only_the_shared_catalog_derivations() -> None:
    modules = _modules(OBSERVABILITY_ROOT)
    _assert_surface(modules, OBSERVABILITY_ANCHORS, "janus.observability")
    violations = {
        module: findings
        for module, source in modules.items()
        if (findings := _catalog_boundary_findings(source))
    }
    calls = {
        name.rsplit(".", 1)[-1]
        for source in modules.values()
        for _, name in _call_names(source)
    }
    expected = {
        "derive_pyiceberg_catalog_name",
        "derive_pyiceberg_catalog_properties",
    }
    assert expected <= calls, (
        f"the catalog sweep found no use of shared derivations: {sorted(expected - calls)}"
    )
    assert not violations, (
        "observability re-derived catalog connection material instead of using "
        f"catalog_properties: {violations}"
    )



def test_execution_preflight_uses_the_shared_catalog_derivations() -> None:
    path = PACKAGE_ROOT / "runtime" / "contract_preflight.py"
    assert path.is_file(), "catalog sweep missed the execution preflight"
    source = path.read_text(encoding="utf-8")
    assert _catalog_boundary_findings(source) == ()
    calls = {name.rsplit(".", 1)[-1] for _, name in _call_names(source)}
    assert {"derive_pyiceberg_catalog_name", "derive_pyiceberg_catalog_properties"} <= calls


def test_catalog_detector_rejects_a_second_derivation_and_accepts_shared_helpers() -> None:
    violation = (
        "def connect(config):\n"
        "    iceberg = config.get('spark', {}).get('iceberg', {})\n"
        "    return {'uri': 'jdbc:sqlite:data/catalog.sqlite'}\n"
    )
    clean = (
        "from janus.utils.catalog_properties import (\n"
        "    derive_pyiceberg_catalog_name,\n"
        "    derive_pyiceberg_catalog_properties,\n"
        ")\n"
        "def connect(config, paths):\n"
        "    return (derive_pyiceberg_catalog_name(config), "
        "derive_pyiceberg_catalog_properties(config, paths))\n"
    )
    assert _catalog_boundary_findings(violation)
    assert _catalog_boundary_findings(clean) == ()


def test_required_ci_jobs_do_not_configure_http_openlineage() -> None:
    workflow_root = _workflow_root()
    workflows = sorted(workflow_root.glob("*.yml")) + sorted(workflow_root.glob("*.yaml"))
    assert workflows, "the workflow sweep matched no YAML files"
    ci_path = workflow_root / "ci.yml"
    assert ci_path in workflows, "the workflow sweep did not find .github/workflows/ci.yml"

    document = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert isinstance(document, Mapping)
    jobs = document.get("jobs")
    assert isinstance(jobs, Mapping)
    assert set(jobs) >= REQUIRED_CI_JOBS, (
        f"the workflow sweep missed required jobs: {sorted(REQUIRED_CI_JOBS - set(jobs))}"
    )
    required_surface = {
        "global_env": document.get("env", {}),
        "jobs": {name: jobs[name] for name in REQUIRED_CI_JOBS},
    }
    assert not _workflow_lineage_findings(required_surface), (
        "a required CI job configures an HTTP OpenLineage destination: "
        f"{_workflow_lineage_findings(required_surface)}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "jobs:\n  fast:\n    env:\n      JANUS_OPENLINEAGE_URL: https://lineage.invalid\n",
        "jobs:\n  spark:\n    env:\n      JANUS_OPENLINEAGE_TRANSPORT: http\n",
        "jobs:\n  fast:\n    steps:\n      - run: JANUS_OPENLINEAGE_ENDPOINT=/api pytest\n",
    ],
)
def test_workflow_detector_rejects_deliberate_http_configuration(source: str) -> None:
    assert _workflow_lineage_findings(yaml.safe_load(source))


def test_workflow_detector_accepts_hermetic_lineage_configuration() -> None:
    source = (
        "jobs:\n"
        "  fast:\n"
        "    env:\n"
        "      JANUS_OPENLINEAGE_TRANSPORT: file\n"
        "      JANUS_OPENLINEAGE_EVENTS_DIR: data/lineage\n"
    )
    assert _workflow_lineage_findings(yaml.safe_load(source)) == ()
