"""Package-level boundaries for the orchestration core and optional adapters."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import janus

PACKAGE_ROOT = Path(inspect.getfile(janus)).parent
ADAPTER_ROOT = PACKAGE_ROOT / "adapters" / "dagster"

CORE_PACKAGES = (
    "models",
    "registry",
    "planner",
    "orchestration",
    "strategies",
    "runtime",
    "cli",
)
CORE_ANCHORS = {
    "cli/run_all.py",
    "main.py",
    "models/source_config.py",
    "orchestration/selection.py",
    "planner/core.py",
    "registry/loader.py",
    "runtime/executor.py",
    "strategies/api/core.py",
}
CORE_FORBIDDEN_IMPORTS = ("airflow", "dagster", "janus.adapters")

ADAPTER_ANCHORS = {
    "collector.py",
    "definitions.py",
    "events.py",
    "manifest.py",
    "runtime.py",
}
ADAPTER_IMPLEMENTATION_IMPORTS = (
    "aiohttp",
    "http.client",
    "httpx",
    "janus.checkpoints",
    "janus.normalizers",
    "janus.quality",
    "janus.readers",
    "janus.runtime.materialize",
    "janus.strategies.api",
    "janus.strategies.catalog",
    "janus.strategies.files",
    "janus.strategies.http",
    "janus.writers",
    "pyiceberg",
    "pyspark",
    "requests",
    "socket",
    "urllib.request",
)

PURE_GRAPH_ANCHORS = {
    "models/dependencies.py",
    "orchestration/planning.py",
    "orchestration/selection.py",
    "registry/dependencies.py",
    "registry/loader.py",
}
PURE_GRAPH_FORBIDDEN_IMPORTS = (
    "airflow",
    "dagster",
    "http.client",
    "pyiceberg",
    "pyspark",
    "requests",
    "socket",
    "sqlalchemy",
    "sqlite3",
    "urllib.request",
    "janus.adapters",
    "janus.checkpoints",
    "janus.normalizers",
    "janus.quality",
    "janus.readers",
    "janus.runtime",
    "janus.strategies",
    "janus.utils.catalog_options",
    "janus.utils.catalog_properties",
    "janus.utils.spark",
    "janus.writers",
)


def _core_modules() -> tuple[Path, ...]:
    modules = [PACKAGE_ROOT / "main.py"]
    for package in CORE_PACKAGES:
        modules.extend((PACKAGE_ROOT / package).rglob("*.py"))
    return tuple(sorted(set(modules)))


def _pure_graph_modules() -> tuple[Path, ...]:
    modules = [PACKAGE_ROOT / "models" / "dependencies.py"]
    modules.extend((PACKAGE_ROOT / "registry").rglob("*.py"))
    modules.extend((PACKAGE_ROOT / "orchestration").rglob("*.py"))
    return tuple(sorted(set(modules)))


def _import_targets(source: str) -> tuple[tuple[int, str], ...]:
    targets: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative_module = f"{'.' * node.level}{node.module or ''}"
            if node.module:
                targets.append((node.lineno, relative_module))
            targets.extend(
                (
                    node.lineno,
                    f"{relative_module}{'.' if node.module else ''}{alias.name}",
                )
                for alias in node.names
            )
    return tuple(targets)


def _matches_root(module: str, root: str) -> bool:
    if module == root or module.startswith(f"{root}."):
        return True
    if not module.startswith("."):
        return False
    relative = module.lstrip(".")
    package_relative_root = root.removeprefix("janus.")
    return relative == package_relative_root or relative.startswith(
        f"{package_relative_root}."
    )


def _forbidden_imports(source: str, roots: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"line {line}: {module}"
        for line, module in _import_targets(source)
        if any(_matches_root(module, root) for root in roots)
    )


def _package_violations(
    modules: tuple[Path, ...], roots: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    return {
        str(module.relative_to(PACKAGE_ROOT)): violations
        for module in modules
        if (
            violations := _forbidden_imports(
                module.read_text(encoding="utf-8"),
                roots,
            )
        )
    }


def test_core_packages_never_import_an_orchestration_adapter() -> None:
    modules = _core_modules()
    matched = {str(module.relative_to(PACKAGE_ROOT)) for module in modules}

    assert matched >= CORE_ANCHORS, (
        "the core import sweep missed an intended package: "
        f"{sorted(CORE_ANCHORS - matched)}"
    )
    assert not _package_violations(modules, CORE_FORBIDDEN_IMPORTS), (
        "core code imported an optional orchestrator or the adapter layer: "
        f"{_package_violations(modules, CORE_FORBIDDEN_IMPORTS)}"
    )


def test_dagster_adapter_does_not_reach_ingestion_implementations_directly() -> None:
    modules = tuple(sorted(ADAPTER_ROOT.rglob("*.py")))
    matched = {module.name for module in modules}

    assert matched >= ADAPTER_ANCHORS, (
        "the adapter import sweep missed an intended module: "
        f"{sorted(ADAPTER_ANCHORS - matched)}"
    )
    assert not _package_violations(modules, ADAPTER_IMPLEMENTATION_IMPORTS), (
        "the Dagster adapter bypassed the shared execution service: "
        f"{_package_violations(modules, ADAPTER_IMPLEMENTATION_IMPORTS)}"
    )


def test_graph_and_selection_modules_cannot_reach_compute_or_catalog_clients() -> None:
    modules = _pure_graph_modules()
    matched = {str(module.relative_to(PACKAGE_ROOT)) for module in modules}

    assert matched >= PURE_GRAPH_ANCHORS, (
        "the pure-module sweep missed an intended package: "
        f"{sorted(PURE_GRAPH_ANCHORS - matched)}"
    )
    assert not _package_violations(modules, PURE_GRAPH_FORBIDDEN_IMPORTS), (
        "pure graph or selection code imported compute, catalog, or execution code: "
        f"{_package_violations(modules, PURE_GRAPH_FORBIDDEN_IMPORTS)}"
    )


@pytest.mark.parametrize(
    ("source", "roots"),
    [
        ("import dagster", CORE_FORBIDDEN_IMPORTS),
        ("from airflow import DAG", CORE_FORBIDDEN_IMPORTS),
        ("from janus import adapters", CORE_FORBIDDEN_IMPORTS),
        ("from .. import adapters", CORE_FORBIDDEN_IMPORTS),
        (
            "from janus.strategies.http.transport import UrllibApiTransport",
            ADAPTER_IMPLEMENTATION_IMPORTS,
        ),
        ("from urllib.request import urlopen", ADAPTER_IMPLEMENTATION_IMPORTS),
        ("from pyspark.sql import SparkSession", PURE_GRAPH_FORBIDDEN_IMPORTS),
        ("from ...strategies.http import transport", PURE_GRAPH_FORBIDDEN_IMPORTS),
    ],
)
def test_import_detector_rejects_deliberate_boundary_violations(
    source: str,
    roots: tuple[str, ...],
) -> None:
    assert _forbidden_imports(source, roots), f"detector missed a violation in {source!r}"


def test_import_detector_accepts_clean_downward_dependencies() -> None:
    clean_core = "from janus.models import SourceConfig\nfrom pathlib import Path"
    clean_adapter = (
        "from janus.orchestration import BatchPlanner\n"
        "from janus.runtime import SourceExecutor"
    )

    assert _forbidden_imports(clean_core, CORE_FORBIDDEN_IMPORTS) == ()
    assert _forbidden_imports(clean_adapter, ADAPTER_IMPLEMENTATION_IMPORTS) == ()
