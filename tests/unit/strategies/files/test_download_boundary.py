"""D-16: every file module composes the shared HTTP mechanics exactly once."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import janus.strategies.files as files_package
from tests.support.http_boundary import BANNED_RETRY_TOKENS

PACKAGE_ROOT = Path(inspect.getfile(files_package)).parent
ANCHORS = {
    "core.py",
    "extraction.py",
    "download.py",
    "download_loop.py",
    "download_window.py",
    "archives.py",
    "resolvers.py",
}
SYNC_PRIMITIVES = {
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Condition",
    "Event",
    "Barrier",
}
# The ordered window currently needs no synchronization primitive. Any future
# allowance must identify its primitive and explain why ordering alone is insufficient.
SYNC_ALLOWANCES: dict[tuple[str, str], str] = {}
THREADING_MODULES = {"concurrent.futures", "threading", "queue", "multiprocessing"}
ORCHESTRATION_MODULES = {"extraction.py", "concurrent_extraction.py", "download_window.py"}


def _package_sources():
    return {path.name: path.read_text("utf-8") for path in sorted(PACKAGE_ROOT.glob("*.py"))}


def _qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_qualified_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _qualified_name(node.func)
    return ""


def _aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for name in node.names:
                aliases[name.asname or name.name] = f"{module}.{name.name}"
    return aliases


def _resolve(node, aliases):
    name = _qualified_name(node)
    first, _, rest = name.partition(".")
    return aliases.get(first, first) + (f".{rest}" if rest else "")


def _enclosing_scopes(node, parents):
    scopes, parent = [], parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            scopes.append(parent.name)
        parent = parents.get(parent)
    return tuple(reversed(scopes))


def _parsed_sources(sources):
    for filename, source in sources.items():
        tree = ast.parse(source)
        yield filename, tree, _aliases(tree)


def _retry_violations(sources):
    return [
        (filename, token)
        for filename, source in sources.items()
        for token in BANNED_RETRY_TOKENS
        if token in source
    ]


def _throttle_violations(sources):
    sites = []
    for filename, tree, aliases in _parsed_sources(sources):
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        sites.extend(
            (filename, _enclosing_scopes(node, parents))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _resolve(node.func, aliases).rsplit(".", 1)[-1] == "HttpRequestThrottle"
        )
    expected = [("extraction.py", ("FileExtractionContext", "build"))]
    return [] if sites == expected else sites or ["no shared throttle construction"]


def _sync_violations(sources):
    violations, observed = [], set()
    for filename, tree, aliases in _parsed_sources(sources):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            primitive = _resolve(node.func, aliases).rsplit(".", 1)[-1]
            if primitive not in SYNC_PRIMITIVES:
                continue
            site = (filename, primitive)
            observed.add(site)
            if filename != "download_window.py" or not SYNC_ALLOWANCES.get(site, "").strip():
                violations.append(site)
    violations.extend((site, "stale allowance") for site in SYNC_ALLOWANCES.keys() - observed)
    return violations


def _threading_violations(sources):
    observed, violations = set(), []
    for filename, tree, _aliases_map in _parsed_sources(sources):
        # Inspect import nodes as well as bindings: `import queue` must not be
        # hidden by a later import reusing its alias.
        imports = (
            [
                name.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for name in node.names
            ]
            + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            + [
                f"{node.module}.{name.name}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for name in node.names
            ]
        )
        for name in imports:
            if not any(
                name == module or name.startswith(f"{module}.") for module in THREADING_MODULES
            ):
                continue
            observed.add(filename)
            if name == "multiprocessing" or name.startswith("multiprocessing."):
                violations.append((filename, "multiprocessing is forbidden"))
    if observed != {"download_window.py"}:
        violations.append((sorted(observed), "expected only download_window.py"))
    return violations


def _order_violations(sources):
    violations = []
    for filename, tree, aliases in _parsed_sources(sources):
        names = [
            _resolve(node, aliases)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name | ast.Attribute)
        ] + list(aliases.values())
        violations.extend(
            (filename, name)
            for name in names
            if name.rsplit(".", 1)[-1] in {"as_completed", "add_done_callback"}
            or name == "concurrent.futures.wait"
        )
    return violations


def _transport_violations(sources):
    return [
        (filename, token)
        for filename, source in sources.items()
        if filename in ORCHESTRATION_MODULES
        for token in ("ApiTransport", "ApiClient", "urllib", "UrllibApiTransport")
        if token in source
    ]


DETECTORS = {
    "retry": _retry_violations,
    "throttle": _throttle_violations,
    "sync": _sync_violations,
    "threading": _threading_violations,
    "order": _order_violations,
    "transport": _transport_violations,
}


def test_sweep_covers_every_module_including_the_download_window():
    sources = _package_sources()
    assert len(sources) >= 10
    assert sources.keys() >= ANCHORS


@pytest.mark.parametrize("rule", DETECTORS)
def test_package_download_boundaries(rule):
    assert not DETECTORS[rule](_package_sources())


_CLEAN = {
    "extraction.py": """
class FileExtractionContext:
    @classmethod
    def build(cls):
        return HttpRequestThrottle(clock=clock, sleeper=sleeper)
""",
    "download_window.py": "from concurrent.futures import ThreadPoolExecutor\n",
}


@pytest.mark.parametrize(
    "rule,filename,violation",
    [
        ("retry", "download.py", "while attempt < 3: pass"),
        ("retry", "download.py", "RETRYABLE_STATUS_CODES = {503}"),
        ("retry", "download.py", "time.sleep(1)"),
        ("throttle", "download.py", "private = HttpRequestThrottle()"),
        ("throttle", "extraction.py", "other = HttpRequestThrottle()"),
        (
            "throttle",
            "download.py",
            "from janus.strategies.http import HttpRequestThrottle as T\nT()",
        ),
        *[
            ("sync", filename, f"from threading import {primitive} as P\nP()")
            for primitive in sorted(SYNC_PRIMITIVES)
            for filename in ("download.py", "download_window.py")
        ],
        ("sync", "download.py", "import threading as t\nt.RLock()"),
        *[
            ("threading", filename, f"import {module} as machinery")
            for module in sorted(THREADING_MODULES)
            for filename in ("download.py", "concurrent_extraction.py")
        ],
        ("threading", "download_window.py", "from multiprocessing import get_context"),
        ("threading", "download.py", "from concurrent import futures as f"),
        ("threading", "download.py", "from queue import Queue as Q"),
        ("threading", "download.py", "import queue as machinery\nimport os as machinery"),
        ("order", "download.py", "from concurrent.futures import wait as await_all"),
        ("order", "download.py", "import concurrent.futures as f\nf.wait(pending)"),
        ("order", "download.py", "for future in as_completed(pending): pass"),
        ("order", "download.py", "from concurrent.futures import as_completed as done"),
        ("order", "download.py", "future.add_done_callback(callback)"),
        *[
            ("transport", filename, f"from transport import {token}")
            for filename in sorted(ORCHESTRATION_MODULES)
            for token in ("ApiTransport", "ApiClient", "UrllibApiTransport")
        ],
        ("transport", "download_window.py", "import urllib.request"),
    ],
)
def test_detector_rejects_a_deliberate_violation(rule, filename, violation):
    sources = dict(_CLEAN)
    sources[filename] = sources.get(filename, "") + "\n" + violation
    assert DETECTORS[rule](sources), (rule, filename, violation)


@pytest.mark.parametrize("rule", DETECTORS)
def test_detector_accepts_clean_composition(rule):
    sources = _CLEAN | {
        "download.py": "from janus.strategies.http import ApiClient\n",
        "concurrent_extraction.py": "future.result()\n",
    }
    assert DETECTORS[rule](sources) == []


@pytest.mark.parametrize("sources", [{}, {"download_window.py": _CLEAN["download_window.py"]}])
def test_throttle_detector_rejects_missing_construction(sources):
    assert _throttle_violations(sources)


def test_throttle_detector_rejects_the_right_count_in_the_wrong_scope():
    assert _throttle_violations({"extraction.py": "def build(): return HttpRequestThrottle()"})


def test_threading_detector_rejects_an_empty_inventory():
    assert _threading_violations({})


def test_sync_allowances_require_a_reason_and_a_live_window_primitive(monkeypatch):
    site = ("download_window.py", "Lock")
    sources = {"download_window.py": "from threading import Lock\nLock()"}
    monkeypatch.setitem(SYNC_ALLOWANCES, site, " ")
    assert _sync_violations(sources)
    monkeypatch.setitem(SYNC_ALLOWANCES, site, "Protects a future window invariant")
    assert not _sync_violations(sources)
    assert _sync_violations({})  # Stale allowances do not silently accumulate.
    assert _sync_violations({"download.py": "Lock()"})
