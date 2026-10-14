"""D-16: every file module composes the shared HTTP mechanics exactly once."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import janus.strategies.files as files_package

PACKAGE_ROOT = Path(inspect.getfile(files_package)).parent


def _api_retry_tokens():
    # The API test imports its sibling conftest by bare name, so importing it during
    # isolated files-test collection fails. Read its declared tuple without executing
    # that module; the API test remains the single source and stays untouched.
    path = Path(__file__).parents[1] / "api/test_extraction_boundary.py"
    tree = ast.parse(path.read_text("utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BANNED_RETRY_TOKENS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, "API retry-token declaration missing or ambiguous"
    return ast.literal_eval(assignments[0].value)


BANNED_RETRY_TOKENS = _api_retry_tokens()


def _package_sources():
    return {path.name: path.read_text("utf-8") for path in sorted(PACKAGE_ROOT.glob("*.py"))}


def _qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_qualified_name(node.value)}.{node.attr}"
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
            for name in node.names:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"
    return aliases


def _resolve(node, aliases):
    name = _qualified_name(node)
    first, _, rest = name.partition(".")
    return aliases.get(first, first) + (f".{rest}" if rest else "")


def _enclosing_scopes(node, parents):
    scopes, parent = [], parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef | ast.FunctionDef):
            scopes.append(parent.name)
        parent = parents.get(parent)
    return tuple(reversed(scopes))


def _rule_violations(sources):
    violations = {
        rule: [] for rule in ("retry", "throttle", "sync", "threading", "order", "transport")
    }
    throttle_sites = []
    for filename, source in sources.items():
        tree = ast.parse(source)
        aliases = _aliases(tree)
        violations["retry"].extend(
            (filename, token) for token in BANNED_RETRY_TOKENS if token in source
        )
        threaded = any(
            name == "threading"
            or name.startswith("threading.")
            or name == "concurrent.futures"
            or name.startswith("concurrent.futures.")
            for name in aliases.values()
        )
        if threaded:
            violations["threading"].append(filename)
        if filename in {"extraction.py", "download_window.py"}:
            violations["transport"].extend(
                (filename, token)
                for token in ("ApiTransport", "ApiClient", "urllib")
                if token in source
            )
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = _resolve(node.func, aliases)
                leaf = called.rsplit(".", 1)[-1]
                if leaf == "HttpRequestThrottle":
                    throttle_sites.append((filename, _enclosing_scopes(node, parents)))
                if (
                    leaf in {"Lock", "RLock", "Semaphore", "Condition", "Event"}
                    and filename != "download_window.py"
                ):
                    violations["sync"].append((filename, called))
            if isinstance(node, ast.Name | ast.Attribute):
                name = _resolve(node, aliases)
                if (
                    name.rsplit(".", 1)[-1] in {"as_completed", "add_done_callback"}
                    or name == "concurrent.futures.wait"
                ):
                    violations["order"].append((filename, name))
        for imported in aliases.values():
            if imported in {"concurrent.futures.as_completed", "concurrent.futures.wait"}:
                violations["order"].append((filename, imported))
    if throttle_sites != [("extraction.py", ("FileExtractionContext", "build"))]:
        violations["throttle"] = throttle_sites or ["no shared throttle construction"]
    return violations


def test_sweep_covers_every_module_including_the_download_window():
    sources = _package_sources()
    assert len(sources) >= 10
    assert {"extraction.py", "download.py", "download_loop.py", "download_window.py"} <= set(
        sources
    )


@pytest.mark.parametrize("rule", ["retry", "throttle", "sync", "order"])
def test_existing_package_boundaries_remain_green(rule):
    assert len(_package_sources()) >= 10
    assert not _rule_violations(_package_sources())[rule]


def test_only_the_download_window_imports_threading_mechanics():
    assert _rule_violations(_package_sources())["threading"] == ["download_window.py"]


def test_existing_extraction_orchestrator_has_no_transport_symbols():
    assert not _rule_violations(_package_sources())["transport"]


def test_download_window_has_no_transport_symbols():
    sources = _package_sources()
    assert "download_window.py" in sources
    assert not _rule_violations(sources)["transport"]


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
        ("sync", "download.py", "from threading import Lock as L\nlock = L()"),
        ("sync", "download.py", "import threading as t\nlock = t.RLock()"),
        ("sync", "download.py", "gate = Semaphore()"),
        ("sync", "download.py", "gate = Condition()"),
        ("sync", "download.py", "gate = Event()"),
        ("threading", "download.py", "import threading"),
        ("threading", "download.py", "from concurrent.futures import Future"),
        ("order", "download.py", "from concurrent.futures import wait as await_all"),
        ("order", "download.py", "import concurrent.futures as futures\nfutures.wait(pending)"),
        ("order", "download.py", "for future in as_completed(pending): pass"),
        ("order", "download.py", "future.add_done_callback(callback)"),
        ("transport", "extraction.py", "from janus.strategies.http import ApiClient"),
        ("transport", "download_window.py", "from janus.strategies.http import ApiTransport"),
        ("transport", "download_window.py", "import urllib.request"),
    ],
)
def test_detector_rejects_a_deliberate_violation(rule, filename, violation):
    sources = dict(_CLEAN)
    sources[filename] = sources.get(filename, "") + "\n" + violation
    detected = _rule_violations(sources)[rule]
    if rule == "threading":
        assert set(detected) == {"download.py", "download_window.py"}
    else:
        assert detected


@pytest.mark.parametrize("rule", ["retry", "throttle", "sync", "threading", "order", "transport"])
def test_detector_accepts_clean_composition(rule):
    detected = _rule_violations(_CLEAN)[rule]
    assert detected == (["download_window.py"] if rule == "threading" else [])
