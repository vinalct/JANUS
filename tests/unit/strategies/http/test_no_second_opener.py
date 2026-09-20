"""One opener, one redirect handler, one place."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import janus as janus_package
import janus.adapters as adapters_package
import janus.observability as observability_package
import janus.strategies as strategies_package

#: ``urllib.request`` names that open a URL, or that build the thing which does.
BANNED_CALLS = frozenset(
    {"build_opener", "urlopen", "urlretrieve", "install_opener", "OpenerDirector"}
)

BANNED_IMPORTS = BANNED_CALLS

REDIRECT_BASE = "HTTPRedirectHandler"


ALLOWED_SITE = "strategies/http/transport.py"
ALLOWED_SITE_REASON = (
    "The single shared transport. FR-1 builds the OpenerDirector here by hand so "
    "File/FTP/Data/Unknown handlers are never installed, and FR-3 installs JanusRedirectHandler "
    "here so credentials cannot follow a cross-origin Location. Every family and the "
    "OpenLineage HTTP transport inherit both by composition."
)

IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__"})

ALWAYS_SWEPT = (
    "strategies/http/transport.py",
    "strategies/files/resolvers.py",
    "strategies/api/requests.py",
    "strategies/catalog/requests.py",
    "observability/openlineage/transport.py",
    "adapters/dagster/runtime.py",
)

MINIMUM_MODULES_SWEPT = 40


# ── the detector ─────────────────────────────────────────────────────────────


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def opener_violations(tree: ast.Module) -> list[ast.AST]:
    """Calls to / imports of the banned urllib.request names, and redirect-handler subclasses.

    Shared by the real sweep **and** by the meta-tests below: a meta-test that reimplemented
    the check would prove nothing about the check.
    """
    redirect_names = {REDIRECT_BASE, *_aliased_redirect_bases(tree)}
    return [
        node
        for node in ast.walk(tree)
        if _is_banned_call(node)
        or _is_banned_import(node)
        or _is_redirect_subclass(node, redirect_names)
    ]


def _is_banned_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _called_name(node) in BANNED_CALLS


def _is_banned_import(node: ast.AST) -> bool:
    """A banned name imported from ``urllib.request``, or the module imported whole."""
    if isinstance(node, ast.ImportFrom) and node.module == "urllib.request":
        return any(alias.name in BANNED_IMPORTS for alias in node.names)
    if isinstance(node, ast.Import):
        return any(alias.name == "urllib.request" for alias in node.names)
    return False


def _is_redirect_subclass(node: ast.AST, redirect_names: set[str]) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        _base_name(base) in redirect_names for base in node.bases
    )


def _aliased_redirect_bases(tree: ast.Module) -> set[str]:
    """Local names bound to ``HTTPRedirectHandler`` by an aliased import."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "urllib.request"
        for alias in node.names
        if alias.name == REDIRECT_BASE
    }


def redirect_subclasses(tree: ast.Module) -> list[ast.ClassDef]:
    """Classes deriving from the redirect handler, including through an import alias."""
    names = {REDIRECT_BASE, *_aliased_redirect_bases(tree)}
    return [
        node
        for node in ast.walk(tree)
        if _is_redirect_subclass(node, names)
    ]


# ── the sweep ────────────────────────────────────────────────────────────────


def _package_dir(package) -> Path:
    """A package's directory, whether or not it carries an ``__init__.py``.

    ``janus.strategies`` is a namespace package, so ``inspect.getfile`` raises on it;
    ``__path__`` answers for both kinds and keeps the sweep from skipping the one package
    it exists to read.
    """
    return Path(next(iter(package.__path__))).resolve()


def _http_speaking_sources() -> list[tuple[str, ast.Module]]:
    """Every module under the three packages that speak HTTP, parsed but never imported."""
    janus_root = _package_dir(janus_package)
    parsed: list[tuple[str, ast.Module]] = []
    for package in (strategies_package, observability_package, adapters_package):
        root = _package_dir(package)
        for path in sorted(root.rglob("*.py")):
            if not IGNORED_DIRECTORY_NAMES.isdisjoint(path.relative_to(root).parts):
                continue
            relative = path.relative_to(janus_root).as_posix()
            parsed.append((relative, ast.parse(path.read_text(encoding="utf-8"))))
    return parsed


def _render(relative: str, node: ast.AST) -> str:
    return f"{relative}:{getattr(node, 'lineno', '?')}: {ast.unparse(node).splitlines()[0]}"


def test_sweep_actually_covers_the_http_speaking_packages():
    """A glob that silently matches nothing would pass every assertion below it."""
    swept = {relative for relative, _ in _http_speaking_sources()}

    assert len(swept) >= MINIMUM_MODULES_SWEPT, (
        f"only {len(swept)} modules swept; the walk looks truncated: {sorted(swept)}"
    )
    for module in ALWAYS_SWEPT:
        assert module in swept, (
            f"the sweep never read {module}, which exists in every environment — the package "
            "roots are wrong and the containment assertion is vacuous"
        )


def test_only_the_shared_transport_builds_an_opener():
    """AC-8: no second opener, no second redirect handler, no direct ``urlopen``."""
    violations = [
        _render(relative, node)
        for relative, tree in _http_speaking_sources()
        for node in opener_violations(tree)
        if relative != ALLOWED_SITE
    ]

    assert not violations, (
        "a module outside the shared transport opens URLs or adopts CPython's redirect "
        "behaviour:\n"
        + "\n".join(f"  {line}" for line in violations)
        + f"\nCompose {ALLOWED_SITE} instead — {ALLOWED_SITE_REASON}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "red until: transport.py must construct the "
        "OpenerDirector and define JanusRedirectHandler for the allowance to be load-bearing"
    ),
)
def test_the_allowed_site_still_needs_its_allowance():
    """An allowance granted to a file that no longer needs it is fiction, not a decision.

    The same stale-allowance rule the catalog-containment sweep applies to its
    ``ALLOWED_SITES``: if ``transport.py`` ever stops constructing the opener and defining the
    redirect handler, the exemption comes out with that commit rather than quietly outliving
    it.
    """
    allowed = dict(_http_speaking_sources()).get(ALLOWED_SITE)

    assert allowed is not None, f"{ALLOWED_SITE} is missing; the allowance cannot be checked"
    constructions = [
        node
        for node in ast.walk(allowed)
        if isinstance(node, ast.Call) and _called_name(node) == "OpenerDirector"
    ]
    assert constructions, (
        f"{ALLOWED_SITE} constructs no OpenerDirector — the allowance is dead weight and the "
        "sweep should be tightened"
    )
    assert redirect_subclasses(allowed), (
        f"{ALLOWED_SITE} defines no {REDIRECT_BASE} subclass — FR-3's handler is either "
        "missing or has moved, and the allowance no longer describes this file"
    )


# ── detector meta-tests ──────────────────────────────────────────────────────

VIOLATING_SNIPPETS = (
    ("urlopen_import_and_call", "from urllib.request import urlopen\nurlopen(u)\n"),
    ("module_import_then_build", "import urllib.request\nurllib.request.build_opener()\n"),
    ("redirect_subclass", "class Mine(HTTPRedirectHandler):\n    pass\n"),
    (
        "aliased_redirect_subclass",
        "from urllib.request import HTTPRedirectHandler as H\n\n\nclass X(H):\n    ...\n",
    ),
    ("opener_director", "o = OpenerDirector()\n"),
    ("install_opener", "import urllib.request\nurllib.request.install_opener(o)\n"),
)

CLEAN_SNIPPETS = (
    ("urllib_parse", "from urllib.parse import urlsplit\n"),
    ("urllib_error", "from urllib.error import URLError\n"),
    ("composes_the_transport", "UrllibApiTransport().send(req)\n"),
    ("unrelated_class", "class RedirectHandlerPolicy:\n    max_hops = 5\n"),
    ("string_literal", 'NAME = "build_opener"\n'),
    ("comment", "# build_opener is what this sweep forbids\nvalue = 1\n"),
)


@pytest.mark.parametrize(
    ("label", "snippet"), VIOLATING_SNIPPETS, ids=[row[0] for row in VIOLATING_SNIPPETS]
)
def test_detector_flags_a_deliberate_violation(label, snippet):
    """The guardrail must fail on a violation, not merely pass on clean code."""
    del label
    assert opener_violations(ast.parse(snippet)), "the opener detector no longer detects anything"


@pytest.mark.parametrize(
    ("label", "snippet"), CLEAN_SNIPPETS, ids=[row[0] for row in CLEAN_SNIPPETS]
)
def test_detector_does_not_flag_clean_code(label, snippet):
    """Counterpart to the meta-test above: an always-failing detector attributes nothing."""
    del label
    found = opener_violations(ast.parse(snippet))
    assert not found, f"the detector flags clean code: {[ast.unparse(n) for n in found]}"
