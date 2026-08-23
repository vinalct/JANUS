"""AC-3/AC-4: the variant data is declared once, and one edit reaches both consumers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import janus
from janus.models.config.constants import SUPPORTED_STRATEGY_VARIANTS
from janus.models.config.strategy_registry import STRATEGY_REGISTRY, StrategyRegistry
from janus.models.source_config import SourceConfig, SourceConfigValidationError
from janus.planner import StrategyCatalog

REGISTRY_NAME = "SUPPORTED_STRATEGY_VARIANTS"

CONSTANTS_MODULE = "models/config/constants.py"

CONFIG_PATH = Path("conf/sources/example/single_source_registry.yaml")

MINIMUM_SWEPT_MODULES = 40

VARIANT_NAMES = frozenset(
    variant for variants in SUPPORTED_STRATEGY_VARIANTS.values() for variant in variants
)


KNOWN_LITERAL_SITES: dict[str, str] = {
    "strategies/api/core.py": (
        "plan-time compatibility checks: each api variant asserts the pagination type (and, "
        "for date_window_api, the extraction mode) its implementation requires. The registry "
        "says the variant exists; only the strategy knows what it needs."
    ),
    "strategies/catalog/document.py": (
        "ROOT_ENTITY_PRIORITY maps the two catalog variants to their entity-ordering, and "
        "falls back to metadata_catalog's. Per-variant traversal behaviour, keyed by variant "
        "because that is what it varies on."
    ),
    "strategies/files/core.py": (
        "archive_package is the one file variant requiring access.format='binary'; the check "
        "belongs with the code that unpacks the archive."
    ),
    "strategies/files/archives.py": (
        "archive_package branch — only that variant expands members out of a downloaded "
        "container."
    ),
    "strategies/files/discovery.py": (
        "static_file branches: a single fixed URL needs neither version discovery nor link "
        "resolution, which the other two file variants do."
    ),
    "strategies/files/download.py": (
        "static_file branch in the download path, for the same reason as discovery.py."
    ),
}

_PACKAGE_DIR = Path(janus.__file__).resolve().parent


@dataclass(frozen=True)
class LiteralFinding:
    """One hardcoded variant name, located precisely enough to act on."""

    module: str
    lineno: int
    variant: str

    def __str__(self) -> str:
        return f"{self.module}:{self.lineno}: {self.variant!r}"


def _swept_modules() -> dict[str, ast.Module]:
    """Every module in ``src/janus``, keyed by its path relative to the package root.

    Package-scoped by construction: a module added by a future split joins both sweeps
    without anyone remembering to add it.
    """
    return {
        path.relative_to(_PACKAGE_DIR).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE_DIR.rglob("*.py"))
    }


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    """Docstrings *describe* variants; they do not declare one. Excluded from the sweep."""
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, holders)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }


def _assigns_the_registry(tree: ast.Module) -> bool:
    """Whether this module *binds* the registry name, as opposed to reading it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == REGISTRY_NAME for target in targets
        ):
            return True
    return False


def _registry_reference_kinds(tree: ast.Module) -> set[str]:
    """How a module mentions the registry name: importing, reading, or naming it in a string."""
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            if any(alias.name.endswith(REGISTRY_NAME) for alias in node.names):
                kinds.add("import")
        elif isinstance(node, ast.Name) and node.id == REGISTRY_NAME:
            kinds.add("read" if isinstance(node.ctx, ast.Load) else "bind")
        elif isinstance(node, ast.Constant) and node.value == REGISTRY_NAME:
            kinds.add("string")
    return kinds


def _hardcoded_variants(tree: ast.Module, *, module: str) -> list[LiteralFinding]:
    """Every variant-name string literal in `tree`, docstrings excluded."""
    docstrings = _docstring_constant_ids(tree)
    return [
        LiteralFinding(module, node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in VARIANT_NAMES
        and id(node) not in docstrings
    ]


# ── AC-3: one definition ─────────────────────────────────────────────────────


def test_the_sweep_actually_covers_the_package():
    """A glob that silently matches nothing passes every absence assertion below it."""
    modules = _swept_modules()

    assert len(modules) >= MINIMUM_SWEPT_MODULES, (
        f"the registry sweep found only {len(modules)} modules under {_PACKAGE_DIR}; it is "
        "meant to read the whole of src/janus"
    )
    assert CONSTANTS_MODULE in modules, (
        f"the registry sweep found {len(modules)} modules but not {CONSTANTS_MODULE}, which "
        "is the one module that must contain the definition it looks for"
    )
    assert VARIANT_NAMES, "the variant literal sweep has no names to look for"


def test_the_variant_data_is_defined_exactly_once():
    """AC-3: one assignment in the tree; everything else imports or names it.

    ``strategy_registry.py`` wraps the literal in behaviour and ``planner/core.py`` cites
    it in an error message — neither is a second copy, and this test is what tells the
    difference between citing the definition and duplicating it.
    """
    modules = _swept_modules()

    assigning = sorted(name for name, tree in modules.items() if _assigns_the_registry(tree))

    assert assigning == [CONSTANTS_MODULE], (
        f"{REGISTRY_NAME} is assigned in {assigning}. It must be defined exactly once, in "
        f"{CONSTANTS_MODULE} — every other module reads it through an import or through "
        "STRATEGY_REGISTRY. A second assignment is the duplication AC-3 exists to prevent."
    )

    illegitimate = {
        name: sorted(kinds - {"import", "read", "string"})
        for name, tree in modules.items()
        if name != CONSTANTS_MODULE
        and (kinds := _registry_reference_kinds(tree)) - {"import", "read", "string"}
    }

    assert not illegitimate, (
        f"modules mention {REGISTRY_NAME} in a way that is neither an import, a read, nor "
        f"an __all__/message string: {illegitimate}"
    )


def test_no_module_hardcodes_a_variant_name():
    """AC-3's other half: no module enumerates variants to decide which ones exist.

    Findings are reported with module and line so a failure is actionable without a grep.
    A new hit is not automatically wrong — it is a decision, and the way to record it is a
    ``KNOWN_LITERAL_SITES`` entry with a written reason, never a loosened sweep.
    """
    findings = [
        finding
        for module, tree in _swept_modules().items()
        if module != CONSTANTS_MODULE and module not in KNOWN_LITERAL_SITES
        for finding in _hardcoded_variants(tree, module=module)
    ]

    assert not findings, (
        "strategy variant names are hardcoded outside the registry:\n"
        + "\n".join(f"  {finding}" for finding in sorted(map(str, findings)))
        + f"\nIf the module dispatches on a variant it implements, add it to "
        "KNOWN_LITERAL_SITES with a reason. If it is deciding which variants exist, it "
        f"must read the registry instead — that is what {CONSTANTS_MODULE} is for."
    )


def test_every_allowlisted_site_still_hardcodes_something():
    """A stale allowance is dead licence — it exempts a module that no longer needs it.

    Same discipline as the ``SOFT_CEILING_ALLOWANCES`` stale check: an entry that stops
    being load-bearing must be deleted, not left behind to silently cover a future hit.
    """
    modules = _swept_modules()

    unknown = sorted(set(KNOWN_LITERAL_SITES) - set(modules))
    assert not unknown, (
        f"KNOWN_LITERAL_SITES names modules that no longer exist: {unknown}. Delete the "
        "entries — an allowance for a missing module can only ever mislead."
    )

    stale = sorted(
        module
        for module in KNOWN_LITERAL_SITES
        if not _hardcoded_variants(modules[module], module=module)
    )
    assert not stale, (
        f"these modules are allowlisted but hardcode no variant name any more: {stale}. "
        "Delete the entries so the next hit in them has to be argued for."
    )

    unreasoned = sorted(module for module, why in KNOWN_LITERAL_SITES.items() if not why.strip())
    assert not unreasoned, (
        f"KNOWN_LITERAL_SITES entries without a written reason: {unreasoned}. An allowlist "
        "with reasons is a guardrail; one without is a hole."
    )


def test_model_and_planner_read_the_same_registry():
    """AC-3: the planner *derives* from the registry rather than agreeing with it.

    ``test_strategy_registry_drift.py`` pins that the two agree under the default
    registry, which a hardcoded planner copy would also satisfy. This restricts the
    registry to a single pair and requires both consumers to follow it down — a copy
    cannot. Restriction is the direction nothing else tests: every other one-edit test in
    this order widens the registry, and widening is the easier half.
    """
    default_bindings = {
        (binding.family, binding.variant)
        for binding in StrategyCatalog.with_defaults().bindings
    }
    assert default_bindings == set(STRATEGY_REGISTRY.dispatch_keys())

    restricted = StrategyRegistry(variants_by_family={"api": frozenset({"page_number_api"})})

    catalog = StrategyCatalog.with_defaults(restricted)
    assert {(binding.family, binding.variant) for binding in catalog.bindings} == {
        ("api", "page_number_api")
    }

    config = SourceConfig.from_mapping(
        _source_mapping(strategy_variant="page_number_api"), CONFIG_PATH, registry=restricted
    )
    assert catalog.resolve(config).variant == "page_number_api"

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(
            _source_mapping(strategy_variant="offset_api"), CONFIG_PATH, registry=restricted
        )
    assert "strategy_variant: must be one of: page_number_api" in str(exc_info.value)


# ── AC-4: one edit ───────────────────────────────────────────────────────────


def test_adding_a_variant_is_one_edit():
    """AC-4 as a measurement: **one** object built, **both** consumers follow it.

    ``test_strategy_registry.py`` proves each consumer accepts *an* extended registry, but
    each of its tests builds its own — so nothing there pins that a single edit serves
    both. That is the actual AC-4 claim, and it is what this test makes: ``extended`` is
    constructed once and is the only thing either consumer is given.
    """
    extended = STRATEGY_REGISTRY.with_family(
        "api", STRATEGY_REGISTRY.variants_for("api") | {"tsv_api"}
    )
    mapping = _source_mapping(strategy_variant="tsv_api")

    with pytest.raises(SourceConfigValidationError):
        SourceConfig.from_mapping(mapping, CONFIG_PATH)

    config = SourceConfig.from_mapping(mapping, CONFIG_PATH, registry=extended)
    binding = StrategyCatalog.with_defaults(extended).resolve(config)

    assert config.strategy_variant == "tsv_api"
    assert (binding.family, binding.variant) == ("api", "tsv_api")
    assert binding.strategy.strategy_family == "api"
    assert STRATEGY_REGISTRY.supports("api", "tsv_api") is False, (
        "the shared default registry was mutated by with_family; injection must produce a "
        "new object or one test can change what the next one validates against (NFR-2)"
    )


# ── detector meta-tests ──────────────────────────────────────────────────────

CLEAN_REGISTRY_USE = (
    "from janus.models.config.constants import SUPPORTED_STRATEGY_VARIANTS\n"
    "__all__ = ['SUPPORTED_STRATEGY_VARIANTS']\n"
    "def families():\n"
    "    return set(SUPPORTED_STRATEGY_VARIANTS)\n"
)

SECOND_DEFINITION = (
    "SUPPORTED_STRATEGY_VARIANTS = {'api': frozenset({'page_number_api'})}\n"
)


def test_the_single_definition_detector_flags_a_second_definition():
    """A guardrail that has never been seen to fail is a guardrail nobody has tested."""
    assert _assigns_the_registry(ast.parse(SECOND_DEFINITION)) is True


def test_the_single_definition_detector_does_not_flag_importing_or_naming_it():
    """The counterpart: a detector that flags every mention would forbid consuming the data.

    Importing, reading and naming the registry in a string are exactly what the modules
    that legitimately depend on it do — ``strategy_registry.py``, ``source_config.py``,
    ``planner/core.py``.
    """
    tree = ast.parse(CLEAN_REGISTRY_USE)

    assert _assigns_the_registry(tree) is False
    assert _registry_reference_kinds(tree) == {"import", "read", "string"}

    real = _swept_modules()["models/config/strategy_registry.py"]
    assert _assigns_the_registry(real) is False


def test_the_literal_detector_flags_a_hardcoded_variant_set():
    """The shape a second registry would actually take, and the detector must see it."""
    offending = ast.parse(
        "SUPPORTED = ('page_number_api', 'offset_api')\n"
        "def is_paged(variant):\n"
        "    return variant == 'page_number_api'\n"
    )

    findings = _hardcoded_variants(offending, module="fake_registry.py")

    assert {finding.variant for finding in findings} == {"page_number_api", "offset_api"}


def test_the_literal_detector_does_not_flag_docstrings_or_unrelated_strings():
    """Docstrings describe variants; a detector that flagged them would push docs out.

    ``coercion.py`` is the live control — real package code, full of string literals, and
    naming no variant at all.
    """
    documented = ast.parse(
        '''"""Handles page_number_api and offset_api sources."""\n'''
        "def run(mode):\n"
        "    return mode == 'full_refresh'\n"
    )

    assert _hardcoded_variants(documented, module="fake_documented.py") == []
    assert (
        _hardcoded_variants(_swept_modules()["models/config/coercion.py"], module="coercion.py")
        == []
    )


def _source_mapping(*, strategy_variant: str) -> dict[str, object]:
    """A source that loads cleanly, so a failure is about the registry, not the fixture."""
    return {
        "source_id": "single_source_registry",
        "name": "single_source_registry",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": strategy_variant,
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/records",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 100,
            },
            "rate_limit": {"requests_per_minute": 10, "concurrency": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/single_source_registry", "format": "json"},
            "bronze": {
                "path": "data/bronze/example/single_source_registry",
                "format": "iceberg",
            },
            "metadata": {
                "path": "data/metadata/example/single_source_registry",
                "format": "json",
            },
        },
        "quality": {"allow_schema_evolution": True},
    }
