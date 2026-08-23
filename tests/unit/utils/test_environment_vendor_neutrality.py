"""Vendor-neutrality sweep over the `janus.utils` package."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import janus.utils as utils_package

# Every module the rule binds, and the ones the coverage check insists on finding.
SWEPT_PACKAGE = "janus/utils"
REQUIRED_MODULES = ("environment.py", "catalog_properties.py")

# Names that would betray a provider-specific branch in a swept module.
VENDOR_TOKENS = (
    "glue",
    "emr",
    "dataproc",
    "databricks",
    "snowflake",
    "gcs",
    "adls",
    "azure",
)


# ── the detector ─────────────────────────────────────────────────────────────


def _branch_condition_nodes(tree: ast.AST) -> list[ast.AST]:
    """Every expression that decides a branch.

    `if`/`elif`/`while` tests, ternary tests, `match` subjects, case patterns
    and guards, and comprehension filters. Bodies are excluded on purpose:
    a module-level constant assignment is not a branch, which is exactly the
    exemption the module docstring records.
    """

    conditions: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If | ast.IfExp | ast.While):
            conditions.append(node.test)
        elif isinstance(node, ast.Match):
            conditions.append(node.subject)
            for case in node.cases:
                conditions.append(case.pattern)
                if case.guard is not None:
                    conditions.append(case.guard)
        elif isinstance(node, ast.comprehension):
            conditions.extend(node.ifs)
    return conditions


def _branch_vocabulary(tree: ast.AST) -> list[str]:
    """Identifiers, attribute names and string literals inside branch conditions."""

    vocabulary: list[str] = []
    for condition in _branch_condition_nodes(tree):
        for node in ast.walk(condition):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                vocabulary.append(node.value)
            elif isinstance(node, ast.Name):
                vocabulary.append(node.id)
            elif isinstance(node, ast.Attribute):
                vocabulary.append(node.attr)
    return vocabulary


def _vendor_violations(source: str) -> list[str]:
    return [
        f"{token!r} in {text!r}"
        for text in _branch_vocabulary(ast.parse(source))
        for token in VENDOR_TOKENS
        if token in text.lower()
    ]


def _swept_modules() -> list[Path]:
    package_root = Path(utils_package.__file__).parent
    return sorted(module for module in package_root.glob("*.py") if module.name != "__init__.py")


# ── the sweep ────────────────────────────────────────────────────────────────


def test_no_module_in_the_utils_package_branches_on_a_vendor():
    """NFR-1: no conditional under `utils/` names a specific provider."""

    violations = [
        f"{module.name}: {violation}"
        for module in _swept_modules()
        for violation in _vendor_violations(module.read_text(encoding="utf-8"))
    ]

    assert not violations, (
        f"vendor-specific names found in a branch of {SWEPT_PACKAGE}/ — swapping "
        "the catalog backing store must stay config-only. Impl-class "
        "strings belong in module-level constants, never in a conditional:\n"
        + "\n".join(violations)
    )


def test_the_sweep_actually_parsed_branching_logic():
    """The sweep cannot pass by matching nothing: those modules have real branches.

    A refactor that emptied a module, a glob that stopped matching it, or a detector that
    stopped finding conditions would make the assertion above vacuously green; this pins the
    sweep to the modules that carry the catalog logic and to a non-empty vocabulary drawn from
    each of them.
    """

    swept = {module.name: module.read_text(encoding="utf-8") for module in _swept_modules()}

    missing = [name for name in REQUIRED_MODULES if name not in swept]
    assert not missing, (
        f"the vendor-neutrality sweep never reached {missing} — either they moved out of "
        f"{SWEPT_PACKAGE}/ or the sweep went blind"
    )
    for name in REQUIRED_MODULES:
        assert _branch_vocabulary(ast.parse(swept[name])), (
            f"the vendor-neutrality sweep found no branch conditions in {SWEPT_PACKAGE}/{name} "
            "— either the module lost all its logic or the detector went blind"
        )


# ── detector meta-tests, per repo convention ─────────────────────────────────


@pytest.mark.parametrize(
    ("label", "snippet"),
    [
        (
            "if_comparison",
            'def pick(provider):\n    if provider == "glue":\n        return 1\n    return 0\n',
        ),
        (
            "elif_comparison",
            "def pick(kind):\n"
            '    if kind == "jdbc":\n'
            "        return 1\n"
            '    elif kind == "databricks":\n'
            "        return 2\n"
            "    return 0\n",
        ),
        (
            "match_case",
            "def pick(provider):\n"
            "    match provider:\n"
            '        case "dataproc":\n'
            "            return 1\n"
            "    return 0\n",
        ),
        (
            "identifier_in_condition",
            "def pick(options, emr_mode):\n    if emr_mode:\n        return 1\n    return 0\n",
        ),
        (
            "ternary",
            'def pick(provider):\n    return 1 if provider == "azure" else 0\n',
        ),
        (
            "comprehension_filter",
            'def pick(keys):\n    return [key for key in keys if key == "snowflake"]\n',
        ),
    ],
)
def test_the_detector_flags_a_deliberate_vendor_branch(label, snippet):
    """Each branching shape the detector claims to cover actually trips it."""

    del label
    assert _vendor_violations(snippet), f"detector missed a vendor branch in:\n{snippet}"


def test_the_detector_does_not_flag_clean_code():
    """Impl-class constants and catalog-type branches are allowed, per the rule.

    The constants carry vendor-ish package segments (`gcs`, `adls`, `azure`)
    precisely because that is the exemption: an impl string is data. The
    branches compare catalog *types*, which are protocol names.
    """

    clean = (
        'S3_FILE_IO_IMPL = "org.apache.iceberg.aws.s3.S3FileIO"\n'
        'GCS_FILE_IO_IMPL = "org.apache.iceberg.gcp.gcs.GCSFileIO"\n'
        'ADLS_FILE_IO_IMPL = "org.apache.iceberg.azure.adlsv2.ADLSFileIO"\n'
        "\n"
        "def branch(catalog_type, options):\n"
        '    if catalog_type == "jdbc":\n'
        '        options["io-impl"] = S3_FILE_IO_IMPL\n'
        '    elif catalog_type in {"rest", "hadoop"}:\n'
        '        options["io-impl"] = GCS_FILE_IO_IMPL\n'
        "    return options\n"
    )

    assert _vendor_violations(clean) == []


def test_a_vendor_named_constant_used_in_a_branch_is_still_flagged():
    """The exemption covers holding a constant, not steering logic with it."""

    steering = (
        'GCS_FILE_IO_IMPL = "org.apache.iceberg.gcp.gcs.GCSFileIO"\n'
        "\n"
        "def branch(io_impl):\n"
        "    if io_impl == GCS_FILE_IO_IMPL:\n"
        "        return 1\n"
        "    return 0\n"
    )

    assert _vendor_violations(steering)
