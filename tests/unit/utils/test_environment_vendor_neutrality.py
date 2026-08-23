"""Vendor-neutrality sweep over `janus.utils.environment`."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import janus.utils.environment as environment

# Names that would betray a provider-specific branch in the environment module.
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


def _environment_module_source() -> str:
    return Path(inspect.getfile(environment)).read_text(encoding="utf-8")


# ── the sweep ────────────────────────────────────────────────────────────────


def test_environment_logic_does_not_branch_on_a_vendor():
    """NFR-1: no conditional in `utils/environment.py` names a specific provider."""

    violations = _vendor_violations(_environment_module_source())

    assert not violations, (
        "vendor-specific names found in a branch of utils/environment.py — swapping "
        "the catalog backing store must stay config-only. Impl-class "
        "strings belong in module-level constants, never in a conditional:\n"
        + "\n".join(violations)
    )


def test_the_sweep_actually_parsed_branching_logic():
    """The sweep cannot pass by matching nothing: the module has real branches.

    A refactor that emptied the module (or a detector that stopped finding
    conditions) would make the assertion above vacuously green; this pins the
    detector to a non-empty vocabulary drawn from the real module.
    """

    vocabulary = _branch_vocabulary(ast.parse(_environment_module_source()))

    assert vocabulary, (
        "the vendor-neutrality sweep found no branch conditions in utils/environment.py "
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
