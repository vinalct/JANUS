"""phase-scope policy lives in exactly one module.

The rules this sweeps for (public-access requirement, federation-level scope, the
strategy == source_type pairing) are *product-phase decisions*, not structural invariants.
Move them into models/config/policy.py so broadening JANUS's scope is a policy
edit rather than type-layer surgery. This test is what keeps that true: a rule that leaks
back into a builder or into from_mapping would restore the debt silently.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

import janus.models.source_config as source_config_module

POLICY_MODULE = "policy.py"

FEDERATION_LEVEL_SCOPE = frozenset({"federal"})

FEDERATION_LITERAL_EXEMPT_MODULES = frozenset({"constants.py"})


@dataclass(frozen=True)
class PhaseScopeFinding:
    """One inline phase-scope decision, located precisely enough to act on."""

    module: str
    lineno: int
    kind: str
    snippet: str

    def __str__(self) -> str:
        return f"{self.module}:{self.lineno}: [{self.kind}] {self.snippet}"


# ── the detector ─────────────────────────────────────────────────────────────


def _operand_tail(node: ast.expr) -> str | None:
    """Trailing name of a Name/Attribute operand: ``cfg.public_access`` -> ``public_access``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_public_access_decision(node: ast.Compare) -> bool:
    """(a) ``public_access is False`` / ``cfg.public_access == True`` and friends."""
    operands = [node.left, *node.comparators]
    tails = [_operand_tail(operand) for operand in operands]
    reads_public_access = any(tail is not None and tail.endswith("public_access") for tail in tails)
    against_bool = any(
        isinstance(operand, ast.Constant) and isinstance(operand.value, bool)
        for operand in operands
    )
    return reads_public_access and against_bool


def _is_source_type_strategy_pairing(node: ast.Compare) -> bool:
    """(c) ``source_type != strategy`` — the phase-1 simplification, not a structural rule."""
    tails = [
        tail for tail in (_operand_tail(operand) for operand in [node.left, *node.comparators])
        if tail is not None
    ]
    return any(tail.endswith("source_type") for tail in tails) and any(
        tail.endswith("strategy") for tail in tails
    )


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    """Docstrings describe a rule; they do not decide one. Excluded from the literal case."""
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


def phase_scope_decisions(tree: ast.Module, *, module: str) -> list[PhaseScopeFinding]:
    """The detector itself.

    Shared by the real guardrail and by its meta-tests below — a meta-test that
    reimplemented the check would prove nothing about the check.
    """
    findings: list[PhaseScopeFinding] = []
    docstrings = _docstring_constant_ids(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if _is_public_access_decision(node):
                findings.append(
                    PhaseScopeFinding(module, node.lineno, "public_access", ast.unparse(node))
                )
            if _is_source_type_strategy_pairing(node):
                findings.append(
                    PhaseScopeFinding(
                        module, node.lineno, "strategy==source_type", ast.unparse(node)
                    )
                )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in FEDERATION_LEVEL_SCOPE
            and id(node) not in docstrings
            and module not in FEDERATION_LITERAL_EXEMPT_MODULES
        ):
            findings.append(
                PhaseScopeFinding(module, node.lineno, "federation_level", ast.unparse(node))
            )

    return sorted(findings, key=lambda finding: (finding.module, finding.lineno, finding.kind))


# ── the swept surface ────────────────────────────────────────────────────────


def _swept_modules() -> dict[str, ast.Module]:
    """``source_config.py`` plus every module of ``janus.models.config``, parsed.

    Package-scoped by construction: a module added by a later split joins the sweep
    without anyone remembering to add it.
    """
    models_dir = Path(inspect.getfile(source_config_module)).parent
    paths = [models_dir / "source_config.py", *sorted((models_dir / "config").glob("*.py"))]
    return {
        path.name: ast.parse(path.read_text(encoding="utf-8")) for path in paths if path.exists()
    }


def test_the_sweep_actually_covers_the_config_package():
    """A glob that silently matches nothing passes every assertion below it."""
    modules = _swept_modules()

    assert modules, "the phase-policy sweep matched no modules at all"
    assert "source_config.py" in modules, (
        f"the phase-policy sweep found {sorted(modules)} — it must always read "
        "source_config.py, which is where the phase-1 rules live today"
    )
    known_builders = {"access.py", "request_inputs.py", "contracts.py", "constants.py"}
    assert known_builders <= set(modules), (
        f"the phase-policy sweep is missing modules it must cover: "
        f"{sorted(known_builders - set(modules))} (found {sorted(modules)})"
    )


@pytest.mark.xfail(
    strict=True,
    reason="moves these rules into models/config/policy.py; "
    "this xfail must turn into an XPASS failure the moment it lands, "
    "at which point the marker is removed in the same commit",
)
def test_no_module_outside_policy_decides_phase_scope():
    findings = [
        finding
        for module, tree in _swept_modules().items()
        if module != POLICY_MODULE
        for finding in phase_scope_decisions(tree, module=module)
    ]

    assert not findings, (
        "inline phase-scope decisions found outside models/config/policy.py:\n"
        + "\n".join(f"  {finding}" for finding in findings)
        + "\nThese are product-phase decisions, not structural invariants — they belong to "
        "the ValidationPolicy so broadening scope needs a policy edit, not a type-layer one."
    )


# ── detector meta-tests ──────────────────────────────────────────────────────

DELIBERATE_VIOLATIONS: dict[str, tuple[str, str]] = {
    "public_access": (
        "public_access",
        "def check(public_access, issues):\n"
        "    if public_access is False:\n"
        "        issues.append(ValidationIssue('public_access', 'must be true'))\n",
    ),
    "federation_level": (
        "federation_level",
        "def check(federation_level, issues):\n"
        "    if federation_level == 'federal':\n"
        "        return True\n"
        "    return False\n",
    ),
    "strategy_pairing": (
        "strategy==source_type",
        "def check(source_type, strategy, issues):\n"
        "    if source_type and strategy and source_type != strategy:\n"
        "        issues.append(ValidationIssue('strategy', 'must match source_type'))\n",
    ),
}


@pytest.mark.parametrize("case", sorted(DELIBERATE_VIOLATIONS))
def test_phase_policy_detector_flags_a_deliberate_violation(case: str, tmp_path: Path):
    """The guardrail must fail on a violation, not merely pass on clean code."""
    expected_kind, source = DELIBERATE_VIOLATIONS[case]
    offending = tmp_path / "fake_builder.py"
    offending.write_text(source, encoding="utf-8")

    findings = phase_scope_decisions(
        ast.parse(offending.read_text(encoding="utf-8")), module=offending.name
    )

    assert findings, f"the phase-policy detector no longer detects the {case} violation"
    assert expected_kind in {finding.kind for finding in findings}, (
        f"the {case} violation was flagged as {[f.kind for f in findings]}, not {expected_kind!r}"
    )


def test_phase_policy_detector_does_not_flag_clean_code(tmp_path: Path):
    """Counterpart to the meta-test above: a detector that flags everything proves nothing.

    ``coercion.py`` is the control — real package code, full of comparisons and string
    literals, and entirely free of phase-scope decisions.
    """
    clean = tmp_path / "fake_clean.py"
    clean.write_text(
        "def check(source_type, strategy_variant, issues):\n"
        "    if source_type not in ('api', 'catalog'):\n"
        "        return None\n"
        "    if strategy_variant == 'page_number_api':\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )

    snippet_findings = phase_scope_decisions(
        ast.parse(clean.read_text(encoding="utf-8")), module=clean.name
    )
    assert not snippet_findings, (
        f"the detector flags clean code: {list(map(str, snippet_findings))}"
    )

    coercion_findings = phase_scope_decisions(
        _swept_modules()["coercion.py"], module="coercion.py"
    )
    assert not coercion_findings, (
        "the detector flags models/config/coercion.py, which decides no phase scope: "
        f"{list(map(str, coercion_findings))}"
    )


def test_the_constants_exemption_is_load_bearing_and_narrow():
    """The exemption must cover a literal that is really there, and nothing more.

    If ``constants.py`` stopped holding the federation-level set, this test would fail and
    the exemption would have to be deleted rather than left behind as dead licence.
    """
    constants_tree = _swept_modules()["constants.py"]

    exempted = phase_scope_decisions(constants_tree, module="constants.py")
    assert not exempted, f"constants.py is exempt yet still flagged: {list(map(str, exempted))}"

    unexempted = phase_scope_decisions(constants_tree, module="not_the_data_module.py")
    assert any(finding.kind == "federation_level" for finding in unexempted), (
        "constants.py no longer holds a federation-level literal, so the exemption in "
        "FEDERATION_LITERAL_EXEMPT_MODULES now licenses nothing — delete it."
    )
