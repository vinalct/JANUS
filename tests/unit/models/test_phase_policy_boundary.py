"""phase-scope policy lives in exactly one module.

``from_mapping`` consults the policy for four decisions — the source-type, strategy and
federation-level value sets, the strategy == source_type pairing, and the public-access
requirement — at five call sites. Those are *product-phase decisions*, not structural
invariants, so they live in models/config/policy.py and broadening JANUS's scope is a
policy edit rather than type-layer surgery. This test is what keeps that true: a rule that
leaks back into a builder or into from_mapping would restore the debt silently.

The fourth detector, ``contract.status == "active"``, is the one rule that moved the other
way. Order-18 made it policy; order-19 made it structural, because the materializer cannot
write without a reviewed contract. Its only licence is therefore one loader function,
and policy.py is swept for it like every other module.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

import janus.models.source_config as source_config_module
import janus.registry as registry_module
from janus.models.data_contracts import SUPPORTED_CONTRACT_STATUSES

POLICY_MODULE = "config/policy.py"

FEDERATION_LEVEL_SCOPE = frozenset({"federal"})
CONTRACT_STATUS_SCOPE = SUPPORTED_CONTRACT_STATUSES

FEDERATION_LITERAL_EXEMPT_MODULES = frozenset({"constants.py", "config/constants.py"})

#: The one place allowed to decide contract status. Structural since order-19 (D-15): an
#: enabled source materializes only under an ``active`` contract, so the rule cannot relax
#: with a policy. Function-scoped, so a second status check in the loader is still flagged.
STRUCTURAL_STATUS_RULE = ("registry/loader.py", "_require_active_contract")


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


def _is_contract_status_decision(node: ast.Compare) -> bool:
    """(d) ``contract.status == "active"`` — only the loader may decide this."""
    operands = [node.left, *node.comparators]
    reads_status = any(
        (tail := _operand_tail(operand)) is not None and tail.endswith("status")
        for operand in operands
    )
    compares_status_literal = any(
        isinstance(operand, ast.Constant)
        and isinstance(operand.value, str)
        and operand.value in CONTRACT_STATUS_SCOPE
        for operand in operands
    )
    return reads_status and compares_status_literal


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
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def in_structural_loader_rule(node: ast.AST) -> bool:
        rule_module, rule_function = STRUCTURAL_STATUS_RULE
        if module != rule_module:
            return False
        current = parents.get(node)
        while current is not None:
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef):
                return current.name == rule_function
            current = parents.get(current)
        return False

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
            if _is_contract_status_decision(node) and not in_structural_loader_rule(node):
                findings.append(
                    PhaseScopeFinding(
                        module, node.lineno, "contract_status", ast.unparse(node)
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
    """Parse config, contract-model and registry packages into one guarded surface."""
    models_dir = Path(inspect.getfile(source_config_module)).parent
    registry_dir = Path(inspect.getfile(registry_module)).parent
    paths = {
        "source_config.py": models_dir / "source_config.py",
        **{
            f"config/{path.name}": path
            for path in sorted((models_dir / "config").glob("*.py"))
        },
        **{
            f"data_contracts/{path.name}": path
            for path in sorted((models_dir / "data_contracts").glob("*.py"))
        },
        **{
            f"registry/{path.name}": path
            for path in sorted(registry_dir.glob("*.py"))
        },
    }
    return {
        module: ast.parse(path.read_text(encoding="utf-8"))
        for module, path in paths.items()
        if path.exists()
    }


def test_the_sweep_actually_covers_the_config_package():
    """A glob that silently matches nothing passes every assertion below it."""
    modules = _swept_modules()

    assert modules, "the phase-policy sweep matched no modules at all"
    assert "source_config.py" in modules, (
        f"the phase-policy sweep found {sorted(modules)} — it must always read "
        "source_config.py, which is where the phase-1 rules live today"
    )
    known_builders = {
        "config/access.py",
        "config/request_inputs.py",
        "config/contracts.py",
        "config/constants.py",
        "data_contracts/model.py",
        "registry/loader.py",
    }
    assert known_builders <= set(modules), (
        f"the phase-policy sweep is missing modules it must cover: "
        f"{sorted(known_builders - set(modules))} (found {sorted(modules)})"
    )


def test_no_module_outside_policy_decides_phase_scope():
    findings = [
        finding
        for module, tree in _swept_modules().items()
        for finding in phase_scope_decisions(tree, module=module)
        if module != POLICY_MODULE or finding.kind == "contract_status"
    ]

    assert not findings, (
        "scope decisions found outside policy or the structural loader rule:\n"
        + "\n".join(f"  {finding}" for finding in findings)
        + "\nProduct-phase decisions belong in policy; active status belongs only "
        "in registry/loader.py::_require_active_contract."
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
    "contract_status": (
        "contract_status",
        "def check(contract, issues):\n"
        "    if contract.status == 'active':\n"
        "        return True\n"
        "    return False\n",
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


def test_active_status_exemption_is_function_scoped():
    inside = ast.parse(
        "def _require_active_contract(contract):\n"
        "    return contract.status == 'active'\n"
    )
    elsewhere = ast.parse(
        "def another_loader_check(contract):\n"
        "    return contract.status == 'active'\n"
    )

    assert not phase_scope_decisions(inside, module="registry/loader.py")
    assert any(
        finding.kind == "contract_status"
        for finding in phase_scope_decisions(elsewhere, module="registry/loader.py")
    )
    assert any(
        finding.kind == "contract_status"
        for finding in phase_scope_decisions(inside, module=POLICY_MODULE)
    ), "policy.py lost its licence for contract status in order-19; the sweep must flag it"


def test_the_structural_status_exemption_is_load_bearing_and_narrow():
    """The loader exemption must cover exactly one real decision, in the named function.

    If ``_require_active_contract`` stopped deciding status, the exemption would license
    nothing and must be deleted; if a second status check appeared anywhere in the loader,
    counting it here would show the one-decision rule had quietly become two.
    """
    rule_module, rule_function = STRUCTURAL_STATUS_RULE
    loader_tree = _swept_modules()[rule_module]

    exempted = phase_scope_decisions(loader_tree, module=rule_module)
    assert not exempted, f"{rule_module} is exempt yet still flagged: {list(map(str, exempted))}"

    unexempted = [
        finding
        for finding in phase_scope_decisions(loader_tree, module="registry/not_the_loader.py")
        if finding.kind == "contract_status"
    ]
    assert len(unexempted) == 1, (
        f"{rule_module} must decide contract status exactly once, in {rule_function}; "
        f"found {list(map(str, unexempted))}"
    )
    rule = next(
        node
        for node in ast.walk(loader_tree)
        if isinstance(node, ast.FunctionDef) and node.name == rule_function
    )
    assert rule.lineno <= unexempted[0].lineno <= (rule.end_lineno or rule.lineno)


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
        "    return 0\n"
        "def check(status: str):\n"
        "    \"\"\"Contract status == 'active' is descriptive text.\"\"\"\n"
        "    return status\n",
        encoding="utf-8",
    )

    snippet_findings = phase_scope_decisions(
        ast.parse(clean.read_text(encoding="utf-8")), module=clean.name
    )
    assert not snippet_findings, (
        f"the detector flags clean code: {list(map(str, snippet_findings))}"
    )

    coercion_findings = phase_scope_decisions(
        _swept_modules()["config/coercion.py"], module="coercion.py"
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
    constants_tree = _swept_modules()["config/constants.py"]

    exempted = phase_scope_decisions(constants_tree, module="constants.py")
    assert not exempted, f"constants.py is exempt yet still flagged: {list(map(str, exempted))}"

    unexempted = phase_scope_decisions(constants_tree, module="not_the_data_module.py")
    assert any(finding.kind == "federation_level" for finding in unexempted), (
        "constants.py no longer holds a federation-level literal, so the exemption in "
        "FEDERATION_LITERAL_EXEMPT_MODULES now licenses nothing — delete it."
    )
