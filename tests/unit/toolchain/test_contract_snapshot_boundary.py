

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import janus

PACKAGE_ROOT = Path(inspect.getfile(janus)).parent

#: Where a contract would most plausibly be re-read: everything downstream of planning.
SNAPSHOT_SCOPED_PACKAGES = (
    "lineage",
    "observability",
    "quality",
    "readers",
    "runtime",
    "scripts",
    "strategies",
    "writers",
)

#: Modules the sweep must actually reach, so a bad glob cannot pass by matching nothing.
SNAPSHOT_SCOPED_ANCHORS = {
    "lineage/models.py",
    "quality/validators.py",
    "runtime/materialize.py",
}

#: Importing the loader is as much a re-read as calling it: the import is the intent.
FORBIDDEN_LOADER_IMPORTS = (
    "janus.models.data_contracts.loader",
)

#: The functions that turn a path into a contract. Every one of them reads a file.
FORBIDDEN_LOADER_CALLS = (
    "load_data_contract",
)

#: The package that *defines* these functions, necessarily calling them among themselves.
DEFINITION_PACKAGE = "models/data_contracts"

#: The only modules in ``src/janus`` allowed to turn a path into a contract, with reasons.
ALLOWED_CONTRACT_READERS: dict[str, str] = {
    "registry/contracts.py": (
        "the snapshot itself: one load reads every declared contract once, keyed by "
        "resolved path, and every other layer receives the object from the plan"
    ),
    "cli/contract.py": (
        "the drafting subcommand writes a contract rather than running a source, so it "
        "is outside the snapshot by construction; it is never reached from run/run-all"
    ),
}


def _modules_under(*packages: str) -> tuple[Path, ...]:
    modules: list[Path] = []
    for package in packages:
        modules.extend((PACKAGE_ROOT / package).rglob("*.py"))
    return tuple(sorted(set(modules)))


def _relative(module: Path) -> str:
    return module.relative_to(PACKAGE_ROOT).as_posix()


def _imported_modules(source: str) -> tuple[tuple[int, str], ...]:
    imported: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.append((node.lineno, node.module))
            imported.extend(
                (node.lineno, f"{node.module}.{alias.name}") for alias in node.names
            )
    return tuple(imported)


def _called_names(source: str) -> tuple[tuple[int, str], ...]:
    called: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.append((node.lineno, func.id))
        elif isinstance(func, ast.Attribute):
            called.append((node.lineno, func.attr))
    return tuple(called)


def _contract_reads(source: str) -> tuple[str, ...]:
    """Report every place one module turns a contract path into a contract object."""
    violations = [
        f"line {line}: imports {module}"
        for line, module in _imported_modules(source)
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for forbidden in FORBIDDEN_LOADER_IMPORTS
        )
    ]
    violations.extend(
        f"line {line}: calls {name}()"
        for line, name in _called_names(source)
        if name in FORBIDDEN_LOADER_CALLS
    )
    return tuple(violations)


def _violations(modules: tuple[Path, ...]) -> dict[str, tuple[str, ...]]:
    return {
        _relative(module): reads
        for module in modules
        if (reads := _contract_reads(module.read_text(encoding="utf-8")))
    }


def test_no_module_downstream_of_planning_reads_a_contract_file() -> None:
    modules = _modules_under(*SNAPSHOT_SCOPED_PACKAGES)
    matched = {_relative(module) for module in modules}

    assert matched >= SNAPSHOT_SCOPED_ANCHORS, (
        "the contract-snapshot sweep missed an intended package: "
        f"{sorted(SNAPSHOT_SCOPED_ANCHORS - matched)}"
    )
    assert not _violations(modules), (
        "a module downstream of planning opened a contract file instead of reading "
        f"plan.data_contract: {_violations(modules)}"
    )


def test_only_the_snapshot_and_the_drafting_cli_turn_a_path_into_a_contract() -> None:
    """The repository-wide form of the same rule, so a new package cannot slip past it."""
    modules = tuple(
        module
        for module in _modules_under(".")
        if not _relative(module).startswith(DEFINITION_PACKAGE)
    )

    assert len(modules) > len(_modules_under(*SNAPSHOT_SCOPED_PACKAGES)), (
        "the repository-wide sweep should cover more than the downstream packages"
    )
    unexpected = {
        path: reads
        for path, reads in _violations(modules).items()
        if path not in ALLOWED_CONTRACT_READERS
    }
    assert not unexpected, (
        "a contract was read outside the registry snapshot: "
        f"{unexpected}. Read it once with the registry and carry it on the plan, or add "
        "the module to ALLOWED_CONTRACT_READERS with a written reason."
    )


def test_every_recorded_allowance_is_load_bearing() -> None:
    """An allowance for a module that no longer reads a contract is a disarmed guardrail."""
    reading = set(_violations(_modules_under(".")))
    stale = sorted(
        path
        for path in ALLOWED_CONTRACT_READERS
        if (PACKAGE_ROOT / path).exists() and path not in reading
    )

    assert not stale, (
        f"ALLOWED_CONTRACT_READERS entries that read no contract any more: {stale}. "
        "Remove the allowance; it is a decision, not a backlog."
    )
    assert "registry/contracts.py" in reading, (
        "the registry snapshot stopped reading contracts, so this sweep now enforces "
        "nothing about where they are read instead"
    )


@pytest.mark.parametrize(
    "source",
    [
        "from janus.models.data_contracts.loader import load_data_contract",
        "contract = load_data_contract(path)",
        "contract = contracts.load_data_contract(path)",
    ],
)
def test_the_detector_flags_a_deliberate_violation(source: str) -> None:
    assert _contract_reads(source)


@pytest.mark.parametrize(
    "source",
    [
        "from janus.models.data_contracts import DataContract",
        "from janus.models.data_contracts.model import ContractProperty",
        "contract = plan.data_contract",
        "contract = registry.contract_for(source_id)",
        "schema = spark_schema_from_contract(plan.data_contract)",
        "load_data_contract = None",
    ],
)
def test_the_detector_does_not_flag_reading_the_snapshot(source: str) -> None:
    assert not _contract_reads(source)
