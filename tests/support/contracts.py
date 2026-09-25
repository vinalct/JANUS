from __future__ import annotations

from pathlib import Path

MINIMAL_CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "contracts" / "minimal_contract.yaml"
)

#: Where a synthetic project keeps it, relative to the project root.
DECLARED_CONTRACT_PATH = "conf/contracts/test/minimal_contract.yaml"

#: The schema block a synthetic source declares.
CONTRACT_SCHEMA_BLOCK = {"contract": DECLARED_CONTRACT_PATH}


def write_minimal_contract(project_root: Path) -> Path:
    """Copy the minimal contract into ``project_root`` and return where it landed."""
    target = project_root / DECLARED_CONTRACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(MINIMAL_CONTRACT_FIXTURE.read_bytes())
    return target


def minimal_contract_yaml() -> str:
    """The contract's text, for a builder that renders a project as strings."""
    return MINIMAL_CONTRACT_FIXTURE.read_text(encoding="utf-8")
