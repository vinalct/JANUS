from __future__ import annotations

from pathlib import Path

from janus.models import ExecutionPlan
from janus.registry.contracts import load_contract_snapshot

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


def with_registry_contract(plan: ExecutionPlan) -> ExecutionPlan:
    """Give a hand-built plan the contract the planner would have attached to it."""
    source_config = plan.source_config
    snapshot = load_contract_snapshot(
        (source_config,),
        project_root=plan.run_context.project_root,
        sources_dir=source_config.config_path.parent,
    )
    return plan.with_data_contract(snapshot.get(source_config.source_id))
