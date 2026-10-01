from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from janus.models import ExecutionPlan
from janus.registry.contracts import load_contract_snapshot

MINIMAL_CONTRACT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "contracts" / "minimal_contract.yaml"
)

#: Where a synthetic project keeps it, relative to the project root.
DECLARED_CONTRACT_PATH = "conf/contracts/test/minimal_contract.yaml"

#: The schema block a synthetic source declares.
CONTRACT_SCHEMA_BLOCK = {"contract": DECLARED_CONTRACT_PATH}

#: Where a synthetic project keeps a contract that declares a primaryKey.
KEYED_CONTRACT_PATH = "conf/contracts/test/keyed_contract.yaml"


def write_minimal_contract(project_root: Path) -> Path:
    """Copy the minimal contract into ``project_root`` and return where it landed."""
    target = project_root / DECLARED_CONTRACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(MINIMAL_CONTRACT_FIXTURE.read_bytes())
    return target


def minimal_contract_yaml() -> str:
    """The contract's text, for a builder that renders a project as strings."""
    return MINIMAL_CONTRACT_FIXTURE.read_text(encoding="utf-8")


def keyed_contract_yaml(
    primary_key: Sequence[str] = ("id",), *, required: Sequence[str] | None = None
) -> str:
    """The minimal contract with ``primary_key`` declared as its key, one string column each.

    Key columns are ``required`` unless ``required`` is given, in which case it names the
    required columns exactly: leaving a key column out of it builds the contract the quality
    gate's ``config.quality_contract`` check refuses. Required columns keep
    ``sourceNullable: "true"``, so the generated read schema is the minimal contract's.
    """
    required_columns = tuple(primary_key) if required is None else tuple(required)
    contract = yaml.safe_load(minimal_contract_yaml())
    contract["id"] = "example.keyed"
    contract["name"] = "Keyed contract"
    contract["description"]["purpose"] = "Minimal contract that declares a primaryKey."
    contract["schema"][0]["properties"] = [
        _keyed_property(name, required=name in required_columns, key=name in primary_key)
        for name in dict.fromkeys(("id", *primary_key, *required_columns))
    ]
    return yaml.safe_dump(contract, sort_keys=False)


def _keyed_property(name: str, *, required: bool, key: bool) -> dict[str, object]:
    prop: dict[str, object] = {"name": name, "logicalType": "string", "physicalType": "string"}
    if required:
        prop["required"] = True
        prop["customProperties"] = [{"property": "sourceNullable", "value": "true"}]
    if key:
        prop["primaryKey"] = True
    return prop


def write_keyed_contract(
    project_root: Path,
    primary_key: Sequence[str] = ("id",),
    *,
    required: Sequence[str] | None = None,
) -> Path:
    """Write :func:`keyed_contract_yaml` at ``KEYED_CONTRACT_PATH`` and return where it landed."""
    target = project_root / KEYED_CONTRACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(keyed_contract_yaml(primary_key, required=required), encoding="utf-8")
    return target


def with_registry_contract(plan: ExecutionPlan) -> ExecutionPlan:
    """Give a hand-built plan the contract the planner would have attached to it."""
    source_config = plan.source_config
    snapshot = load_contract_snapshot(
        (source_config,),
        project_root=plan.run_context.project_root,
        sources_dir=source_config.config_path.parent,
    )
    return plan.with_data_contract(snapshot.get(source_config.source_id))
