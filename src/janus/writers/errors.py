"""Writer failures that carry a stable runtime failure stage."""

from janus.quality.contract_checks import ContractEnforcementError
from janus.writers.evolution import EvolutionPlan


class SchemaEvolutionRefusedError(ContractEnforcementError):
    """The live table cannot evolve under the declared compatibility mode."""

    failure_stage = "schema_evolution"

    def __init__(self, evolution: EvolutionPlan) -> None:
        self.evolution = evolution
        details = "; ".join(
            f"{item.column}: {item.kind} ({item.detail})" for item in evolution.refusals
        )
        super().__init__(f"{evolution.reason}: {details}")
