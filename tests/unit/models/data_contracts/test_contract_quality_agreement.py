from __future__ import annotations

from pathlib import Path

from janus.registry.loader import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_quality_expectations_are_recorded_in_every_active_contract() -> None:
    """Active contracts record quality fields; unreviewed drafts make no such claim."""
    registry = load_registry(PROJECT_ROOT)
    declared = [source for source in registry.sources if source.schema.contract]

    assert len(declared) == 31
    assert len({source.schema.contract for source in declared}) == 30

    for source in declared:
        contract = registry.contract_for(source.source_id)
        assert contract is not None, source.source_id

        if contract.status == "draft":
            assert source.enabled is False, source.source_id
            assert not contract.required_columns, source.source_id
            assert not contract.primary_key, source.source_id
            continue

        assert contract.status == "active", source.source_id
        assert set(source.quality.required_fields) <= set(contract.required_columns), (
            source.source_id,
            source.quality.required_fields,
            contract.required_columns,
        )
        if source.quality.unique_fields:
            assert set(source.quality.unique_fields) == set(contract.primary_key), (
                source.source_id,
                source.quality.unique_fields,
                contract.primary_key,
            )
