"""Whole-registry guards for the checked-in source tree.

The cheapest defence against a future incremental source landing keyless: load every config
under ``conf/sources/**`` and assert the tree validates clean, then assert every incremental
source's contract declares the ``primaryKey`` its upsert write path merges on.

"Validates" means structure and meaning: ``load_registry`` runs the registry's semantic pass
(``janus.registry.semantics``), so this sweep rejects exactly what the planner, ``run-all`` and
``janus validate`` reject. Each test makes exactly one ``load_registry`` call;
no rule is re-run or restated here. The third test pins an inventory, not a rule.
"""

from __future__ import annotations

from pathlib import Path

from janus.registry import load_registry, unverified_required_fields

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SERVIDORES_POR_ORGAO = "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"
SERVIDORES_POR_ORGAO_FIELDS = frozenset({"qntPessoas", "qntVinculos", "codOrgaoExercicioSiape"})

EXPECTED_UNVERIFIED: dict[str, tuple[str, ...]] = {}


def test_every_checked_in_source_validates() -> None:
    """load_registry validates every config, the graph and the semantic pass, or raises."""
    registry = load_registry(PROJECT_ROOT)

    assert len(registry.list_sources(enabled_only=False)) > 0


def test_every_incremental_source_declares_a_primary_key() -> None:
    registry = load_registry(PROJECT_ROOT)
    incremental = [
        source
        for source in registry.list_sources(enabled_only=False)
        if source.extraction.mode == "incremental"
    ]

    keyless_incremental = [
        source.source_id
        for source in incremental
        if not getattr(registry.contract_for(source.source_id), "primary_key", ())
    ]

    assert incremental, "expected at least one incremental source in conf/sources"
    assert not keyless_incremental, (
        "incremental sources need a primaryKey in their contract for idempotent upserts: "
        f"{keyless_incremental}"
    )


def test_no_source_declares_required_fields_it_cannot_verify() -> None:
    registry = load_registry(PROJECT_ROOT)
    sources = registry.list_sources(enabled_only=False)

    unverified = dict(unverified_required_fields(sources, contracts=registry.contracts))
    declaring_servidores_fields: list[str] = []
    for source in sources:
        contract = registry.contract_for(source.source_id)
        if contract is None:
            continue
        if contract.status == "draft" and contract.required_columns:
            unverified[source.source_id] = contract.required_columns
        if SERVIDORES_POR_ORGAO_FIELDS & set(contract.required_columns):
            declaring_servidores_fields.append(source.source_id)

    assert unverified == EXPECTED_UNVERIFIED, (
        f"required columns with no reviewed contract behind them (D-1): {unverified}"
    )
    assert declaring_servidores_fields == [SERVIDORES_POR_ORGAO], (
        "only the servidores-por-órgão endpoint serves "
        f"{sorted(SERVIDORES_POR_ORGAO_FIELDS)}: {declaring_servidores_fields}"
    )
