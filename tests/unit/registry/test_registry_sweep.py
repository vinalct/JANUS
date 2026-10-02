"""Whole-registry guards for the checked-in source tree.

The cheapest defence against a future incremental source landing keyless: load every config
under ``conf/sources/**`` and assert the tree validates clean, then assert every incremental
source's contract declares the ``primaryKey`` its upsert write path merges on.

"Validates" means structure and meaning: ``load_registry`` runs the registry's semantic pass
(``janus.registry.semantics``), so this sweep rejects exactly what the planner, ``run-all`` and
``janus validate`` reject. Each test makes exactly one ``load_registry`` call;
no rule is re-run or restated here.
"""

from __future__ import annotations

from pathlib import Path

from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
