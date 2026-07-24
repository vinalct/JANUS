"""Whole-registry guards for the checked-in source tree.

The cheapest defence against a future incremental source landing keyless: load every config
under ``conf/sources/**`` and assert the tree validates clean, then assert every incremental
source declares the idempotency keys its upsert write path depends on.
"""

from __future__ import annotations

from pathlib import Path

from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_every_checked_in_source_validates() -> None:
    """load_registry validates every discovered config; a broken one would raise here."""
    registry = load_registry(PROJECT_ROOT)

    assert len(registry.list_sources(enabled_only=False)) > 0


def test_every_incremental_source_declares_unique_fields() -> None:
    registry = load_registry(PROJECT_ROOT)

    keyless_incremental = [
        source.source_id
        for source in registry.list_sources(enabled_only=False)
        if source.extraction.mode == "incremental" and not source.quality.unique_fields
    ]

    assert not keyless_incremental, (
        "incremental sources must declare quality.unique_fields for idempotent upserts: "
        f"{keyless_incremental}"
    )
