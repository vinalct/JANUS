"""Architectural guardrails for the shared catalog strategy boundary.

These tests enforce the rule: the shared catalog core must never branch on
source_id or emit source-specific field mappings. Source-local behavior belongs
exclusively in CatalogHook implementations.

If any of these tests break it means someone is trying to smuggle
one-source logic into the shared parsing path — the change should be
rejected in review and redirected to a bounded hook instead.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import janus.strategies.catalog.core as catalog_core
from janus.models import ExtractedArtifact
from janus.strategies.api import ApiRequest, ApiResponse
from janus.strategies.api.pagination import PaginationState
from janus.strategies.catalog.core import (
    CatalogHook,
    _normalize_catalog_record,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_core_ast() -> ast.Module:
    source_path = Path(inspect.getfile(catalog_core))
    return ast.parse(source_path.read_text(encoding="utf-8"))


def _compare_references_source_id(node: ast.Compare) -> bool:
    """True when any operand of a Compare node is a .source_id attribute access."""
    all_operands = [node.left, *node.comparators]
    return any(isinstance(n, ast.Attribute) and n.attr == "source_id" for n in all_operands)


# ── Guardrail: no source_id branching ────────────────────────────────────────


def test_catalog_core_has_no_source_id_comparisons():
    """Shared catalog core must not branch on source_id — that belongs in hooks.

    Any if/elif/match that reads source_id to select behavior for one specific
    source is a boundary violation. Move it to a CatalogHook subclass instead.
    """
    tree = _parse_core_ast()
    violations = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and _compare_references_source_id(node)
    ]
    assert not violations, (
        "source_id comparisons found in catalog/core.py — move to a CatalogHook:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ── Contract: normalized record schema is stable and generic ─────────────────

# These are the only fields the shared strategy is allowed to emit per record.
# Adding a source-specific field here is a boundary violation.
_NORMALIZED_RECORD_CONTRACT = frozenset(
    {
        # entity identity
        "entity_type",
        "entity_key",
        "entity_id",
        # parent reference
        "parent_entity_type",
        "parent_entity_key",
        "parent_entity_id",
        # provenance
        "catalog_collection_path",
        "catalog_record_path",
        "catalog_request_url",
        "catalog_request_index",
        "catalog_page_number",
        "catalog_offset",
        "catalog_cursor",
        "catalog_received_at",
        "catalog_raw_artifact_path",
        # full original payload (source-specific fields live HERE, not at root)
        "payload",
    }
)


def _fake_request() -> ApiRequest:
    return ApiRequest(
        method="GET",
        url="https://example.invalid/catalog",
        timeout_seconds=30,
        headers=(),
        params=(),
    )


def _fake_response(request: ApiRequest) -> ApiResponse:
    return ApiResponse(
        request=request,
        status_code=200,
        body=b"{}",
        received_at=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )


def _fake_pagination_state() -> PaginationState:
    return PaginationState(request_index=1, page_number=1)


def _fake_artifact() -> ExtractedArtifact:
    return ExtractedArtifact(
        path="/data/raw/page-0001.json",
        format="json",
        checksum="abc123",
    )


def test_normalized_catalog_record_emits_only_generic_fields():
    """_normalize_catalog_record must emit exactly the generic field contract — no extras.

    If this test fails because a new field was added, verify it belongs in the
    generic contract (useful for any source) before expanding _NORMALIZED_RECORD_CONTRACT.
    If the new field is source-specific, it must live inside payload instead.
    """
    request = _fake_request()
    response = _fake_response(request)
    record: dict[str, Any] = {
        "id": "ds-1",
        "title": "Dataset One",
        "notes": "A description",
        "metadata_created": "2026-01-01T00:00:00Z",
        "resources": [],
        "owner_org": "org-1",
    }
    result = _normalize_catalog_record(
        entity_type="dataset",
        record=record,
        collection_path="payload.results",
        record_path="payload.results[0]",
        request=request,
        response=response,
        pagination_state=_fake_pagination_state(),
        raw_artifact=_fake_artifact(),
        parent=None,
    )
    emitted_fields = frozenset(result.keys())
    extra_fields = emitted_fields - _NORMALIZED_RECORD_CONTRACT
    missing_fields = _NORMALIZED_RECORD_CONTRACT - emitted_fields
    assert not extra_fields, (
        "Unexpected source-specific fields in normalized record — move them into payload:\n"
        + "\n".join(f"  {f}" for f in sorted(extra_fields))
    )
    assert not missing_fields, (
        "Contract fields missing from normalized record — update _normalize_catalog_record:\n"
        + "\n".join(f"  {f}" for f in sorted(missing_fields))
    )


def test_normalized_catalog_record_payload_field_carries_original_data():
    """payload must be the full original record dict — the safe carrier for source fields."""
    request = _fake_request()
    response = _fake_response(request)
    record: dict[str, Any] = {
        "id": "ds-1",
        "title": "Dataset One",
        "custom_source_field": "should-be-here",
    }
    result = _normalize_catalog_record(
        entity_type="dataset",
        record=record,
        collection_path="payload",
        record_path="payload[0]",
        request=request,
        response=response,
        pagination_state=_fake_pagination_state(),
        raw_artifact=_fake_artifact(),
        parent=None,
    )
    assert result["payload"] == json.dumps(record, sort_keys=True, ensure_ascii=False)
    assert "custom_source_field" in result["payload"]
    assert "custom_source_field" not in result


# ── Boundary: CatalogHook exposes the projection surface ─────────────────────


def test_catalog_hook_exposes_transform_payload_for_source_local_shaping():
    """CatalogHook.transform_payload is the sanctioned entry point for
    source-local payload mutation."""
    hook = CatalogHook()
    assert callable(hook.transform_payload), "transform_payload must be callable"


def test_catalog_hook_exposes_page_records_for_source_local_counting():
    """CatalogHook.page_records is the sanctioned entry point for source-local record counting."""
    hook = CatalogHook()
    assert callable(hook.page_records), "page_records must be callable"


def test_catalog_hook_base_transform_payload_is_identity():
    """Default CatalogHook.transform_payload must be a no-op — sources opt in, not opt out."""
    hook = CatalogHook()
    sentinel: Any = {"some": "payload"}
    result = hook.transform_payload(None, None, None, sentinel)  # type: ignore[arg-type]
    assert result is sentinel


def test_catalog_hook_base_page_records_returns_none():
    """Default CatalogHook.page_records returns None — signals the strategy to
    use generic counting."""
    hook = CatalogHook()
    result = hook.page_records(None, None, None, None)  # type: ignore[arg-type]
    assert result is None
