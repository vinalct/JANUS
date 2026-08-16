from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.catalog.artifacts import _persist_generic_artifacts
from janus.strategies.catalog.document import (
    CATALOG_EDGES_FILE,
    CATALOG_NODES_FILE,
    CatalogBatch,
    CatalogParseSummary,
    DocumentNode,
    NodeClassification,
    _build_generic_catalog_edge,
    _build_generic_catalog_node,
    _classification_confidence,
    _compute_parse_summary,
    _payload_hash,
    _root_batches,
    classify_catalog_node,
    walk_document,
)
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

# --- walk_document ---


def test_walk_document_list_style_page():
    """List-style catalog pages: root array of objects yields one node per item."""
    payload = [{"id": "1", "title": "A"}, {"id": "2", "title": "B"}]
    nodes = walk_document(payload, path="payload")
    assert len(nodes) == 2
    assert nodes[0] == DocumentNode(
        path="payload[0]",
        collection_path="payload",
        data={"id": "1", "title": "A"},
    )
    assert nodes[1] == DocumentNode(
        path="payload[1]",
        collection_path="payload",
        data={"id": "2", "title": "B"},
    )


def test_walk_document_single_object_detail():
    """Single-object detail payloads: root mapping yields one node with the root path."""
    payload = {"id": "ds-1", "title": "Dataset One", "notes": "a description"}
    nodes = walk_document(payload, path="payload")
    assert len(nodes) == 1
    assert nodes[0].path == "payload"
    assert nodes[0].collection_path == "payload"
    assert nodes[0].data == payload


def test_walk_document_wrapped_payload():
    """Wrapped payloads: walker descends into inner object and yields it."""
    payload = {"id": "ds-1", "title": "Dataset One"}
    nodes = walk_document({"data": [payload]}, path="payload")
    assert len(nodes) == 2
    assert nodes[0].path == "payload"
    assert nodes[0].collection_path == "payload"
    assert nodes[1].path == "payload.data[0]"
    assert nodes[1].collection_path == "payload.data"
    assert nodes[1].data == payload


def test_walk_document_nested_arrays_of_child_objects():
    """Nested arrays: walker yields root node then each child array item."""
    payload = {
        "id": "ds-1",
        "resources": [
            {"id": "r-1", "format": "CSV"},
            {"id": "r-2", "format": "JSON"},
        ],
    }
    nodes = walk_document(payload, path="payload")
    assert len(nodes) == 3
    assert nodes[0].path == "payload"
    assert nodes[0].collection_path == "payload"
    assert nodes[1].path == "payload.resources[0]"
    assert nodes[1].collection_path == "payload.resources"
    assert nodes[2].path == "payload.resources[1]"
    assert nodes[2].collection_path == "payload.resources"


def test_walk_document_deterministic_key_order():
    """Keys are sorted so traversal order is deterministic regardless of dict insertion order."""
    payload = {"z": [{"id": "z-child"}], "a": [{"id": "a-child"}]}
    nodes = walk_document(payload, path="payload")
    paths = [n.path for n in nodes]
    assert paths == ["payload", "payload.a[0]", "payload.z[0]"]


def test_walk_document_empty_mapping():
    """Empty mapping yields one node with empty data."""
    nodes = walk_document({}, path="payload")
    assert len(nodes) == 1
    assert nodes[0].data == {}


def test_walk_document_non_mapping_array_items_skipped():
    """Non-mapping items in arrays (scalars, None) are skipped by the walker."""
    payload = [{"id": "1"}, None, "scalar", 42]
    nodes = walk_document(payload, path="payload")
    assert len(nodes) == 1
    assert nodes[0].data == {"id": "1"}


# --- _root_batches fallback integration ---


def test_root_batches_single_object_detail_returns_batch():
    """Single-object detail endpoint: walker fallback emits one batch from root object."""
    payload = {
        "id": "ds-1",
        "title": "Dataset One",
        "notes": "some notes",
        "metadata_created": "2026-01-01T00:00:00Z",
        "resources": [],
        "owner_org": "org-1",
    }
    batches = _root_batches(payload, "metadata_catalog", path="payload")
    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch, CatalogBatch)
    assert batch.collection_path == "payload"
    assert len(batch.records) == 1
    assert batch.records[0]["id"] == "ds-1"


def test_root_batches_wrapped_detail_endpoint_returns_batch():
    """Wrapper around a detail object: falls through wrapper check into walker fallback."""
    payload = {"result": {"id": "ds-2", "title": "Dataset Two", "owner_org": "org-1"}}
    batches = _root_batches(payload, "metadata_catalog", path="payload")
    assert len(batches) == 1
    assert batches[0].records[0]["id"] == "ds-2"


def test_root_batches_existing_ckan_list_still_works():
    """List-style CKAN payloads continue to use the alias-based path, not the walker."""
    payload = {"result": {"results": [{"id": "d-1"}, {"id": "d-2"}]}}
    batches = _root_batches(payload, "metadata_catalog", path="payload")
    assert len(batches) == 1
    assert batches[0].collection_path == "payload.result.results"
    assert len(batches[0].records) == 2


def test_root_batches_empty_mapping_returns_empty():
    """Empty root mapping produces no batches (walker guard on non-empty data)."""
    batches = _root_batches({}, "metadata_catalog", path="payload")
    assert batches == ()


# --- generic catalog node/edge helpers ---

_SAMPLE_ENTITY_RECORD = {
    "entity_type": "dataset",
    "entity_key": "ds-1",
    "entity_id": "ds-1",
    "parent_entity_type": None,
    "parent_entity_key": None,
    "parent_entity_id": None,
    "catalog_collection_path": "payload.result.results",
    "catalog_record_path": "payload.result.results[0]",
    "catalog_request_url": "https://example.invalid/catalog?page=1",
    "catalog_request_index": 1,
    "catalog_page_number": 1,
    "catalog_offset": None,
    "catalog_cursor": None,
    "catalog_received_at": "2026-04-10T12:00:00+00:00",
    "catalog_raw_artifact_path": "/data/raw/page-0001.json",
    "payload": {"id": "ds-1", "title": "Dataset One", "resources": [], "owner_org": "org-1"},
}

_SAMPLE_CHILD_RECORD = {
    **_SAMPLE_ENTITY_RECORD,
    "entity_type": "resource",
    "entity_key": "res-1",
    "entity_id": "res-1",
    "parent_entity_type": "dataset",
    "parent_entity_key": "ds-1",
    "parent_entity_id": "ds-1",
    "catalog_collection_path": "payload.result.results[0].resources",
    "catalog_record_path": "payload.result.results[0].resources[0]",
    "payload": {"id": "res-1", "format": "CSV", "url": "https://example.invalid/f.csv"},
}


def test_build_generic_catalog_node_has_all_required_fields():
    """Generic node contains every field from the handoff spec including classification_signals."""
    node = _build_generic_catalog_node(
        _SAMPLE_ENTITY_RECORD,
        parent_node_path=None,
        variant="metadata_catalog",
    )
    required_fields = {
        "node_key", "node_path", "parent_node_key", "parent_node_path",
        "root_document_key", "payload", "payload_hash", "request_url",
        "raw_artifact_path", "discovered_at", "entity_type_guess",
        "classification_confidence", "classification_signals", "parse_status",
    }
    assert required_fields <= set(node)
    assert node["node_key"] == "ds-1"
    assert node["node_path"] == "payload.result.results[0]"
    assert node["parent_node_key"] is None
    assert node["parent_node_path"] is None
    assert node["root_document_key"] == "/data/raw/page-0001.json"
    assert node["payload"] == _SAMPLE_ENTITY_RECORD["payload"]
    assert node["request_url"] == "https://example.invalid/catalog?page=1"
    assert node["raw_artifact_path"] == "/data/raw/page-0001.json"
    assert node["discovered_at"] == "2026-04-10T12:00:00+00:00"
    assert node["entity_type_guess"] == "dataset"
    assert isinstance(node["classification_signals"], list)
    assert node["parse_status"] == "ok"


def test_build_generic_catalog_node_resolves_parent_node_path():
    """parent_node_path is passed through from the lookup table, not re-derived."""
    node = _build_generic_catalog_node(
        _SAMPLE_CHILD_RECORD,
        parent_node_path="payload.result.results[0]",
        variant="metadata_catalog",
    )
    assert node["parent_node_key"] == "ds-1"
    assert node["parent_node_path"] == "payload.result.results[0]"


def test_build_generic_catalog_edge_has_correct_fields():
    """Edge record links parent and child node keys."""
    edge = _build_generic_catalog_edge(_SAMPLE_CHILD_RECORD)
    assert edge["parent_node_key"] == "ds-1"
    assert edge["child_node_key"] == "res-1"
    assert edge["relationship_type"] == "contains"
    assert edge["raw_artifact_path"] == "/data/raw/page-0001.json"
    assert edge["discovered_at"] == "2026-04-10T12:00:00+00:00"


def test_payload_hash_is_deterministic():
    """Same payload always produces the same hash regardless of dict insertion order."""
    payload_a = {"id": "1", "title": "T", "notes": "N"}
    payload_b = {"notes": "N", "id": "1", "title": "T"}
    assert _payload_hash(payload_a) == _payload_hash(payload_b)


def test_payload_hash_differs_for_different_payloads():
    """Different payloads produce different hashes."""
    assert _payload_hash({"id": "1"}) != _payload_hash({"id": "2"})


def test_classification_confidence_high_for_strong_hints():
    """Record with ≥2 strong hint keys for its entity type → 'high' confidence."""
    dataset_payload = {"resources": [...], "owner_org": "org-1", "metadata_created": "2026-01-01"}
    assert _classification_confidence("dataset", dataset_payload) == "high"


def test_classification_confidence_heuristic_for_single_hint():
    """Record with exactly 1 matching hint key → 'heuristic' confidence."""
    resource_payload = {"format": "CSV"}
    assert _classification_confidence("resource", resource_payload) == "heuristic"


def test_classification_confidence_low_for_no_hints():
    """Record with no matching hint keys → 'low' confidence."""
    bare_payload = {"id": "x", "title": "X"}
    assert _classification_confidence("dataset", bare_payload) == "low"


def _build_plan_for_generic_tests(tmp_path: Path) -> ExecutionPlan:
    source_config = SourceConfig.from_mapping(
        {
            "source_id": "test_generic",
            "name": "test_generic",
            "owner": "janus",
            "enabled": True,
            "source_type": "catalog",
            "strategy": "catalog",
            "strategy_variant": "metadata_catalog",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "base_url": "https://example.invalid",
                "path": "/catalog",
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "pagination": {"type": "none"},
                "rate_limit": {"requests_per_minute": 10, "concurrency": 1, "backoff_seconds": 1},
            },
            "extraction": {
                "mode": "full_refresh",
                "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "jsonl", "write_mode": "append"},
            "outputs": {
                "raw": {"path": "data/raw/example/test_generic", "format": "json"},
                "bronze": {"path": "data/bronze/example/test_generic", "format": "iceberg"},
                "metadata": {"path": "data/metadata/example/test_generic", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / "test_generic.yaml",
    )
    run_context = RunContext.create(
        run_id="run-test-generic",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "runtime",
                "raw_dir": "runtime/raw",
                "bronze_dir": "runtime/bronze",
                "metadata_dir": "runtime/metadata",
            }
        },
        tmp_path,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_persist_generic_artifacts_collection_page_emits_nodes_and_edges(tmp_path):
    """Collection with parent-child entities emits catalog node and edge artifacts."""
    plan = _build_plan_for_generic_tests(tmp_path)
    raw_writer = RawArtifactWriter(_storage_layout(tmp_path))
    normalized_records = {
        "organization": [],
        "group": [],
        "dataset": [dict(_SAMPLE_ENTITY_RECORD)],
        "resource": [dict(_SAMPLE_CHILD_RECORD)],
    }

    artifacts, _ = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    artifact_names = [Path(a.path).name for a in artifacts]

    assert f"{CATALOG_NODES_FILE}.jsonl" in artifact_names
    assert f"{CATALOG_EDGES_FILE}.jsonl" in artifact_names

    nodes_path = next(
        Path(a.path)
        for a in artifacts
        if Path(a.path).name == f"{CATALOG_NODES_FILE}.jsonl"
    )
    edges_path = next(
        Path(a.path)
        for a in artifacts
        if Path(a.path).name == f"{CATALOG_EDGES_FILE}.jsonl"
    )

    nodes = _read_jsonl(nodes_path)
    edges = _read_jsonl(edges_path)

    assert len(nodes) == 2
    assert {n["node_key"] for n in nodes} == {"ds-1", "res-1"}
    assert all(n["parse_status"] == "ok" for n in nodes)
    assert all("payload_hash" in n for n in nodes)

    assert len(edges) == 1
    assert edges[0]["parent_node_key"] == "ds-1"
    assert edges[0]["child_node_key"] == "res-1"
    assert edges[0]["relationship_type"] == "contains"


def test_persist_generic_artifacts_detail_document_emits_node_no_edge(tmp_path):
    """Single root-level entity emits a node artifact without edge artifacts."""
    plan = _build_plan_for_generic_tests(tmp_path)
    raw_writer = RawArtifactWriter(_storage_layout(tmp_path))
    normalized_records = {
        "organization": [],
        "group": [],
        "dataset": [dict(_SAMPLE_ENTITY_RECORD)],
        "resource": [],
    }

    artifacts, _ = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    artifact_names = [Path(a.path).name for a in artifacts]

    assert f"{CATALOG_NODES_FILE}.jsonl" in artifact_names
    assert f"{CATALOG_EDGES_FILE}.jsonl" not in artifact_names

    nodes_path = next(
        Path(a.path)
        for a in artifacts
        if Path(a.path).name == f"{CATALOG_NODES_FILE}.jsonl"
    )
    nodes = _read_jsonl(nodes_path)

    assert len(nodes) == 1
    assert nodes[0]["node_key"] == "ds-1"
    assert nodes[0]["parent_node_key"] is None


def test_persist_generic_artifacts_parent_node_path_resolved_from_sibling_record(tmp_path):
    """parent_node_path in a child node is resolved from the parent record's catalog_record_path."""
    plan = _build_plan_for_generic_tests(tmp_path)
    raw_writer = RawArtifactWriter(_storage_layout(tmp_path))
    normalized_records = {
        "organization": [],
        "group": [],
        "dataset": [dict(_SAMPLE_ENTITY_RECORD)],
        "resource": [dict(_SAMPLE_CHILD_RECORD)],
    }

    artifacts, _ = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    nodes_path = next(
        Path(a.path)
        for a in artifacts
        if Path(a.path).name == f"{CATALOG_NODES_FILE}.jsonl"
    )
    nodes = _read_jsonl(nodes_path)

    resource_node = next(n for n in nodes if n["node_key"] == "res-1")
    assert resource_node["parent_node_path"] == "payload.result.results[0]"


# --- _compute_parse_summary ---


def _make_node(*, confidence: str, parent_key: str | None = None) -> dict:
    return {
        "node_key": "k",
        "parent_node_key": parent_key,
        "classification_confidence": confidence,
    }


def test_parse_summary_empty_when_no_nodes():
    summary = _compute_parse_summary([], [])
    assert isinstance(summary, CatalogParseSummary)
    assert summary.parse_status == "empty"
    assert summary.node_count == 0
    assert summary.edge_count == 0
    assert summary.classified_node_count == 0
    assert summary.unknown_node_count == 0
    assert summary.parse_warning_count == 0
    assert summary.root_node_count == 0


def test_parse_summary_ambiguous_when_all_nodes_unknown():
    nodes = [_make_node(confidence="low"), _make_node(confidence="low")]
    summary = _compute_parse_summary(nodes, [])
    assert summary.parse_status == "ambiguous"
    assert summary.node_count == 2
    assert summary.classified_node_count == 0
    assert summary.unknown_node_count == 2
    assert summary.parse_warning_count == 2


def test_parse_summary_parsed_when_all_nodes_classified():
    nodes = [
        _make_node(confidence="high"),
        _make_node(confidence="heuristic"),
    ]
    summary = _compute_parse_summary(nodes, [])
    assert summary.parse_status == "parsed"
    assert summary.classified_node_count == 2
    assert summary.unknown_node_count == 0
    assert summary.parse_warning_count == 0


def test_parse_summary_parsed_with_warnings_for_mixed_nodes():
    nodes = [
        _make_node(confidence="high"),
        _make_node(confidence="low"),
    ]
    summary = _compute_parse_summary(nodes, [{"x": 1}])
    assert summary.parse_status == "parsed_with_warnings"
    assert summary.node_count == 2
    assert summary.classified_node_count == 1
    assert summary.unknown_node_count == 1
    assert summary.parse_warning_count == 1
    assert summary.edge_count == 1


def test_parse_summary_root_node_count_excludes_children():
    nodes = [
        _make_node(confidence="high", parent_key=None),
        _make_node(confidence="high", parent_key="parent-key"),
    ]
    summary = _compute_parse_summary(nodes, [])
    assert summary.root_node_count == 1
    assert summary.node_count == 2


def test_persist_generic_artifacts_deterministic_across_reruns(tmp_path):
    """Calling _persist_generic_artifacts twice with identical input produces identical hashes."""
    plan = _build_plan_for_generic_tests(tmp_path)
    raw_writer = RawArtifactWriter(_storage_layout(tmp_path))
    normalized_records = {
        "organization": [],
        "group": [],
        "dataset": [dict(_SAMPLE_ENTITY_RECORD)],
        "resource": [dict(_SAMPLE_CHILD_RECORD)],
    }

    artifacts1, _ = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    nodes1 = _read_jsonl(
        next(Path(a.path) for a in artifacts1 if Path(a.path).name == f"{CATALOG_NODES_FILE}.jsonl")
    )

    artifacts2, _ = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    nodes2 = _read_jsonl(
        next(Path(a.path) for a in artifacts2 if Path(a.path).name == f"{CATALOG_NODES_FILE}.jsonl")
    )

    hashes1 = [n["payload_hash"] for n in nodes1]
    hashes2 = [n["payload_hash"] for n in nodes2]
    assert hashes1 == hashes2


# --- classify_catalog_node ---


def test_classify_catalog_node_strong_dataset_returns_high_confidence():
    """Payload with ≥2 dataset hint keys → dataset / high confidence."""
    payload = {"resources": [{"id": "r1"}], "owner_org": "org-1", "metadata_created": "2026-01-01"}
    result = classify_catalog_node(payload, variant="metadata_catalog")
    assert isinstance(result, NodeClassification)
    assert result.entity_type_guess == "dataset"
    assert result.classification_confidence == "high"
    assert set(result.classification_signals) >= {"owner_org", "metadata_created"}


def test_classify_catalog_node_single_resource_hint_is_heuristic():
    """Payload with exactly 1 resource hint key → resource / heuristic confidence."""
    payload = {"format": "CSV", "id": "r-1"}
    result = classify_catalog_node(payload, variant="metadata_catalog")
    assert result.entity_type_guess == "resource"
    assert result.classification_confidence == "heuristic"
    assert "format" in result.classification_signals


def test_classify_catalog_node_unknown_when_no_hints():
    """No hint keys present → unknown entity with low confidence and empty signals."""
    payload = {"id": "x", "title": "Something Generic"}
    result = classify_catalog_node(payload, variant="metadata_catalog")
    assert result.entity_type_guess == "unknown"
    assert result.classification_confidence == "low"
    assert result.classification_signals == ()


def test_classify_catalog_node_parent_dataset_implies_resource_no_payload_hints():
    """Parent type 'dataset' → structural inference of resource even without payload hints."""
    payload = {"id": "r-1"}
    result = classify_catalog_node(payload, variant="metadata_catalog", parent_type="dataset")
    assert result.entity_type_guess == "resource"
    assert result.classification_confidence == "heuristic"
    assert result.classification_signals == ("parent:dataset",)


def test_classify_catalog_node_parent_dataset_with_resource_hints_high_confidence():
    """Parent type 'dataset' + ≥2 resource hint keys → high confidence."""
    payload = {"format": "CSV", "url": "https://example.invalid/f.csv", "mimetype": "text/csv"}
    result = classify_catalog_node(payload, variant="metadata_catalog", parent_type="dataset")
    assert result.entity_type_guess == "resource"
    assert result.classification_confidence == "high"
    assert "format" in result.classification_signals


def test_classify_catalog_node_parent_organization_implies_dataset():
    """Parent type 'organization' → structural inference of dataset."""
    payload = {"id": "ds-1"}
    result = classify_catalog_node(payload, variant="metadata_catalog", parent_type="organization")
    assert result.entity_type_guess == "dataset"
    assert result.classification_confidence == "heuristic"


def test_classify_catalog_node_resource_catalog_variant_prefers_resource():
    """resource_catalog variant breaks ties in favour of resource."""
    payload = {"format": "JSON"}
    result = classify_catalog_node(payload, variant="resource_catalog")
    assert result.entity_type_guess == "resource"


def test_classify_catalog_node_deterministic():
    """Same payload + variant always produces the same NodeClassification."""
    payload = {"resources": [1, 2], "owner_org": "org-1", "notes": "desc"}
    r1 = classify_catalog_node(payload, variant="metadata_catalog")
    r2 = classify_catalog_node(payload, variant="metadata_catalog")
    assert r1 == r2


def test_classify_catalog_node_signals_present_in_generic_node(tmp_path):
    """classification_signals is serialised as a list in the catalog_nodes.jsonl artifact."""
    plan = _build_plan_for_generic_tests(tmp_path)
    raw_writer = RawArtifactWriter(_storage_layout(tmp_path))
    normalized_records = {
        "organization": [],
        "group": [],
        "dataset": [dict(_SAMPLE_ENTITY_RECORD)],
        "resource": [],
    }

    artifacts, _ = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    nodes_path = next(
        Path(a.path)
        for a in artifacts
        if Path(a.path).name == f"{CATALOG_NODES_FILE}.jsonl"
    )
    nodes = _read_jsonl(nodes_path)

    node = nodes[0]
    assert "classification_signals" in node
    assert isinstance(node["classification_signals"], list)


def test_classify_catalog_node_unknown_reflected_in_parse_summary(tmp_path):
    """Low-confidence unknown classification is counted in the parse summary."""
    plan = _build_plan_for_generic_tests(tmp_path)
    raw_writer = RawArtifactWriter(_storage_layout(tmp_path))
    bare_record = {
        **_SAMPLE_ENTITY_RECORD,
        "entity_type": "dataset",
        "payload": {"id": "x", "title": "bare"},
    }
    normalized_records = {
        "organization": [],
        "group": [],
        "dataset": [bare_record],
        "resource": [],
    }

    _, summary = _persist_generic_artifacts(plan, raw_writer, normalized_records)
    assert summary.unknown_node_count == 1
    assert summary.classified_node_count == 0
    assert summary.parse_status == "ambiguous"
