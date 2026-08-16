from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

WRAPPER_CONTAINER_KEYS = ("result", "data", "payload", "response", "value")
GENERIC_COLLECTION_KEYS = ("results", "items")
ENTITY_FILE_NAMES = {
    "organization": "organizations",
    "group": "groups",
    "dataset": "datasets",
    "resource": "resources",
}
ENTITY_TYPE_ORDER = ("organization", "group", "dataset", "resource")
ROOT_ENTITY_PRIORITY = {
    "metadata_catalog": ("dataset", "organization", "group", "resource"),
    "resource_catalog": ("resource", "dataset", "organization", "group"),
}
COLLECTION_ALIASES = {
    "organization": ("organizations", "organization", "organization_list"),
    "group": ("groups", "group", "group_list"),
    "dataset": ("datasets", "dataset", "packages", "package"),
    "resource": ("resources", "resource"),
}
IDENTIFIER_KEYS = ("id", "identifier", "name", "slug", "guid", "uri", "url")
NAME_KEYS = ("name", "display_name", "label")
TITLE_KEYS = ("title", "display_name", "label", "name")
DESCRIPTION_KEYS = ("description", "notes", "summary", "excerpt")
URL_KEYS = ("url", "uri", "homepage", "download_url")
FORMAT_KEYS = ("format", "mimetype", "media_type", "type")
CREATED_AT_KEYS = ("created", "metadata_created", "issued", "publication_date")
UPDATED_AT_KEYS = (
    "updated",
    "metadata_modified",
    "modified",
    "revision_timestamp",
    "last_updated",
)
STATE_KEYS = ("state", "status", "visibility")
DATASET_HINT_KEYS = (
    "resources",
    "organization",
    "groups",
    "metadata_created",
    "metadata_modified",
    "owner_org",
    "notes",
    "private",
)
RESOURCE_HINT_KEYS = ("format", "mimetype", "download_url", "url_type", "resource_type")
ORGANIZATION_HINT_KEYS = ("packages", "dataset_count", "package_count", "image_url")
GROUP_HINT_KEYS = ("packages", "dataset_count", "package_count")
CATALOG_NODES_FILE = "catalog_nodes"
CATALOG_EDGES_FILE = "catalog_edges"
ENTITY_HINT_KEYS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "dataset": DATASET_HINT_KEYS,
    "resource": RESOURCE_HINT_KEYS,
    "organization": ORGANIZATION_HINT_KEYS,
    "group": GROUP_HINT_KEYS,
}
UNKNOWN_ENTITY_TYPE = "unknown"

HIGH_CONFIDENCE_SIGNAL_COUNT = 2


@dataclass(frozen=True, slots=True)
class CatalogEntityReference:
    """Minimal parent reference attached to nested catalog records."""

    entity_type: str
    entity_key: str
    entity_id: str | None = None
    entity_name: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogBatch:
    """One entity collection discovered in a catalog payload."""

    entity_type: str
    records: tuple[dict[str, Any], ...]
    collection_path: str

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_FILE_NAMES:
            allowed = ", ".join(sorted(ENTITY_FILE_NAMES))
            raise ValueError(f"entity_type must be one of: {allowed}")
        if not self.collection_path.strip():
            raise ValueError("collection_path must not be empty")


@dataclass(frozen=True, slots=True)
class DocumentNode:
    """Structural node discovered during generic JSON document traversal."""

    path: str
    collection_path: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CatalogParseSummary:
    """Parse quality metrics for one catalog extraction run."""

    parse_status: str
    root_node_count: int
    node_count: int
    edge_count: int
    classified_node_count: int
    unknown_node_count: int
    parse_warning_count: int


@dataclass(frozen=True, slots=True)
class NodeClassification:
    """Inferred entity role for a generic catalog node."""

    entity_type_guess: str
    classification_confidence: str
    classification_signals: tuple[str, ...]


def walk_document(payload: Any, *, path: str = "payload") -> tuple[DocumentNode, ...]:
    """Walk any JSON payload and return structural document nodes depth-first, deterministically."""
    nodes: list[DocumentNode] = []
    _collect_document_nodes(payload, path=path, collection_path=path, nodes=nodes)
    return tuple(nodes)


def _collect_document_nodes(
    value: Any,
    *,
    path: str,
    collection_path: str,
    nodes: list[DocumentNode],
) -> None:
    if isinstance(value, Mapping):
        record = _normalize_mapping(value)
        nodes.append(DocumentNode(path=path, collection_path=collection_path, data=record))
        for key in sorted(str(k) for k in value):
            child = value[key]
            if isinstance(child, Sequence) and not isinstance(child, str | bytes | bytearray):
                child_path = f"{path}.{key}"
                for index, item in enumerate(child):
                    if isinstance(item, Mapping):
                        _collect_document_nodes(
                            item,
                            path=f"{child_path}[{index}]",
                            collection_path=child_path,
                            nodes=nodes,
                        )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                _collect_document_nodes(
                    item,
                    path=f"{path}[{index}]",
                    collection_path=path,
                    nodes=nodes,
                )


def _coerce_records(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping):
        return (_normalize_mapping(value),)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return ()

    records: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            records.append(_normalize_mapping(item))
            continue
        if item is None:
            continue
        records.append({"value": item, "name": str(item)})
    return tuple(records)


def _normalize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def _root_batches(
    payload: Any,
    variant: str,
    *,
    path: str,
) -> tuple[CatalogBatch, ...]:
    if isinstance(payload, Mapping):
        direct_batches = _batches_from_mapping(
            payload, variant=variant, parent_type=None, path=path
        )
        if direct_batches:
            return direct_batches

        for key in WRAPPER_CONTAINER_KEYS:
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                nested_batches = _root_batches(
                    nested,
                    variant,
                    path=f"{path}.{key}",
                )
                if nested_batches:
                    return nested_batches

        root_nodes = [
            n for n in walk_document(payload, path=path)
            if n.collection_path == path and n.data
        ]
        if root_nodes:
            entity_type = _infer_entity_type(
                [n.data for n in root_nodes[:5]], variant=variant, parent_type=None
            )
            return (
                CatalogBatch(
                    entity_type=entity_type,
                    records=tuple(n.data for n in root_nodes),
                    collection_path=path,
                ),
            )
        return ()

    records = _coerce_records(payload)
    if not records:
        return ()
    entity_type = _infer_entity_type(records, variant=variant, parent_type=None)
    return (CatalogBatch(entity_type=entity_type, records=records, collection_path=path),)


def _nested_batches(
    record: Mapping[str, Any],
    *,
    variant: str,
    parent_type: str,
    path: str,
) -> tuple[CatalogBatch, ...]:
    direct_batches = _batches_from_mapping(
        record, variant=variant, parent_type=parent_type, path=path
    )
    if direct_batches:
        return direct_batches

    nested_batches: list[CatalogBatch] = []
    for key in WRAPPER_CONTAINER_KEYS:
        nested = record.get(key)
        if not isinstance(nested, Mapping):
            continue
        nested_batches.extend(
            _batches_from_mapping(
                nested,
                variant=variant,
                parent_type=parent_type,
                path=f"{path}.{key}",
            )
        )
    return tuple(nested_batches)


def _batches_from_mapping(
    payload: Mapping[str, Any],
    *,
    variant: str,
    parent_type: str | None,
    path: str,
) -> tuple[CatalogBatch, ...]:
    batches: list[CatalogBatch] = []
    for entity_type in ROOT_ENTITY_PRIORITY.get(variant, ROOT_ENTITY_PRIORITY["metadata_catalog"]):
        for alias in COLLECTION_ALIASES[entity_type]:
            if alias not in payload:
                continue
            records = _coerce_records(payload[alias])
            if not records:
                continue
            batches.append(
                CatalogBatch(
                    entity_type=entity_type,
                    records=records,
                    collection_path=f"{path}.{alias}",
                )
            )
            break

    if batches:
        return tuple(batches)

    for alias in GENERIC_COLLECTION_KEYS:
        if alias not in payload:
            continue
        records = _coerce_records(payload[alias])
        if not records:
            continue
        entity_type = _infer_entity_type(records, variant=variant, parent_type=parent_type)
        return (
            CatalogBatch(
                entity_type=entity_type,
                records=records,
                collection_path=f"{path}.{alias}",
            ),
        )
    return ()


def _infer_entity_type(
    records: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    parent_type: str | None,
) -> str:
    if parent_type == "dataset":
        return "resource"
    if parent_type in {"organization", "group"}:
        return "dataset"

    scores = {
        "organization": 0,
        "group": 0,
        "dataset": 0,
        "resource": 0,
    }
    for record in records[:5]:
        scores["dataset"] += _score_record(record, DATASET_HINT_KEYS)
        scores["resource"] += _score_record(record, RESOURCE_HINT_KEYS)
        scores["organization"] += _score_record(record, ORGANIZATION_HINT_KEYS)
        scores["group"] += _score_record(record, GROUP_HINT_KEYS)

    highest_score = max(scores.values())
    priority = ROOT_ENTITY_PRIORITY.get(variant, ROOT_ENTITY_PRIORITY["metadata_catalog"])
    if highest_score > 0:
        return max(
            priority,
            key=lambda entity_type: (scores[entity_type], -priority.index(entity_type)),
        )
    return priority[0]


def _score_record(record: Mapping[str, Any], keys: Sequence[str]) -> int:
    return sum(1 for key in keys if key in record and record.get(key) not in (None, "", (), [], {}))


def _first_string(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _build_entity_reference(
    entity_type: str,
    record: Mapping[str, Any],
    record_path: str,
) -> CatalogEntityReference:
    entity_id = _first_string(record, IDENTIFIER_KEYS)
    entity_name = _first_string(record, NAME_KEYS) or _first_string(record, TITLE_KEYS)
    entity_key = entity_id or entity_name or record_path
    return CatalogEntityReference(
        entity_type=entity_type,
        entity_key=entity_key,
        entity_id=entity_id,
        entity_name=entity_name,
    )


def classify_catalog_node(
    payload: Mapping[str, Any],
    *,
    variant: str,
    parent_type: str | None = None,
) -> NodeClassification:
    """Score payload signals and return an inferred role with explicit confidence.

    Returns ``unknown`` when no hint keys match and no structural parent context is available.
    Classification is deterministic for the same payload, variant, and parent type.
    """
    if parent_type == "dataset":
        signals = _matched_signals(payload, RESOURCE_HINT_KEYS)
        confidence = (
            "high" if len(signals) >= HIGH_CONFIDENCE_SIGNAL_COUNT else "heuristic"
        )
        return NodeClassification("resource", confidence, signals or (f"parent:{parent_type}",))

    if parent_type in {"organization", "group"}:
        signals = _matched_signals(payload, DATASET_HINT_KEYS)
        confidence = (
            "high" if len(signals) >= HIGH_CONFIDENCE_SIGNAL_COUNT else "heuristic"
        )
        return NodeClassification("dataset", confidence, signals or (f"parent:{parent_type}",))

    priority = ROOT_ENTITY_PRIORITY.get(variant, ROOT_ENTITY_PRIORITY["metadata_catalog"])
    all_signals = {
        etype: _matched_signals(payload, ENTITY_HINT_KEYS_BY_TYPE[etype])
        for etype in priority
    }
    all_scores = {etype: len(sigs) for etype, sigs in all_signals.items()}

    best_score = max(all_scores.values())
    if best_score == 0:
        return NodeClassification(UNKNOWN_ENTITY_TYPE, "low", ())

    best_type = max(priority, key=lambda t: (all_scores[t], -priority.index(t)))
    signals = all_signals[best_type]
    confidence = "high" if best_score >= HIGH_CONFIDENCE_SIGNAL_COUNT else "heuristic"
    return NodeClassification(best_type, confidence, signals)


def _matched_signals(payload: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(k for k in keys if k in payload and payload.get(k) not in (None, "", (), [], {}))


def _classification_confidence(entity_type: str, payload: Mapping[str, Any]) -> str:
    hint_keys = ENTITY_HINT_KEYS_BY_TYPE.get(entity_type, ())
    score = _score_record(payload, hint_keys)
    if score >= HIGH_CONFIDENCE_SIGNAL_COUNT:
        return "high"
    if score >= 1:
        return "heuristic"
    return "low"


def _payload_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _build_generic_catalog_node(
    entity_record: Mapping[str, Any],
    *,
    parent_node_path: str | None,
    variant: str,
) -> dict[str, Any]:
    payload = entity_record.get("payload") or {}
    parent_type = entity_record.get("parent_entity_type")
    classification = classify_catalog_node(payload, variant=variant, parent_type=parent_type)
    return {
        "node_key": entity_record["entity_key"],
        "node_path": entity_record["catalog_record_path"],
        "parent_node_key": entity_record["parent_entity_key"],
        "parent_node_path": parent_node_path,
        "root_document_key": entity_record["catalog_raw_artifact_path"],
        "payload": payload,
        "payload_hash": _payload_hash(payload),
        "request_url": entity_record["catalog_request_url"],
        "raw_artifact_path": entity_record["catalog_raw_artifact_path"],
        "discovered_at": entity_record["catalog_received_at"],
        "entity_type_guess": classification.entity_type_guess,
        "classification_confidence": classification.classification_confidence,
        "classification_signals": list(classification.classification_signals),
        "parse_status": "ok",
    }


def _build_generic_catalog_edge(entity_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_node_key": entity_record["parent_entity_key"],
        "child_node_key": entity_record["entity_key"],
        "relationship_type": "contains",
        "raw_artifact_path": entity_record["catalog_raw_artifact_path"],
        "discovered_at": entity_record["catalog_received_at"],
    }


def _compute_parse_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> CatalogParseSummary:
    node_count = len(nodes)
    edge_count = len(edges)
    root_node_count = sum(1 for n in nodes if not n.get("parent_node_key"))
    classified_node_count = sum(
        1 for n in nodes if n.get("classification_confidence") in ("high", "heuristic")
    )
    unknown_node_count = node_count - classified_node_count
    parse_warning_count = unknown_node_count

    if node_count == 0:
        parse_status = "empty"
    elif classified_node_count == 0:
        parse_status = "ambiguous"
    elif unknown_node_count > 0:
        parse_status = "parsed_with_warnings"
    else:
        parse_status = "parsed"

    return CatalogParseSummary(
        parse_status=parse_status,
        root_node_count=root_node_count,
        node_count=node_count,
        edge_count=edge_count,
        classified_node_count=classified_node_count,
        unknown_node_count=unknown_node_count,
        parse_warning_count=parse_warning_count,
    )
