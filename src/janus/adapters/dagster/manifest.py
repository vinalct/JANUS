"""Non-secret Dagster run manifest used by workers and terminal collectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from janus.lineage import compute_config_version
from janus.models.dependencies import SourceDependencyEdge
from janus.orchestration import SourceSelection
from janus.registry import SourceRegistry
from janus.utils.storage import StorageLayout

ADAPTER_TAG = "janus/adapter"
MANIFEST_TAG = "janus/manifest"
SNAPSHOT_TAG = "janus/snapshot_id"
ADAPTER_NAME = "dagster"
MANIFEST_SCHEMA_VERSION = 1
_PAIR_LENGTH = 2


@dataclass(frozen=True, slots=True)
class DagsterRunManifest:
    """Definition-time facts required to aggregate a run after its code reloads."""

    job_name: str
    environment: str
    project_root: Path
    requested_tags: tuple[str, ...]
    requested_domains: tuple[str, ...]
    root_ids: tuple[str, ...]
    included_upstream_ids: tuple[str, ...]
    source_order: tuple[str, ...]
    edges: tuple[SourceDependencyEdge, ...]
    config_versions: tuple[tuple[str, str], ...]
    op_names: tuple[tuple[str, str], ...]
    storage_paths: tuple[tuple[str, str], ...]
    environment_config_digest: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dagster manifest schema {self.schema_version!r}")
        if not self.job_name.strip() or not self.environment.strip():
            raise ValueError("job_name and environment must not be empty")
        if not self.project_root.is_absolute():
            raise ValueError("project_root must be absolute")
        if not self.environment_config_digest.strip():
            raise ValueError("environment_config_digest must not be empty")
        _validate_membership(self)
        _validate_edges(self)
        _validate_storage_paths(self.storage_paths)

    @classmethod
    def create(
        cls,
        *,
        job_name: str,
        environment: str,
        registry: SourceRegistry,
        selection: SourceSelection,
        environment_config: Mapping[str, Any],
        op_names: Mapping[str, str],
    ) -> Self:
        """Capture the selected validated graph without credentials or runtime objects."""
        versions = tuple(
            (
                source_id,
                compute_config_version(registry.get_source(source_id).config_path),
            )
            for source_id in selection.source_ids
        )
        layout = StorageLayout.from_environment_config(
            environment_config,
            registry.project_root,
        )
        return cls(
            job_name=job_name,
            environment=environment,
            project_root=registry.project_root,
            requested_tags=selection.selection.tags,
            requested_domains=selection.selection.domains,
            root_ids=selection.root_ids,
            included_upstream_ids=selection.included_upstream_ids,
            source_order=selection.source_ids,
            edges=selection.graph.edges,
            config_versions=versions,
            op_names=tuple((source_id, op_names[source_id]) for source_id in selection.source_ids),
            storage_paths=tuple(
                (key, str(value))
                for key, value in layout.as_dict().items()
                if key != "project_root"
            ),
            environment_config_digest=_digest_json(environment_config),
        )

    @classmethod
    def from_json(cls, value: str) -> Self:
        """Load and validate a manifest copied into immutable Dagster run tags."""
        try:
            payload = json.loads(value)
            if not isinstance(payload, Mapping):
                raise TypeError("manifest must be a JSON object")
            edges = tuple(
                SourceDependencyEdge(
                    producer_id=_required_string(edge, "producer_id"),
                    consumer_id=_required_string(edge, "consumer_id"),
                    table=_required_string(edge, "table"),
                    input_paths=_string_tuple(edge.get("input_paths")),
                )
                for edge in _mapping_sequence(payload.get("edges"))
            )
            return cls(
                schema_version=int(payload.get("schema_version", 0)),
                job_name=_required_string(payload, "job_name"),
                environment=_required_string(payload, "environment"),
                project_root=Path(_required_string(payload, "project_root")),
                requested_tags=_string_tuple(payload.get("requested_tags")),
                requested_domains=_string_tuple(payload.get("requested_domains")),
                root_ids=_string_tuple(payload.get("root_ids")),
                included_upstream_ids=_string_tuple(payload.get("included_upstream_ids")),
                source_order=_string_tuple(payload.get("source_order")),
                edges=edges,
                config_versions=_string_pairs(payload.get("config_versions")),
                op_names=_string_pairs(payload.get("op_names")),
                storage_paths=_string_pairs(payload.get("storage_paths")),
                environment_config_digest=_required_string(
                    payload,
                    "environment_config_digest",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid JANUS Dagster run manifest: {exc}") from exc

    @property
    def snapshot_id(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_name": self.job_name,
            "environment": self.environment,
            "project_root": str(self.project_root),
            "requested_tags": list(self.requested_tags),
            "requested_domains": list(self.requested_domains),
            "root_ids": list(self.root_ids),
            "included_upstream_ids": list(self.included_upstream_ids),
            "source_order": list(self.source_order),
            "edges": [
                {
                    "producer_id": edge.producer_id,
                    "consumer_id": edge.consumer_id,
                    "table": edge.table,
                    "input_paths": list(edge.input_paths),
                }
                for edge in self.edges
            ],
            "config_versions": [list(pair) for pair in self.config_versions],
            "op_names": [list(pair) for pair in self.op_names],
            "storage_paths": [list(pair) for pair in self.storage_paths],
            "environment_config_digest": self.environment_config_digest,
        }

    def upstreams_of(self, source_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(edge.producer_id for edge in self.edges if edge.consumer_id == source_id)
        )

    def storage_layout(self) -> StorageLayout:
        paths = {key: Path(value) for key, value in self.storage_paths}
        return StorageLayout(project_root=self.project_root, **paths)


def _validate_membership(manifest: DagsterRunManifest) -> None:
    if not manifest.source_order or len(set(manifest.source_order)) != len(
        manifest.source_order
    ):
        raise ValueError("source_order must contain unique sources")
    source_ids = set(manifest.source_order)
    if set(manifest.root_ids) - source_ids:
        raise ValueError("every root must occur in source_order")
    if set(manifest.included_upstream_ids) != source_ids - set(manifest.root_ids):
        raise ValueError("included_upstream_ids must be exactly the non-root sources")
    if tuple(source_id for source_id, _ in manifest.config_versions) != manifest.source_order:
        raise ValueError("config_versions must follow source_order")
    if tuple(source_id for source_id, _ in manifest.op_names) != manifest.source_order:
        raise ValueError("op_names must follow source_order")
    if len({name for _, name in manifest.op_names}) != len(manifest.op_names):
        raise ValueError("op_names must be unique")


def _validate_edges(manifest: DagsterRunManifest) -> None:
    source_ids = set(manifest.source_order)
    order = {source_id: index for index, source_id in enumerate(manifest.source_order)}
    for edge in manifest.edges:
        if edge.producer_id not in source_ids or edge.consumer_id not in source_ids:
            raise ValueError("manifest edge endpoint is outside source_order")
        if order[edge.producer_id] >= order[edge.consumer_id]:
            raise ValueError("source_order must be topological for every manifest edge")


def _validate_storage_paths(storage_paths: tuple[tuple[str, str], ...]) -> None:
    required_paths = {"root_dir", "raw_dir", "bronze_dir", "metadata_dir"}
    paths = dict(storage_paths)
    if set(paths) != required_paths:
        raise ValueError(f"storage_paths must contain exactly {sorted(required_paths)}")
    if any(not Path(value).is_absolute() for value in paths.values()):
        raise ValueError("every manifest storage path must be absolute")


def _digest_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("expected a JSON string list")
    return tuple(value)


def _string_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON pair list")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != _PAIR_LENGTH
            or any(not isinstance(part, str) for part in item)
        ):
            raise TypeError("expected a JSON pair list")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise TypeError("expected a JSON object list")
    return tuple(value)
