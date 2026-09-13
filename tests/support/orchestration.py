"""Synthetic registry inputs and expected contracts, not a DAG implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import yaml

from janus.utils.storage import bronze_table_identifier


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    upstreams: tuple[str, ...] = ()
    enabled: bool = True
    family: str = "api"
    domain: str = "reference"
    tags: tuple[str, ...] = ("reference",)
    table_name: str | None = None


@dataclass(frozen=True)
class GraphCase:
    sources: tuple[SourceSpec, ...]
    expected_order: tuple[str, ...] = ()
    expected_error: str | None = None


A = SourceSpec("A")
B = SourceSpec("B", ("A",), domain="reporting", tags=("report",))
C = SourceSpec("C", domain="independent", tags=("independent",))
D = SourceSpec("D", ("B",), domain="reporting", tags=("report", "terminal"))

GRAPH_CASES = {
    "independent": GraphCase((C, A), ("A", "C")),
    "chain": GraphCase((B, A), ("A", "B")),
    "chain_and_peer": GraphCase((D, C, B, A), ("A", "B", "C", "D")),
    "fan_in": GraphCase((SourceSpec("D", ("A", "B")), SourceSpec("B"), A), ("A", "B", "D")),
    "diamond": GraphCase(
        (SourceSpec("D", ("B", "C")), SourceSpec("C", ("A",)), B, A),
        ("A", "B", "C", "D"),
    ),
    "catalog_consumer": GraphCase((SourceSpec("B", ("A",), family="catalog"), A), ("A", "B")),
    "self_cycle": GraphCase((SourceSpec("A", ("A",)),), expected_error="cycle"),
    "cycle": GraphCase((SourceSpec("A", ("D",)), B, D), expected_error="cycle"),
    "missing_producer": GraphCase((SourceSpec("B", ("missing",)),), expected_error="missing"),
    "disabled_producer": GraphCase((B, SourceSpec("A", enabled=False)), expected_error="disabled"),
    "disabled_subgraph": GraphCase(
        (SourceSpec("B", ("A",), enabled=False), SourceSpec("A", enabled=False), C),
        ("C",),
    ),
    "duplicate_producer": GraphCase(
        (SourceSpec("A", table_name="Shared-Target"), SourceSpec("B", table_name="shared target")),
        expected_error="ambiguous",
    ),
}

SELECTION_CASES = (
    ("tag", ("terminal",), ("D",), ("A", "B", "D")),
    ("tag", ("terminal", "independent"), ("C", "D"), ("A", "B", "C", "D")),
    ("domain", ("reporting",), ("B", "D"), ("A", "B", "D")),
    ("domain", ("reporting", "independent"), ("B", "C", "D"), ("A", "B", "C", "D")),
    ("tag", ("absent",), (), ()),  # Must raise an empty-selection error; never execute.
)


def source_payload(spec: SourceSpec) -> dict[str, Any]:
    """A small, valid source with no credentials, delays, hooks, or live endpoints."""
    return {
        "source_id": spec.source_id,
        "name": f"Synthetic {spec.source_id}",
        "owner": "janus-tests",
        "enabled": spec.enabled,
        "source_type": spec.family,
        "strategy": spec.family,
        "strategy_variant": "metadata_catalog" if spec.family == "catalog" else "page_number_api",
        "federation_level": "federal",
        "domain": spec.domain,
        "public_access": True,
        "tags": list(spec.tags),
        "access": {
            "base_url": "https://fixtures.invalid",
            "path": f"/{spec.source_id}",
            "method": "GET",
            "format": "json",
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "size",
                "page_size": 10,
            },
            "rate_limit": {"concurrency": 1, "backoff_seconds": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "checkpoint_strategy": "none",
            "retry": {"max_attempts": 1, "backoff_seconds": 1},
        },
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "jsonl" if spec.family == "catalog" else "json",
            "write_mode": "overwrite",
            "partition_by": ["ingestion_date"],
        },
        "outputs": {
            zone: {
                "path": f"data/{zone}/{spec.source_id}",
                "format": "iceberg" if zone == "bronze" else "json",
                **({"table_name": spec.table_name} if zone == "bronze" and spec.table_name else {}),
            }
            for zone in ("raw", "bronze", "metadata")
        },
        "quality": {"allow_schema_evolution": True},
    }


def source_documents(case: GraphCase, *, declare_upstreams: bool = True) -> list[dict[str, Any]]:
    sources = {spec.source_id: source_payload(spec) for spec in case.sources}
    for spec in case.sources:
        leaves = []
        for upstream in spec.upstreams:
            producer = sources.get(upstream, source_payload(SourceSpec(upstream)))
            output = producer["outputs"]["bronze"]
            namespace, table = bronze_table_identifier(
                output["path"], fallback_name=upstream, table_name=output.get("table_name")
            ).split(".")
            leaves.append(
                {
                    "type": "iceberg_rows",
                    "namespace": namespace,
                    "table_name": table,
                    "columns": {f"{upstream.lower()}_id": "id"},
                    **({"upstream_source_id": upstream} if declare_upstreams else {}),
                }
            )
        if leaves:
            access = sources[spec.source_id]["access"]
            access["request_inputs"] = (
                leaves[0] if len(leaves) == 1 else {"type": "combined", "inputs": leaves}
            )
            access["parameter_bindings"] = {
                field: {"from": f"request_input.{field}"}
                for leaf in leaves
                for field in leaf["columns"]
            }
    return list(sources.values())


def write_project(root: Path, documents: list[dict[str, Any]], *, grouped: bool = True) -> Path:
    """Write an isolated registry without loading/validating its future graph contract."""
    sources_dir = root / "conf" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (root / "conf" / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: '*.yaml'\n", encoding="utf-8"
    )
    if grouped:
        (sources_dir / "sources.yaml").write_text(
            yaml.safe_dump({"sources": documents}, sort_keys=False), encoding="utf-8"
        )
    else:
        for index, document in enumerate(documents):
            (sources_dir / f"{index:02d}.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
            )
    for zone in ("raw", "bronze", "metadata"):
        (root / "data" / zone).mkdir(parents=True, exist_ok=True)
    return root


def build_graph_project(
    tmp_path: Path, name: str, *, grouped: bool = True, declare_upstreams: bool = True
) -> Path:
    root = Path(mkdtemp(prefix=f"{name}-", dir=tmp_path))
    return write_project(
        root,
        source_documents(GRAPH_CASES[name], declare_upstreams=declare_upstreams),
        grouped=grouped,
    )
