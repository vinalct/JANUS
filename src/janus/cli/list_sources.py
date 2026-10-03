"""`janus list`: every configured source, its dispatch, and its place in the graph.

The listing reads the source registry and nothing else: no environment profile, no plan,
no Spark session. Each row comes from the loaded `SourceConfig` and from `registry.graph`.
Upstreams, downstreams and `bronze_table` are the graph's own answers (the writer's table
identity, never a warehouse lookup), and `--graph` prints `topological_order()` as the
graph computed it, never re-sorted.

`--tag` and `--domain` mean what they mean to `run-all`: both build one `BatchSelection`,
so a combined filter is refused with the same message. `--family` and `--enabled-only` are
list-only refinements applied after it. Two things differ from `run-all` on purpose:

- **No closure.** `run-all` adds every upstream of a selected consumer, because a consumer
  scheduled without its producer reads stale data. Listing has no such hazard, and a
  listing that added rows nobody asked for would be a worse answer, so `list` shows exactly
  what matched. The UPSTREAMS column and `--graph` are where the dependencies live.
- **An empty result exits 0**, with the header, zero rows and `0 source(s)`. In `run-all` a
  selection matching nothing is an error, because listing nothing is a true answer to a
  question, while running nothing is a silent no-op.

The output is deterministic (D-18): no timestamp, run id or absolute path. Rows are sorted
by `source_id` in every format. Table columns are padded to the widest value in the listed
set, so a filter changes the padding and nothing else. Counts describe the listed set:
its sources, the enabled ones among them, and the edges between them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from janus.models.config.strategy_registry import STRATEGY_REGISTRY
from janus.models.dependencies import SourceDependencyEdge, SourceDependencyGraph
from janus.models.source_config import SourceConfig
from janus.orchestration import BatchSelection
from janus.registry import SourceRegistry, load_registry

ARGUMENT_ERROR = 2
EMPTY_CELL = "-"
TABLE_COLUMNS = ("SOURCE_ID", "FAMILY", "VARIANT", "MODE", "EN", "HOOK", "UPSTREAMS", "TAGS")
COLUMN_GAP = "  "

_EPILOG = (
    "Filters list exactly the sources they match. Unlike run-all, list never adds the "
    "upstreams a matching consumer depends on: the UPSTREAMS column names them, and --graph "
    "shows the dependency order. An empty listing exits 0. Only the source registry is read: "
    "no environment profile, no plan and no Spark session."
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Defaults to table.",
    )
    # Not an argparse exclusive group: `BatchSelection` refuses the combination, so `list`
    # and `run-all` reject it with one message from one place.
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="List sources carrying this tag; repeat for an any-match filter. Cannot be "
        "combined with --domain.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="List sources in this domain; repeat for an any-match filter. Cannot be "
        "combined with --tag.",
    )
    parser.add_argument(
        "--family",
        choices=sorted(STRATEGY_REGISTRY.families),
        help="List only sources of this strategy family.",
    )
    parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="List only enabled sources. Off by default: disabled sources are listed too.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Print the dependency graph of the listed sources instead of their rows: the "
        "topological order, then every edge with its table and input path.",
    )
    parser.epilog = _EPILOG


def list_command(args: argparse.Namespace) -> int:
    """Exit 0 with the listing, empty or not; 2 when the filter or the registry is refused."""
    try:
        selection = BatchSelection.create(tags=args.tag, domains=args.domain)
        registry = load_registry(args.project_root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        # `SelectionFilterError` is a `ValueError`: printed as `run-all` prints it.
        print(str(exc), file=sys.stderr)
        return ARGUMENT_ERROR

    listed = _listed_sources(
        registry, selection, family=args.family, enabled_only=args.enabled_only
    )
    subgraph = registry.graph.subgraph(source.source_id for source in listed)
    counts = {
        "sources": len(listed),
        "enabled": sum(1 for source in listed if source.enabled),
        "edges": len(subgraph.edges),
    }

    if args.graph:
        document = _graph_document(subgraph, counts)
        if args.format == "json":
            print(_render_json(document))
        else:
            scope = _describe_filter(selection, family=args.family, enabled_only=args.enabled_only)
            totals = (len(registry.sources), len(registry.graph.edges))
            print(_render_graph(document, scope=scope, totals=totals))
        return 0

    listing = {
        "counts": counts,
        "sources": [_source_record(source, registry.graph) for source in listed],
    }
    print(_render_json(listing) if args.format == "json" else _render_table(listing))
    return 0


def _listed_sources(
    registry: SourceRegistry,
    selection: BatchSelection,
    *,
    family: str | None,
    enabled_only: bool,
) -> tuple[SourceConfig, ...]:
    """The sources the filters match, sorted by id. No upstream is ever added."""
    return tuple(
        sorted(
            (
                source
                for source in registry.list_sources(enabled_only=enabled_only)
                if selection.matches(source) and family in (None, source.strategy)
            ),
            key=lambda source: source.source_id,
        )
    )


def _source_record(source: SourceConfig, graph: SourceDependencyGraph) -> dict[str, Any]:
    """One source as both formats show it. The graph columns read the whole registry's graph,
    so a consumer listed without its producer still names it."""
    return {
        "source_id": source.source_id,
        "name": source.name,
        "family": source.strategy,
        "variant": source.strategy_variant,
        "mode": source.extraction.mode,
        "enabled": source.enabled,
        "domain": source.domain,
        "hook": source.source_hook,
        # `SourceConfig.tags` keeps YAML order; sorted here, at render time, not in the model.
        "tags": sorted(source.tags),
        "upstreams": list(graph.upstreams(source.source_id)),
        "downstreams": list(graph.downstreams(source.source_id)),
        "bronze_table": graph.node(source.source_id).bronze_table,
    }


def _graph_document(graph: SourceDependencyGraph, counts: dict[str, int]) -> dict[str, Any]:
    order = graph.topological_order()
    return {
        "counts": counts,
        "topological_order": list(order),
        "edges": [
            {
                "producer_id": edge.producer_id,
                "consumer_id": edge.consumer_id,
                "table": edge.table,
                "input_paths": list(edge.input_paths),
            }
            for edge in _edges_in_dependency_order(graph, order)
        ],
    }


def _edges_in_dependency_order(
    graph: SourceDependencyGraph, order: Sequence[str]
) -> list[SourceDependencyEdge]:
    leaving: dict[str, list[SourceDependencyEdge]] = {}
    for edge in graph.edges:
        leaving.setdefault(edge.producer_id, []).append(edge)
    return [edge for source_id in order for edge in leaving.get(source_id, ())]


def _describe_filter(
    selection: BatchSelection, *, family: str | None, enabled_only: bool
) -> str | None:
    """The filter as the graph header names it, or None when nothing was filtered."""
    parts = [] if selection.is_empty else [selection.describe()]
    if family is not None:
        parts.append(f"family {family}")
    if enabled_only:
        parts.append("enabled only")
    return ", ".join(parts) or None


def _render_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True)


def _render_table(listing: dict[str, Any]) -> str:
    rows = [TABLE_COLUMNS, *(_table_row(source) for source in listing["sources"])]
    # The last column is left unpadded, so no line carries trailing whitespace.
    widths = [max(len(row[index]) for row in rows) for index in range(len(TABLE_COLUMNS) - 1)]
    lines = [
        COLUMN_GAP.join(
            [*(cell.ljust(width) for cell, width in zip(row[:-1], widths, strict=True)), row[-1]]
        )
        for row in rows
    ]
    counts = listing["counts"]
    summary = (
        f"{counts['sources']} source(s); {counts['enabled']} enabled; {counts['edges']} edge(s)"
    )
    return "\n".join([*lines, "", summary])


def _table_row(source: dict[str, Any]) -> tuple[str, ...]:
    return (
        source["source_id"],
        source["family"],
        source["variant"],
        source["mode"],
        "yes" if source["enabled"] else "no",
        source["hook"] or EMPTY_CELL,
        ",".join(source["upstreams"]) or EMPTY_CELL,
        ",".join(source["tags"]) or EMPTY_CELL,
    )


def _render_graph(document: dict[str, Any], *, scope: str | None, totals: tuple[int, int]) -> str:
    order = document["topological_order"]
    edges = document["edges"]
    if scope is None:
        order_header = f"topological order ({len(order)})"
        edge_header = f"edges ({len(edges)})"
    else:
        order_header = f"topological order ({len(order)} of {totals[0]}; filtered: {scope})"
        edge_header = f"edges ({len(edges)} of {totals[1]}; between the filtered sources)"
    lines = [
        order_header,
        *(f"{position:>4}  {source_id}" for position, source_id in enumerate(order, start=1)),
        "",
        edge_header,
    ]
    for edge in edges:
        lines.extend(
            (
                f"  {edge['producer_id']}  ->  {edge['consumer_id']}",
                f"      table {edge['table']}",
                f"      via   {', '.join(edge['input_paths'])}",
            )
        )
    return "\n".join(lines)
