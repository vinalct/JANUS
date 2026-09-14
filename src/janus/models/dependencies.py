"""The source→source dependency vocabulary: declared references, and the validated graph.

This module sits below the registry and the runtime. It turns an already-validated
``RequestInputsConfig`` into the flat list of references the config declares, and it
owns the immutable graph records those references resolve into. It resolves nothing
itself: whether a named producer exists, is enabled, or actually writes the referenced
table is a whole-registry question, and answering it here would need an import — of the
registry, a strategy, a catalog client or Spark — that this layer must not have.

The graph records carry their own invariants (sorted, unique, acyclic, endpoints
present), so a batch planner or an orchestration adapter handed a
``SourceDependencyGraph`` has a DAG by construction rather than by trusting its builder.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from janus.models.config.types import (
    CombinedRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    RequestInputsConfig,
)

#: Where an atomic request-input config sits inside a source document.
ROOT_REQUEST_INPUT_PATH = "access.request_inputs"


@dataclass(frozen=True, slots=True)
class IcebergInputReference:
    """One declared producer and its unchanged table reference, with leaf provenance."""

    upstream_source_id: str
    namespace: str
    table_name: str
    input_path: str

    @property
    def table_reference(self) -> str:
        """Return the dotted ``namespace.table`` this leaf reads."""
        return f"{self.namespace}.{self.table_name}"


def iter_iceberg_input_references(
    config: RequestInputsConfig,
    *,
    input_path: str = ROOT_REQUEST_INPUT_PATH,
) -> Iterator[IcebergInputReference]:
    """Visit every Iceberg leaf of a validated atomic or flat combined config."""
    entries = (
        ((f"{input_path}.inputs[{index}]", leaf) for index, leaf in enumerate(config.inputs))
        if isinstance(config, CombinedRequestInputsConfig)
        else iter(((input_path, config),))
    )
    for path, leaf in entries:
        if isinstance(leaf, IcebergRowsRequestInputsConfig):
            yield IcebergInputReference(
                upstream_source_id=leaf.upstream_source_id,
                namespace=leaf.namespace,
                table_name=leaf.table_name,
                input_path=path,
            )


@dataclass(frozen=True, slots=True)
class SourceDependencyNode:
    """One configured source and the physical bronze table it produces, if any.

    ``bronze_table`` is ``None`` for a source whose bronze output is not Iceberg: it can
    still consume, but nothing can depend on it, because a Parquet directory is not a
    table another source can read as ``namespace.table``.
    """

    source_id: str
    enabled: bool
    bronze_table: str | None = None

    def __post_init__(self) -> None:
        """Reject a node that names nobody or claims an empty table."""
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.bronze_table is not None and not self.bronze_table.strip():
            raise ValueError("bronze_table must be None or a non-empty identifier")


@dataclass(frozen=True, slots=True)
class SourceDependencyEdge:
    """One producer→consumer dependency, carrying every leaf that asked for it."""

    producer_id: str
    consumer_id: str
    table: str
    input_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject an edge with no endpoint, no table, or no provenance."""
        if not self.producer_id.strip() or not self.consumer_id.strip():
            raise ValueError("edge endpoints must not be empty")
        if not self.table.strip():
            raise ValueError("table must not be empty")
        if not self.input_paths:
            raise ValueError("input_paths must record at least one originating leaf")
        if len(set(self.input_paths)) != len(self.input_paths):
            raise ValueError("input_paths must not repeat a leaf path")


@dataclass(frozen=True, slots=True)
class SourceDependencyGraph:
    """A validated source→source DAG: sorted nodes, sorted unique edges, no cycles.

    The invariants are enforced here rather than by the builder alone, so a graph handed
    to a batch planner or an orchestration adapter is a DAG by construction — there is no
    second, unvalidated way to make one.
    """

    nodes: tuple[SourceDependencyNode, ...] = ()
    edges: tuple[SourceDependencyEdge, ...] = ()

    def __post_init__(self) -> None:
        """Enforce ordering, uniqueness, endpoint existence and acyclicity."""
        source_ids = [node.source_id for node in self.nodes]
        if source_ids != sorted(source_ids):
            raise ValueError("nodes must be sorted by source_id")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("nodes must not repeat a source_id")

        edge_keys = [(edge.producer_id, edge.consumer_id) for edge in self.edges]
        if edge_keys != sorted(edge_keys):
            raise ValueError("edges must be sorted by (producer_id, consumer_id)")
        if len(set(edge_keys)) != len(edge_keys):
            raise ValueError("edges must hold one entry per producer/consumer pair")

        known = set(source_ids)
        unknown = sorted(
            {endpoint for key in edge_keys for endpoint in key} - known,
        )
        if unknown:
            raise ValueError(f"edges reference sources outside the graph: {unknown}")

        cycles = find_dependency_cycles(self.edges)
        if cycles:
            rendered = "; ".join(render_dependency_cycle(cycle) for cycle in cycles)
            raise ValueError(f"dependency graph is cyclic: {rendered}")

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Return every node id, sorted."""
        return tuple(node.source_id for node in self.nodes)

    def node(self, source_id: str) -> SourceDependencyNode:
        """Return one node, or raise ``LookupError`` when the graph does not hold it."""
        for node in self.nodes:
            if node.source_id == source_id:
                return node
        raise LookupError(f"Source {source_id!r} is not part of the dependency graph")

    def upstreams(self, source_id: str) -> tuple[str, ...]:
        """Return the direct producers this source reads from, sorted."""
        self.node(source_id)
        return tuple(
            sorted(edge.producer_id for edge in self.edges if edge.consumer_id == source_id)
        )

    def downstreams(self, source_id: str) -> tuple[str, ...]:
        """Return the direct consumers of this source, sorted.

        This is the reverse adjacency failure propagation needs: a failed source blocks
        exactly these, and they are answerable from the graph without rescanning configs.
        """
        self.node(source_id)
        return tuple(
            sorted(edge.consumer_id for edge in self.edges if edge.producer_id == source_id)
        )

    def topological_order(self) -> tuple[str, ...]:
        """Return every node in dependency order, breaking ties lexicographically.

        Deterministic by construction (NFR-2): the ready set is a heap of source ids, so
        discovery order cannot reach the result.
        """
        successors: dict[str, list[str]] = {node.source_id: [] for node in self.nodes}
        indegree = {node.source_id: 0 for node in self.nodes}
        for edge in self.edges:
            successors[edge.producer_id].append(edge.consumer_id)
            indegree[edge.consumer_id] += 1

        ready = [source_id for source_id, degree in indegree.items() if degree == 0]
        heapify(ready)
        ordered: list[str] = []
        while ready:
            source_id = heappop(ready)
            ordered.append(source_id)
            for consumer_id in sorted(successors[source_id]):
                indegree[consumer_id] -= 1
                if indegree[consumer_id] == 0:
                    heappush(ready, consumer_id)

        if len(ordered) != len(self.nodes):
            raise ValueError("dependency graph is cyclic")
        return tuple(ordered)


def find_dependency_cycles(
    edges: Iterable[SourceDependencyEdge],
) -> tuple[tuple[str, ...], ...]:
    """Return one canonical cycle per cyclic component, deterministically.

    Enumerating *every* elementary cycle is exponential and would make the diagnostic
    depend on how many ways there are to say the same thing. One shortest cycle through
    the smallest member of each strongly connected component is reproducible, and naming
    it is enough to find the edge to delete.
    """
    successors: dict[str, set[str]] = {}
    for edge in edges:
        successors.setdefault(edge.producer_id, set()).add(edge.consumer_id)
        successors.setdefault(edge.consumer_id, set())

    reachable = {node: _reachable_from(node, successors) for node in successors}
    components: list[tuple[str, ...]] = []
    assigned: set[str] = set()
    for node in sorted(successors):
        if node in assigned or node not in reachable[node]:
            continue
        component = tuple(
            sorted(
                other
                for other in successors
                if other in reachable[node] and node in reachable[other]
            )
        )
        assigned.update(component)
        components.append(component)

    return tuple(_canonical_cycle(component, successors) for component in components)


def render_dependency_cycle(cycle: Iterable[str]) -> str:
    """Render a cycle the way the error message shows it: ``A → B → C → A``."""
    return " → ".join(cycle)


def _reachable_from(start: str, successors: dict[str, set[str]]) -> set[str]:
    """Return every node reachable from ``start`` along one or more edges."""
    seen: set[str] = set()
    pending = sorted(successors[start])
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(sorted(successors[node]))
    return seen


def _canonical_cycle(
    component: tuple[str, ...],
    successors: dict[str, set[str]],
) -> tuple[str, ...]:
    """Return the shortest cycle through the smallest member of ``component``."""
    start = component[0]
    members = set(component)
    if start in successors[start]:
        return (start, start)

    for first in sorted(successors[start] & members):
        path = _shortest_path(first, start, members, successors)
        if path is not None:
            return (start, *path)
    raise ValueError(f"component {component} was reported cyclic but holds no cycle")


def _shortest_path(
    start: str,
    target: str,
    members: set[str],
    successors: dict[str, set[str]],
) -> tuple[str, ...] | None:
    """Breadth-first path from ``start`` to ``target`` inside ``members``, or ``None``."""
    queue: deque[tuple[str, ...]] = deque([(start,)])
    seen = {start}
    while queue:
        path = queue.popleft()
        for successor in sorted(successors[path[-1]] & members):
            if successor == target:
                return (*path, target)
            if successor not in seen:
                seen.add(successor)
                queue.append((*path, successor))
    return None
