"""Turn a filter into the exact set of sources a batch will run, in a stable order.

Selection has two halves, and they answer to different authorities. **Which sources an
operator asked for** is a filter over tags or domains — a product decision. **Which
sources those cannot run without** is the graph's answer, not the operator's: selecting a
consumer selects its producers, however far up the chain and whatever tag or domain they
carry, because a consumer scheduled without its producer is a consumer reading whatever
happened to be in the table last time.

The order comes from the induced subgraph's own topological sort, so the batch orders
itself with the one algorithm the graph already owns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from janus.models.dependencies import SourceDependencyGraph
from janus.models.source_config import SourceConfig
from janus.orchestration.errors import (
    BatchPlanningError,
    DisabledUpstreamError,
    EmptySelectionError,
    SelectionFilterError,
)


@dataclass(frozen=True, slots=True)
class BatchSelection:
    """The requested filter: nothing, one or more tags, or one or more domains."""

    tags: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a combined filter, and pin selectors to their canonical form."""
        if self.tags and self.domains:
            raise SelectionFilterError(
                "A batch is selected by tag or by domain, not by both: "
                f"tags={list(self.tags)} and domains={list(self.domains)} were both requested"
            )
        for name, values in (("tags", self.tags), ("domains", self.domains)):
            if any(not value.strip() for value in values):
                raise SelectionFilterError(f"{name} must not hold an empty selector")
            if list(values) != sorted(set(values)):
                raise SelectionFilterError(
                    f"{name} must be sorted and free of repeats; build the selection with "
                    "BatchSelection.create, which normalizes repeated flags"
                )

    @classmethod
    def create(
        cls,
        *,
        tags: Iterable[str] = (),
        domains: Iterable[str] = (),
    ) -> Self:
        """Normalize repeatable CLI flags into one canonical selection.

        ``--tag a --tag b`` and ``--tag b --tag a --tag b`` are the same request, so they
        must produce the same selection record and the same plan (NFR-2).
        """
        return cls(tags=_normalized(tags), domains=_normalized(domains))

    @property
    def is_empty(self) -> bool:
        """Return True when no filter was requested, which selects every enabled source."""
        return not self.tags and not self.domains

    def describe(self) -> str:
        """Render the request the way a refusal names it."""
        if self.tags:
            return f"tag in ({', '.join(self.tags)})"
        if self.domains:
            return f"domain in ({', '.join(self.domains)})"
        return "no filter (all enabled sources)"

    def matches(self, source: SourceConfig) -> bool:
        """Return True when this source is a *root* of the selection.

        Being a root is the only thing a filter decides. A source can still join the
        batch as somebody's upstream without matching anything.
        """
        if self.tags:
            return any(tag in self.tags for tag in source.tags)
        if self.domains:
            return source.domain in self.domains
        return True


NO_SELECTION = BatchSelection()


@dataclass(frozen=True, slots=True)
class SourceSelection:
    """The resolved batch membership: roots, the closure, and the induced subgraph."""

    selection: BatchSelection
    root_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    graph: SourceDependencyGraph

    def __post_init__(self) -> None:
        """Keep membership, order and graph describing one and the same set."""
        if not self.root_ids:
            raise BatchPlanningError("a resolved selection must hold at least one root")
        if set(self.root_ids) - set(self.source_ids):
            raise BatchPlanningError("every root must appear in the expanded selection")
        if sorted(self.source_ids) != list(self.graph.source_ids):
            raise BatchPlanningError("the expanded selection and its subgraph must agree")

    @property
    def included_upstream_ids(self) -> tuple[str, ...]:
        """Return the sources pulled in as dependencies rather than requested, sorted."""
        return tuple(sorted(set(self.source_ids) - set(self.root_ids)))

    def is_root(self, source_id: str) -> bool:
        """Return True when this source was selected directly, not as an upstream."""
        return source_id in self.root_ids


def select_sources(
    graph: SourceDependencyGraph,
    sources: Sequence[SourceConfig],
    selection: BatchSelection = NO_SELECTION,
) -> SourceSelection:
    """Resolve one filter against a validated graph into an ordered batch membership."""
    by_id = {source.source_id: source for source in sources}
    unknown = sorted(set(graph.source_ids) - by_id.keys())
    if unknown:
        raise BatchPlanningError(
            f"The dependency graph holds sources the registry snapshot does not: {unknown}. "
            "Selection must run against the graph the same load validated."
        )

    roots = tuple(
        sorted(
            node.source_id
            for node in graph.nodes
            if node.enabled and selection.matches(by_id[node.source_id])
        )
    )
    if not roots:
        raise EmptySelectionError(_empty_selection_message(graph, selection))

    expanded: set[str] = set(roots)
    for root in roots:
        expanded.update(graph.ancestors(root))

    disabled = sorted(
        source_id for source_id in expanded if not graph.node(source_id).enabled
    )
    if disabled:
        raise DisabledUpstreamError(
            f"{selection.describe()} selects sources that require the disabled upstream(s) "
            f"{disabled}. Enable them in their source config, or narrow the selection; a "
            "batch never enables a source on your behalf, and never reuses what a disabled "
            "producer wrote as this run's output."
        )

    subgraph = graph.subgraph(expanded)
    return SourceSelection(
        selection=selection,
        root_ids=roots,
        source_ids=subgraph.topological_order(),
        graph=subgraph,
    )


def _empty_selection_message(
    graph: SourceDependencyGraph,
    selection: BatchSelection,
) -> str:
    """Explain an empty selection with what *is* enabled, so the next attempt lands."""
    enabled = sorted(node.source_id for node in graph.nodes if node.enabled)
    if not enabled:
        return (
            f"No source is enabled, so {selection.describe()} selects nothing. A batch that "
            "runs no source is a configuration problem, not an empty success."
        )
    return (
        f"No enabled source matches {selection.describe()}. Enabled sources: "
        f"{enabled}. A selection that matches nothing is a configuration problem, not an "
        "empty success."
    )


def _normalized(values: Iterable[str]) -> tuple[str, ...]:
    """Strip, deduplicate and sort selector values so repeats cannot change a plan."""
    return tuple(sorted({value.strip() for value in values}))
