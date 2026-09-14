"""Resolve declared Iceberg request inputs into one validated source dependency graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from janus.models.config.issues import ValidationIssue
from janus.models.dependencies import (
    IcebergInputReference,
    SourceDependencyEdge,
    SourceDependencyGraph,
    SourceDependencyNode,
    find_dependency_cycles,
    iter_iceberg_input_references,
    render_dependency_cycle,
)
from janus.models.source_config import SourceConfig
from janus.utils.storage import bronze_table_identifier

#: The bronze output format an ``iceberg_rows`` dependency can be satisfied by.
ICEBERG_OUTPUT_FORMAT = "iceberg"

#: Field path reported for a producer-side problem, mirroring the source YAML block.
BRONZE_OUTPUT_PATH = "outputs.bronze"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Where a source is actually editable: its file, and its entry in a grouped file."""

    source_id: str
    config_path: Path
    entry: str | None = None

    def describe(self) -> str:
        """Render the location the way diagnostics name it."""
        if self.entry:
            return f"{self.config_path}:{self.entry}"
        return str(self.config_path)


class SourceGraphValidationError(ValueError):
    """Every independent dependency-graph problem found in one registry load."""

    def __init__(self, sources_dir: Path, issues: Sequence[ValidationIssue]) -> None:
        """Build a readable error listing each collected graph issue."""
        self.sources_dir = sources_dir
        self.issues = tuple(issues)
        message_lines = [f"Invalid source dependency graph: {sources_dir}"]
        message_lines.extend(f"- {issue.render()}" for issue in self.issues)
        super().__init__("\n".join(message_lines))


def producer_table_identifier(source: SourceConfig) -> str | None:
    """Return the Iceberg table this source produces, or ``None`` if it produces none.

    A raw, metadata or Parquet bronze output cannot satisfy an Iceberg dependency, so it
    contributes no producer identity rather than a plausible-looking one.
    """
    bronze = source.outputs.bronze
    if bronze.format.strip().lower() != ICEBERG_OUTPUT_FORMAT:
        return None
    return bronze_table_identifier(
        bronze.path,
        fallback_name=source.source_id,
        namespace=bronze.namespace,
        table_name=bronze.table_name,
    )


def build_source_dependency_graph(
    sources: Sequence[SourceConfig],
    *,
    locations: Iterable[SourceLocation] = (),
    sources_dir: Path | None = None,
) -> SourceDependencyGraph:
    """Return the validated graph for a whole registry, or raise with every problem.

    ``locations`` only sharpens diagnostics: without it a message names the file, with it
    the exact grouped entry. Validation itself never depends on it, so a registry built
    for batch use without locations is held to the same invariants.
    """
    described = {location.source_id: location.describe() for location in locations}
    producers: dict[str, list[SourceConfig]] = {}
    nodes: list[SourceDependencyNode] = []
    for source in sorted(sources, key=lambda candidate: candidate.source_id):
        table = producer_table_identifier(source)
        nodes.append(
            SourceDependencyNode(
                source_id=source.source_id,
                enabled=source.enabled,
                bronze_table=table,
            )
        )
        if table is not None:
            producers.setdefault(table, []).append(source)

    issues = _shared_table_issues(producers, described)
    edges, reference_issues = _resolve_references(sources, producers, described)
    issues.extend(reference_issues)
    issues.extend(_cycle_issues(edges))

    if issues:
        raise SourceGraphValidationError(sources_dir or Path("conf/sources"), issues)
    return SourceDependencyGraph(nodes=tuple(nodes), edges=tuple(edges))


def _shared_table_issues(
    producers: Mapping[str, list[SourceConfig]],
    described: Mapping[str, str],
) -> list[ValidationIssue]:
    """Allow co-writers that declare each other; reject a collision nobody declared."""
    issues: list[ValidationIssue] = []
    for table, candidates in sorted(producers.items()):
        ordered = sorted(candidates, key=lambda candidate: candidate.source_id)
        writers = [source.source_id for source in ordered]
        for source in ordered:
            declared = set(source.outputs.bronze.shared_with)
            expected = set(writers) - {source.source_id}
            if declared == expected:
                continue
            issues.append(
                _shared_table_issue(source, table, declared, expected, ordered, described)
            )
    return issues


def _shared_table_issue(
    source: SourceConfig,
    table: str,
    declared: set[str],
    expected: set[str],
    writers: Sequence[SourceConfig],
    described: Mapping[str, str],
) -> ValidationIssue:
    """Explain one side of a mis-declared table, naming what to add or remove."""
    field_path = f"{source.source_id} ({_describe(source, described)}).{BRONZE_OUTPUT_PATH}"
    if not expected:
        return ValidationIssue(
            f"{field_path}.shared_with",
            f"declares co-writers {sorted(declared)} for {table!r}, but no other configured "
            "source writes that table; remove the declaration or fix the table identity",
        )

    named = ", ".join(
        f"{writer.source_id} ({_describe(writer, described)})"
        for writer in writers
        if writer.source_id != source.source_id
    )
    if not declared:
        return ValidationIssue(
            f"{field_path}.shared_with",
            f"writes {table!r}, which {named} also writes: an undeclared collision is an "
            "ambiguous producer target. Two pipelines may deliberately share one bronze "
            "table — a full-refresh rebuild and an incremental delta job for one dataset — "
            "but each must then name the other in shared_with",
        )
    return ValidationIssue(
        f"{field_path}.shared_with",
        f"declares {sorted(declared)} for {table!r}, but its actual co-writers are "
        f"{sorted(expected)}; the declaration must name every other source writing the "
        "table, and only those",
    )


def _resolve_references(
    sources: Sequence[SourceConfig],
    producers: Mapping[str, list[SourceConfig]],
    described: Mapping[str, str],
) -> tuple[list[SourceDependencyEdge], list[ValidationIssue]]:
    """Turn every declared leaf into one edge per producer/consumer pair, or an issue."""
    by_id = {source.source_id: source for source in sources}
    issues: list[ValidationIssue] = []
    paths_by_pair: dict[tuple[str, str], list[str]] = {}
    tables_by_pair: dict[tuple[str, str], str] = {}

    for source in sorted(sources, key=lambda candidate: candidate.source_id):
        for reference in iter_iceberg_input_references(source.access.request_inputs):
            field_path = _reference_path(source, reference, described)
            producer = _validated_producer(
                source,
                reference,
                by_id=by_id,
                producers=producers,
                described=described,
                field_path=field_path,
                issues=issues,
            )
            if producer is None:
                continue
            pair = (producer.source_id, source.source_id)
            paths_by_pair.setdefault(pair, []).append(reference.input_path)
            tables_by_pair[pair] = reference.table_reference

    edges = [
        SourceDependencyEdge(
            producer_id=producer_id,
            consumer_id=consumer_id,
            table=tables_by_pair[(producer_id, consumer_id)],
            input_paths=tuple(paths_by_pair[(producer_id, consumer_id)]),
        )
        for producer_id, consumer_id in sorted(paths_by_pair)
    ]
    return edges, issues


def _validated_producer(
    source: SourceConfig,
    reference: IcebergInputReference,
    *,
    by_id: Mapping[str, SourceConfig],
    producers: Mapping[str, list[SourceConfig]],
    described: Mapping[str, str],
    field_path: str,
    issues: list[ValidationIssue],
) -> SourceConfig | None:
    """Return the one configured producer this leaf may depend on, or record why not."""
    if not _is_unqualified(reference.namespace) or not _is_unqualified(reference.table_name):
        issues.append(
            ValidationIssue(
                field_path,
                f"reads {reference.table_reference!r}, which is not a supported table "
                "reference: namespace and table_name must each be one unqualified "
                "identifier, and a catalog-qualified or cross-catalog name is not resolvable "
                "against a configured producer",
            )
        )
        return None

    producer = by_id.get(reference.upstream_source_id)
    if producer is None:
        issues.append(
            ValidationIssue(
                field_path,
                f"declares upstream_source_id {reference.upstream_source_id!r}, which is "
                "missing from the registry: no configured source has that id",
            )
        )
        return None

    produced = producer_table_identifier(producer)
    if produced is None:
        issues.append(
            ValidationIssue(
                field_path,
                f"declares upstream source {producer.source_id!r}, whose bronze output "
                f"format is {producer.outputs.bronze.format!r}; an iceberg_rows input can "
                "only depend on a source that writes an iceberg bronze table",
            )
        )
        return None

    if produced != reference.table_reference:
        issues.append(
            ValidationIssue(
                field_path,
                f"reads {reference.table_reference!r}, but its declared upstream source "
                f"{producer.source_id!r} ({_describe(producer, described)}) produces "
                f"{produced!r}{_actual_producer_clause(reference, producers, described)}",
            )
        )
        return None

    if source.enabled and not producer.enabled:
        issues.append(
            ValidationIssue(
                field_path,
                f"is enabled, but its upstream source {producer.source_id!r} "
                f"({_describe(producer, described)}) is disabled; a run never enables an "
                "upstream on its behalf",
            )
        )
    return producer


def _cycle_issues(edges: Sequence[SourceDependencyEdge]) -> list[ValidationIssue]:
    """Report each cyclic component once, with a reproducible path and its leaves."""
    provenance = {(edge.producer_id, edge.consumer_id): edge for edge in edges}
    issues: list[ValidationIssue] = []
    for cycle in find_dependency_cycles(edges):
        explanations: list[str] = []
        for producer_id, consumer_id in pairwise(cycle):
            edge = provenance[(producer_id, consumer_id)]
            for input_path in edge.input_paths:
                explanations.append(f"{consumer_id}.{input_path} reads {edge.table}")
        issues.append(
            ValidationIssue(
                render_dependency_cycle(cycle),
                "is a source dependency cycle and can never be scheduled "
                f"({'; '.join(explanations)})",
            )
        )
    return issues


def _actual_producer_clause(
    reference: IcebergInputReference,
    producers: Mapping[str, list[SourceConfig]],
    described: Mapping[str, str],
) -> str:
    """Name who really writes the referenced table, so the fix is the shorter edit."""
    candidates = producers.get(reference.table_reference, [])
    if not candidates:
        return "; no configured source produces the referenced table"
    named = ", ".join(
        f"{source.source_id} ({_describe(source, described)})"
        for source in sorted(candidates, key=lambda candidate: candidate.source_id)
    )
    return f"; the referenced table is produced by {named}"


def _reference_path(
    source: SourceConfig,
    reference: IcebergInputReference,
    described: Mapping[str, str],
) -> str:
    """Compose the editable location of one leaf: source, file/entry, and input path."""
    return f"{source.source_id} ({_describe(source, described)}).{reference.input_path}"


def _describe(source: SourceConfig, described: Mapping[str, str]) -> str:
    """Return the grouped-entry location when the loader supplied one, else the file."""
    return described.get(source.source_id, str(source.config_path))


def _is_unqualified(segment: str) -> bool:
    """Return True when a reference segment names one identifier, not a dotted path."""
    return bool(segment.strip()) and "." not in segment
