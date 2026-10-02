"""The registry's semantic pass: configs that parse, load, and still cannot be true."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from janus.models.config.issues import ValidationIssue
from janus.models.config.types import CombinedRequestInputsConfig, IcebergRowsRequestInputsConfig
from janus.models.data_contracts import DataContract
from janus.models.dependencies import IcebergInputReference, iter_iceberg_input_references
from janus.models.source_config import SourceConfig
from janus.models.write_intent import NORMALIZATION_METADATA_COLUMNS


@dataclass(frozen=True, slots=True)
class SemanticContext:
    """Everything the rules may read. Nothing else is in scope."""

    sources_by_id: Mapping[str, SourceConfig]
    contracts: Mapping[str, DataContract]
    hook_ids: frozenset[str]


SemanticRule = Callable[[SourceConfig, SemanticContext], list[ValidationIssue]]


def expected_fields(
    source: SourceConfig, *, contracts: Mapping[str, DataContract]
) -> tuple[str, ...] | None:
    """The column names this source declares, or ``None`` when it declares no contract."""
    contract = contracts.get(source.source_id)
    return None if contract is None else contract.column_names


def _primary_key_in_required(
    source: SourceConfig, context: SemanticContext
) -> list[ValidationIssue]:
    """(b) A ``primaryKey`` column that may be null cannot key a merge.

    The run-time half, ``quality/validators.py::validate_quality_contract``, stays: a plan
    built directly in a test never passes through the loader. This module may not import
    the quality layer, so the sentence is restated here and pinned to that check's own by a
    parity test.
    """
    contract = context.contracts.get(source.source_id)
    if contract is None:
        return []
    required = set(contract.required_columns)
    missing = [column for column in contract.primary_key if column not in required]
    if not missing:
        return []
    return [
        ValidationIssue(
            "schema.contract",
            "primaryKey columns must also be required: " + ", ".join(missing),
        )
    ]


def _iceberg_columns_in_producer_contract(
    source: SourceConfig, context: SemanticContext
) -> list[ValidationIssue]:
    """(c) Every column an ``iceberg_rows`` leaf maps must exist in its producer's contract.

    Only the top-level segment is compared: bronze declares ``payload``, not its leaves, so
    ``payload.label`` is judged by ``payload``. Silent when the producer is missing —
    ``registry/dependencies.py`` already refuses that, with a better message — and when the
    producer declares no contract, because there is nothing to compare against.
    """
    issues: list[ValidationIssue] = []
    for reference, leaf in _iceberg_leaves(source):
        producer = context.sources_by_id.get(reference.upstream_source_id)
        if producer is None:
            continue
        declared = expected_fields(producer, contracts=context.contracts)
        if declared is None:
            continue
        known = set(declared)
        missing = sorted(
            {column for column in leaf.columns.values() if column.split(".", 1)[0] not in known}
        )
        if missing:
            issues.append(
                ValidationIssue(
                    f"{reference.input_path}.columns",
                    f"reads column(s) the producer {reference.upstream_source_id} "
                    f"does not declare: {', '.join(missing)}",
                )
            )
    return issues


def _source_hook_resolves(source: SourceConfig, context: SemanticContext) -> list[ValidationIssue]:
    """(e) A declared ``source_hook`` must resolve: ``HookCatalog.resolve``'s test, at load."""
    if source.source_hook is None or source.source_hook in context.hook_ids:
        return []
    known = ", ".join(sorted(context.hook_ids))
    return [ValidationIssue("source_hook", f"is not a registered hook; known hooks: {known}")]


def _partition_columns_known(
    source: SourceConfig, context: SemanticContext
) -> list[ValidationIssue]:
    """(f) Partition only by a normalization metadata column or a contract column.

    Silent without a contract, where the written columns cannot be known before a run.
    """
    declared = expected_fields(source, contracts=context.contracts)
    if declared is None:
        return []
    known = {*NORMALIZATION_METADATA_COLUMNS, *declared}
    unknown = sorted({column for column in source.spark.partition_by if column not in known})
    if not unknown:
        return []
    return [
        ValidationIssue(
            "spark.partition_by",
            "names column(s) that are neither normalization metadata nor contract columns: "
            + ", ".join(unknown),
        )
    ]


_RULES: Final[tuple[tuple[str, SemanticRule], ...]] = (
    ("primary_key_in_required", _primary_key_in_required),
    ("iceberg_columns_in_producer_contract", _iceberg_columns_in_producer_contract),
    ("source_hook_resolves", _source_hook_resolves),
    ("partition_columns_known", _partition_columns_known),
)

RULE_IDS: Final[tuple[str, ...]] = tuple(rule_id for rule_id, _rule in _RULES)


def collect_semantic_issues(
    sources: Sequence[SourceConfig],
    *,
    contracts: Mapping[str, DataContract],
    hook_ids: frozenset[str] | None = None,
) -> list[tuple[str, ValidationIssue]]:
    """Every cross-block and cross-source problem, collected, never raised.

    Sources are visited sorted by id and the rules in ``RULE_IDS`` order, so discovery order
    never reaches the result. ``hook_ids`` is the catalog rule (e) checks against; ``None``
    means the built-in hooks, the ones ``HookCatalog.with_defaults()`` resolves.
    """
    context = SemanticContext(
        sources_by_id={source.source_id: source for source in sources},
        contracts=contracts,
        hook_ids=_hook_catalog(sources, hook_ids),
    )
    return [
        (source.source_id, issue)
        for source in sorted(sources, key=lambda item: item.source_id)
        for _rule_id, rule in _RULES
        for issue in rule(source, context)
    ]


def unverified_required_fields(
    sources: Sequence[SourceConfig], *, contracts: Mapping[str, DataContract]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``(source_id, fields)`` for each source declaring required_fields with no contract (D-3).

    Not an issue: a note ``janus validate`` prints, because there is nothing to check the
    declaration against. Empty for the checked-in registry, where every entry has a contract.
    """
    return tuple(
        (source.source_id, source.quality.required_fields)
        for source in sorted(sources, key=lambda item: item.source_id)
        if source.quality.required_fields
        and expected_fields(source, contracts=contracts) is None
    )


def _hook_catalog(
    sources: Sequence[SourceConfig], hook_ids: frozenset[str] | None
) -> frozenset[str]:
    """The ids rule (e) accepts: the injected catalog, else the built-in hooks.

    ``janus.hooks`` is imported here, and only when some source names a hook, because it
    reaches the strategy and runtime packages and, through them, ``janus.planner`` — which
    imports this package. At module scope that is an import cycle.
    """
    if hook_ids is not None:
        return frozenset(hook_ids)
    if all(source.source_hook is None for source in sources):
        return frozenset()
    from janus.hooks import built_in_hooks

    return frozenset(hook_id for hook_id, _hook in built_in_hooks())


def _iceberg_leaves(
    source: SourceConfig,
) -> Iterator[tuple[IcebergInputReference, IcebergRowsRequestInputsConfig]]:
    """Pair each Iceberg reference with the leaf that declared it, for the leaf's ``columns``.

    ``iter_iceberg_input_references`` owns the walk and the provenance path. It visits leaves
    in declaration order, so the declared Iceberg leaves line up with it one to one; the
    strict zip turns any future drift between the two into a loud error, not a wrong pairing.
    """
    inputs = source.access.request_inputs
    leaves = inputs.inputs if isinstance(inputs, CombinedRequestInputsConfig) else (inputs,)
    iceberg = [leaf for leaf in leaves if isinstance(leaf, IcebergRowsRequestInputsConfig)]
    return zip(iter_iceberg_input_references(inputs), iceberg, strict=True)


__all__ = [
    "RULE_IDS",
    "SemanticContext",
    "collect_semantic_issues",
    "expected_fields",
    "unverified_required_fields",
]
