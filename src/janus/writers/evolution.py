"""Pure decisions for contract governed Iceberg schema evolution.

``EvolutionPlan.render()`` produces the stable metadata vocabulary: ``none``,
``added:a,b;promoted:c(integer->long)``, ``breaking_replace``, or
``refused:<kind>,<kind>``. The writer consumes the plan. this module never runs
DDL. Column order and changes to an existing column's ``required`` flag are
ignored: writes address columns by name, and the pre-write check enforces
required values. Nested container evolution is outside the verified promotion
set. callers must provide a differing type signature to flag nested drift.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from janus.models.data_contracts import DataContract, parse_physical_type
from janus.models.data_contracts.vocabulary import UnknownPhysicalTypeError
from janus.normalizers.base import NORMALIZATION_METADATA_COLUMNS

PLAN_METADATA_KEY = "schema_evolution"
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
PROMOTIONS: frozenset[tuple[str, str]] = frozenset({("integer", "long"), ("float", "double")})


@dataclass(frozen=True, slots=True)
class LiveColumn:
    name: str
    physical_type: str
    required: bool


@dataclass(frozen=True, slots=True)
class EvolutionRefusal:
    kind: str
    column: str
    detail: str


@dataclass(frozen=True, slots=True)
class EvolutionPlan:
    outcome: str
    add_columns: tuple[tuple[str, str], ...]
    promote_columns: tuple[tuple[str, str, str], ...]
    refusals: tuple[EvolutionRefusal, ...]
    reason: str
    recorded_major: int
    contract_major: int

    @property
    def is_noop(self) -> bool:
        return self.outcome == "noop"

    @property
    def evolves(self) -> bool:
        return self.outcome == "evolve"

    def render(self) -> str:
        if self.outcome == "noop":
            return "none"
        if self.outcome == "breaking_replace":
            return "breaking_replace"
        if self.outcome == "refused":
            return "refused:" + ",".join(sorted({item.kind for item in self.refusals}))
        parts = []
        if self.add_columns:
            parts.append("added:" + ",".join(name for name, _ in self.add_columns))
        if self.promote_columns:
            parts.append(
                "promoted:"
                + ",".join(
                    f"{name}({before}->{after})" for name, before, after in self.promote_columns
                )
            )
        return ";".join(parts)


def contract_major(version: str) -> int:
    """Return the major component of a validated contract version."""
    if not isinstance(version, str) or _SEMVER.fullmatch(version) is None:
        raise ValueError(f"invalid contract version: {version!r}")
    return int(version.split(".", 1)[0])


def _decimal_relation(before: str, after: str) -> str | None:
    """Classify a same-scale decimal precision change, if there is one."""
    try:
        old = parse_physical_type(before)
        new = parse_physical_type(after)
    except UnknownPhysicalTypeError:
        return None
    if old.name != "decimal(p,s)" or new.name != "decimal(p,s)":
        return None
    if old.scale != new.scale:
        return None
    if old.precision < new.precision:
        return "promote"
    if old.precision > new.precision:
        return "narrowed"
    return None


def _type_relation(before: str, after: str) -> str:
    try:
        parse_physical_type(before)
    except UnknownPhysicalTypeError:
        return "retyped"
    if before == after:
        return "same"
    if (before, after) in PROMOTIONS:
        return "promote"
    if (after, before) in PROMOTIONS:
        return "narrowed"
    return _decimal_relation(before, after) or "retyped"


def _diff_columns(
    contract: DataContract, live_columns: Sequence[LiveColumn]
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[EvolutionRefusal]]:
    """Classify top-level differences without applying compatibility policy."""
    live = {
        column.name: column
        for column in live_columns
        if column.name not in NORMALIZATION_METADATA_COLUMNS
    }
    declared = {prop.name: prop for prop in contract.schema.properties}
    adds: list[tuple[str, str]] = []
    promotions: list[tuple[str, str, str]] = []
    refusals: list[EvolutionRefusal] = []
    missing_names = declared.keys() - live.keys()
    extra_names = live.keys() - declared.keys()

    for prop in contract.schema.properties:
        if prop.name in missing_names:
            if prop.required:
                refusals.append(
                    EvolutionRefusal(
                        "newly_required",
                        prop.name,
                        "new required column cannot be added to existing rows",
                    )
                )
            else:
                adds.append((prop.name, prop.physical_type))
            continue
        before = live[prop.name].physical_type
        relation = _type_relation(before, prop.physical_type)
        if relation == "promote":
            promotions.append((prop.name, before, prop.physical_type))
        elif relation != "same":
            refusals.append(
                EvolutionRefusal(relation, prop.name, f"{before} -> {prop.physical_type}")
            )

    # An unmatched live field with a new field of the same type is a likely
    # rename. Otherwise a trailing unknown field is diagnosed as undeclared;
    # a field within the declared sequence is diagnosed as dropped. Both are
    # equally unsafe and refused. Without a stored prior schema this distinction
    # is only diagnostic, not a permission decision.
    added_types = {declared[name].physical_type for name in missing_names}
    last_shared = max(
        (i for i, col in enumerate(live_columns) if col.name in declared),
        default=-1,
    )
    for i, column in enumerate(live_columns):
        if column.name not in extra_names:
            continue
        if column.physical_type in added_types:
            kind = "renamed_or_dropped"
        elif i > last_shared:
            kind = "undeclared_live_column"
        else:
            kind = "dropped"
        refusals.append(
            EvolutionRefusal(kind, column.name, "live column is absent from the declared contract")
        )
    return adds, promotions, refusals


def _apply_compatibility(
    compatibility: str,
    additions: list[tuple[str, str]],
    promotions: list[tuple[str, str, str]],
    refusals: list[EvolutionRefusal],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[EvolutionRefusal]]:
    if compatibility == "frozen":
        changed = {
            *(name for name, _ in additions),
            *(name for name, _, _ in promotions),
            *(item.column for item in refusals),
        }
        return (
            [],
            [],
            [
                EvolutionRefusal("frozen", name, "frozen contract permits no schema changes")
                for name in changed
            ],
        )
    if compatibility == "additive":
        refusals.extend(
            EvolutionRefusal(
                "retyped", name, f"{before} -> {after} requires backward compatibility"
            )
            for name, before, after in promotions
        )
        return additions, [], refusals
    return additions, promotions, refusals


def plan_schema_evolution(
    *,
    contract: DataContract,
    live_columns: Sequence[LiveColumn],
    recorded_contract_version: str | None,
    write_strategy: str,
    batch_index: int = 1,
) -> EvolutionPlan:
    """Plan the allowed table change from plain contract and live column values.

    A major bump permits a refused change only on the first ``replace_table``
    batch. A malformed recorded version fails closed so an invalid table stamp
    cannot accidentally authorize replacement.
    """
    major = contract_major(contract.version)
    try:
        recorded_major = (
            contract_major(recorded_contract_version)
            if recorded_contract_version is not None
            else 0
        )
    except (ValueError, TypeError):
        recorded_major = 0
        invalid_stamp = True
    else:
        invalid_stamp = False

    additions, promotions, refusals = _diff_columns(contract, live_columns)
    additions, promotions, refusals = _apply_compatibility(
        contract.janus.compatibility, additions, promotions, refusals
    )
    if invalid_stamp:
        refusals.append(
            EvolutionRefusal(
                "invalid_recorded_version",
                "<table>",
                f"invalid recorded contract version: {recorded_contract_version!r}",
            )
        )
    ordered_refusals = tuple(sorted(refusals, key=lambda item: (item.column, item.kind)))
    add_columns = tuple(additions)
    promote_columns = tuple(promotions)
    if ordered_refusals:
        allowed_replace = (
            not invalid_stamp
            and major > recorded_major
            and write_strategy == "replace_table"
            and batch_index == 1
        )
        outcome = "breaking_replace" if allowed_replace else "refused"
        kinds = ", ".join(sorted({item.kind for item in ordered_refusals}))
        reason = (
            f"contract major version {recorded_major} -> {major} authorizes "
            f"full refresh table replacement ({kinds})"
            if allowed_replace
            else f"schema change refused ({kinds}); bump the contract MAJOR and run a full refresh"
        )
    elif add_columns or promote_columns:
        outcome = "evolve"
        reason = (
            "schema can evolve in place: "
            + EvolutionPlan(
                "evolve", add_columns, promote_columns, (), "", recorded_major, major
            ).render()
        )
    else:
        outcome = "noop"
        reason = "live schema matches the contract"
    return EvolutionPlan(
        outcome,
        add_columns,
        promote_columns,
        ordered_refusals,
        reason,
        recorded_major,
        major,
    )
