"""The pure structural check: is this frame what the contract declares?

Engine-free: frames arrive as the JSON types a ``StructType`` serialises to (the one adapter is
``schema_contracts.frame_columns_from_spark_schema``), the contract becomes the same JSON through
the vocabulary, and both sides compare in vocabulary spellings, never Spark's. A report, not a
decision: it never raises.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from janus.models.data_contracts import (
    ContractProperty,
    DataContract,
    VocabularyError,
    physical_type_from_spark_json,
    spark_json_type,
)
from janus.quality.models import ValidationCheck

CORRUPT_RECORD_COLUMN = "_janus_corrupt_record"


@dataclass(frozen=True, slots=True)
class FrameColumn:
    """One Spark frame column, engine-free: its type is the JSON value a StructType carries."""

    name: str
    spark_json_type: Any
    nullable: bool


@dataclass(frozen=True, slots=True)
class ContractMismatch:
    """One difference between a frame and its contract, in vocabulary spellings."""

    kind: str  
    column: str  
    expected: str | None = None
    observed: str | None = None 

    def render(self) -> str:
        sides = [
            f"{side} {value}"
            for side, value in (("contract", self.expected), ("frame", self.observed))
            if value is not None
        ]
        text = f"{self.column}: {self.kind.replace('_', ' ')}"
        return f"{text} ({', '.join(sides)})" if sides else text


@dataclass(frozen=True, slots=True)
class ContractCheck:
    """What one frame looked like against one contract."""

    contract_id: str
    contract_version: str
    schema_version: str
    mismatches: tuple[ContractMismatch, ...]
    nullability_relaxed: tuple[str, ...]
    checked_columns: int

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def to_validation_check(self) -> ValidationCheck:
        """The ``data.schema_expectations`` check the validation JSON reports."""
        label = f"data contract {self.contract_id} v{self.contract_version}"
        details = {
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "checked_columns": self.checked_columns,
            "mismatches": json.dumps([mismatch.render() for mismatch in self.mismatches]),
            "nullability_relaxed": json.dumps(list(self.nullability_relaxed)),
        }
        verdict = "matches" if self.ok else "does not match"
        message = "; ".join([f"Frame {verdict} {label}", *(m.render() for m in self.mismatches)])
        build = ValidationCheck.passed if self.ok else ValidationCheck.failed
        return build("data", "schema_expectations", message, details=details)


class ContractEnforcementError(RuntimeError):
    """Base of every 'the contract said no' failure; failure_stage names where.

    The materializer attaches what it knew when it refused: ``evidence`` holds the pre-write
    evidence of every batch it checked, the refused one last, and ``committed_results`` the bronze
    writes of the batches before it, so an entry point can report both without re-reading.
    """

    failure_stage: str = "contract_check"
    checks: tuple[ValidationCheck, ...] = ()
    evidence: tuple[Any, ...] = ()
    committed_results: tuple[Any, ...] = ()


class MissingContractError(ContractEnforcementError):
    """A plan without a data contract: there is nothing to check a batch against, so no write."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        super().__init__(
            f"source {source_id!r} has no data contract; bronze is materialized only under one "
            "(declare schema.contract)"
        )


class ContractViolationError(ContractEnforcementError):
    """A batch's frame does not match its contract, so that batch must not be written."""

    failure_stage = "contract_check"

    def __init__(self, check: ContractCheck, *, batch_index: int = 1, batch_count: int = 1) -> None:
        self.check = check
        self.batch_index = batch_index
        self.batch_count = batch_count
        self.checks = (check.to_validation_check(),)
        header = (
            f"{check.contract_id} v{check.contract_version}: frame does not match the contract "
            f"(batch {batch_index}/{batch_count}):"
        )
        lines = [header, *(f"  {mismatch.render()}" for mismatch in check.mismatches)]
        super().__init__("\n".join(lines))


def check_frame_against_contract(
    frame: Sequence[FrameColumn],
    contract: DataContract,
    *,
    allow_corrupt_column: bool = False,
) -> ContractCheck:
    """Compare a frame's names and types to its contract; total, it returns for every input."""
    columns = [column for column in frame if column.name != CORRUPT_RECORD_COLUMN]
    leaked = [column.spark_json_type for column in frame if column.name == CORRUPT_RECORD_COLUMN]
    properties = contract.schema.properties
    mismatches = _compare_names(
        "",
        [(prop.name, _contract_json(prop)) for prop in properties],
        [(column.name, column.spark_json_type) for column in columns],
    )
    if not allow_corrupt_column:
        mismatches.extend(
            ContractMismatch("corrupt_column_leaked", CORRUPT_RECORD_COLUMN, observed=_spelling(t))
            for t in leaked
        )
    return ContractCheck(
        contract_id=contract.id,
        contract_version=contract.version,
        schema_version=contract.schema_version,
        mismatches=tuple(sorted(mismatches, key=lambda mismatch: (mismatch.column, mismatch.kind))),
        nullability_relaxed=_relaxed_nullability(contract.required_columns, columns),
        checked_columns=len(properties),
    )


def _compare_names(
    prefix: str, expected: list[tuple[str, Any]], observed: list[tuple[str, Any]]
) -> list[ContractMismatch]:
    """Exact and case-sensitive, undeclared columns included (D-4); shared names go on to types."""
    present = dict(observed)
    mismatches: list[ContractMismatch] = []
    for name, value in expected:
        path = _path(prefix, name)
        if name in present:
            mismatches.extend(_compare_types(path, value, present[name]))
        else:
            mismatches.append(ContractMismatch("missing_column", path, expected=_spelling(value)))
    declared = {name for name, _ in expected}
    mismatches.extend(
        ContractMismatch("unexpected_column", _path(prefix, name), observed=_spelling(value))
        for name, value in observed
        if name not in declared
    )
    return mismatches


def _compare_types(path: str, expected: Any, observed: Any) -> list[ContractMismatch]:
    """One mismatch where the kinds differ; otherwise recurse into the container's children."""
    kind = _kind(expected)
    if kind is None or kind != _kind(observed):
        return [_type_mismatch(path, expected, observed)]
    if kind == "struct":
        return _compare_names(path, _struct_fields(expected), _struct_fields(observed))
    if kind == "array":
        return _compare_types(f"{path}[]", expected["elementType"], observed["elementType"])
    if kind == "map":
        if _spelling(expected["keyType"]) != _spelling(observed["keyType"]):
            return [_type_mismatch(path, expected, observed)]
        return _compare_types(f"{path}{{}}", expected["valueType"], observed["valueType"])
    return []


def _relaxed_nullability(required: tuple[str, ...], columns: list[FrameColumn]) -> tuple[str, ...]:
    """Required in the contract, nullable in the frame: recorded, never judged (D-3)."""
    nullable = {column.name for column in columns if column.nullable}
    return tuple(name for name in required if name in nullable)


def _type_mismatch(path: str, expected: Any, observed: Any) -> ContractMismatch:
    return ContractMismatch("type_mismatch", path, _spelling(expected), _spelling(observed))


def _path(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _contract_json(prop: ContractProperty) -> Any:
    """The Spark JSON type the reader applied; one outside the vocabulary matches no frame."""
    try:
        return spark_json_type(prop)
    except VocabularyError:
        return {"type": prop.physical_type}


def _kind(value: Any) -> str | None:
    """The vocabulary spelling of a Spark JSON type's own kind, or ``None`` when it has none."""
    if isinstance(value, str):
        try:
            return physical_type_from_spark_json(value)
        except VocabularyError:
            return None
    if isinstance(value, Mapping) and _is_container(value):
        return str(value["type"])
    return None


def _is_container(value: Mapping[str, Any]) -> bool:
    """A struct, array or map carrying every key its children are read from."""
    kind = value.get("type")
    if kind == "struct":
        fields = value.get("fields")
        return isinstance(fields, list) and all(
            isinstance(field, Mapping) and isinstance(field.get("name"), str) and "type" in field
            for field in fields
        )
    if kind == "array":
        return "elementType" in value
    return kind == "map" and "keyType" in value and "valueType" in value


def _struct_fields(value: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [(field["name"], field["type"]) for field in value["fields"]]


def _spelling(value: Any) -> str:
    """The full vocabulary spelling (``map<string,long>``), raw where the vocabulary has none."""
    kind = _kind(value)
    if kind == "struct":
        fields = ",".join(f"{name}:{_spelling(child)}" for name, child in _struct_fields(value))
        return f"struct<{fields}>"
    if kind == "array":
        return f"array<{_spelling(value['elementType'])}>"
    if kind == "map":
        return f"map<{_spelling(value['keyType'])},{_spelling(value['valueType'])}>"
    if kind is not None:
        return kind
    raw = value.get("type") if isinstance(value, Mapping) else value
    return raw if isinstance(raw, str) else repr(value)
