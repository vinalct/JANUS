"""The pre-write pass: structural check and one shared data aggregation per batch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NoReturn

from janus.models.data_contracts import DataContract
from janus.quality.contract_checks import (
    CORRUPT_RECORD_COLUMN,
    ContractCheck,
    ContractMismatch,
    ContractViolationError,
    check_frame_against_contract,
)
from janus.quality.malformed_rows import MalformedRowsError, bounded_samples, malformed_rows_check
from janus.quality.models import ValidationCheck

STRICT = "strict"
REQUIRED_NULL = "required_null"

_OUTCOME_PRIORITY = ("failed", "skipped", "passed")


@dataclass(frozen=True, slots=True)
class PreWriteEvidence:
    """What the one pre-write pass measured for one batch."""

    batch_index: int
    batch_count: int
    contract_check: ContractCheck
    required_null_counts: Mapping[str, int] = field(default_factory=dict)
    malformed_count: int | None = None
    malformed_samples: tuple[str, ...] = ()
    malformed_truncated: bool = False
    max_malformed_rows: int = 0

    def checks(self, *, enforcement: str) -> tuple[ValidationCheck, ...]:
        """Render schema, required-field and malformed-row results for this batch."""
        rendered = [self.contract_check.to_validation_check()]
        if enforcement == STRICT:
            rendered.append(self._required_fields_check())
        rendered.append(
            malformed_rows_check(
                self.malformed_count,
                enforcement=enforcement,
                threshold=self.max_malformed_rows,
                samples=self.malformed_samples,
                truncated=self.malformed_truncated,
                batch_index=self.batch_index,
            )
        )
        return tuple(rendered)

    def _required_fields_check(self) -> ValidationCheck:
        if any(mismatch.kind != REQUIRED_NULL for mismatch in self.contract_check.mismatches):
            return ValidationCheck.skipped(
                "data",
                "required_fields",
                "Required fields were not counted: the batch failed the structural check first.",
            )
        return required_fields_check(self.required_null_counts)


def run_pre_write_pass(
    dataframe: Any,
    contract: DataContract,
    *,
    enforcement: str,
    batch_index: int,
    batch_count: int,
    max_malformed_rows: int = 0,
    tracks_corrupt: bool = False,
) -> tuple[Any, PreWriteEvidence]:
    """Check schema and data before the write, returning a frame without reader evidence."""
    check = structural_check(dataframe, contract, allow_corrupt_column=tracks_corrupt)
    if not check.ok:
        _refuse(PreWriteEvidence(batch_index, batch_count, check))

    fields = contract.required_columns if enforcement == STRICT else ()
    counts: dict[str, int] = {}
    malformed_count: int | None = None
    if pre_write_aggregates(contract, enforcement=enforcement, tracks_corrupt=tracks_corrupt):
        from pyspark.sql.functions import col, when
        from pyspark.sql.functions import sum as spark_sum

        aggregations = required_null_aggregations(dataframe, fields)
        if tracks_corrupt:
            aggregations.append(
                spark_sum(when(col(CORRUPT_RECORD_COLUMN).isNotNull(), 1).otherwise(0)).alias(
                    "__malformed"
                )
            )
        row = dataframe.agg(*aggregations).first()
        if row is None:
            raise RuntimeError("Spark returned no row for the pre-write aggregation")
        counts = {name: int(row[name] or 0) for name in fields}
        if tracks_corrupt:
            malformed_count = int(row["__malformed"] or 0)

    samples: tuple[str, ...] = ()
    truncated = False
    if malformed_count:
        from pyspark.sql.functions import col

        rows = (
            dataframe.where(col(CORRUPT_RECORD_COLUMN).isNotNull())
            .select(CORRUPT_RECORD_COLUMN)
            .limit(5)
            .collect()
        )
        samples, truncated = bounded_samples(rows)

    violations = [
        ContractMismatch(REQUIRED_NULL, name, observed=f"{count} null/blank")
        for name, count in counts.items()
        if count > 0
    ]
    if violations:
        mismatches = sorted([*check.mismatches, *violations], key=lambda m: (m.column, m.kind))
        _refuse(
            PreWriteEvidence(
                batch_index,
                batch_count,
                replace(check, mismatches=tuple(mismatches)),
                counts,
                malformed_count,
                samples,
                truncated,
                max_malformed_rows,
            )
        )

    if tracks_corrupt:
        dataframe = dataframe.drop(CORRUPT_RECORD_COLUMN)
        clean_check = structural_check(dataframe, contract)
        if not clean_check.ok:
            _refuse(PreWriteEvidence(batch_index, batch_count, clean_check))

    evidence = PreWriteEvidence(
        batch_index,
        batch_count,
        check,
        counts,
        malformed_count,
        samples,
        truncated,
        max_malformed_rows,
    )
    if (
        enforcement == STRICT
        and malformed_count is not None
        and malformed_count > max_malformed_rows
    ):
        error = MalformedRowsError(
            malformed_count, samples, max_malformed_rows, batch_index, batch_count
        )
        error.evidence = (evidence,)
        error.checks = (
            malformed_rows_check(
                malformed_count,
                enforcement=enforcement,
                threshold=max_malformed_rows,
                samples=samples,
                truncated=truncated,
                batch_index=batch_index,
            ),
        )
        raise error
    return dataframe, evidence


def pre_write_aggregates(
    contract: DataContract, *, enforcement: str, tracks_corrupt: bool = False
) -> bool:
    """Whether a batch needs a persisted scan for the shared data aggregation."""
    return tracks_corrupt or (enforcement == STRICT and bool(contract.required_columns))


def structural_check(
    dataframe: Any,
    contract: DataContract,
    *,
    ignored_columns: Sequence[str] = (),
    allow_corrupt_column: bool = False,
) -> ContractCheck:
    from janus.schema_contracts import frame_columns_from_spark_schema

    declared = set(contract.column_names)
    columns = [
        column
        for column in frame_columns_from_spark_schema(dataframe.schema)
        if column.name in declared or column.name not in ignored_columns
    ]
    return check_frame_against_contract(
        columns, contract, allow_corrupt_column=allow_corrupt_column
    )


def required_null_aggregations(dataframe: Any, fields: Sequence[str]) -> list[Any]:
    """One ``sum`` per field counting nulls, and blanks for string columns, aliased by the field."""
    from pyspark.sql.functions import col, trim, when
    from pyspark.sql.functions import sum as spark_sum
    from pyspark.sql.types import StringType

    schema_by_name = {column.name: column.dataType for column in dataframe.schema.fields}
    aggregations = []
    for name in fields:
        invalid = col(name).isNull()
        if isinstance(schema_by_name[name], StringType):
            invalid = invalid | (trim(col(name)) == "")
        aggregations.append(spark_sum(when(invalid, 1).otherwise(0)).alias(name))
    return aggregations


def required_fields_check(counts: Mapping[str, int]) -> ValidationCheck:
    """The ``data.required_fields`` check for measured null/blank counts, pre- or post-write."""
    if not counts:
        return ValidationCheck.skipped(
            "data", "required_fields", "No required_fields were configured."
        )
    failing = {name: count for name, count in counts.items() if count > 0}
    if failing:
        rendered = ", ".join(f"{name} ({count})" for name, count in failing.items())
        return ValidationCheck.failed(
            "data",
            "required_fields",
            "Required fields contain null or blank values: " + rendered,
            details=failing,
        )
    return ValidationCheck.passed(
        "data",
        "required_fields",
        "All required fields are present and populated.",
        details={"required_field_count": len(counts)},
    )


def summarize_pre_write_evidence(
    evidence: Sequence[PreWriteEvidence], *, enforcement: str
) -> dict[str, ValidationCheck]:
    """Merge each batch's checks into one per name: failed if any batch failed, then skipped."""
    per_name: dict[str, list[tuple[PreWriteEvidence, ValidationCheck]]] = {}
    for item in evidence:
        for check in item.checks(enforcement=enforcement):
            per_name.setdefault(check.name, []).append((item, check))
    return {name: _merge(entries) for name, entries in per_name.items()}


def _merge(entries: list[tuple[PreWriteEvidence, ValidationCheck]]) -> ValidationCheck:
    outcome = next(o for o in _OUTCOME_PRIORITY if any(c.outcome == o for _, c in entries))
    chosen = [(item, check) for item, check in entries if check.outcome == outcome]
    first_item, first = chosen[0]
    message = first.message
    if outcome != "passed" and first_item.batch_count > 1:
        message = "; ".join(
            f"batch {item.batch_index}/{item.batch_count}: {check.message}"
            for item, check in chosen
        )
    details = {**first.details_as_dict(), "batches": str(len(entries))}
    if first.name == "malformed_rows" and outcome != "skipped":
        measured = [check.details_as_dict() for _, check in entries if check.outcome != "skipped"]
        count = sum(int(item["count"]) for item in measured)
        samples = [sample for item in measured for sample in json.loads(item["samples"])][:5]
        details.update(
            count=str(count),
            samples=json.dumps(samples, ensure_ascii=False),
            truncated=str(any(item["truncated"] == "true" for item in measured)).lower(),
        )
        if len(entries) > 1 and outcome == "passed":
            if "severity" in details:
                message = f"WARNING: {count} malformed rows observed across {len(entries)} batches."
            else:
                message = (
                    f"{count} malformed rows observed across {len(entries)} batches; "
                    "each batch within max_malformed_rows."
                )
    return replace(first, message=message, details=tuple(sorted(details.items())))


def _refuse(evidence: PreWriteEvidence) -> NoReturn:
    error = ContractViolationError(
        evidence.contract_check,
        batch_index=evidence.batch_index,
        batch_count=evidence.batch_count,
    )
    error.evidence = (evidence,)
    raise error


__all__ = [
    "REQUIRED_NULL",
    "STRICT",
    "PreWriteEvidence",
    "pre_write_aggregates",
    "required_fields_check",
    "required_null_aggregations",
    "run_pre_write_pass",
    "structural_check",
    "summarize_pre_write_evidence",
]
