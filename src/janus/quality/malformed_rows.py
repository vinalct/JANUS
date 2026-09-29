"""Malformed JSON/CSV evidence and the pre-write refusal it can trigger."""

from __future__ import annotations

import json
from typing import Any

from janus.quality.contract_checks import ContractEnforcementError
from janus.quality.models import ValidationCheck
from janus.utils.logging import sanitize_log_payload

SAMPLE_LIMIT = 5
SAMPLE_CHARACTER_LIMIT = 500


def bounded_samples(rows: list[Any]) -> tuple[tuple[str, ...], bool]:
    """Scrub collected raw records, then bound what may enter a validation report."""
    samples: list[str] = []
    truncated = False
    for row in rows[:SAMPLE_LIMIT]:
        raw = str(row[0])
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            safe = str(sanitize_log_payload(raw))
        else:
            safe = json.dumps(sanitize_log_payload(parsed), ensure_ascii=False)
            safe = str(sanitize_log_payload(safe))
        if len(safe) > SAMPLE_CHARACTER_LIMIT:
            safe = safe[: SAMPLE_CHARACTER_LIMIT - 1] + "…"
            truncated = True
        samples.append(safe)
    return tuple(samples), truncated


def malformed_rows_check(
    count: int | None,
    *,
    enforcement: str,
    threshold: int,
    samples: tuple[str, ...] = (),
    truncated: bool = False,
    batch_index: int = 1,
) -> ValidationCheck:
    if count is None:
        return ValidationCheck.skipped(
            "data", "malformed_rows", "typed by construction: parquet handoff."
        )
    details = {
        "count": count,
        "threshold": threshold,
        "samples": json.dumps(list(samples), ensure_ascii=False),
        "truncated": str(truncated).lower(),
        "batch_index": batch_index,
    }
    if enforcement == "strict" and count > threshold:
        return ValidationCheck.failed(
            "data",
            "malformed_rows",
            f"{count} malformed rows exceed max_malformed_rows {threshold}.",
            details=details,
        )
    if enforcement == "lenient":
        details["severity"] = "warning"
        return ValidationCheck.passed(
            "data",
            "malformed_rows",
            f"WARNING: {count} malformed rows observed.",
            details=details,
        )
    return ValidationCheck.passed(
        "data",
        "malformed_rows",
        f"{count} malformed rows within max_malformed_rows {threshold}.",
        details=details,
    )


class MalformedRowsError(ContractEnforcementError):
    """Strict enforcement refused a batch with more malformed rows than allowed."""

    failure_stage = "malformed_rows"

    def __init__(
        self,
        count: int,
        samples: tuple[str, ...],
        threshold: int,
        batch_index: int,
        batch_count: int = 1,
    ) -> None:
        self.count = count
        self.samples = samples
        self.threshold = threshold
        self.batch_index = batch_index
        self.batch_count = batch_count
        super().__init__(
            f"{count} malformed rows exceed max_malformed_rows {threshold} "
            f"(batch {batch_index}/{batch_count})"
        )
