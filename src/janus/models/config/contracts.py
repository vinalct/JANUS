"""Cross-block invariants — rules no single builder can see on its own.

These run after every block is built, because each one relates two blocks that are
validated independently.
"""

from __future__ import annotations

from janus.models.config.constants import CONCURRENT_PAGINATION_TYPES
from janus.models.config.issues import ValidationIssue
from janus.models.config.types import AccessConfig, ExtractionConfig, QualityConfig


def _validate_incremental_contract(
    extraction: ExtractionConfig,
    quality: QualityConfig,
    issues: list[ValidationIssue],
) -> None:
    """Require idempotency keys for incremental sources.

    Incremental writes are upserted on ``quality.unique_fields``; without a key there is
    no definable "same row", so the loader could only duplicate silently.
    """
    if extraction.mode == "incremental" and not quality.unique_fields:
        issues.append(
            ValidationIssue(
                "quality.unique_fields",
                "is required when extraction.mode is 'incremental' — incremental writes are "
                "upserted on these keys and have no defined idempotency without them",
            )
        )


def _validate_concurrency_contract(
    source_type: str,
    access: AccessConfig,
    issues: list[ValidationIssue],
) -> None:
    """Concurrency is a speculative pagination capability, not a generic knob."""

    if source_type != "api":
        return
    if access.rate_limit.concurrency <= 1:
        return
    if access.pagination.type not in CONCURRENT_PAGINATION_TYPES:
        issues.append(
            ValidationIssue(
                "access.rate_limit.concurrency",
                "must be 1 unless access.pagination.type is 'page_number' or 'offset' — "
                "concurrent pagination speculates on the next page index and cannot predict "
                f"{access.pagination.type!r} pagination",
            )
        )


def _validate_retry_status_contract(
    access: AccessConfig,
    extraction: ExtractionConfig,
    issues: list[ValidationIssue],
) -> None:
    """A status cannot mean both "the stream ended" and "try that again".

    The retry loop resolves the overlap deterministically — terminal is checked first — but
    a config that has to be read alongside the loop's branch order to be understood is a
    config that will eventually be misread. Rejecting the overlap keeps each status with
    exactly one meaning.
    """

    overlap = sorted(
        set(extraction.retry.retryable_status_codes)
        & set(access.pagination.past_end_status_codes)
    )
    if not overlap:
        return

    listed = ", ".join(str(code) for code in overlap)
    issues.append(
        ValidationIssue(
            "extraction.retry.retryable_status_codes",
            f"must not also appear in access.pagination.past_end_status_codes ({listed}) — "
            "a status is either evidence the stream ended or a transient failure worth "
            "re-sending, never both",
        )
    )
