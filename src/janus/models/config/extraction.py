"""Builders for the ``extraction`` block — mode, checkpointing, and retry policy."""

from __future__ import annotations

from typing import Any

from janus.models.config.coercion import (
    _optional_enum,
    _optional_int,
    _optional_int_list,
    _optional_string,
    _require_enum,
    _require_mapping,
)
from janus.models.config.constants import (
    CLIENT_ERROR_STATUS_MIN,
    DEFAULT_RETRYABLE_STATUS_CODES,
    SERVER_ERROR_STATUS_MAX_EXCLUSIVE,
    SUPPORTED_BACKOFF_STRATEGIES,
    SUPPORTED_CHECKPOINT_STRATEGIES,
    SUPPORTED_EXTRACTION_MODES,
)
from janus.models.config.issues import ValidationIssue
from janus.models.config.types import ExtractionConfig, RetryConfig


def _build_extraction_config(raw_value: Any, issues: list[ValidationIssue]) -> ExtractionConfig:
    """Validate extraction semantics such as mode, checkpointing, and retries."""
    data = _require_mapping(raw_value, "extraction", issues)
    mode = _require_enum(data, "mode", SUPPORTED_EXTRACTION_MODES, issues, "extraction")
    checkpoint_field = _optional_string(data, "checkpoint_field", issues, "extraction")
    checkpoint_strategy = _optional_enum(
        data,
        "checkpoint_strategy",
        SUPPORTED_CHECKPOINT_STRATEGIES,
        issues,
        "extraction",
        default="none",
    )
    lookback_days = _optional_int(data, "lookback_days", issues, "extraction", minimum=0)
    dead_letter_max_items = _optional_int(
        data,
        "dead_letter_max_items",
        issues,
        "extraction",
        default=0,
        minimum=0,
    )
    retry = _build_retry_config(data.get("retry"), issues)

    if mode == "incremental":
        if not checkpoint_field:
            issues.append(
                ValidationIssue(
                    "extraction.checkpoint_field",
                    "is required when extraction.mode is 'incremental'",
                )
            )
        if checkpoint_strategy == "none":
            issues.append(
                ValidationIssue(
                    "extraction.checkpoint_strategy",
                    "must not be 'none' when extraction.mode is 'incremental'",
                )
            )

    return ExtractionConfig(
        mode=mode,
        checkpoint_field=checkpoint_field,
        checkpoint_strategy=checkpoint_strategy,
        lookback_days=lookback_days,
        dead_letter_max_items=dead_letter_max_items,
        retry=retry,
    )


def _build_retry_config(raw_value: Any, issues: list[ValidationIssue]) -> RetryConfig:
    """Validate retry settings and fill in the small defaults used by the registry."""
    data = _require_mapping(raw_value, "extraction.retry", issues)
    max_attempts = _optional_int(
        data, "max_attempts", issues, "extraction.retry", default=3, minimum=1
    )
    backoff_strategy = _optional_enum(
        data,
        "backoff_strategy",
        SUPPORTED_BACKOFF_STRATEGIES,
        issues,
        "extraction.retry",
        default="fixed",
    )
    backoff_seconds = _optional_int(
        data, "backoff_seconds", issues, "extraction.retry", default=1, minimum=1
    )
    raw_retryable = _optional_int_list(
        data, "retryable_status_codes", issues, "extraction.retry"
    )
    retryable_status_codes = _resolve_retryable_status_codes(raw_retryable, issues)

    return RetryConfig(
        max_attempts=max_attempts,
        backoff_strategy=backoff_strategy,
        backoff_seconds=backoff_seconds,
        retryable_status_codes=retryable_status_codes,
    )


def _resolve_retryable_status_codes(
    raw_codes: list[int] | None,
    issues: list[ValidationIssue],
) -> tuple[int, ...]:
    """Normalize the declared retryable statuses into a deterministic, validated tuple.

    An explicit empty list is honoured as "never retry on status" — a source that wants
    every non-2xx to fail on the first attempt is expressing a policy, not omitting one.
    Only error statuses may be listed: 2xx never reaches the classification branch, and a
    3xx is resolved by the transport, so naming either is a config mistake rather than a
    preference the loop could act on.
    """

    if raw_codes is None:
        return DEFAULT_RETRYABLE_STATUS_CODES

    accepted: set[int] = set()
    for index, code in enumerate(raw_codes):
        child_path = f"extraction.retry.retryable_status_codes[{index}]"
        if not CLIENT_ERROR_STATUS_MIN <= code < SERVER_ERROR_STATUS_MAX_EXCLUSIVE:
            issues.append(
                ValidationIssue(
                    child_path, "must be a 4xx or 5xx error status code"
                )
            )
            continue
        accepted.add(code)

    return tuple(sorted(accepted))
