"""The frozen block dataclasses a validated source config is made of.

Pure shape: no parsing, no issue collection. ``SourceConfig`` itself stays in
``janus.models.source_config`` because it carries ``from_mapping``, which imports every
builder — keeping it here would invert the package's dependency arrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final

from janus.models.config.constants import (
    DEFAULT_MAX_ARCHIVE_RATIO,
    DEFAULT_MAX_ARCHIVE_TOTAL_BYTES,
    DEFAULT_MAX_PAYLOAD_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_PAST_END_STATUS_CODES,
    DEFAULT_RETRYABLE_STATUS_CODES,
    MAX_REDIRECTS_CEILING,
)


@dataclass(frozen=True, slots=True)
class AuthConfig:
    type: str
    env_var: str | None = None
    header_name: str | None = None
    query_param: str | None = None
    username_env_var: str | None = None
    password_env_var: str | None = None
    token_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class PaginationConfig:
    """Pagination shape plus the end-of-stream evidence the strategy may rely on.

    ``past_end_status_codes`` are the client-error statuses this API returns when a
    page beyond the last one is requested — evidence that the stream ended, not that
    the request failed. ``total_count_field`` is a dotted path to a total-record count
    in the payload, used to cap speculative look-ahead when the API exposes one.
    """

    type: str
    page_param: str | None = None
    size_param: str | None = None
    page_size: int | None = None
    offset_param: str | None = None
    limit_param: str | None = None
    cursor_param: str | None = None
    past_end_status_codes: tuple[int, ...] = DEFAULT_PAST_END_STATUS_CODES
    total_count_field: str | None = None


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    requests_per_minute: int | None = None
    concurrency: int = 1
    backoff_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """Per-source ceilings on what remote content may cost. Structural, never policy."""

    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    max_archive_member_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_archive_total_bytes: int = DEFAULT_MAX_ARCHIVE_TOTAL_BYTES
    max_archive_ratio: int = DEFAULT_MAX_ARCHIVE_RATIO

    def __post_init__(self) -> None:
        positive_limits = {
            "max_payload_bytes": self.max_payload_bytes,
            "max_archive_member_bytes": self.max_archive_member_bytes,
            "max_archive_total_bytes": self.max_archive_total_bytes,
            "max_archive_ratio": self.max_archive_ratio,
        }
        for name, value in positive_limits.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if (
            not isinstance(self.max_redirects, int)
            or isinstance(self.max_redirects, bool)
            or not 0 <= self.max_redirects <= MAX_REDIRECTS_CEILING
        ):
            raise ValueError(
                f"max_redirects must be an integer between 0 and {MAX_REDIRECTS_CEILING}"
            )

        if self.max_archive_member_bytes > self.max_archive_total_bytes:
            raise ValueError(
                "max_archive_member_bytes must not exceed max_archive_total_bytes"
            )


@dataclass(frozen=True, slots=True)
class RequestInputsConfig:
    type: str

    @property
    def requires_spark(self) -> bool:
        """Return whether loading these request inputs needs a live SparkSession.

        Answered from config alone, before any session exists, so callers can decide
        whether extraction must hold compute at all. ``none`` and ``date_window`` are
        synthesized in pure Python and never need one.
        """
        return False


_INVALID_DATE_BOUND: Final[date] = date(1, 1, 1)


@dataclass(frozen=True, slots=True)
class DateWindowRequestInputsConfig(RequestInputsConfig):
    start: date
    end: date
    step: str

    def __post_init__(self) -> None:
        """Reject the placeholder values a failed parse used to substitute."""
        if _INVALID_DATE_BOUND in (self.start, self.end):
            raise ValueError(
                "date_window start/end must be real dates; 0001-01-01 is a "
                "parse-failure placeholder and is not a valid window bound"
            )
        if not self.step:
            raise ValueError("date_window step must not be empty")


@dataclass(frozen=True, slots=True)
class IcebergRowsRequestInputsConfig(RequestInputsConfig):
    """One upstream bronze table read, and the source declared to produce it.

    ``upstream_source_id`` names the producer; ``namespace``/``table_name`` stay the
    data dependency. The declaration never substitutes for the table reference — a
    consumer that names a producer and reads a table that producer does not write is
    an inconsistency for registry validation to reject, not one to paper over here.
    """

    upstream_source_id: str
    namespace: str
    table_name: str
    columns: dict[str, str]
    distinct: bool = False

    def __post_init__(self) -> None:
        """Reject a producer declaration that identifies nobody.

        The parser already collects this as an issue; the invariant is here so a
        directly constructed config — a test fixture, a future caller — cannot hold an
        edge the dependency graph would have to guess at.
        """
        if not isinstance(self.upstream_source_id, str) or not self.upstream_source_id.strip():
            raise ValueError("iceberg_rows upstream_source_id must be a non-empty string")

    @property
    def requires_spark(self) -> bool:
        """Return True: the projected rows are read from an upstream Iceberg table."""
        return True


@dataclass(frozen=True, slots=True)
class CombinedRequestInputsConfig(RequestInputsConfig):
    inputs: tuple[RequestInputsConfig, ...]

    @property
    def requires_spark(self) -> bool:
        """Return True when any sub-input needs Spark, since all of them are loaded."""
        return any(sub_input.requires_spark for sub_input in self.inputs)


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    from_: str
    format: str | None = None


@dataclass(frozen=True, slots=True)
class AccessConfig:
    format: str
    method: str
    timeout_seconds: int
    auth: AuthConfig
    pagination: PaginationConfig
    rate_limit: RateLimitConfig
    request_inputs: RequestInputsConfig
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    base_url: str | None = None
    path: str | None = None
    url: str | None = None
    discovery_pattern: str | None = None
    remote_file_pattern: str | None = None
    file_pattern: str | None = None
    headers: dict[str, str] | None = None
    params: dict[str, str] | None = None
    parameter_bindings: dict[str, ParameterBinding] | None = None
    link_resolver: str = "auto"
    allowed_hosts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """How one HTTP attempt is repeated, and which responses are worth repeating.

    ``retryable_status_codes`` is per-source because "this status is transient" is a
    property of the upstream API, not of HTTP. Portal da Transparência, for one, answers a
    perfectly valid page with ``400`` under load; with the hard-coded set that single
    response ended a multi-hour run on its first attempt, because a status outside the set
    is raised without consuming ``max_attempts``. Declaring it here makes ``max_attempts``
    actually govern it.
    """

    max_attempts: int
    backoff_strategy: str
    backoff_seconds: int
    retryable_status_codes: tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    mode: str
    retry: RetryConfig
    checkpoint_field: str | None = None
    checkpoint_strategy: str = "none"
    lookback_days: int | None = None
    dead_letter_max_items: int = 0


@dataclass(frozen=True, slots=True)
class SchemaConfig:
    """How one source declares the shape of what it writes."""

    contract: str


@dataclass(frozen=True, slots=True)
class SparkConfig:
    input_format: str
    write_mode: str
    repartition: int | None = None
    partition_by: tuple[str, ...] = ()
    read_options: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class OutputTarget:
    """One configured output zone, plus the Iceberg identity a bronze target carries."""

    path: str
    format: str
    namespace: str | None = None
    table_name: str | None = None
    shared_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a co-writer list that names nobody, or names somebody twice."""
        if any(not peer.strip() for peer in self.shared_with):
            raise ValueError("shared_with entries must be non-empty source ids")
        if len(set(self.shared_with)) != len(self.shared_with):
            raise ValueError("shared_with must not repeat a source id")


@dataclass(frozen=True, slots=True)
class OutputsConfig:
    raw: OutputTarget
    bronze: OutputTarget
    metadata: OutputTarget


@dataclass(frozen=True, slots=True)
class QualityConfig:
    required_fields: tuple[str, ...] = ()
    unique_fields: tuple[str, ...] = ()
