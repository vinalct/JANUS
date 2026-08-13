"""The frozen block dataclasses a validated source config is made of.

Pure shape: no parsing, no issue collection. ``SourceConfig`` itself stays in
``janus.models.source_config`` because it carries ``from_mapping``, which imports every
builder — keeping it here would invert the package's dependency arrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

from janus.models.config.constants import DEFAULT_PAST_END_STATUS_CODES


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
    namespace: str
    table_name: str
    columns: dict[str, str]
    distinct: bool = False

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


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int
    backoff_strategy: str
    backoff_seconds: int


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
    mode: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class SparkConfig:
    input_format: str
    write_mode: str
    repartition: int | None = None
    partition_by: tuple[str, ...] = ()
    read_options: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class OutputTarget:
    path: str
    format: str
    namespace: str | None = None
    table_name: str | None = None


@dataclass(frozen=True, slots=True)
class OutputsConfig:
    raw: OutputTarget
    bronze: OutputTarget
    metadata: OutputTarget


@dataclass(frozen=True, slots=True)
class QualityConfig:
    required_fields: tuple[str, ...] = ()
    unique_fields: tuple[str, ...] = ()
    allow_schema_evolution: bool = False
