"""Closed value sets the source-config contract validates against.

The bottom of the package layering: this module imports nothing from
``janus.models.config``, so every other module in it may depend on this one.
"""

from __future__ import annotations

SUPPORTED_SOURCE_TYPES = frozenset({"api", "catalog", "file"})
SUPPORTED_STRATEGIES = SUPPORTED_SOURCE_TYPES
SUPPORTED_STRATEGY_VARIANTS = {
    "api": frozenset(
        {"cursor_api", "date_window_api", "offset_api", "page_number_api"}
    ),
    "catalog": frozenset({"metadata_catalog", "resource_catalog"}),
    "file": frozenset({"archive_package", "static_file", "versioned_file"}),
}
SUPPORTED_AUTH_TYPES = frozenset(
    {"basic", "bearer_token", "header_token", "none", "query_token"}
)
SUPPORTED_EXTRACTION_MODES = frozenset({"full_refresh", "incremental", "snapshot"})
SUPPORTED_CHECKPOINT_STRATEGIES = frozenset({"date_window", "max_value", "none"})
SUPPORTED_PAGINATION_TYPES = frozenset({"cursor", "none", "offset", "page_number"})
DEFAULT_PAST_END_STATUS_CODES: tuple[int, ...] = (404, 416)
RETRYABLE_CLIENT_STATUS_CODES: frozenset[int] = frozenset({408, 429})
CONCURRENT_PAGINATION_TYPES: frozenset[str] = frozenset({"page_number", "offset"})
SUPPORTED_SCHEMA_MODES = frozenset({"explicit", "infer"})
SUPPORTED_DATA_FORMATS = frozenset(
    {"binary", "csv", "iceberg", "json", "jsonl", "parquet", "text"}
)
SUPPORTED_WRITE_MODES = frozenset({"append", "ignore", "overwrite"})
SUPPORTED_BACKOFF_STRATEGIES = frozenset({"exponential", "fixed"})
SUPPORTED_HTTP_METHODS = frozenset({"DELETE", "GET", "PATCH", "POST", "PUT"})
SUPPORTED_FEDERATION_LEVELS = frozenset({"federal"})
SUPPORTED_REQUEST_INPUT_TYPES = frozenset({"combined", "date_window", "iceberg_rows", "none"})
SUPPORTED_LINK_RESOLVERS = frozenset({"auto", "direct", "html_links", "nextcloud_webdav"})
SUPPORTED_REQUEST_INPUT_STEPS = frozenset({"day", "month"})
_SUPPORTED_SUB_REQUEST_INPUT_TYPES = frozenset({"date_window", "iceberg_rows"})
SUPPORTED_PARAMETER_BINDING_WINDOW_SOURCES = frozenset(
    {"request_input.window_end", "request_input.window_start"}
)
REQUEST_INPUT_BINDING_PREFIX = "request_input."
