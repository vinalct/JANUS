"""Builders for the ``access`` block — URL shape, auth, pagination, and rate limiting.

The top of the builder layer: ``_build_access_config`` composes the request-input and
parameter-binding builders in the order their issues must be reported, which is why it
is the one builder module that depends on siblings.
"""

from __future__ import annotations

from typing import Any

from janus.models.config.bindings import _build_parameter_bindings_config
from janus.models.config.coercion import (
    _optional_enum,
    _optional_int,
    _optional_int_list,
    _optional_string,
    _optional_string_mapping,
    _require_enum,
    _require_mapping,
)
from janus.models.config.constants import (
    ALLOWED_HOST_PATTERN,
    CLIENT_ERROR_STATUS_MAX_EXCLUSIVE,
    CLIENT_ERROR_STATUS_MIN,
    DEFAULT_MAX_ARCHIVE_RATIO,
    DEFAULT_MAX_ARCHIVE_TOTAL_BYTES,
    DEFAULT_MAX_PAYLOAD_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_PAST_END_STATUS_CODES,
    MAX_REDIRECTS_CEILING,
    RETRYABLE_CLIENT_STATUS_CODES,
    SUPPORTED_AUTH_TYPES,
    SUPPORTED_DATA_FORMATS,
    SUPPORTED_HTTP_METHODS,
    SUPPORTED_LINK_RESOLVERS,
    SUPPORTED_PAGINATION_TYPES,
)
from janus.models.config.issues import ValidationIssue
from janus.models.config.request_inputs import _build_request_inputs_config
from janus.models.config.types import (
    AccessConfig,
    AuthConfig,
    LimitsConfig,
    PaginationConfig,
    RateLimitConfig,
)


def _build_access_config(
    raw_value: Any, source_type: str, issues: list[ValidationIssue]
) -> AccessConfig:
    """Validate and normalize the access block shared by all source families."""
    data = _require_mapping(raw_value, "access", issues)

    format_name = _require_enum(data, "format", SUPPORTED_DATA_FORMATS, issues, "access")
    method = _require_enum(data, "method", SUPPORTED_HTTP_METHODS, issues, "access")
    timeout_seconds = _optional_int(
        data, "timeout_seconds", issues, "access", default=60, minimum=1
    )
    base_url = _optional_string(data, "base_url", issues, "access")
    path = _optional_string(data, "path", issues, "access")
    url = _optional_string(data, "url", issues, "access")
    discovery_pattern = _optional_string(data, "discovery_pattern", issues, "access")
    remote_file_pattern = _optional_string(data, "remote_file_pattern", issues, "access")
    file_pattern = _optional_string(data, "file_pattern", issues, "access")
    headers = _optional_string_mapping(data, "headers", issues, "access")
    params = _optional_string_mapping(data, "params", issues, "access")
    request_inputs = _build_request_inputs_config(
        data.get("request_inputs"),
        source_type,
        issues,
    )
    parameter_bindings = _build_parameter_bindings_config(
        data.get("parameter_bindings"),
        source_type,
        request_inputs,
        issues,
    )

    if source_type in {"api", "catalog"} and not (base_url or url):
        issues.append(
            ValidationIssue(
                "access.base_url",
                "or access.url is required for api and catalog sources",
            )
        )

    if source_type == "file" and not (url or path or discovery_pattern):
        issues.append(
            ValidationIssue(
                "access.url",
                "or access.path or access.discovery_pattern is required for file sources",
            )
        )

    auth = _build_auth_config(data.get("auth"), issues)
    pagination = _build_pagination_config(data.get("pagination"), issues)
    rate_limit = _build_rate_limit_config(data.get("rate_limit"), issues)
    limits = _build_limits_config(data.get("limits"), issues)

    if params and parameter_bindings:
        duplicate_keys = sorted(set(params).intersection(parameter_bindings))
        for key in duplicate_keys:
            issues.append(
                ValidationIssue(
                    f"access.parameter_bindings.{key}",
                    (
                        f"duplicates access.params.{key}; declare the parameter in "
                        "only one place"
                    ),
                )
            )

    link_resolver = _optional_enum(
        data, "link_resolver", SUPPORTED_LINK_RESOLVERS, issues, "access", default="auto"
    )
    allowed_hosts = _build_allowed_hosts(data.get("allowed_hosts"), issues)

    configured_urls = (url, base_url)
    if any(
        value is not None and not value.lower().startswith(("http://", "https://"))
        for value in configured_urls
    ):
        issues.append(ValidationIssue("access.url", "must use the http or https scheme"))

    return AccessConfig(
        format=format_name,
        method=method,
        timeout_seconds=timeout_seconds,
        base_url=base_url,
        path=path,
        url=url,
        discovery_pattern=discovery_pattern,
        remote_file_pattern=remote_file_pattern,
        file_pattern=file_pattern,
        headers=headers,
        params=params,
        parameter_bindings=parameter_bindings,
        auth=auth,
        pagination=pagination,
        rate_limit=rate_limit,
        limits=limits,
        request_inputs=request_inputs,
        link_resolver=link_resolver,
        allowed_hosts=allowed_hosts,
    )


def _build_limits_config(raw_value: Any, issues: list[ValidationIssue]) -> LimitsConfig:
    """Validate per-source remote-content ceilings without consulting policy."""
    if raw_value is None:
        return LimitsConfig()

    issue_count = len(issues)
    data = _require_mapping(raw_value, "access.limits", issues)
    if len(issues) != issue_count:
        return LimitsConfig()

    max_payload_bytes = _read_limit_int(
        data, "max_payload_bytes", issues, DEFAULT_MAX_PAYLOAD_BYTES, minimum=1
    )
    max_redirects = _read_limit_int(
        data, "max_redirects", issues, DEFAULT_MAX_REDIRECTS, minimum=0
    )
    max_archive_member_bytes = _read_limit_int(
        data, "max_archive_member_bytes", issues, max_payload_bytes, minimum=1
    )
    max_archive_total_bytes = _read_limit_int(
        data,
        "max_archive_total_bytes",
        issues,
        DEFAULT_MAX_ARCHIVE_TOTAL_BYTES,
        minimum=1,
    )
    max_archive_ratio = _read_limit_int(
        data, "max_archive_ratio", issues, DEFAULT_MAX_ARCHIVE_RATIO, minimum=1
    )

    if max_redirects > MAX_REDIRECTS_CEILING:
        issues.append(
            ValidationIssue(
                "access.limits.max_redirects",
                f"must be at most {MAX_REDIRECTS_CEILING}",
            )
        )
    if (
        max_archive_member_bytes >= 1
        and max_archive_total_bytes >= 1
        and max_archive_member_bytes > max_archive_total_bytes
    ):
        issues.append(
            ValidationIssue(
                "access.limits.max_archive_member_bytes",
                "must not exceed access.limits.max_archive_total_bytes",
            )
        )

    if len(issues) != issue_count:
        return LimitsConfig()

    # Unknown keys are intentionally ignored, matching the other nested builders.
    return LimitsConfig(
        max_payload_bytes=max_payload_bytes,
        max_redirects=max_redirects,
        max_archive_member_bytes=max_archive_member_bytes,
        max_archive_total_bytes=max_archive_total_bytes,
        max_archive_ratio=max_archive_ratio,
    )


def _read_limit_int(
    data: Any,
    field_name: str,
    issues: list[ValidationIssue],
    default: int,
    *,
    minimum: int,
) -> int:
    """Read one limit, treating an explicit null as malformed rather than absent."""
    if field_name in data and data[field_name] is None:
        issues.append(
            ValidationIssue(f"access.limits.{field_name}", "must be an integer")
        )
        return default
    return _optional_int(
        data,
        field_name,
        issues,
        "access.limits",
        default=default,
        minimum=minimum,
    )


def _build_allowed_hosts(raw_value: Any, issues: list[ValidationIssue]) -> tuple[str, ...]:
    """Normalize the optional hostname allow-list and collect every malformed entry."""
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        issues.append(ValidationIssue("access.allowed_hosts", "must be a list of hostnames"))
        return ()

    accepted: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_value):
        field_path = f"access.allowed_hosts[{index}]"
        if not isinstance(entry, str) or not entry.strip():
            issues.append(ValidationIssue(field_path, "must be a non-empty string"))
            continue

        normalized = entry.strip().lower()
        if normalized in {"*", "*."}:
            issues.append(ValidationIssue(field_path, "an allow-all wildcard is not an allow-list"))
            continue
        if ALLOWED_HOST_PATTERN.fullmatch(normalized) is None:
            issues.append(
                ValidationIssue(
                    field_path,
                    (
                        "must be a bare hostname, optionally prefixed with '*.' "
                        "(no scheme, port or path)"
                    ),
                )
            )
            continue
        if normalized in seen:
            issues.append(ValidationIssue(field_path, "duplicates an earlier entry"))
            continue

        accepted.append(normalized)
        seen.add(normalized)

    return tuple(accepted)


def _build_auth_config(raw_value: Any, issues: list[ValidationIssue]) -> AuthConfig:
    """Validate and normalize the nested auth settings inside the access block."""
    data = _require_mapping(raw_value, "access.auth", issues)
    auth_type = _require_enum(data, "type", SUPPORTED_AUTH_TYPES, issues, "access.auth")
    env_var = _optional_string(data, "env_var", issues, "access.auth")
    header_name = _optional_string(data, "header_name", issues, "access.auth")
    query_param = _optional_string(data, "query_param", issues, "access.auth")
    username_env_var = _optional_string(data, "username_env_var", issues, "access.auth")
    password_env_var = _optional_string(data, "password_env_var", issues, "access.auth")
    token_prefix = _optional_string(data, "token_prefix", issues, "access.auth")

    if auth_type in {"header_token", "bearer_token"} and not env_var:
        issues.append(
            ValidationIssue("access.auth.env_var", "is required for token-based auth")
        )

    if auth_type == "header_token" and not header_name:
        issues.append(
            ValidationIssue(
                "access.auth.header_name",
                "is required when access.auth.type is 'header_token'",
            )
        )

    if auth_type == "query_token" and not query_param:
        issues.append(
            ValidationIssue(
                "access.auth.query_param",
                "is required when access.auth.type is 'query_token'",
            )
        )

    if auth_type == "basic":
        if not username_env_var:
            issues.append(
                ValidationIssue(
                    "access.auth.username_env_var",
                    "is required when access.auth.type is 'basic'",
                )
            )
        if not password_env_var:
            issues.append(
                ValidationIssue(
                    "access.auth.password_env_var",
                    "is required when access.auth.type is 'basic'",
                )
            )

    if auth_type == "bearer_token" and header_name is None:
        header_name = "Authorization"

    return AuthConfig(
        type=auth_type,
        env_var=env_var,
        header_name=header_name,
        query_param=query_param,
        username_env_var=username_env_var,
        password_env_var=password_env_var,
        token_prefix=token_prefix,
    )


def _build_pagination_config(raw_value: Any, issues: list[ValidationIssue]) -> PaginationConfig:
    """Validate pagination settings and enforce the fields required by each mode."""
    data = _require_mapping(raw_value, "access.pagination", issues)
    pagination_type = _require_enum(
        data, "type", SUPPORTED_PAGINATION_TYPES, issues, "access.pagination"
    )
    page_param = _optional_string(data, "page_param", issues, "access.pagination")
    size_param = _optional_string(data, "size_param", issues, "access.pagination")
    page_size = _optional_int(
        data, "page_size", issues, "access.pagination", minimum=1
    )
    offset_param = _optional_string(data, "offset_param", issues, "access.pagination")
    limit_param = _optional_string(data, "limit_param", issues, "access.pagination")
    cursor_param = _optional_string(data, "cursor_param", issues, "access.pagination")
    raw_past_end = _optional_int_list(data, "past_end_status_codes", issues, "access.pagination")
    past_end_status_codes = _resolve_past_end_status_codes(raw_past_end, issues)
    total_count_field = _optional_string(data, "total_count_field", issues, "access.pagination")
    if total_count_field is not None:
        _validate_dotted_path(
            total_count_field, "access.pagination.total_count_field", issues
        )

    if pagination_type == "page_number":
        if not page_param:
            issues.append(
                ValidationIssue(
                    "access.pagination.page_param",
                    "is required when access.pagination.type is 'page_number'",
                )
            )
        if not size_param:
            issues.append(
                ValidationIssue(
                    "access.pagination.size_param",
                    "is required when access.pagination.type is 'page_number'",
                )
            )
        if page_size is None:
            issues.append(
                ValidationIssue(
                    "access.pagination.page_size",
                    "is required when access.pagination.type is 'page_number'",
                )
            )

    if pagination_type == "offset":
        if not offset_param:
            issues.append(
                ValidationIssue(
                    "access.pagination.offset_param",
                    "is required when access.pagination.type is 'offset'",
                )
            )
        if not limit_param:
            issues.append(
                ValidationIssue(
                    "access.pagination.limit_param",
                    "is required when access.pagination.type is 'offset'",
                )
            )
        if page_size is None:
            issues.append(
                ValidationIssue(
                    "access.pagination.page_size",
                    "is required when access.pagination.type is 'offset'",
                )
            )

    if pagination_type == "cursor" and not cursor_param:
        issues.append(
            ValidationIssue(
                "access.pagination.cursor_param",
                "is required when access.pagination.type is 'cursor'",
            )
        )

    return PaginationConfig(
        type=pagination_type,
        page_param=page_param,
        size_param=size_param,
        page_size=page_size,
        offset_param=offset_param,
        limit_param=limit_param,
        cursor_param=cursor_param,
        past_end_status_codes=past_end_status_codes,
        total_count_field=total_count_field,
    )


def _resolve_past_end_status_codes(
    raw_codes: list[int] | None,
    issues: list[ValidationIssue],
) -> tuple[int, ...]:
    """Normalize the declared past-end statuses into a deterministic, validated tuple."""

    if raw_codes is None:
        return DEFAULT_PAST_END_STATUS_CODES

    accepted: set[int] = set()
    for index, code in enumerate(raw_codes):
        child_path = f"access.pagination.past_end_status_codes[{index}]"
        if not CLIENT_ERROR_STATUS_MIN <= code < CLIENT_ERROR_STATUS_MAX_EXCLUSIVE:
            issues.append(
                ValidationIssue(child_path, "must be a 4xx client-error status code")
            )
            continue
        if code in RETRYABLE_CLIENT_STATUS_CODES:
            retryable = ", ".join(str(item) for item in sorted(RETRYABLE_CLIENT_STATUS_CODES))
            issues.append(
                ValidationIssue(
                    child_path,
                    f"must not be a retryable status code ({retryable})",
                )
            )
            continue
        accepted.add(code)

    return tuple(sorted(accepted))


def _validate_dotted_path(
    value: str,
    field_path: str,
    issues: list[ValidationIssue],
) -> None:
    """Check the shape of a dotted payload path; resolution semantics live downstream."""
    if any(not segment.strip() for segment in value.split(".")):
        issues.append(
            ValidationIssue(
                field_path,
                "must be a dotted path without empty segments, e.g. 'meta.total'",
            )
        )


def _build_rate_limit_config(raw_value: Any, issues: list[ValidationIssue]) -> RateLimitConfig:
    """Validate the rate-limit block and apply safe numeric defaults where allowed."""
    data = _require_mapping(raw_value, "access.rate_limit", issues)
    requests_per_minute = _optional_int(
        data, "requests_per_minute", issues, "access.rate_limit", minimum=1
    )
    concurrency = _optional_int(
        data, "concurrency", issues, "access.rate_limit", default=1, minimum=1
    )
    backoff_seconds = _optional_int(
        data, "backoff_seconds", issues, "access.rate_limit", minimum=1
    )

    return RateLimitConfig(
        requests_per_minute=requests_per_minute,
        concurrency=concurrency,
        backoff_seconds=backoff_seconds,
    )
