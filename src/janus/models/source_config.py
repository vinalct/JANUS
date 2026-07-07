from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Self

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


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str

    def render(self) -> str:
        """Return the issue in the same path-first format used in validation errors."""
        return f"{self.path}: {self.message}"


class SourceConfigValidationError(ValueError):
    def __init__(self, config_path: Path, issues: list[ValidationIssue]) -> None:
        """Build a readable validation error for a single source config file."""
        self.config_path = config_path
        self.issues = tuple(issues)
        message_lines = [f"Invalid source config: {config_path}"]
        message_lines.extend(f"- {issue.render()}" for issue in self.issues)
        super().__init__("\n".join(message_lines))


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
    type: str
    page_param: str | None = None
    size_param: str | None = None
    page_size: int | None = None
    offset_param: str | None = None
    limit_param: str | None = None
    cursor_param: str | None = None


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    requests_per_minute: int | None = None
    concurrency: int = 1
    backoff_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class RequestInputsConfig:
    type: str


@dataclass(frozen=True, slots=True)
class DateWindowRequestInputsConfig(RequestInputsConfig):
    start: date
    end: date
    step: str


@dataclass(frozen=True, slots=True)
class IcebergRowsRequestInputsConfig(RequestInputsConfig):
    namespace: str
    table_name: str
    columns: dict[str, str]
    distinct: bool = False


@dataclass(frozen=True, slots=True)
class CombinedRequestInputsConfig(RequestInputsConfig):
    inputs: tuple[RequestInputsConfig, ...]


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


@dataclass(frozen=True, slots=True)
class SourceConfig:
    config_path: Path
    source_id: str
    name: str
    owner: str
    enabled: bool
    source_type: str
    strategy: str
    strategy_variant: str
    federation_level: str
    domain: str
    public_access: bool
    access: AccessConfig
    extraction: ExtractionConfig
    schema: SchemaConfig
    spark: SparkConfig
    outputs: OutputsConfig
    quality: QualityConfig
    description: str | None = None
    source_hook: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], config_path: Path) -> Self:
        """Validate a raw source mapping and return the typed source contract."""
        issues: list[ValidationIssue] = []

        source_id = _require_string(data, "source_id", issues)
        name = _require_string(data, "name", issues)
        owner = _require_string(data, "owner", issues)
        enabled = _require_bool(data, "enabled", issues)
        source_type = _require_enum(data, "source_type", SUPPORTED_SOURCE_TYPES, issues)
        strategy = _require_enum(data, "strategy", SUPPORTED_STRATEGIES, issues)
        strategy_variant = _require_string(data, "strategy_variant", issues)
        federation_level = _require_enum(
            data, "federation_level", SUPPORTED_FEDERATION_LEVELS, issues
        )
        domain = _require_string(data, "domain", issues)
        public_access = _require_bool(data, "public_access", issues)
        description = _optional_string(data, "description", issues)
        source_hook = _optional_string(data, "source_hook", issues)
        tags = _optional_string_list(data, "tags", issues)

        if source_type and strategy and source_type != strategy:
            issues.append(
                ValidationIssue(
                    "strategy",
                    (
                        f"must match source_type {source_type!r} "
                        "for the current JANUS strategy families"
                    ),
                )
            )

        if strategy and strategy_variant:
            supported_variants = SUPPORTED_STRATEGY_VARIANTS.get(strategy, frozenset())
            if strategy_variant not in supported_variants:
                allowed_variants = ", ".join(sorted(supported_variants))
                issues.append(
                    ValidationIssue(
                        "strategy_variant",
                        f"must be one of: {allowed_variants}",
                    )
                )

        if public_access is False:
            issues.append(
                ValidationIssue(
                    "public_access",
                    "must be true because JANUS only supports public federal sources in phase 1",
                )
            )

        access = _build_access_config(data.get("access"), source_type, issues)
        extraction = _build_extraction_config(data.get("extraction"), issues)
        schema = _build_schema_config(data.get("schema"), issues)
        spark = _build_spark_config(data.get("spark"), issues)
        outputs = _build_outputs_config(data.get("outputs"), issues)
        quality = _build_quality_config(data.get("quality"), issues)

        if issues:
            raise SourceConfigValidationError(config_path, issues)

        return cls(
            config_path=config_path,
            source_id=source_id,
            name=name,
            description=description,
            owner=owner,
            enabled=enabled,
            source_type=source_type,
            strategy=strategy,
            strategy_variant=strategy_variant,
            source_hook=source_hook,
            federation_level=federation_level,
            domain=domain,
            public_access=public_access,
            tags=tuple(tags),
            access=access,
            extraction=extraction,
            schema=schema,
            spark=spark,
            outputs=outputs,
            quality=quality,
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
        request_inputs=request_inputs,
        link_resolver=link_resolver,
    )


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


def _build_request_inputs_config(
    raw_value: Any,
    source_type: str,
    issues: list[ValidationIssue],
) -> RequestInputsConfig:
    """Validate and normalize API request-input configuration with a safe default."""
    if raw_value is None:
        return RequestInputsConfig(type="none")

    if source_type not in ("api", "catalog"):
        issues.append(
            ValidationIssue(
                "access.request_inputs",
                "is only supported for api and catalog sources",
            )
        )
        return RequestInputsConfig(type="none")

    data = _require_mapping(raw_value, "access.request_inputs", issues)
    request_input_type = _require_enum(
        data,
        "type",
        SUPPORTED_REQUEST_INPUT_TYPES,
        issues,
        "access.request_inputs",
    )

    if request_input_type == "combined":
        return _build_combined_request_inputs_config(data, issues)

    return _parse_request_input_entry(
        data, request_input_type, "access.request_inputs", issues
    )


def _parse_request_input_entry(
    data: Mapping[str, Any],
    input_type: str,
    prefix: str,
    issues: list[ValidationIssue],
) -> RequestInputsConfig:
    """Parse one atomic request-input config from an already-validated dict."""
    if input_type == "date_window":
        start = _require_date(data, "start", issues, prefix)
        end = _require_date(data, "end", issues, prefix)
        step = _require_enum(data, "step", SUPPORTED_REQUEST_INPUT_STEPS, issues, prefix)

        if start is not None and end is not None and start > end:
            issues.append(
                ValidationIssue(
                    f"{prefix}.end",
                    f"must be on or after {prefix}.start",
                )
            )

        return DateWindowRequestInputsConfig(
            type=input_type,
            start=start or date.min,
            end=end or date.min,
            step=step,
        )

    if input_type == "iceberg_rows":
        namespace = _require_string(data, "namespace", issues, prefix)
        table_name = _require_string(data, "table_name", issues, prefix)
        columns_value = data.get("columns")
        columns = _require_non_empty_string_mapping(
            columns_value,
            f"{prefix}.columns",
            issues,
        )
        if isinstance(columns_value, Mapping) and not columns_value:
            issues.append(
                ValidationIssue(
                    f"{prefix}.columns",
                    "must not be empty",
                )
            )
        distinct = _optional_bool(data, "distinct", issues, prefix, default=False)

        return IcebergRowsRequestInputsConfig(
            type=input_type,
            namespace=namespace,
            table_name=table_name,
            columns=columns,
            distinct=distinct,
        )

    return RequestInputsConfig(type="none")


def _build_combined_request_inputs_config(
    data: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> RequestInputsConfig:
    """Validate and build a combined request-input config from a list of sub-inputs."""
    inputs_raw = data.get("inputs")
    if not isinstance(inputs_raw, list):
        issues.append(
            ValidationIssue(
                "access.request_inputs.inputs",
                "is required and must be a list when type is 'combined'",
            )
        )
        return CombinedRequestInputsConfig(type="combined", inputs=())

    if len(inputs_raw) < 2:
        issues.append(
            ValidationIssue(
                "access.request_inputs.inputs",
                "must contain at least 2 entries when type is 'combined'",
            )
        )
        return CombinedRequestInputsConfig(type="combined", inputs=())

    sub_configs: list[RequestInputsConfig] = []
    seen_fields: set[str] = set()

    for idx, sub_raw in enumerate(inputs_raw):
        sub_prefix = f"access.request_inputs.inputs[{idx}]"
        if not isinstance(sub_raw, Mapping):
            issues.append(ValidationIssue(sub_prefix, "must be a mapping"))
            continue

        sub_type = _require_enum(
            sub_raw,
            "type",
            _SUPPORTED_SUB_REQUEST_INPUT_TYPES,
            issues,
            sub_prefix,
        )
        if not sub_type:
            continue

        sub_config = _parse_request_input_entry(sub_raw, sub_type, sub_prefix, issues)

        sub_fields = _request_input_field_names_for(sub_config)
        conflicts = seen_fields.intersection(sub_fields)
        if conflicts:
            conflicting = ", ".join(sorted(conflicts))
            issues.append(
                ValidationIssue(
                    sub_prefix,
                    f"field name(s) {conflicting!r} conflict with another input "
                    "in this combined config",
                )
            )
        seen_fields.update(sub_fields)
        sub_configs.append(sub_config)

    return CombinedRequestInputsConfig(type="combined", inputs=tuple(sub_configs))


def _request_input_field_names_for(config: RequestInputsConfig) -> frozenset[str]:
    """Return the set of field names that a request-input config exposes at runtime."""
    if config.type == "date_window":
        return frozenset({"window_start", "window_end"})
    if config.type == "iceberg_rows" and isinstance(config, IcebergRowsRequestInputsConfig):
        return frozenset(config.columns.keys())
    return frozenset()


def _build_parameter_bindings_config(
    raw_value: Any,
    source_type: str,
    request_inputs: RequestInputsConfig,
    issues: list[ValidationIssue],
) -> dict[str, ParameterBinding] | None:
    """Validate declarative runtime request-parameter bindings for API sources."""
    if raw_value is None:
        return None

    if source_type not in ("api", "catalog"):
        issues.append(
            ValidationIssue(
                "access.parameter_bindings",
                "is only supported for api and catalog sources",
            )
        )
        return None

    data = _require_mapping(raw_value, "access.parameter_bindings", issues)
    bindings: dict[str, ParameterBinding] = {}

    for key, item in data.items():
        binding_path = f"access.parameter_bindings.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(binding_path, "keys must be strings"))
            continue

        parameter_name = key.strip()
        if not parameter_name:
            issues.append(ValidationIssue(binding_path, "must not be empty"))
            continue
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(binding_path, "must be a mapping"))
            continue

        from_source = _require_string(item, "from", issues, binding_path)
        output_format = _optional_string(item, "format", issues, binding_path)
        if from_source:
            _validate_parameter_binding_source(
                from_source,
                request_inputs,
                issues,
                f"{binding_path}.from",
            )

        bindings[parameter_name] = ParameterBinding(
            from_=from_source,
            format=output_format,
        )

    return bindings


def _validate_parameter_binding_source(
    from_source: str,
    request_inputs: RequestInputsConfig,
    issues: list[ValidationIssue],
    field_path: str,
) -> None:
    """Validate the limited phase-1 binding sources supported by the API contract."""
    if from_source == "checkpoint_value":
        return

    if not from_source.startswith(REQUEST_INPUT_BINDING_PREFIX):
        issues.append(
            ValidationIssue(
                field_path,
                (
                    "must be 'checkpoint_value', 'request_input.window_start', "
                    "'request_input.window_end', or 'request_input.<field>'"
                ),
            )
        )
        return

    request_input_field = from_source.removeprefix(REQUEST_INPUT_BINDING_PREFIX).strip()
    if not request_input_field:
        issues.append(
            ValidationIssue(
                field_path,
                "request_input bindings must reference a field name",
            )
        )
        return

    if request_inputs.type == "none":
        issues.append(
            ValidationIssue(
                field_path,
                "requires access.request_inputs to declare a non-'none' type",
            )
        )
        return

    if request_inputs.type == "date_window":
        if from_source not in SUPPORTED_PARAMETER_BINDING_WINDOW_SOURCES:
            issues.append(
                ValidationIssue(
                    field_path,
                    (
                        "must be 'request_input.window_start' or "
                        "'request_input.window_end' when "
                        "access.request_inputs.type is 'date_window'"
                    ),
                )
            )
        return

    if request_inputs.type == "iceberg_rows":
        if not isinstance(request_inputs, IcebergRowsRequestInputsConfig):
            return

        if request_input_field in {"window_start", "window_end"}:
            issues.append(
                ValidationIssue(
                    field_path,
                    "must reference one of access.request_inputs.columns when "
                    "access.request_inputs.type is 'iceberg_rows'",
                )
            )
            return

        if request_input_field not in request_inputs.columns:
            allowed_fields = ", ".join(sorted(request_inputs.columns))
            issues.append(
                ValidationIssue(
                    field_path,
                    f"must reference one of access.request_inputs.columns: {allowed_fields}",
                )
            )

    if request_inputs.type == "combined":
        if not isinstance(request_inputs, CombinedRequestInputsConfig):
            return

        all_fields = frozenset(
            field
            for sub in request_inputs.inputs
            for field in _request_input_field_names_for(sub)
        )
        if request_input_field not in all_fields:
            allowed_fields = ", ".join(sorted(all_fields))
            issues.append(
                ValidationIssue(
                    field_path,
                    f"must reference one of the combined input fields: {allowed_fields}",
                )
            )


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

    return RetryConfig(
        max_attempts=max_attempts,
        backoff_strategy=backoff_strategy,
        backoff_seconds=backoff_seconds,
    )


def _build_schema_config(raw_value: Any, issues: list[ValidationIssue]) -> SchemaConfig:
    """Validate the schema block and require a path when explicit schemas are declared."""
    data = _require_mapping(raw_value, "schema", issues)
    mode = _require_enum(data, "mode", SUPPORTED_SCHEMA_MODES, issues, "schema")
    path = _optional_string(data, "path", issues, "schema")

    if mode == "explicit" and not path:
        issues.append(
            ValidationIssue(
                "schema.path",
                "is required when schema.mode is 'explicit'",
            )
        )

    return SchemaConfig(mode=mode, path=path)


def _build_spark_config(raw_value: Any, issues: list[ValidationIssue]) -> SparkConfig:
    """Validate the Spark-facing options that later tasks will consume."""
    data = _require_mapping(raw_value, "spark", issues)
    input_format = _require_enum(data, "input_format", SUPPORTED_DATA_FORMATS, issues, "spark")
    write_mode = _require_enum(data, "write_mode", SUPPORTED_WRITE_MODES, issues, "spark")
    repartition = _optional_int(data, "repartition", issues, "spark", minimum=1)
    partition_by = tuple(_optional_string_list(data, "partition_by", issues, "spark"))
    read_options = _optional_string_mapping(data, "read_options", issues, "spark")

    return SparkConfig(
        input_format=input_format,
        write_mode=write_mode,
        repartition=repartition,
        partition_by=partition_by,
        read_options=read_options,
    )


def _build_outputs_config(raw_value: Any, issues: list[ValidationIssue]) -> OutputsConfig:
    """Validate the output zone contract for raw, bronze, and metadata targets."""
    data = _require_mapping(raw_value, "outputs", issues)
    return OutputsConfig(
        raw=_build_output_target(data.get("raw"), "outputs.raw", issues),
        bronze=_build_output_target(data.get("bronze"), "outputs.bronze", issues),
        metadata=_build_output_target(data.get("metadata"), "outputs.metadata", issues),
    )


def _build_output_target(
    raw_value: Any, field_path: str, issues: list[ValidationIssue]
) -> OutputTarget:
    """Validate one concrete output target inside the outputs block."""
    data = _require_mapping(raw_value, field_path, issues)
    path = _require_string(data, "path", issues, field_path)
    format_name = _require_enum(data, "format", SUPPORTED_DATA_FORMATS, issues, field_path)
    namespace = _optional_string(data, "namespace", issues, field_path)
    table_name = _optional_string(data, "table_name", issues, field_path)

    if "table" in data and data["table"] is not None:
        issues.append(
            ValidationIssue(
                f"{field_path}.table",
                "is not supported; use table_name",
            )
        )

    if field_path != "outputs.bronze":
        if namespace is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.namespace",
                    "is only supported for outputs.bronze",
                )
            )
        if table_name is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.table_name",
                    "is only supported for outputs.bronze",
                )
            )

    if format_name != "iceberg":
        if namespace is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.namespace",
                    "requires format='iceberg'",
                )
            )
        if table_name is not None:
            issues.append(
                ValidationIssue(
                    f"{field_path}.table_name",
                    "requires format='iceberg'",
                )
            )

    return OutputTarget(
        path=path,
        format=format_name,
        namespace=namespace,
        table_name=table_name,
    )


def _build_quality_config(raw_value: Any, issues: list[ValidationIssue]) -> QualityConfig:
    """Validate the quality rules that travel with a source definition."""
    data = _require_mapping(raw_value, "quality", issues)
    required_fields = tuple(_optional_string_list(data, "required_fields", issues, "quality"))
    unique_fields = tuple(_optional_string_list(data, "unique_fields", issues, "quality"))
    allow_schema_evolution = _optional_bool(
        data, "allow_schema_evolution", issues, "quality", default=False
    )

    return QualityConfig(
        required_fields=required_fields,
        unique_fields=unique_fields,
        allow_schema_evolution=allow_schema_evolution,
    )


def _require_mapping(
    value: Any, field_path: str, issues: list[ValidationIssue]
) -> Mapping[str, Any]:
    """Return a mapping value or record a validation issue when the field is malformed."""
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return {}
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(field_path, "must be a mapping"))
        return {}
    return value


def _require_string(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str:
    """Read a required non-empty string field and register a clear error otherwise."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return ""
    if not isinstance(value, str):
        issues.append(ValidationIssue(field_path, "must be a string"))
        return ""
    value = value.strip()
    if not value:
        issues.append(ValidationIssue(field_path, "must not be empty"))
        return ""
    return value


def _optional_string(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str | None:
    """Read an optional string field while reusing the required-string validation rules."""
    if field_name not in data or data[field_name] is None:
        return None
    return _require_string(data, field_name, issues, prefix)


def _require_bool(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> bool:
    """Read a required boolean field and register an issue when the type is wrong."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return False
    if not isinstance(value, bool):
        issues.append(ValidationIssue(field_path, "must be a boolean"))
        return False
    return value


def _optional_bool(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: bool = False,
) -> bool:
    """Read an optional boolean field and fall back to the provided default."""
    if field_name not in data or data[field_name] is None:
        return default
    return _require_bool(data, field_name, issues, prefix)


def _require_enum(
    data: Mapping[str, Any],
    field_name: str,
    allowed_values: frozenset[str],
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str:
    """Read a required string field and ensure it belongs to the allowed value set."""
    value = _require_string(data, field_name, issues, prefix)
    if value and value not in allowed_values:
        issues.append(
            ValidationIssue(
                _field_path(field_name, prefix),
                f"must be one of: {', '.join(sorted(allowed_values))}",
            )
        )
    return value


def _optional_enum(
    data: Mapping[str, Any],
    field_name: str,
    allowed_values: frozenset[str],
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: str | None = None,
) -> str:
    """Read an optional enum field and return the configured default when absent."""
    if field_name not in data or data[field_name] is None:
        return default or ""
    return _require_enum(data, field_name, allowed_values, issues, prefix)


def _optional_int(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: int | None = None,
    minimum: int | None = None,
) -> int | None:
    """Read an optional integer field and enforce a minimum when one is provided."""
    if field_name not in data or data[field_name] is None:
        return default

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(ValidationIssue(field_path, "must be an integer"))
        return default
    if minimum is not None and value < minimum:
        issues.append(ValidationIssue(field_path, f"must be >= {minimum}"))
    return value


def _require_date(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> date | None:
    """Read a required ISO date field while accepting YAML-native date scalars."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            issues.append(ValidationIssue(field_path, "must not be empty"))
            return None
        try:
            return date.fromisoformat(stripped_value)
        except ValueError:
            issues.append(ValidationIssue(field_path, "must be a YYYY-MM-DD date"))
            return None

    issues.append(ValidationIssue(field_path, "must be a YYYY-MM-DD date"))
    return None


def _optional_string_mapping(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> dict[str, str] | None:
    """Read an optional mapping whose keys and values must both be strings."""
    if field_name not in data or data[field_name] is None:
        return None

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(field_path, "must be a mapping"))
        return None

    result: dict[str, str] = {}
    for key, item in value.items():
        child_path = f"{field_path}.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(child_path, "keys must be strings"))
            continue
        if not isinstance(item, str):
            issues.append(ValidationIssue(child_path, "values must be strings"))
            continue
        result[key] = item
    return result


def _require_non_empty_string_mapping(
    value: Any,
    field_path: str,
    issues: list[ValidationIssue],
) -> dict[str, str]:
    """Read a required mapping whose keys and values must be non-empty strings."""
    data = _require_mapping(value, field_path, issues)
    result: dict[str, str] = {}

    for key, item in data.items():
        child_path = f"{field_path}.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(child_path, "keys must be strings"))
            continue
        normalized_key = key.strip()
        if not normalized_key:
            issues.append(ValidationIssue(child_path, "keys must not be empty"))
            continue
        if not isinstance(item, str):
            issues.append(ValidationIssue(child_path, "values must be strings"))
            continue
        normalized_item = item.strip()
        if not normalized_item:
            issues.append(ValidationIssue(child_path, "values must not be empty"))
            continue
        result[normalized_key] = normalized_item

    return result


def _optional_string_list(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> list[str]:
    """Read an optional list of non-empty strings and report invalid entries inline."""
    if field_name not in data or data[field_name] is None:
        return []

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, list):
        issues.append(ValidationIssue(field_path, "must be a list"))
        return []

    result: list[str] = []
    for index, item in enumerate(value):
        child_path = f"{field_path}[{index}]"
        if not isinstance(item, str):
            issues.append(ValidationIssue(child_path, "must be a string"))
            continue
        stripped_item = item.strip()
        if not stripped_item:
            issues.append(ValidationIssue(child_path, "must not be empty"))
            continue
        result.append(stripped_item)
    return result


def _field_path(field_name: str, prefix: str | None) -> str:
    """Compose the dotted path used in nested validation messages."""
    if prefix:
        return f"{prefix}.{field_name}"
    return field_name
