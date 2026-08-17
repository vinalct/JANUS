"""The source configuration contract and its load-time entry point.

The per-block builders live in ``janus.models.config``. This module
keeps ``SourceConfig`` and ``from_mapping`` — the entry point — and re-exports every
name it exported before the split, so ``from janus.models.source_config import
AuthConfig`` keeps working.

``SourceConfig`` is the one block dataclass that did not move to ``config/types.py``,
because it carries ``from_mapping``, which imports every builder; defining it below the
builders would invert the package's dependency arrow and force a function-local import.

``from_mapping`` also owns the **only** raise site in the whole load path. Builders
append to the shared ``issues`` list and never raise, so a config with five problems
reports five.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Self, overload

from janus.models.config.access import (
    _build_access_config,
    _build_auth_config,
    _build_pagination_config,
    _build_rate_limit_config,
    _resolve_past_end_status_codes,
    _validate_dotted_path,
)
from janus.models.config.bindings import (
    _build_parameter_bindings_config,
    _validate_parameter_binding_source,
)
from janus.models.config.coercion import (
    _field_path,
    _optional_bool,
    _optional_enum,
    _optional_int,
    _optional_int_list,
    _optional_string,
    _optional_string_list,
    _optional_string_mapping,
    _require_bool,
    _require_date,
    _require_enum,
    _require_mapping,
    _require_non_empty_string_mapping,
    _require_string,
)
from janus.models.config.constants import (
    _SUPPORTED_SUB_REQUEST_INPUT_TYPES,
    CONCURRENT_PAGINATION_TYPES,
    DEFAULT_PAST_END_STATUS_CODES,
    DEFAULT_RETRYABLE_STATUS_CODES,
    REQUEST_INPUT_BINDING_PREFIX,
    RETRYABLE_CLIENT_STATUS_CODES,
    SUPPORTED_AUTH_TYPES,
    SUPPORTED_BACKOFF_STRATEGIES,
    SUPPORTED_CHECKPOINT_STRATEGIES,
    SUPPORTED_DATA_FORMATS,
    SUPPORTED_EXTRACTION_MODES,
    SUPPORTED_FEDERATION_LEVELS,
    SUPPORTED_HTTP_METHODS,
    SUPPORTED_LINK_RESOLVERS,
    SUPPORTED_PAGINATION_TYPES,
    SUPPORTED_PARAMETER_BINDING_WINDOW_SOURCES,
    SUPPORTED_REQUEST_INPUT_STEPS,
    SUPPORTED_REQUEST_INPUT_TYPES,
    SUPPORTED_SCHEMA_MODES,
    SUPPORTED_SOURCE_TYPES,
    SUPPORTED_STRATEGIES,
    SUPPORTED_STRATEGY_VARIANTS,
    SUPPORTED_WRITE_MODES,
)
from janus.models.config.contracts import (
    _validate_concurrency_contract,
    _validate_incremental_contract,
    _validate_retry_status_contract,
)
from janus.models.config.extraction import (
    _build_extraction_config,
    _build_retry_config,
    _resolve_retryable_status_codes,
)
from janus.models.config.issues import SourceConfigValidationError, ValidationIssue
from janus.models.config.outputs import (
    _build_output_target,
    _build_outputs_config,
    _build_quality_config,
    _build_schema_config,
    _build_spark_config,
)
from janus.models.config.request_inputs import (
    _build_combined_request_inputs_config,
    _build_request_inputs_config,
    _parse_request_input_entry,
    _request_input_field_names_for,
)
from janus.models.config.types import (
    _INVALID_DATE_BOUND,
    AccessConfig,
    AuthConfig,
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    ExtractionConfig,
    IcebergRowsRequestInputsConfig,
    OutputsConfig,
    OutputTarget,
    PaginationConfig,
    ParameterBinding,
    QualityConfig,
    RateLimitConfig,
    RequestInputsConfig,
    RetryConfig,
    SchemaConfig,
    SparkConfig,
)

__all__ = [
    "CONCURRENT_PAGINATION_TYPES",
    "DEFAULT_PAST_END_STATUS_CODES",
    "DEFAULT_RETRYABLE_STATUS_CODES",
    "REQUEST_INPUT_BINDING_PREFIX",
    "RETRYABLE_CLIENT_STATUS_CODES",
    "SUPPORTED_AUTH_TYPES",
    "SUPPORTED_BACKOFF_STRATEGIES",
    "SUPPORTED_CHECKPOINT_STRATEGIES",
    "SUPPORTED_DATA_FORMATS",
    "SUPPORTED_EXTRACTION_MODES",
    "SUPPORTED_FEDERATION_LEVELS",
    "SUPPORTED_HTTP_METHODS",
    "SUPPORTED_LINK_RESOLVERS",
    "SUPPORTED_PAGINATION_TYPES",
    "SUPPORTED_PARAMETER_BINDING_WINDOW_SOURCES",
    "SUPPORTED_REQUEST_INPUT_STEPS",
    "SUPPORTED_REQUEST_INPUT_TYPES",
    "SUPPORTED_SCHEMA_MODES",
    "SUPPORTED_SOURCE_TYPES",
    "SUPPORTED_STRATEGIES",
    "SUPPORTED_STRATEGY_VARIANTS",
    "SUPPORTED_WRITE_MODES",
    "_INVALID_DATE_BOUND",
    "_SUPPORTED_SUB_REQUEST_INPUT_TYPES",
    "AccessConfig",
    "AuthConfig",
    "CombinedRequestInputsConfig",
    "DateWindowRequestInputsConfig",
    "ExtractionConfig",
    "IcebergRowsRequestInputsConfig",
    "OutputTarget",
    "OutputsConfig",
    "PaginationConfig",
    "ParameterBinding",
    "QualityConfig",
    "RateLimitConfig",
    "RequestInputsConfig",
    "RetryConfig",
    "SchemaConfig",
    "SourceConfig",
    "SourceConfigValidationError",
    "SparkConfig",
    "ValidationIssue",
    "_build_access_config",
    "_build_auth_config",
    "_build_combined_request_inputs_config",
    "_build_extraction_config",
    "_build_output_target",
    "_build_outputs_config",
    "_build_pagination_config",
    "_build_parameter_bindings_config",
    "_build_quality_config",
    "_build_rate_limit_config",
    "_build_request_inputs_config",
    "_build_retry_config",
    "_build_schema_config",
    "_build_spark_config",
    "_field_path",
    "_optional_bool",
    "_optional_enum",
    "_optional_int",
    "_optional_int_list",
    "_optional_string",
    "_optional_string_list",
    "_optional_string_mapping",
    "_parse_request_input_entry",
    "_request_input_field_names_for",
    "_require_bool",
    "_require_date",
    "_require_enum",
    "_require_mapping",
    "_require_non_empty_string_mapping",
    "_require_string",
    "_resolve_past_end_status_codes",
    "_resolve_retryable_status_codes",
    "_validate_concurrency_contract",
    "_validate_dotted_path",
    "_validate_incremental_contract",
    "_validate_parameter_binding_source",
    "_validate_retry_status_contract",
]


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

        _validate_incremental_contract(extraction, quality, issues)
        _validate_concurrency_contract(source_type, access, issues)
        _validate_retry_status_contract(access, extraction, issues)

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
