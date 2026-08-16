"""Per-block builders and value types behind ``SourceConfig.from_mapping``.

The package layers strictly downward::

    constants -> issues -> coercion -> types -> request_inputs -> bindings -> access
              |                             -> extraction, outputs
              |                             -> contracts -> source_config
              -> strategy_registry -> policy, source_config, janus.planner

``access`` composes ``request_inputs`` and ``bindings`` because it must report their
issues in a fixed order; those are the only builder-to-builder edges. Nothing in this
package imports ``janus.models.source_config`` — that module sits above it and holds the
``from_mapping`` entry point. ``tests/unit/models/test_config_package_imports.py``
enforces both rules.

Only the value types and constants are re-exported here; the ``_build_*`` builders stay
private to their modules and are imported directly by ``from_mapping``.
"""

from __future__ import annotations

from janus.models.config.constants import (
    CONCURRENT_PAGINATION_TYPES,
    DEFAULT_PAST_END_STATUS_CODES,
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
from janus.models.config.issues import SourceConfigValidationError, ValidationIssue
from janus.models.config.policy import (
    DEFAULT_VALIDATION_POLICY,
    PhaseValidationPolicy,
    ValidationPolicy,
)
from janus.models.config.strategy_registry import STRATEGY_REGISTRY, StrategyRegistry
from janus.models.config.types import (
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
    "DEFAULT_VALIDATION_POLICY",
    "REQUEST_INPUT_BINDING_PREFIX",
    "RETRYABLE_CLIENT_STATUS_CODES",
    "STRATEGY_REGISTRY",
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
    "PhaseValidationPolicy",
    "QualityConfig",
    "RateLimitConfig",
    "RequestInputsConfig",
    "RetryConfig",
    "SchemaConfig",
    "SourceConfigValidationError",
    "SparkConfig",
    "StrategyRegistry",
    "ValidationIssue",
    "ValidationPolicy",
]
