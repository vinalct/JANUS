"""AC-5 / FR-3: the import surface is allowed to reorganise but not to break."""

from __future__ import annotations

import importlib

import pytest

import janus.models as models

MODELS_ALL: tuple[str, ...] = (
    "AccessConfig",
    "AuthConfig",
    "BRONZE_WRITE_STRATEGIES",
    "BronzeWriteIntent",
    "CONCURRENT_PAGINATION_TYPES",
    "CombinedRequestInputsConfig",
    "DEFAULT_PAST_END_STATUS_CODES",
    "DEFAULT_VALIDATION_POLICY",
    "DEFAULT_RETRYABLE_STATUS_CODES",
    "DateWindowRequestInputsConfig",
    "ExecutionPlan",
    "ExtractedArtifact",
    "ExtractionConfig",
    "ExtractionResult",
    "IcebergRowsRequestInputsConfig",
    "OutputTarget",
    "OutputsConfig",
    "PaginationConfig",
    "ParameterBinding",
    "PhaseValidationPolicy",
    "QualityConfig",
    "RETRYABLE_CLIENT_STATUS_CODES",
    "RateLimitConfig",
    "RequestInputsConfig",
    "RetryConfig",
    "RunContext",
    "SUPPORTED_OUTPUT_ZONES",
    "SchemaConfig",
    "SourceConfig",
    "SourceConfigValidationError",
    "SourceReference",
    "SparkConfig",
    "ValidationIssue",
    "ValidationPolicy",
    "WriteResult",
    "resolve_bronze_write_intent",
)

#: Import paths that exist today and must keep working.
PUBLIC_SURFACE: dict[str, tuple[str, ...]] = {
    "janus.models": MODELS_ALL,
    "janus.models.source_config": (
        "AccessConfig",
        "AuthConfig",
        "CONCURRENT_PAGINATION_TYPES",
        "CombinedRequestInputsConfig",
        "DEFAULT_PAST_END_STATUS_CODES",
        "DEFAULT_VALIDATION_POLICY",
        "DEFAULT_RETRYABLE_STATUS_CODES",
        "DateWindowRequestInputsConfig",
        "ExtractionConfig",
        "IcebergRowsRequestInputsConfig",
        "OutputTarget",
        "OutputsConfig",
        "PaginationConfig",
        "ParameterBinding",
        "PhaseValidationPolicy",
        "QualityConfig",
        "RETRYABLE_CLIENT_STATUS_CODES",
        "RateLimitConfig",
        "RequestInputsConfig",
        "RetryConfig",
        "SUPPORTED_STRATEGY_VARIANTS",
        "SchemaConfig",
        "SourceConfig",
        "SourceConfigValidationError",
        "SparkConfig",
        "ValidationIssue",
        "ValidationPolicy",
        "_parse_request_input_entry",
    ),
    "janus.strategies.api": (
        "ApiHook",
        "ApiRequest",
        "ApiResponse",
        "ApiStrategy",
        "ApiTransport",
        "ApiTransportError",
        "AuthResolutionError",
        "PaginationState",
        "UrllibApiTransport",
        "build_paginator",
    ),
    "janus.strategies.api.core": (
        "ApiHook",
        "ApiPastEndConflictError",
        "ApiPayloadError",
        "ApiResponseError",
        "ApiStrategy",
        "ApiStrategyError",
        "CONCURRENCY_ONLY_METADATA_KEYS",
        "_raw_relative_path",
        "_request_input_key",
    ),
    "janus.strategies.catalog": (
        "CatalogPayloadError",
        "CatalogStrategy",
    ),
    "janus.strategies.catalog.core": (
        "CatalogHook",
        "CatalogPayloadError",
        "CatalogResponseError",
        "CatalogStrategy",
        "CatalogStrategyError",
        "ENTITY_TYPE_ORDER",
        "_apply_per_input_params",
        "_normalize_catalog_record",
        "_persist_generic_artifacts",
        "_rediscover_catalog_input_artifacts",
        "_replay_catalog_entities_from_dir",
    ),
    "janus.strategies.files": (
        "DiscoveredFile",
        "FileHook",
        "FileIntegrityError",
        "FileStrategy",
    ),
    "janus.strategies.files.core": (
        "ArchiveExtractionError",
        "DiscoveredFile",
        "FileDiscoveryError",
        "FileDownloadError",
        "FileHook",
        "FileIntegrityError",
        "FileStrategy",
        "FileStrategyError",
        "_archive_member_payloads",
        "_filter_members",
        "_infer_handoff_format",
        "_raw_extracted_relative_path",
        "_read_checksum_sidecar",
    ),
    "janus.scripts.raw_to_bronze": (
        "RawToBronzeLoader",
        "RawToBronzeRun",
        "ingest_raw_to_bronze",
        "_artifact_format_for_path",
        "_rediscover_raw_artifacts",
        "_sha256",
    ),
}


@pytest.mark.parametrize("module_path", sorted(PUBLIC_SURFACE))
def test_module_still_exports_its_documented_names(module_path: str):
    module = importlib.import_module(module_path)
    missing = [name for name in PUBLIC_SURFACE[module_path] if not hasattr(module, name)]
    assert not missing, (
        f"{module_path} no longer exports {missing}. moves code between modules "
        "but must not move it out from under an existing import — add a compatibility "
        "re-export in the module that used to define it (FR-3)."
    )


def test_janus_models_all_is_unchanged():
    """``janus.models.__all__`` is the repo's most externally visible list — pin it exactly.

    Compared sorted so the assertion is about *membership*, not about the import-block
    ordering ruff's isort rules happen to produce.
    """
    assert tuple(sorted(models.__all__)) == MODELS_ALL, (
        "janus.models.__all__ changed. is a pure refactor (NFR-1): it may move "
        "where a name is defined, never whether janus.models exports it."
    )
    assert len(models.__all__) == len(set(models.__all__)), (
        "janus.models.__all__ contains duplicates — likely a re-export added twice while "
        "moving a definition between modules."
    )


def test_everything_janus_models_advertises_is_actually_reachable():
    """``__all__`` must not outlive its names: a stale entry breaks ``import *`` only."""
    missing = [name for name in models.__all__ if not hasattr(models, name)]
    assert not missing, (
        f"janus.models.__all__ advertises {missing}, which the module does not define. "
        "A compat re-export was dropped, or a moved name lost its import."
    )
