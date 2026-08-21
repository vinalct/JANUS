
from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import janus.models as models
import janus.models.config as config_package
import janus.models.source_config as source_config_module

PUBLIC_NAMES: tuple[str, ...] = (
    "AccessConfig",
    "Any",
    "AuthConfig",
    "CONCURRENT_PAGINATION_TYPES",
    "CombinedRequestInputsConfig",
    "DEFAULT_PAST_END_STATUS_CODES",
    "DEFAULT_RETRYABLE_STATUS_CODES",
    "DEFAULT_VALIDATION_POLICY",
    "DateWindowRequestInputsConfig",
    "ExtractionConfig",
    "Final",
    "IcebergRowsRequestInputsConfig",
    "Mapping",
    "OutputTarget",
    "OutputsConfig",
    "PaginationConfig",
    "ParameterBinding",
    "Path",
    "PhaseValidationPolicy",
    "QualityConfig",
    "REQUEST_INPUT_BINDING_PREFIX",
    "RETRYABLE_CLIENT_STATUS_CODES",
    "RateLimitConfig",
    "RequestInputsConfig",
    "RetryConfig",
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
    "SchemaConfig",
    "Self",
    "SourceConfig",
    "SourceConfigValidationError",
    "SparkConfig",
    "StrategyRegistry",
    "ValidationIssue",
    "ValidationPolicy",
    "annotations",
    "dataclass",
    "date",
    "datetime",
    "overload",
)


PRIVATE_NAMES: tuple[str, ...] = (
    "_INVALID_DATE_BOUND",
    "_SUPPORTED_SUB_REQUEST_INPUT_TYPES",
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
    "_validate_concurrency_contract",
    "_validate_dotted_path",
    "_validate_incremental_contract",
    "_validate_parameter_binding_source",
)


MODELS_ALL: tuple[str, ...] = (
    "AccessConfig",
    "AuthConfig",
    "BRONZE_WRITE_STRATEGIES",
    "BronzeWriteIntent",
    "CONCURRENT_PAGINATION_TYPES",
    "CombinedRequestInputsConfig",
    "DEFAULT_PAST_END_STATUS_CODES",
    "DEFAULT_RETRYABLE_STATUS_CODES",
    "DEFAULT_VALIDATION_POLICY",
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


LAYER_RANK: dict[str, int] = {
    "constants": 0,
    "issues": 1,
    "strategy_registry": 1,  # imports constants (0) only; never imports issues, and vice versa
    "coercion": 2,
    "policy": 3,  # imports constants, issues and strategy_registry; never crosses types
    "types": 3,
    "extraction": 4,
    "outputs": 4,
    "request_inputs": 4,
    "bindings": 5,
    "access": 6,
    "contracts": 7,
    "__init__": 8,
}


BUILDER_MODULES = frozenset(
    {"access", "bindings", "contracts", "extraction", "outputs", "request_inputs"}
)


ALLOWED_BUILDER_EDGES = frozenset(
    {
        ("access", "bindings"),
        ("access", "request_inputs"),
        ("bindings", "request_inputs"),
    }
)

_PACKAGE_DIR = Path(config_package.__file__).resolve().parent
_SRC_ROOT = _PACKAGE_DIR.parents[2]


def _config_modules() -> dict[str, ast.Module]:
    """Parse every module in ``janus.models.config``, keyed by bare module name."""
    return {
        path.stem: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE_DIR.glob("*.py"))
    }


def _package_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Return every ``janus.models.config.*`` module this tree imports, with its line."""
    edges: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "janus.models.config":
                edges.append(("__init__", node.lineno))
            elif node.module.startswith("janus.models.config."):
                edges.append((node.module.rsplit(".", 1)[1], node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("janus.models.config."):
                    edges.append((alias.name.rsplit(".", 1)[1], node.lineno))
    return edges


def _upward_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Return imports of the modules that sit *above* the config package."""
    above: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "janus.models",
            "janus.models.source_config",
        }:
            above.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"janus.models", "janus.models.source_config"}:
                    above.append((alias.name, node.lineno))
    return above


def test_the_module_sweep_covers_the_whole_package():
    """A glob that silently matches nothing turns every assertion below into a no-op."""
    found = set(_config_modules())

    assert found == set(LAYER_RANK), (
        f"the config package holds {sorted(found)} but the layering table describes "
        f"{sorted(LAYER_RANK)}. A new module must be given a rank — that is the review "
        "conversation about where it belongs, and it is the point of the table."
    )


def test_source_config_module_still_exports_every_public_name():
    """FR-3: the split may move a definition, never move it out from under an import."""
    exported = tuple(sorted(n for n in dir(source_config_module) if not n.startswith("_")))

    assert exported == PUBLIC_NAMES, (
        "janus.models.source_config's public name surface changed. Missing "
        f"{sorted(set(PUBLIC_NAMES) - set(exported))}, unexpected "
        f"{sorted(set(exported) - set(PUBLIC_NAMES))}. A name that moved into "
        "janus.models.config must be re-exported here (FR-3)."
    )


def test_source_config_module_still_exports_its_private_helpers():
    """The private surface is de facto public: tests reach through this module for it."""
    missing = [name for name in PRIVATE_NAMES if not hasattr(source_config_module, name)]

    assert not missing, (
        f"janus.models.source_config no longer exports {missing}. These are imported by "
        "tests (see tests/unit/models/test_date_window_fail_closed.py), so moving them "
        "into janus.models.config requires a compatibility re-export here."
    )


def test_janus_models_all_is_unchanged():
    """``janus.models.__all__`` is compared sorted: membership, not import ordering."""
    assert tuple(sorted(models.__all__)) == MODELS_ALL, (
        "janus.models.__all__ changed. TASK-06 repoints where janus.models imports each "
        "name from; it must not change which names it exports."
    )
    missing = [name for name in models.__all__ if not hasattr(models, name)]
    assert not missing, f"janus.models.__all__ advertises {missing}, which it does not define."


@pytest.mark.parametrize("module_name", sorted(LAYER_RANK))
def test_config_package_has_no_import_cycles(module_name: str):
    """Each module must import standalone, in a fresh interpreter, in any order."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_SRC_ROOT), env.get("PYTHONPATH")) if part
    )
    target = "janus.models.config" if module_name == "__init__" else (
        f"janus.models.config.{module_name}"
    )

    result = subprocess.run(
        [sys.executable, "-c", f"import {target}"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, (
        f"importing {target} on its own failed:\n{result.stderr}\n"
        "A module in this package must not depend on being imported after its siblings."
    )


def test_config_layering_is_one_directional():
    """Every intra-package import points at a strictly lower layer, so cycles cannot form."""
    violations = [
        f"{name}.py:{lineno} imports {target} "
        f"(rank {LAYER_RANK[name]} -> {LAYER_RANK[target]})"
        for name, tree in _config_modules().items()
        for target, lineno in _package_imports(tree)
        if LAYER_RANK[target] >= LAYER_RANK[name]
    ]

    assert not violations, (
        "janus.models.config imports must run strictly downward:\n  "
        + "\n  ".join(violations)
        + "\nEither the import belongs in the other direction, or the layering table "
        "needs a decision made about it."
    )


def test_constants_and_coercion_depend_on_nothing_above_them():
    """The two most-imported modules stay at the bottom, where everything can reach them."""
    modules = _config_modules()

    assert _package_imports(modules["constants"]) == [], (
        "constants.py must import no sibling — it is the layer every other module "
        "depends on, and a dependency of its own would invert the package."
    )

    coercion_targets = {target for target, _ in _package_imports(modules["coercion"])}
    assert coercion_targets <= {"constants", "issues"}, (
        f"coercion.py imports {sorted(coercion_targets)}; it may only depend on "
        "constants and issues. Coercion is about primitives, not about config blocks."
    )


def test_no_builder_imports_a_builder_it_does_not_compose():
    """The three composition edges are a decision; a fourth should be one too."""
    unexpected = [
        f"{name}.py:{lineno} imports {target}"
        for name, tree in _config_modules().items()
        if name in BUILDER_MODULES
        for target, lineno in _package_imports(tree)
        if target in BUILDER_MODULES and (name, target) not in ALLOWED_BUILDER_EDGES
    ]

    assert not unexpected, (
        "a new builder-to-builder dependency appeared:\n  "
        + "\n  ".join(unexpected)
        + f"\nOnly {sorted(ALLOWED_BUILDER_EDGES)} are sanctioned. Builders are meant to "
        "be composable by from_mapping, not chained to each other."
    )


def test_nothing_in_the_config_package_imports_source_config():
    """``source_config`` sits above the package; an import from below would be a cycle."""
    offenders = [
        f"{name}.py:{lineno} imports {target}"
        for name, tree in _config_modules().items()
        for target, lineno in _upward_imports(tree)
    ]

    assert not offenders, (
        "janus.models.config must not import janus.models or janus.models.source_config:"
        "\n  " + "\n  ".join(offenders) + "\nSourceConfig lives above the builders "
        "precisely so this arrow never has to exist."
    )


def test_from_mapping_is_still_the_entry_point():
    """FR-2: the builders moved, the way a config is loaded did not."""
    config = source_config_module.SourceConfig.from_mapping(_valid_source_mapping(), CONFIG_PATH)

    assert config.source_id == "config_package_source"
    assert config.access.pagination.page_size == 100
    assert config.access.pagination.past_end_status_codes == (404, 416)
    assert config.extraction.retry.max_attempts == 3
    assert config.outputs.bronze.format == "iceberg"
    assert config.quality.allow_schema_evolution is True


def test_a_config_with_problems_in_four_blocks_still_reports_all_four():
    """No builder may raise: one load reports every problem, not the first one found."""
    broken = _valid_source_mapping()
    broken["strategy_variant"] = "not_real"
    broken["access"]["auth"] = {"type": "header_token"}
    broken["extraction"]["mode"] = "incremental"
    broken["schema"] = {"mode": "explicit"}

    with pytest.raises(source_config_module.SourceConfigValidationError) as exc_info:
        source_config_module.SourceConfig.from_mapping(broken, CONFIG_PATH)

    message = str(exc_info.value)
    assert "strategy_variant: must be one of" in message
    assert "access.auth.env_var: is required for token-based auth" in message
    assert (
        "extraction.checkpoint_field: is required when extraction.mode is 'incremental'"
        in message
    )
    assert "schema.path: is required when schema.mode is 'explicit'" in message


def test_importing_a_builder_by_its_new_path_works():
    """AC-3: the package is a real import target, not only an implementation detail."""
    access = importlib.import_module("janus.models.config.access")
    types = importlib.import_module("janus.models.config.types")

    assert access._build_access_config is source_config_module._build_access_config
    assert types.AccessConfig is source_config_module.AccessConfig


CONFIG_PATH = Path("conf/sources/example/config_package_source.yaml")


def _valid_source_mapping() -> dict[str, Any]:
    """A minimal source that loads cleanly, so a failure is about the split, not the fixture."""
    return {
        "source_id": "config_package_source",
        "name": "config_package_source",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/records",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 100,
            },
            "rate_limit": {"requests_per_minute": 10, "concurrency": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/config_package_source", "format": "json"},
            "bronze": {
                "path": "data/bronze/example/config_package_source",
                "format": "iceberg",
            },
            "metadata": {
                "path": "data/metadata/example/config_package_source",
                "format": "json",
            },
        },
        "quality": {"allow_schema_evolution": True},
    }
