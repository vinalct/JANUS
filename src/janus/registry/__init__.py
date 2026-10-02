"""Source registry loading — the entry point every JANUS run goes through.

``load_registry`` and ``SourceRegistry.load`` take a keyword-only ``policy`` and
``strategy_registry``, so a whole registry can be loaded under a broadened phase scope
without touching the contract layer. Both types are owned by ``janus.models``
(``ValidationPolicy`` / ``DEFAULT_VALIDATION_POLICY`` and ``StrategyRegistry`` /
``STRATEGY_REGISTRY``) and are deliberately not re-exported here — this package consumes
them, and a second import path for a name is a second place for it to drift.

Loading also resolves the inter-source graph the ``iceberg_rows`` declarations describe.
That resolution belongs here, not in ``janus.models``: it needs the whole registry and the
writer's table identity, and it must run before a planner, an engine or a catalog does.

The semantic pass (``janus.registry.semantics``) is the registry-wide half of validation:
cross-block and cross-source facts that parse and cannot be true. Its rules live there and
only there; a command reports what the pass found, it never re-implements a rule.
"""

from janus.registry.dependencies import (
    SourceGraphValidationError,
    SourceLocation,
    bronze_output_table_identifier,
    build_source_dependency_graph,
    producer_table_identifier,
)
from janus.registry.loader import (
    AppConfig,
    AppConfigValidationError,
    RegistrySettings,
    SourceNotFoundError,
    SourceRegistry,
    load_app_config,
    load_registry,
)
from janus.registry.semantics import (
    RULE_IDS,
    collect_semantic_issues,
    expected_fields,
    unverified_required_fields,
)

__all__ = [
    "RULE_IDS",
    "AppConfig",
    "AppConfigValidationError",
    "RegistrySettings",
    "SourceGraphValidationError",
    "SourceLocation",
    "SourceNotFoundError",
    "SourceRegistry",
    "bronze_output_table_identifier",
    "build_source_dependency_graph",
    "collect_semantic_issues",
    "expected_fields",
    "load_app_config",
    "load_registry",
    "producer_table_identifier",
    "unverified_required_fields",
]
