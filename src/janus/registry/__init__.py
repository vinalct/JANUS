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
"""

from janus.registry.dependencies import (
    SourceGraphValidationError,
    SourceLocation,
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

__all__ = [
    "AppConfig",
    "AppConfigValidationError",
    "RegistrySettings",
    "SourceGraphValidationError",
    "SourceLocation",
    "SourceNotFoundError",
    "SourceRegistry",
    "build_source_dependency_graph",
    "load_app_config",
    "load_registry",
    "producer_table_identifier",
]
