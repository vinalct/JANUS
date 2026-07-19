from janus.lineage.models import (
    SUPPORTED_RUN_STATUSES,
    ArtifactSnapshot,
    ConfiguredOutput,
    LineageRecord,
    MaterializedOutput,
    RunMetadata,
    compute_config_version,
    configured_outputs_from_plan,
)
from janus.lineage.persistence import MetadataZonePaths, read_json_mapping, write_json_atomic

__all__ = [
    "SUPPORTED_RUN_STATUSES",
    "ArtifactSnapshot",
    "ConfiguredOutput",
    "LineageRecord",
    "LineageStore",
    "MaterializedOutput",
    "MetadataZonePaths",
    "PersistedArtifacts",
    "RunMetadata",
    "RunMetadataStore",
    "RunObserver",
    "compute_config_version",
    "configured_outputs_from_plan",
    "read_json_mapping",
    "write_json_atomic",
]

def __getattr__(name: str):
    if name in {"LineageStore", "PersistedArtifacts", "RunMetadataStore", "RunObserver"}:
        from janus.lineage.store import (
            LineageStore,
            PersistedArtifacts,
            RunMetadataStore,
            RunObserver,
        )
        exports = {
            "LineageStore": LineageStore,
            "PersistedArtifacts": PersistedArtifacts,
            "RunMetadataStore": RunMetadataStore,
            "RunObserver": RunObserver,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
