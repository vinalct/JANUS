"""Pinned OpenLineage identities shared by the pure mapping layer."""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5

from janus import __version__

PROJECT_URL = "https://github.com/vinalct/janus"
OPENLINEAGE_SPEC_VERSION = "2-0-2"
OPENLINEAGE_SCHEMA_URL = (
    f"https://openlineage.io/spec/{OPENLINEAGE_SPEC_VERSION}/OpenLineage.json"
    "#/$defs/RunEvent"
)
OPENLINEAGE_PRODUCER = f"{PROJECT_URL}/tree/v{__version__}"

DOCUMENTATION_JOB_FACET_SCHEMA_URL = (
    "https://openlineage.io/spec/facets/1-1-0/DocumentationJobFacet.json"
    "#/$defs/DocumentationJobFacet"
)
ERROR_MESSAGE_RUN_FACET_SCHEMA_URL = (
    "https://openlineage.io/spec/facets/1-0-1/ErrorMessageRunFacet.json"
    "#/$defs/ErrorMessageRunFacet"
)
JOB_TYPE_JOB_FACET_SCHEMA_URL = (
    "https://openlineage.io/spec/facets/2-0-4/JobTypeJobFacet.json"
    "#/$defs/JobTypeJobFacet"
)
OUTPUT_STATISTICS_FACET_SCHEMA_URL = (
    "https://openlineage.io/spec/facets/1-0-2/OutputStatisticsOutputDatasetFacet.json"
    "#/$defs/OutputStatisticsOutputDatasetFacet"
)
SOURCE_CODE_LOCATION_JOB_FACET_SCHEMA_URL = (
    "https://openlineage.io/spec/facets/1-1-0/SourceCodeLocationJobFacet.json"
    "#/$defs/SourceCodeLocationJobFacet"
)

JANUS_RUN_FACET_SCHEMA_VERSION = "1-0-0"
JANUS_RUN_FACET_SCHEMA_URL = (
    f"https://raw.githubusercontent.com/vinalct/janus/v{__version__}/"
    "docs/schemas/openlineage/JanusRunFacet.json#/$defs/JanusRunFacet"
)

# UUIDv5 over this project URL gives JANUS a stable UUID namespace without keeping state.
JANUS_RUN_UUID_NAMESPACE: UUID = uuid5(NAMESPACE_URL, f"{PROJECT_URL}/openlineage/run")


def openlineage_run_id(janus_run_id: str) -> str:
    """Derive the OpenLineage UUID deterministically from the readable JANUS run id."""
    if not janus_run_id.strip():
        raise ValueError("janus_run_id must not be empty")
    return str(uuid5(JANUS_RUN_UUID_NAMESPACE, janus_run_id))


def facet_base(schema_url: str) -> dict[str, str]:
    """Return the two fields every standard or custom OpenLineage facet requires."""
    return {
        "_producer": OPENLINEAGE_PRODUCER,
        "_schemaURL": schema_url,
    }
