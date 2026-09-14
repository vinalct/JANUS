"""Pure dependency references from validated request inputs, before registry resolution.

This module sits below the registry and the runtime: it turns an already-validated
``RequestInputsConfig`` into the flat list of source→source edges the config declares.
It resolves nothing. Whether the named producer exists, is enabled, or actually writes
the referenced table is a whole-registry question, and answering it here would need an
import — of the registry, a strategy, a catalog client or Spark — that this layer must
not have.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from janus.models.config.types import (
    CombinedRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    RequestInputsConfig,
)

#: Where an atomic request-input config sits inside a source document.
ROOT_REQUEST_INPUT_PATH = "access.request_inputs"


@dataclass(frozen=True, slots=True)
class IcebergInputReference:
    """One declared producer and its unchanged table reference, with leaf provenance."""

    upstream_source_id: str
    namespace: str
    table_name: str
    input_path: str

    @property
    def table_reference(self) -> str:
        """Return the dotted ``namespace.table`` this leaf reads."""
        return f"{self.namespace}.{self.table_name}"


def iter_iceberg_input_references(
    config: RequestInputsConfig,
    *,
    input_path: str = ROOT_REQUEST_INPUT_PATH,
) -> Iterator[IcebergInputReference]:
    """Visit every Iceberg leaf of a validated atomic or flat combined config."""
    entries = (
        ((f"{input_path}.inputs[{index}]", leaf) for index, leaf in enumerate(config.inputs))
        if isinstance(config, CombinedRequestInputsConfig)
        else iter(((input_path, config),))
    )
    for path, leaf in entries:
        if isinstance(leaf, IcebergRowsRequestInputsConfig):
            yield IcebergInputReference(
                upstream_source_id=leaf.upstream_source_id,
                namespace=leaf.namespace,
                table_name=leaf.table_name,
                input_path=path,
            )
