"""Frame doubles for fakes that feed the materializer's pre-write contract check."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from janus.models.data_contracts import DataContract
from janus.schema_contracts import spark_schema_from_contract


@dataclass(frozen=True, slots=True)
class SchemaFrame:
    """A frame double whose only Spark-facing surface is its schema."""

    schema: Any

    @property
    def columns(self) -> list[str]:
        return list(self.schema.fieldNames())


class SchemaRows(list):
    """A memory double's rows, carrying the schema the read applied to them."""

    def __init__(self, rows: Iterable[Any] = (), *, schema: Any) -> None:
        super().__init__(rows)
        self.schema = schema


def read_frame(schema: Any, inferred_schema: Any = None) -> SchemaFrame:
    """What a read returns: the schema it was handed, else the one Spark would have inferred."""
    return SchemaFrame(schema if schema is not None else inferred_schema)


def contract_schema(contract: DataContract | None) -> Any:
    """The schema a read shaped by ``contract`` carries — Spark's own type when it is installed."""
    return None if contract is None else spark_schema_from_contract(contract)
