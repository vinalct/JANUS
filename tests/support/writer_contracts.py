"""Contracts for direct bronze-writer integration fixtures."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from janus.models.data_contracts import (
    ContractProperty,
    ContractSchema,
    DataContract,
    JanusContractOptions,
    odcs_logical_type_for,
    physical_type_from_spark_json,
)
from janus.normalizers.base import NORMALIZATION_METADATA_COLUMNS


def contract_for_columns(
    columns: Sequence[tuple[str, str]],
    *,
    source_id: str,
    version: str = "1.0.0",
    compatibility: str = "additive",
) -> DataContract:
    declared = tuple(
        (name, physical) for name, physical in columns if name not in NORMALIZATION_METADATA_COLUMNS
    )
    properties = tuple(
        ContractProperty(
            name=name,
            physical_type=physical,
            logical_type=odcs_logical_type_for(physical),
        )
        for name, physical in declared
    )
    return DataContract(
        contract_path=Path("tests/fixtures/writer_contract"),
        api_version="v3.2.0",
        id=f"tests.{source_id}",
        name=f"{source_id} writer fixture",
        version=version,
        status="active",
        domain="tests",
        purpose="Direct bronze writer integration fixture",
        owners=("janus-tests",),
        tags=(),
        schema=ContractSchema(source_id, "table", properties),
        janus=JanusContractOptions(compatibility, "strict"),
        schema_version=sha256(repr(declared).encode()).hexdigest(),
    )


def contract_for_frame(
    frame: Any,
    *,
    source_id: str,
    version: str = "1.0.0",
    compatibility: str = "additive",
) -> DataContract:
    columns = tuple(
        (field.name, physical_type_from_spark_json(field.dataType.jsonValue()))
        for field in frame.schema.fields
    )
    return contract_for_columns(
        columns, source_id=source_id, version=version, compatibility=compatibility
    )
