from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan
from janus.models.data_contracts import (
    ContractProperty,
    DataContract,
    contract_properties_from_spark_json,
    spark_struct_json,
)
from janus.quality.contract_checks import CORRUPT_RECORD_COLUMN, FrameColumn
from janus.utils.environment import resolve_project_path


def resolve_declared_path(
    project_root: Path, config_path: Path, configured: str | None
) -> Path | None:
    """Resolve one declared schema file against the project, then the config's parents."""
    if not configured:
        return None

    declared_path = Path(configured)
    if declared_path.is_absolute():
        return declared_path

    runtime_path = resolve_project_path(project_root, str(declared_path))
    if runtime_path.exists():
        return runtime_path

    for parent in config_path.resolve().parents:
        candidate = parent / declared_path
        if candidate.exists():
            return candidate

    return runtime_path


def _resolve_declared_path_for_plan(plan: ExecutionPlan, configured: str | None) -> Path | None:
    """Run the shared search with the project root and config path this plan carries."""
    return resolve_declared_path(
        plan.run_context.project_root, plan.source_config.config_path, configured
    )


def resolve_contract_path_for_plan(plan: ExecutionPlan) -> Path | None:
    """Return the data contract configured for one plan, if any."""
    return _resolve_declared_path_for_plan(plan, plan.source_config.schema.contract)


def resolve_spark_schema_for_plan(plan: ExecutionPlan) -> Any | None:
    """Return the Spark schema generated from the contract this plan carries."""
    contract = plan.data_contract
    if contract is None:
        return None
    return spark_schema_from_contract(contract)


def spark_schema_from_contract(contract: DataContract, *, with_corrupt_record: bool = False) -> Any:
    """Build the Spark schema for one contract — the single generator in `src`."""
    struct_json = spark_struct_json(contract.schema.properties)
    if with_corrupt_record:
        struct_json["fields"].append(
            {"name": CORRUPT_RECORD_COLUMN, "type": "string", "nullable": True, "metadata": {}}
        )
    try:
        from pyspark.sql.types import StructType
    except ImportError:
        return _FieldNameSchema(contract.schema.properties, with_corrupt_record)
    return StructType.fromJson(struct_json)


def contract_properties_from_spark_schema(struct_type: Any) -> tuple[ContractProperty, ...]:
    """Invert :func:`spark_schema_from_contract` for the drafting CLI."""
    return contract_properties_from_spark_json(struct_type.jsonValue())


def frame_columns_from_spark_schema(schema: Any) -> tuple[FrameColumn, ...]:
    """StructType -> FrameColumn tuple via ``schema.jsonValue()['fields']`` — the ONE adapter."""
    return tuple(
        FrameColumn(
            name=field["name"],
            spark_json_type=field["type"],
            nullable=bool(field.get("nullable", True)),
        )
        for field in schema.jsonValue()["fields"]
    )


@dataclass(frozen=True, slots=True)
class _FieldNameSchema:
    """Minimal schema facade for unit tests that run without PySpark installed."""

    properties: tuple[ContractProperty, ...]
    with_corrupt_record: bool = False

    def fieldNames(self) -> list[str]:
        names = [prop.name for prop in self.properties]
        if self.with_corrupt_record:
            names.append(CORRUPT_RECORD_COLUMN)
        return names

    def jsonValue(self) -> dict[str, Any]:
        """The struct JSON ``StructType.fromJson`` reads, which its ``jsonValue()`` returns."""
        schema = spark_struct_json(self.properties)
        if self.with_corrupt_record:
            schema["fields"].append(
                {"name": CORRUPT_RECORD_COLUMN, "type": "string", "nullable": True, "metadata": {}}
            )
        return schema


__all__ = [
    "contract_properties_from_spark_schema",
    "frame_columns_from_spark_schema",
    "resolve_contract_path_for_plan",
    "resolve_declared_path",
    "resolve_spark_schema_for_plan",
    "spark_schema_from_contract",
]
