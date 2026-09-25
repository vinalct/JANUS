"""The generated Spark schema equals a hand-built ``StructType`` and inverts back."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyspark.sql")
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

from janus.models.data_contracts import (
    ContractProperty,
    ContractSchema,
    DataContract,
    JanusContractOptions,
    UnsupportedSparkTypeError,
    load_data_contract,
)
from janus.schema_contracts import (
    contract_properties_from_spark_schema,
    spark_schema_from_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONTRACT = (
    PROJECT_ROOT / "conf" / "contracts" / "example" / "federal_open_data_example.yaml"
)


def _property(
    name: str, physical_type: str, logical_type: str, **overrides: object
) -> ContractProperty:
    return ContractProperty(
        name=name, physical_type=physical_type, logical_type=logical_type, **overrides
    )


def _every_vocabulary_property() -> tuple[ContractProperty, ...]:
    """One property per vocabulary entry, with both struct levels and both containers."""
    return (
        _property("flag", "boolean", "boolean", required=True),
        _property("small_count", "integer", "integer"),
        _property("big_count", "long", "integer"),
        _property("ratio", "float", "number"),
        _property("weight", "double", "number"),
        _property("amount", "decimal(18,2)", "number"),
        _property("label", "string", "string", required=True),
        _property("blob", "binary", "string"),
        _property("reference_day", "date", "date"),
        _property("observed_at", "timestamp", "date"),
        _property("emitted_at", "timestamptz", "date"),
        _property(
            "nested",
            "struct",
            "object",
            properties=(
                _property("leaf", "string", "string", required=True),
                _property(
                    "deeper",
                    "struct",
                    "object",
                    properties=(_property("bottom", "long", "integer"),),
                ),
            ),
        ),
        _property(
            "tags",
            "array",
            "array",
            items=_property("element", "string", "string"),
        ),
        _property(
            "attributes",
            "map",
            "object",
            keys=_property("key", "string", "string", required=True),
            values=_property("value", "long", "integer"),
        ),
    )


def _hand_written_struct_type() -> StructType:
    """The same schema written directly in PySpark, field by field."""
    return StructType(
        [
            StructField("flag", BooleanType(), False),
            StructField("small_count", IntegerType(), True),
            StructField("big_count", LongType(), True),
            StructField("ratio", FloatType(), True),
            StructField("weight", DoubleType(), True),
            StructField("amount", DecimalType(18, 2), True),
            StructField("label", StringType(), False),
            StructField("blob", BinaryType(), True),
            StructField("reference_day", DateType(), True),
            StructField("observed_at", TimestampNTZType(), True),
            StructField("emitted_at", TimestampType(), True),
            StructField(
                "nested",
                StructType(
                    [
                        StructField("leaf", StringType(), False),
                        StructField(
                            "deeper",
                            StructType([StructField("bottom", LongType(), True)]),
                            True,
                        ),
                    ]
                ),
                True,
            ),
            StructField("tags", ArrayType(StringType(), containsNull=True), True),
            StructField(
                "attributes",
                MapType(StringType(), LongType(), valueContainsNull=True),
                True,
            ),
        ]
    )


def _contract(properties: tuple[ContractProperty, ...]) -> DataContract:
    return DataContract(
        contract_path=Path("synthetic.yaml"),
        api_version="v3.2.0",
        id="example.vocabulary",
        name="Vocabulary coverage",
        version="1.0.0",
        status="active",
        domain="example",
        purpose="Synthetic contract exercising every vocabulary entry.",
        owners=("janus-tests",),
        tags=(),
        schema=ContractSchema(
            name="vocabulary", physical_type="table", properties=properties
        ),
        janus=JanusContractOptions(compatibility="additive", enforcement="lenient"),
        schema_version="0" * 64,
    )


def test_generated_schema_equals_a_hand_written_struct_type():
    contract = _contract(_every_vocabulary_property())

    generated = spark_schema_from_contract(contract)

    assert generated.jsonValue() == _hand_written_struct_type().jsonValue()
    assert generated == _hand_written_struct_type()


def test_generated_schema_inverts_back_to_the_same_properties():
    properties = _every_vocabulary_property()
    contract = _contract(properties)

    rebuilt = contract_properties_from_spark_schema(spark_schema_from_contract(contract))

    assert tuple(prop.name for prop in rebuilt) == tuple(
        prop.name for prop in properties
    )
    assert tuple(prop.physical_type for prop in rebuilt) == tuple(
        prop.physical_type for prop in properties
    )
    assert tuple(prop.required for prop in rebuilt) == tuple(
        prop.required for prop in properties
    )
    assert spark_schema_from_contract(_contract(rebuilt)) == spark_schema_from_contract(
        contract
    )


def test_the_timestamp_pin_holds_against_real_spark_types():
    contract = _contract(
        (
            _property("observed_at", "timestamp", "date"),
            _property("emitted_at", "timestamptz", "date"),
        )
    )

    generated = spark_schema_from_contract(contract)

    assert isinstance(generated["observed_at"].dataType, TimestampNTZType)
    assert isinstance(generated["emitted_at"].dataType, TimestampType)


def test_the_example_contract_generates_its_declared_columns():
    contract = load_data_contract(EXAMPLE_CONTRACT)

    generated = spark_schema_from_contract(contract)

    assert generated == StructType(
        [
            StructField("id", StringType(), False),
            StructField("updated_at", StringType(), False),
        ]
    )
    assert generated.fieldNames() == list(contract.column_names)


def test_a_spark_schema_outside_the_vocabulary_is_refused_when_inverted():
    struct_type = StructType([StructField("tiny", ShortType(), True)])

    with pytest.raises(UnsupportedSparkTypeError):
        contract_properties_from_spark_schema(struct_type)
