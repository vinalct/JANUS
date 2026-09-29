"""The closed type vocabulary and its two-way Spark/Iceberg mapping."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from janus.models.data_contracts import ContractProperty
from janus.models.data_contracts import vocabulary as vocabulary_module
from janus.models.data_contracts.vocabulary import (
    CONTAINER_TYPE_NAMES,
    DECIMAL_TYPE_NAME,
    VOCABULARY,
    UnknownPhysicalTypeError,
    UnsupportedIcebergTypeError,
    UnsupportedSparkTypeError,
    VocabularyType,
    contract_properties_from_spark_json,
    declared_physical_types,
    iceberg_type_name,
    odcs_logical_type_for,
    parse_physical_type,
    physical_type_from_iceberg_name,
    physical_type_from_spark_json,
    spark_json_type,
    spark_struct_json,
)
from tests.unit.models.data_contracts.test_contract_package_imports import (
    _forbidden_imports,
)

#: AC-2 names these verbatim; the test below asserts each one is in the vocabulary.
REQUIRED_VOCABULARY_NAMES = (
    "decimal(p,s)",
    "date",
    "timestamp",
    "timestamptz",
    "boolean",
    "double",
)

#: The ODCS v3.2.0 logical types the JANUS subset uses.
ALLOWED_ODCS_LOGICAL_TYPES = frozenset(
    {"array", "boolean", "date", "integer", "number", "object", "string"}
)

_SCALAR_ENTRIES = tuple(entry for entry in VOCABULARY if entry.name not in CONTAINER_TYPE_NAMES)


def _scalar(name: str, *, required: bool = False) -> ContractProperty:
    """One scalar property spelled ``name``, with the representative decimal substituted."""
    physical_type = "decimal(18,2)" if name == DECIMAL_TYPE_NAME else name
    return ContractProperty(
        name="column",
        physical_type=physical_type,
        logical_type=odcs_logical_type_for(physical_type),
        required=required,
    )


def _array_of(element: ContractProperty) -> ContractProperty:
    return ContractProperty(
        name="values",
        physical_type="array",
        logical_type="array",
        items=element,
    )


def _map_of(key: ContractProperty, value: ContractProperty) -> ContractProperty:
    return ContractProperty(
        name="attributes",
        physical_type="map",
        logical_type="object",
        keys=key,
        values=value,
    )


def _struct_of(*children: ContractProperty) -> ContractProperty:
    return ContractProperty(
        name="outer",
        physical_type="struct",
        logical_type="object",
        properties=children,
    )


def _named(prop: ContractProperty, name: str) -> ContractProperty:
    return ContractProperty(
        name=name,
        physical_type=prop.physical_type,
        logical_type=prop.logical_type,
        required=prop.required,
        properties=prop.properties,
        items=prop.items,
        keys=prop.keys,
        values=prop.values,
    )


def _container_cases() -> tuple[ContractProperty, ...]:
    """The representative containers AC-2's round trip names, two struct levels included."""
    return (
        _array_of(_named(_scalar("string"), "element")),
        _map_of(
            _named(_scalar("string", required=True), "key"),
            _named(_scalar("long"), "value"),
        ),
        _struct_of(
            _named(_scalar("integer", required=True), "leaf"),
            _named(
                _struct_of(_named(_scalar("timestamp"), "deep")),
                "middle",
            ),
        ),
    )


def test_vocabulary_is_non_empty_and_names_are_unique():
    names = [entry.name for entry in VOCABULARY]

    assert names
    assert len(names) == len(set(names))
    assert declared_physical_types() == tuple(sorted(names))


@pytest.mark.parametrize("entry", VOCABULARY, ids=lambda entry: entry.name)
def test_every_entry_has_all_three_spellings(entry: VocabularyType):
    assert entry.name.strip() == entry.name and entry.name
    assert entry.odcs_logical_type in ALLOWED_ODCS_LOGICAL_TYPES
    assert entry.iceberg_name.strip() == entry.iceberg_name and entry.iceberg_name
    assert entry.spark_json_type.strip() == entry.spark_json_type
    assert entry.spark_json_type


def test_required_vocabulary_entries_are_present():
    names = {entry.name for entry in VOCABULARY}

    assert set(REQUIRED_VOCABULARY_NAMES) <= names


@pytest.mark.parametrize("entry", _SCALAR_ENTRIES, ids=lambda entry: entry.name)
def test_every_scalar_entry_round_trips_through_spark_json(entry: VocabularyType):
    prop = _scalar(entry.name)

    assert physical_type_from_spark_json(spark_json_type(prop)) == prop.physical_type


@pytest.mark.parametrize("prop", _container_cases(), ids=lambda prop: prop.physical_type)
def test_every_container_round_trips_through_spark_json(prop: ContractProperty):
    assert physical_type_from_spark_json(spark_json_type(prop)) == prop.physical_type


@pytest.mark.parametrize("entry", _SCALAR_ENTRIES, ids=lambda entry: entry.name)
def test_every_scalar_entry_round_trips_through_iceberg_name(entry: VocabularyType):
    assert physical_type_from_iceberg_name(entry.iceberg_name) == entry.name


@pytest.mark.parametrize("spelling", ["decimal(18,2)", "decimal(18, 2)"])
def test_iceberg_decimal_spacings_map_to_the_same_physical_type(spelling: str):
    assert physical_type_from_iceberg_name(spelling) == "decimal(18,2)"


@pytest.mark.parametrize(
    ("spelling", "physical"),
    [
        ("struct<1: a: optional string>", "struct"),
        ("list<string>", "array"),
        ("map<string, long>", "map"),
    ],
)
def test_iceberg_container_names_map_to_their_top_level_type(spelling: str, physical: str):
    assert physical_type_from_iceberg_name(spelling) == physical


@pytest.mark.parametrize("spelling", ["uuid", "fixed[16]", "time", "timestamp_ns"])
def test_unsupported_iceberg_types_are_refused(spelling: str):
    with pytest.raises(UnsupportedIcebergTypeError):
        physical_type_from_iceberg_name(spelling)


def test_representative_decimal_round_trips_with_its_precision_and_scale():
    prop = _scalar(DECIMAL_TYPE_NAME)

    assert prop.physical_type == "decimal(18,2)"
    assert spark_json_type(prop) == "decimal(18,2)"
    assert iceberg_type_name(prop) == "decimal(18,2)"
    assert physical_type_from_spark_json("decimal(18,2)") == "decimal(18,2)"


def test_the_timestamp_pin_crosses_spark_and_iceberg_in_both_directions():
    zoneless = _scalar("timestamp")
    zoned = _scalar("timestamptz")

    assert spark_json_type(zoneless) == "timestamp_ntz"
    assert iceberg_type_name(zoneless) == "timestamp"
    assert spark_json_type(zoned) == "timestamp"
    assert iceberg_type_name(zoned) == "timestamptz"
    assert physical_type_from_spark_json("timestamp_ntz") == "timestamp"
    assert physical_type_from_spark_json("timestamp") == "timestamptz"


def test_odcs_logical_types_follow_the_pinned_table():
    assert {entry.name: entry.odcs_logical_type for entry in VOCABULARY} == {
        "boolean": "boolean",
        "integer": "integer",
        "long": "integer",
        "float": "number",
        "double": "number",
        "decimal(p,s)": "number",
        "string": "string",
        "binary": "string",
        "date": "date",
        "timestamp": "date",
        "timestamptz": "date",
        "struct": "object",
        "array": "array",
        "map": "object",
    }


def test_container_iceberg_names_recurse_through_their_children():
    array_prop, map_prop, struct_prop = _container_cases()

    assert iceberg_type_name(array_prop) == "list<string>"
    assert iceberg_type_name(map_prop) == "map<string, long>"
    assert iceberg_type_name(struct_prop) == ("struct<leaf: int, middle: struct<deep: timestamp>>")


def test_container_spark_json_carries_child_nullability():
    array_prop, map_prop, struct_prop = _container_cases()

    assert spark_json_type(array_prop) == {
        "type": "array",
        "elementType": "string",
        "containsNull": True,
    }
    assert spark_json_type(map_prop) == {
        "type": "map",
        "keyType": "string",
        "valueType": "long",
        "valueContainsNull": True,
    }
    assert spark_json_type(struct_prop) == {
        "type": "struct",
        "fields": [
            {"name": "leaf", "type": "integer", "nullable": False, "metadata": {}},
            {
                "name": "middle",
                "type": {
                    "type": "struct",
                    "fields": [
                        {
                            "name": "deep",
                            "type": "timestamp_ntz",
                            "nullable": True,
                            "metadata": {},
                        }
                    ],
                },
                "nullable": True,
                "metadata": {},
            },
        ],
    }


def test_required_array_items_become_containsnull_false():
    prop = _array_of(_named(_scalar("string", required=True), "element"))

    assert spark_json_type(prop) == {
        "type": "array",
        "elementType": "string",
        "containsNull": False,
    }


@pytest.mark.parametrize(
    "spelling",
    ["decimal", "decimal(40,2)", "decimal(4,6)", "decimal(p,s)", "Int", "short", "byte", ""],
)
def test_unknown_and_malformed_spellings_are_rejected(spelling: str):
    with pytest.raises(UnknownPhysicalTypeError):
        parse_physical_type(spelling)


def test_the_unknown_type_message_names_the_whole_vocabulary():
    with pytest.raises(UnknownPhysicalTypeError) as exc_info:
        parse_physical_type("not-a-type")

    message = str(exc_info.value)
    assert "not-a-type" in message
    for name in declared_physical_types():
        assert name in message


@pytest.mark.parametrize("spelling", ["short", "byte", "void", "timestamptz", "decimal"])
def test_spark_types_outside_the_vocabulary_are_refused_not_widened(spelling: str):
    with pytest.raises(UnsupportedSparkTypeError):
        physical_type_from_spark_json(spelling)


@pytest.mark.parametrize(
    "value",
    [
        {"type": "struct", "fields": [{"name": "x", "type": "short"}]},
        {"type": "array", "elementType": "short", "containsNull": True},
        {"type": "map", "keyType": "string", "valueType": "short"},
        {"type": "udt", "class": "example"},
        {"type": "array", "containsNull": True},
        {"type": "map", "keyType": "string"},
        42,
    ],
)
def test_nested_unsupported_spark_types_are_refused(value: Any):
    with pytest.raises(UnsupportedSparkTypeError):
        physical_type_from_spark_json(value)


def test_contract_properties_invert_the_generated_struct_json():
    properties = (
        _scalar("string", required=True),
        *_container_cases(),
    )
    struct_json = spark_struct_json(properties)

    rebuilt = contract_properties_from_spark_json(struct_json)

    assert spark_struct_json(rebuilt) == struct_json
    assert tuple(prop.physical_type for prop in rebuilt) == tuple(
        prop.physical_type for prop in properties
    )
    assert tuple(prop.required for prop in rebuilt) == tuple(prop.required for prop in properties)
    assert tuple(prop.logical_type for prop in rebuilt) == tuple(
        odcs_logical_type_for(prop.physical_type) for prop in properties
    )


def test_inverted_containers_keep_the_conventional_child_names():
    struct_json = spark_struct_json(_container_cases())

    array_prop, map_prop, _struct_prop = contract_properties_from_spark_json(struct_json)

    assert array_prop.items is not None and array_prop.items.name == "element"
    assert map_prop.keys is not None and map_prop.keys.name == "key"
    assert map_prop.values is not None and map_prop.values.name == "value"
    assert map_prop.keys.required is True


def test_malformed_struct_json_is_refused_rather_than_partially_read():
    with pytest.raises(UnsupportedSparkTypeError):
        contract_properties_from_spark_json({"type": "array", "elementType": "string"})
    with pytest.raises(UnsupportedSparkTypeError):
        contract_properties_from_spark_json({"type": "struct", "fields": [{"type": "string"}]})


def test_runs_table_iceberg_names_are_vocabulary_spellings():
    from janus.observability.runs_table import IcebergType

    spellings = {entry.iceberg_name for entry in VOCABULARY}
    declared = {member.value for member in IcebergType} - {"list<string>"}

    assert declared
    assert declared <= spellings
    assert IcebergType.STRING_LIST.value == "list<string>"


def test_vocabulary_module_imports_no_engine():
    source = Path(vocabulary_module.__file__).read_text(encoding="utf-8")

    assert _forbidden_imports(source) == ()
    assert ast.parse(source).body
