"""The one closed type vocabulary behind every JANUS data contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from janus.models.data_contracts.model import ContractProperty

DECIMAL_TYPE_NAME = "decimal(p,s)"
MAX_DECIMAL_PRECISION = 38

#: Container types whose spelling is recursive, so they are never a bare Spark JSON string.
CONTAINER_TYPE_NAMES = frozenset({"array", "map", "struct"})

_DECIMAL_SPELLING = re.compile(r"^decimal\((\d{1,2}),(\d{1,2})\)$")
#: Any Spark JSON type value: a scalar spelling, or a container object.
_SparkJson = Mapping[str, Any] | str

_ARRAY_ELEMENT_NAME = "element"
_MAP_KEY_NAME = "key"
_MAP_VALUE_NAME = "value"


class VocabularyError(ValueError):
    """A type spelling the closed JANUS vocabulary cannot express."""


class UnknownPhysicalTypeError(VocabularyError):
    """A contract declared a ``physicalType`` outside the vocabulary."""


class UnsupportedSparkTypeError(VocabularyError):
    """A Spark schema carries a type no vocabulary entry can express."""


class UnsupportedIcebergTypeError(VocabularyError):
    """An Iceberg schema carries a type no vocabulary entry can express."""


@dataclass(frozen=True, slots=True)
class VocabularyType:
    """One vocabulary entry and its spelling in each of the three dialects."""

    name: str
    odcs_logical_type: str
    iceberg_name: str
    spark_json_type: str


VOCABULARY: tuple[VocabularyType, ...] = (
    VocabularyType("boolean", "boolean", "boolean", "boolean"),
    VocabularyType("integer", "integer", "int", "integer"),
    VocabularyType("long", "integer", "long", "long"),
    VocabularyType("float", "number", "float", "float"),
    VocabularyType("double", "number", "double", "double"),
    VocabularyType(DECIMAL_TYPE_NAME, "number", DECIMAL_TYPE_NAME, DECIMAL_TYPE_NAME),
    VocabularyType("string", "string", "string", "string"),
    VocabularyType("binary", "string", "binary", "binary"),
    VocabularyType("date", "date", "date", "date"),
    VocabularyType("timestamp", "date", "timestamp", "timestamp_ntz"),
    VocabularyType("timestamptz", "date", "timestamptz", "timestamp"),
    VocabularyType("struct", "object", "struct<...>", "struct"),
    VocabularyType("array", "array", "list<...>", "array"),
    VocabularyType("map", "object", "map<...>", "map"),
)

_BY_NAME: dict[str, VocabularyType] = {entry.name: entry for entry in VOCABULARY}

_BY_SPARK_JSON: dict[str, VocabularyType] = {
    entry.spark_json_type: entry
    for entry in VOCABULARY
    if entry.name not in CONTAINER_TYPE_NAMES and entry.name != DECIMAL_TYPE_NAME
}
_BY_ICEBERG_NAME: dict[str, VocabularyType] = {
    entry.iceberg_name: entry
    for entry in VOCABULARY
    if entry.name not in CONTAINER_TYPE_NAMES and entry.name != DECIMAL_TYPE_NAME
}
_DECIMAL_TYPE = _BY_NAME[DECIMAL_TYPE_NAME]


def declared_physical_types() -> tuple[str, ...]:
    """Every vocabulary spelling, sorted, for use in an error message."""
    return tuple(sorted(_BY_NAME))


@dataclass(frozen=True, slots=True)
class ParsedType:
    """One resolved vocabulary entry, carrying a decimal's precision and scale."""

    type: VocabularyType
    precision: int | None = None
    scale: int | None = None

    @property
    def name(self) -> str:
        return self.type.name

    @property
    def physical_type(self) -> str:
        """The contract spelling this parse came from."""
        return self._render(self.type.name)

    @property
    def iceberg_name(self) -> str:
        """The Iceberg spelling for a scalar; containers go through :func:`iceberg_type_name`."""
        return self._render(self.type.iceberg_name)

    @property
    def spark_json_type(self) -> str:
        """The Spark JSON spelling for a scalar; containers render as a JSON object."""
        return self._render(self.type.spark_json_type)

    def _render(self, spelling: str) -> str:
        if self.precision is None or self.scale is None:
            return spelling
        return spelling.replace("(p,s)", f"({self.precision},{self.scale})")


def parse_physical_type(text: str) -> ParsedType:
    """Resolve one contract ``physicalType`` spelling, or refuse it by name."""
    if not isinstance(text, str) or not text.strip():
        raise UnknownPhysicalTypeError(_unknown_message(text))

    spelling = text.strip()
    if spelling != DECIMAL_TYPE_NAME and spelling in _BY_NAME:
        return ParsedType(_BY_NAME[spelling])
    if spelling.startswith("decimal"):
        return _parse_decimal(spelling)
    raise UnknownPhysicalTypeError(_unknown_message(spelling))


def odcs_logical_type_for(physical: str) -> str:
    """The ODCS ``logicalType`` one physical spelling must declare."""
    return parse_physical_type(physical).type.odcs_logical_type


def spark_sql_type(physical_type: str) -> str:
    """Return the scalar Spark SQL spelling used by Iceberg schema DDL."""
    parsed = parse_physical_type(physical_type)
    if parsed.name in CONTAINER_TYPE_NAMES:
        raise UnknownPhysicalTypeError(
            f"nested type {physical_type!r} requires a property tree for Spark SQL DDL"
        )
    if parsed.name == "long":
        return "bigint"
    if parsed.name == "integer":
        return "int"
    return parsed.spark_json_type


def iceberg_type_name(prop: ContractProperty) -> str:
    """The Iceberg type name for one property, recursing through containers."""
    parsed = parse_physical_type(prop.physical_type)
    if parsed.name == "struct":
        fields = ", ".join(f"{child.name}: {iceberg_type_name(child)}" for child in prop.properties)
        return f"struct<{fields}>"
    if parsed.name == "array":
        return f"list<{iceberg_type_name(_array_items(prop))}>"
    if parsed.name == "map":
        keys, values = _map_children(prop)
        return f"map<{iceberg_type_name(keys)}, {iceberg_type_name(values)}>"
    return parsed.iceberg_name


def physical_type_from_iceberg_name(text: str) -> str:
    """Map an Iceberg type spelling to the contract vocabulary.

    Container children are read from the Iceberg field objects by callers; the
    rendered container spelling identifies only its top-level kind here.
    """
    if not isinstance(text, str):
        raise UnsupportedIcebergTypeError(f"unsupported Iceberg type: {text!r}")
    spelling = text.strip()
    entry = _BY_ICEBERG_NAME.get(spelling)
    if entry is not None:
        return entry.name
    if spelling == DECIMAL_TYPE_NAME:
        return DECIMAL_TYPE_NAME
    if spelling.startswith("decimal"):
        compact = re.sub(r",\s+", ",", spelling)
        try:
            return _parse_decimal(compact).physical_type
        except UnknownPhysicalTypeError as exc:
            raise UnsupportedIcebergTypeError(str(exc)) from exc
    for iceberg_prefix, physical_type in (
        ("struct<", "struct"),
        ("list<", "array"),
        ("map<", "map"),
    ):
        if spelling.startswith(iceberg_prefix) and spelling.endswith(">"):
            return physical_type
    raise UnsupportedIcebergTypeError(f"unsupported Iceberg type: {text!r}")


def spark_json_type(prop: ContractProperty) -> dict[str, Any] | str:
    """The Spark JSON type value ``StructType.fromJson`` would read for one property."""
    parsed = parse_physical_type(prop.physical_type)
    if parsed.name == "struct":
        return spark_struct_json(prop.properties)
    if parsed.name == "array":
        items = _array_items(prop)
        return {
            "type": "array",
            "elementType": spark_json_type(items),
            "containsNull": _source_nullable(items),
        }
    if parsed.name == "map":
        keys, values = _map_children(prop)
        return {
            "type": "map",
            "keyType": spark_json_type(keys),
            "valueType": spark_json_type(values),
            "valueContainsNull": _source_nullable(values),
        }
    return parsed.spark_json_type


def spark_struct_json(properties: tuple[ContractProperty, ...]) -> dict[str, Any]:
    """The Spark ``struct`` JSON for a list of properties — one contract table or one struct."""
    return {
        "type": "struct",
        "fields": [
            {
                "name": prop.name,
                "type": spark_json_type(prop),
                "nullable": _source_nullable(prop),
                "metadata": {},
            }
            for prop in properties
        ],
    }


def physical_type_from_spark_json(value: _SparkJson) -> str:
    """The contract spelling for one Spark JSON type value — the inverse of the above."""
    if isinstance(value, str):
        return _scalar_physical_type_from_spark_json(value)
    if isinstance(value, Mapping):
        return _container_physical_type_from_spark_json(value)
    raise UnsupportedSparkTypeError(_unsupported_message(value))


def contract_properties_from_spark_json(
    value: _SparkJson,
) -> tuple[ContractProperty, ...]:
    """Contract properties for one Spark ``struct`` JSON value, used when drafting."""
    fields = _spark_struct_fields(value)
    return tuple(
        _property_from_spark_type(
            _spark_field_name(field),
            field["type"],
            required=not field.get("nullable", True),
            source_nullable=bool(field.get("nullable", True)),
        )
        for field in fields
    )


def _parse_decimal(spelling: str) -> ParsedType:
    match = _DECIMAL_SPELLING.fullmatch(spelling)
    if match is None:
        raise UnknownPhysicalTypeError(
            f"'{spelling}' must be spelled decimal(p,s) with digits only, "
            f"1 <= p <= {MAX_DECIMAL_PRECISION} and 0 <= s <= p"
        )
    precision, scale = int(match.group(1)), int(match.group(2))
    if not 1 <= precision <= MAX_DECIMAL_PRECISION:
        raise UnknownPhysicalTypeError(
            f"'{spelling}' precision must be between 1 and {MAX_DECIMAL_PRECISION}"
        )
    if scale > precision:
        raise UnknownPhysicalTypeError(f"'{spelling}' scale must not be greater than its precision")
    return ParsedType(_DECIMAL_TYPE, precision=precision, scale=scale)


def _scalar_physical_type_from_spark_json(value: str) -> str:
    entry = _BY_SPARK_JSON.get(value)
    if entry is not None:
        return entry.name
    if value.startswith("decimal"):
        # A Spark-side spelling problem is reported as one, not as a contract-side one.
        try:
            return _parse_decimal(value).physical_type
        except UnknownPhysicalTypeError as exc:
            raise UnsupportedSparkTypeError(str(exc)) from exc
    raise UnsupportedSparkTypeError(_unsupported_message(value))


def _container_physical_type_from_spark_json(value: Mapping[str, Any]) -> str:
    kind = value.get("type")
    if kind == "struct":
        for field in _spark_struct_fields(value):
            physical_type_from_spark_json(field["type"])
        return "struct"
    if kind == "array":
        physical_type_from_spark_json(_required_key(value, "elementType"))
        return "array"
    if kind == "map":
        physical_type_from_spark_json(_required_key(value, "keyType"))
        physical_type_from_spark_json(_required_key(value, "valueType"))
        return "map"
    raise UnsupportedSparkTypeError(_unsupported_message(kind))


def _required_key(value: Mapping[str, Any], key: str) -> Any:
    """One structural key of a Spark container, refused by name when it is missing."""
    if key not in value:
        raise UnsupportedSparkTypeError(f"a Spark {value.get('type')!r} type must declare {key!r}")
    return value[key]


def _property_from_spark_type(
    name: str,
    value: _SparkJson,
    *,
    required: bool,
    source_nullable: bool,
) -> ContractProperty:
    physical_type = physical_type_from_spark_json(value)
    nested: dict[str, Any] = {}
    if physical_type == "struct":
        nested["properties"] = contract_properties_from_spark_json(value)
    elif physical_type == "array":
        element = _spark_mapping(value)
        nested["items"] = _property_from_spark_type(
            _ARRAY_ELEMENT_NAME,
            _required_key(element, "elementType"),
            required=not element.get("containsNull", True),
            source_nullable=bool(element.get("containsNull", True)),
        )
    elif physical_type == "map":
        entries = _spark_mapping(value)
        nested["keys"] = _property_from_spark_type(
            _MAP_KEY_NAME,
            _required_key(entries, "keyType"),
            required=True,
            source_nullable=False,
        )
        nested["values"] = _property_from_spark_type(
            _MAP_VALUE_NAME,
            _required_key(entries, "valueType"),
            required=not entries.get("valueContainsNull", True),
            source_nullable=bool(entries.get("valueContainsNull", True)),
        )
    return ContractProperty(
        name=name,
        physical_type=physical_type,
        logical_type=odcs_logical_type_for(physical_type),
        required=required,
        source_nullable=source_nullable,
        **nested,
    )


def _source_nullable(prop: ContractProperty) -> bool:
    """Keep source nullability independent from a contract quality expectation."""
    return prop.source_nullable if prop.source_nullable is not None else not prop.required


def _spark_struct_fields(value: _SparkJson) -> tuple[Mapping[str, Any], ...]:
    struct = _spark_mapping(value)
    if struct.get("type") != "struct" or not isinstance(struct.get("fields"), list):
        raise UnsupportedSparkTypeError("expected a Spark struct JSON object with a 'fields' list")
    fields = struct["fields"]
    for field in fields:
        if not isinstance(field, Mapping) or "type" not in field:
            raise UnsupportedSparkTypeError(
                "every Spark struct field must be an object carrying a 'type'"
            )
    return tuple(fields)


def _spark_field_name(field: Mapping[str, Any]) -> str:
    name = field.get("name")
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedSparkTypeError("every Spark struct field must carry a name")
    return name


def _spark_mapping(value: _SparkJson) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnsupportedSparkTypeError(_unsupported_message(value))
    return value


def _array_items(prop: ContractProperty) -> ContractProperty:
    if prop.items is None:
        raise VocabularyError(f"array property '{prop.name}' declares no items")
    return prop.items


def _map_children(prop: ContractProperty) -> tuple[ContractProperty, ContractProperty]:
    if prop.keys is None or prop.values is None:
        raise VocabularyError(f"map property '{prop.name}' declares no key and value")
    return prop.keys, prop.values


def _unknown_message(value: Any) -> str:
    allowed = ", ".join(declared_physical_types())
    return f"unknown physical type {value!r}; the vocabulary is: {allowed}"


def _unsupported_message(value: Any) -> str:
    allowed = ", ".join(sorted([*_BY_SPARK_JSON, DECIMAL_TYPE_NAME]))
    return (
        f"Spark type {value!r} has no JANUS vocabulary spelling; scalars are: {allowed}, "
        "and array, map and struct arrive as JSON objects"
    )


__all__ = [
    "CONTAINER_TYPE_NAMES",
    "DECIMAL_TYPE_NAME",
    "MAX_DECIMAL_PRECISION",
    "VOCABULARY",
    "ParsedType",
    "UnknownPhysicalTypeError",
    "UnsupportedSparkTypeError",
    "VocabularyError",
    "VocabularyType",
    "contract_properties_from_spark_json",
    "declared_physical_types",
    "iceberg_type_name",
    "odcs_logical_type_for",
    "parse_physical_type",
    "physical_type_from_spark_json",
    "spark_json_type",
    "spark_struct_json",
]
