"""Runtime-loader guardrails for the JANUS ODCS subset."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

import janus.models.data_contracts.loader as loader_module
from janus.models.data_contracts import (
    ContractProperty,
    ContractSchema,
    ContractValidationError,
    DataContract,
    JanusContractOptions,
    compute_schema_version,
    load_data_contract,
)
from janus.models.data_contracts.loader import PINNED_ODCS_API_VERSION
from janus.models.data_contracts.vocabulary import iceberg_type_name

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "contracts"
EXAMPLE_CONTRACT = (
    PROJECT_ROOT / "conf" / "contracts" / "example" / "federal_open_data_example.yaml"
)


def _minimal_mapping() -> dict[str, Any]:
    loaded = yaml.safe_load(
        (CONTRACT_FIXTURES / "minimal_contract.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def _write_contract(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_example_contract_loads_into_the_complete_frozen_model():
    expected_hash = hashlib.sha256(EXAMPLE_CONTRACT.read_bytes()).hexdigest()

    contract = load_data_contract(EXAMPLE_CONTRACT)

    assert contract == DataContract(
        contract_path=EXAMPLE_CONTRACT,
        api_version="v3.2.0",
        id="example.federal_open_data_example",
        name="Federal open data example",
        version="1.0.1",
        status="active",
        domain="example",
        purpose=(
            "Contract example that exercises the api family against example.invalid; "
            "not a live source."
        ),
        owners=("janus",),
        tags=("example", "api"),
        schema=ContractSchema(
            name="federal_open_data_example",
            physical_type="table",
            properties=(
                ContractProperty(
                    name="id",
                    physical_type="string",
                    logical_type="string",
                    business_name="Record id",
                    description="Upstream record identifier, verbatim.",
                    required=True,
                    unique=True,
                    primary_key=True,
                    classification="public",
                    source_field="id",
                    source_format="json",
                    source_nullable=True,
                ),
                ContractProperty(
                    name="updated_at",
                    physical_type="string",
                    logical_type="string",
                    business_name="Updated at",
                    description=("Upstream update timestamp, kept as the string the API sent."),
                    required=True,
                    classification="public",
                    source_field="updated_at",
                    source_format="json",
                    source_nullable=True,
                ),
            ),
        ),
        janus=JanusContractOptions(
            compatibility="additive",
            enforcement="strict",
        ),
        schema_version=expected_hash,
    )
    assert contract.column_names == ("id", "updated_at")
    assert contract.required_columns == ("id", "updated_at")
    assert contract.primary_key == ("id",)
    assert compute_schema_version(EXAMPLE_CONTRACT) == expected_hash


def test_loader_pin_matches_the_vendored_schema_filename():
    pinned_schema = (
        PROJECT_ROOT
        / "docs"
        / "schemas"
        / "odcs"
        / f"odcs-json-schema-{PINNED_ODCS_API_VERSION}.json"
    )

    assert pinned_schema.is_file()


def test_load_data_contract_owns_the_single_raise_site():
    source = Path(loader_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_data_contract"
    )
    raise_lines = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Raise)]

    assert len(raise_lines) == 1


def test_minimal_contract_loads_with_optional_defaults():
    contract = load_data_contract(CONTRACT_FIXTURES / "minimal_contract.yaml")

    assert contract.tags == ()
    assert contract.column_names == ("id",)
    assert contract.required_columns == ()
    assert contract.primary_key == ()
    assert contract.schema.properties[0].business_name is None
    assert contract.janus.drafted_from is None


def test_two_schemas_fail_closed():
    path = CONTRACT_FIXTURES / "odcs_invalid_two_schemas.yaml"

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(path)

    assert exc_info.value.issues[0].path == "schema"
    assert exc_info.value.issues[0].message == ("must declare exactly one table; found 2")


def test_five_independent_problems_are_collected_and_sorted(tmp_path):
    data = _minimal_mapping()
    data.pop("kind")
    data["version"] = "1.0"
    data["status"] = "unknown"
    duplicate = copy.deepcopy(data["schema"][0]["properties"][0])
    data["schema"][0]["properties"].append(duplicate)
    data["customProperties"].append({"property": "janus.foo", "value": "unsupported"})
    path = _write_contract(tmp_path, data)

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(path)

    assert [issue.path for issue in exc_info.value.issues] == [
        "customProperties[2].property",
        "kind",
        "schema[0].properties[1].name",
        "status",
        "version",
    ]
    assert len(str(exc_info.value).splitlines()) == 5
    assert all(line.startswith(f"{path}: ") for line in str(exc_info.value).splitlines())


def test_nested_property_paths_keep_every_index(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"].append(
        {
            "name": "outer",
            "logicalType": "object",
            "physicalType": "struct",
            "properties": [
                {
                    "name": "middle",
                    "logicalType": "object",
                    "physicalType": "struct",
                    "properties": [
                        {
                            "name": " bad ",
                            "logicalType": "string",
                            "physicalType": "string",
                        }
                    ],
                }
            ],
        }
    )
    path = _write_contract(tmp_path, data)

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(path)

    assert [issue.path for issue in exc_info.value.issues] == [
        "schema[0].properties[1].properties[0].properties[0].name"
    ]


def test_struct_and_array_require_their_nested_declarations(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"].extend(
        [
            {"name": "nested", "logicalType": "object", "physicalType": "struct"},
            {"name": "values", "logicalType": "array", "physicalType": "array"},
        ]
    )
    path = _write_contract(tmp_path, data)

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(path)

    assert [issue.path for issue in exc_info.value.issues] == [
        "schema[0].properties[1].properties",
        "schema[0].properties[2].items",
    ]


def test_unknown_top_level_odcs_keys_are_ignored(tmp_path):
    data = _minimal_mapping()
    data["servers"] = [{"server": "ignored"}]
    data["slaProperties"] = [{"property": "ignored"}]

    contract = load_data_contract(_write_contract(tmp_path, data))

    assert contract.id == "example.minimal"


def test_unpinned_api_version_names_the_pin(tmp_path):
    data = _minimal_mapping()
    data["apiVersion"] = "v3.1.0"

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [(issue.path, issue.message) for issue in exc_info.value.issues] == [
        (
            "apiVersion",
            f"must equal pinned ODCS API version '{PINNED_ODCS_API_VERSION}'",
        )
    ]


def test_numeric_yaml_version_requires_a_quoted_semver(tmp_path):
    data = _minimal_mapping()
    data["version"] = 1.0

    with pytest.raises(ContractValidationError, match="must be a quoted semver string"):
        load_data_contract(_write_contract(tmp_path, data))


def test_invalid_yaml_is_reported_as_a_contract_issue(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("schema: [", encoding="utf-8")

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(path)

    assert [issue.path for issue in exc_info.value.issues] == ["<root>"]
    assert "must be valid YAML" in exc_info.value.issues[0].message


def test_unknown_physical_type_is_rejected_by_the_closed_vocabulary(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"][0]["physicalType"] = "not-a-type"

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [issue.path for issue in exc_info.value.issues] == [
        "schema[0].properties[0].physicalType"
    ]
    assert "unknown physical type" in exc_info.value.issues[0].message
    assert "timestamptz" in exc_info.value.issues[0].message


def test_a_declared_logical_type_must_agree_with_its_physical_type(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"][0]["physicalType"] = "long"

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [(issue.path, issue.message) for issue in exc_info.value.issues] == [
        (
            "schema[0].properties[0].logicalType",
            "must be 'integer' for physicalType 'long'",
        )
    ]


def test_a_non_string_logical_type_reports_one_problem_once(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"][0]["logicalType"] = 7

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [(issue.path, issue.message) for issue in exc_info.value.issues] == [
        ("schema[0].properties[0].logicalType", "must be a string")
    ]


def test_an_omitted_logical_type_is_derived_from_the_physical_type(tmp_path):
    data = _minimal_mapping()
    prop = data["schema"][0]["properties"][0]
    prop.pop("logicalType")
    prop["physicalType"] = "decimal(18,2)"

    contract = load_data_contract(_write_contract(tmp_path, data))

    assert contract.schema.properties[0].physical_type == "decimal(18,2)"
    assert contract.schema.properties[0].logical_type == "number"


def test_a_malformed_decimal_spelling_is_rejected(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"][0]["physicalType"] = "decimal"

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [issue.path for issue in exc_info.value.issues] == [
        "schema[0].properties[0].physicalType"
    ]


def test_a_map_reads_its_odcs_key_and_value_block(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"].append(
        {
            "name": "attributes",
            "logicalType": "object",
            "physicalType": "map",
            "map": {
                "key": {
                    "name": "key",
                    "logicalType": "string",
                    "physicalType": "string",
                    "required": True,
                },
                "value": {
                    "name": "value",
                    "logicalType": "integer",
                    "physicalType": "long",
                },
            },
        }
    )

    contract = load_data_contract(_write_contract(tmp_path, data))
    attributes = contract.schema.properties[1]

    assert attributes.physical_type == "map"
    assert attributes.keys is not None and attributes.keys.physical_type == "string"
    assert attributes.keys.required is True
    assert attributes.values is not None and attributes.values.physical_type == "long"
    assert iceberg_type_name(attributes) == "map<string, long>"


def test_a_map_without_its_key_and_value_block_is_an_issue(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"].append(
        {"name": "attributes", "logicalType": "object", "physicalType": "map"}
    )

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [(issue.path, issue.message) for issue in exc_info.value.issues] == [
        ("schema[0].properties[1].map", "is required for physicalType 'map'")
    ]


def test_a_nested_shape_belonging_to_another_container_is_rejected(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"].append(
        {
            "name": "tags",
            "logicalType": "array",
            "physicalType": "array",
            "items": {"name": "element", "logicalType": "string", "physicalType": "string"},
            "map": {"key": {}, "value": {}},
        }
    )

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert [(issue.path, issue.message) for issue in exc_info.value.issues] == [
        ("schema[0].properties[1].map", "is not allowed for physicalType 'array'")
    ]


@pytest.mark.parametrize("value", ["-1", "x", "1.5", "+2"])
def test_max_malformed_rows_rejects_invalid_strings(tmp_path, value):
    data = _minimal_mapping()
    data["customProperties"].append({"property": "janus.maxMalformedRows", "value": value})

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert any(
        issue.path == "customProperties.janus.maxMalformedRows" for issue in exc_info.value.issues
    )


def test_max_malformed_rows_parses_and_defaults_to_zero(tmp_path):
    data = _minimal_mapping()
    assert load_data_contract(_write_contract(tmp_path, data)).janus.max_malformed_rows == 0
    data["customProperties"].append({"property": "janus.maxMalformedRows", "value": "3"})
    assert load_data_contract(_write_contract(tmp_path, data)).janus.max_malformed_rows == 3


def test_corrupt_record_column_is_reserved_even_when_nested(tmp_path):
    data = _minimal_mapping()
    data["schema"][0]["properties"].append(
        {"name": "_janus_corrupt_record", "physicalType": "string"}
    )

    with pytest.raises(ContractValidationError) as exc_info:
        load_data_contract(_write_contract(tmp_path, data))

    assert any(
        issue.path == "schema[0].properties[1].name" and "reserved" in issue.message
        for issue in exc_info.value.issues
    )


def test_corrupt_schema_flag_appends_one_nullable_string_and_preserves_default():
    from janus.schema_contracts import spark_schema_from_contract

    contract = load_data_contract(CONTRACT_FIXTURES / "hostile" / "base.yaml")
    default_schema = spark_schema_from_contract(contract).jsonValue()
    tracked_schema = spark_schema_from_contract(contract, with_corrupt_record=True).jsonValue()

    assert tracked_schema["fields"][:-1] == default_schema["fields"]
    assert tracked_schema["fields"][-1] == {
        "name": "_janus_corrupt_record",
        "type": "string",
        "nullable": True,
        "metadata": {},
    }
