"""Fail-closed structural loader for the JANUS subset of ODCS v3.2.0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import yaml

from janus.models.config.coercion import (
    _optional_bool,
    _optional_string,
    _optional_string_list,
    _require_enum,
    _require_mapping,
    _require_string,
)
from janus.models.config.issues import ValidationIssue
from janus.models.data_contracts.errors import ContractValidationError
from janus.models.data_contracts.model import (
    SUPPORTED_COMPATIBILITY_MODES,
    SUPPORTED_CONTRACT_STATUSES,
    SUPPORTED_ENFORCEMENT_MODES,
    ContractProperty,
    ContractSchema,
    DataContract,
    JanusContractOptions,
    _is_semver,
)

PINNED_ODCS_API_VERSION = "v3.2.0"
_ALLOWED_JANUS_PROPERTIES = frozenset(
    {"janus.compatibility", "janus.draftedFrom", "janus.enforcement"}
)


@dataclass(frozen=True, slots=True)
class _Identity:
    api_version: str
    id: str
    name: str
    version: str
    status: str
    domain: str
    purpose: str
    tags: tuple[str, ...]


def compute_schema_version(path: Path) -> str:
    """Return the SHA-256 identity of the exact contract bytes at path."""
    return sha256(path.read_bytes()).hexdigest()


def load_data_contract(path: Path) -> DataContract:
    """Load one complete contract or report all structural issues found in it."""
    contract_bytes = path.read_bytes()
    schema_version = sha256(contract_bytes).hexdigest()
    issues: list[ValidationIssue] = []

    try:
        payload: Any = yaml.safe_load(contract_bytes)
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        issues.append(ValidationIssue("<root>", f"must be valid YAML: {problem}"))
        payload = None

    identity: _Identity | None = None
    owners: tuple[str, ...] = ()
    schema: ContractSchema | None = None
    janus: JanusContractOptions | None = None

    if not issues:
        data = _require_mapping(payload, "<root>", issues)
        identity = _read_identity(data, issues)
        owners = _read_team(data.get("team"), issues)
        schema = _read_schema(data.get("schema"), issues)
        janus = _read_janus_options(data.get("customProperties"), issues)

    if issues:
        raise ContractValidationError(path, issues)

    valid_identity = cast(_Identity, identity)
    return DataContract(
        contract_path=path,
        api_version=valid_identity.api_version,
        id=valid_identity.id,
        name=valid_identity.name,
        version=valid_identity.version,
        status=valid_identity.status,
        domain=valid_identity.domain,
        purpose=valid_identity.purpose,
        owners=owners,
        tags=valid_identity.tags,
        schema=cast(ContractSchema, schema),
        janus=cast(JanusContractOptions, janus),
        schema_version=schema_version,
    )


def _read_identity(data: Mapping[str, Any], issues: list[ValidationIssue]) -> _Identity:
    api_version = _require_string(data, "apiVersion", issues)
    if api_version and api_version != PINNED_ODCS_API_VERSION:
        issues.append(
            ValidationIssue(
                "apiVersion",
                f"must equal pinned ODCS API version '{PINNED_ODCS_API_VERSION}'",
            )
        )

    kind = _require_string(data, "kind", issues)
    if kind and kind != "DataContract":
        issues.append(ValidationIssue("kind", "must equal 'DataContract'"))

    description = _require_mapping(data.get("description"), "description", issues)
    version = _read_version(data, issues)
    return _Identity(
        api_version=api_version,
        id=_require_string(data, "id", issues),
        name=_require_string(data, "name", issues),
        version=version,
        status=_require_enum(data, "status", SUPPORTED_CONTRACT_STATUSES, issues),
        domain=_require_string(data, "domain", issues),
        purpose=_require_string(description, "purpose", issues, "description"),
        tags=tuple(_optional_string_list(data, "tags", issues)),
    )


def _read_version(data: Mapping[str, Any], issues: list[ValidationIssue]) -> str:
    value = data.get("version")
    if value is None:
        issues.append(ValidationIssue("version", "is required"))
        return ""
    if not isinstance(value, str):
        issues.append(
            ValidationIssue(
                "version", "must be a quoted semver string (MAJOR.MINOR.PATCH)"
            )
        )
        return ""
    if not _is_semver(value):
        issues.append(
            ValidationIssue("version", "must be MAJOR.MINOR.PATCH using digits only")
        )
    return value


def _read_team(value: Any, issues: list[ValidationIssue]) -> tuple[str, ...]:
    if value is None:
        issues.append(ValidationIssue("team", "is required"))
        return ()
    if not isinstance(value, list):
        issues.append(ValidationIssue("team", "must be a list"))
        return ()

    owners: list[str] = []
    has_owner_role = False
    for index, item in enumerate(value):
        path = f"team[{index}]"
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(path, "must be a mapping"))
            continue
        username = _require_string(item, "username", issues, path)
        role = _require_string(item, "role", issues, path)
        if role == "owner":
            has_owner_role = True
            if username:
                owners.append(username)

    if not has_owner_role:
        issues.append(
            ValidationIssue("team", "must include at least one member with role 'owner'")
        )
    return tuple(owners)


def _read_schema(value: Any, issues: list[ValidationIssue]) -> ContractSchema | None:
    if value is None:
        issues.append(ValidationIssue("schema", "is required"))
        return None
    if not isinstance(value, list):
        issues.append(ValidationIssue("schema", "must be a list"))
        return None
    if len(value) != 1:
        issues.append(
            ValidationIssue(
                "schema", f"must declare exactly one table; found {len(value)}"
            )
        )

    schemas = [
        _read_schema_entry(item, f"schema[{index}]", issues)
        for index, item in enumerate(value)
    ]
    return schemas[0] if schemas else None


def _read_schema_entry(
    value: Any, path: str, issues: list[ValidationIssue]
) -> ContractSchema | None:
    start = len(issues)
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(path, "must be a mapping"))
        return None

    name = _require_string(value, "name", issues, path)
    physical_type = _require_string(value, "physicalType", issues, path)
    if physical_type and physical_type != "table":
        issues.append(ValidationIssue(f"{path}.physicalType", "must equal 'table'"))
    properties = _read_properties(value.get("properties"), f"{path}.properties", issues)

    if len(issues) != start:
        return None
    return ContractSchema(name=name, physical_type=physical_type, properties=properties)


def _read_properties(
    value: Any, path: str, issues: list[ValidationIssue]
) -> tuple[ContractProperty, ...]:
    if value is None:
        issues.append(ValidationIssue(path, "is required"))
        return ()
    if not isinstance(value, list):
        issues.append(ValidationIssue(path, "must be a list"))
        return ()
    if not value:
        issues.append(ValidationIssue(path, "must contain at least one property"))
        return ()

    properties: list[ContractProperty] = []
    seen_names: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(item, Mapping):
            _record_duplicate_name(item.get("name"), item_path, seen_names, issues)
        prop = _read_property(item, item_path, issues)
        if prop is not None:
            properties.append(prop)
    return tuple(properties)


def _record_duplicate_name(
    value: Any,
    path: str,
    seen_names: set[str],
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        return
    if value in seen_names:
        issues.append(
            ValidationIssue(
                f"{path}.name", f"must be unique within its level; found '{value}'"
            )
        )
    seen_names.add(value)


def _read_property(
    value: Any, path: str, issues: list[ValidationIssue]
) -> ContractProperty | None:
    start = len(issues)
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(path, "must be a mapping"))
        return None

    name = _read_property_name(value, path, issues)
    physical_type = _require_string(value, "physicalType", issues, path)
    logical_type = _optional_string(value, "logicalType", issues, path) or physical_type
    custom = _read_custom_properties(
        value.get("customProperties"), f"{path}.customProperties", issues
    )
    properties, items = _read_nested_declarations(
        value, physical_type, path, issues
    )

    business_name = _optional_string(value, "businessName", issues, path)
    description = _optional_string(value, "description", issues, path)
    required = _optional_bool(value, "required", issues, path)
    unique = _optional_bool(value, "unique", issues, path)
    primary_key = _optional_bool(value, "primaryKey", issues, path)
    classification = _optional_string(value, "classification", issues, path)
    if len(issues) != start:
        return None
    return ContractProperty(
        name=name,
        physical_type=physical_type,
        logical_type=logical_type,
        business_name=business_name,
        description=description,
        required=required,
        unique=unique,
        primary_key=primary_key,
        classification=classification,
        source_field=custom.get("sourceField"),
        source_format=custom.get("sourceFormat"),
        properties=properties,
        items=items,
    )


def _read_property_name(
    data: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> str:
    value = data.get("name")
    field_path = f"{path}.name"
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return ""
    if not isinstance(value, str):
        issues.append(ValidationIssue(field_path, "must be a string"))
        return ""
    if not value.strip():
        issues.append(ValidationIssue(field_path, "must not be empty"))
    elif value != value.strip():
        issues.append(
            ValidationIssue(field_path, "must not contain surrounding whitespace")
        )
    return value


def _read_nested_declarations(
    data: Mapping[str, Any],
    physical_type: str,
    path: str,
    issues: list[ValidationIssue],
) -> tuple[tuple[ContractProperty, ...], ContractProperty | None]:
    has_properties = "properties" in data
    has_items = "items" in data
    properties: tuple[ContractProperty, ...] = ()
    items: ContractProperty | None = None

    if physical_type == "struct":
        properties = _read_properties(
            data.get("properties"), f"{path}.properties", issues
        )
        if has_items:
            issues.append(
                ValidationIssue(
                    f"{path}.items", "is not allowed for physicalType 'struct'"
                )
            )
    elif physical_type == "array":
        items = _read_array_item(data.get("items"), f"{path}.items", issues)
        if has_properties:
            issues.append(
                ValidationIssue(
                    f"{path}.properties", "is not allowed for physicalType 'array'"
                )
            )
    else:
        _reject_scalar_nested_declarations(
            has_properties, has_items, physical_type, path, issues
        )
    return properties, items


def _read_array_item(
    value: Any, path: str, issues: list[ValidationIssue]
) -> ContractProperty | None:
    if value is None:
        issues.append(
            ValidationIssue(path, "is required for physicalType 'array'")
        )
        return None
    return _read_property(value, path, issues)


def _reject_scalar_nested_declarations(
    has_properties: bool,
    has_items: bool,
    physical_type: str,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if has_properties:
        issues.append(
            ValidationIssue(
                f"{path}.properties",
                f"is not allowed for physicalType '{physical_type}'",
            )
        )
    if has_items:
        issues.append(
            ValidationIssue(
                f"{path}.items",
                f"is not allowed for physicalType '{physical_type}'",
            )
        )


def _read_custom_properties(
    value: Any, path: str, issues: list[ValidationIssue]
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, list):
        issues.append(ValidationIssue(path, "must be a list"))
        return {}

    result: dict[str, str] = {}
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(item_path, "must be a mapping"))
            continue
        property_name = _require_string(item, "property", issues, item_path)
        property_value = _require_string(item, "value", issues, item_path)
        if (
            property_name.startswith("janus.")
            and property_name not in _ALLOWED_JANUS_PROPERTIES
        ):
            issues.append(
                ValidationIssue(
                    f"{item_path}.property",
                    f"unknown JANUS custom property '{property_name}'",
                )
            )
        if property_name and property_value:
            result[property_name] = property_value
    return result


def _read_janus_options(
    value: Any, issues: list[ValidationIssue]
) -> JanusContractOptions | None:
    start = len(issues)
    custom = _read_custom_properties(value, "customProperties", issues)
    compatibility = _require_enum(
        custom,
        "janus.compatibility",
        SUPPORTED_COMPATIBILITY_MODES,
        issues,
        "customProperties",
    )
    enforcement = _require_enum(
        custom,
        "janus.enforcement",
        SUPPORTED_ENFORCEMENT_MODES,
        issues,
        "customProperties",
    )
    drafted_from = _optional_string(
        custom, "janus.draftedFrom", issues, "customProperties"
    )
    if len(issues) != start:
        return None
    return JanusContractOptions(
        compatibility=compatibility,
        enforcement=enforcement,
        drafted_from=drafted_from,
    )
