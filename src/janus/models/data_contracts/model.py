"""Frozen, engine-neutral value objects for JANUS data contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_CONTRACT_STATUSES = frozenset({"active", "deprecated", "draft"})
SUPPORTED_COMPATIBILITY_MODES = frozenset({"additive", "backward", "frozen"})
SUPPORTED_ENFORCEMENT_MODES = frozenset({"lenient", "strict"})

_SEMVER_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_semver(value: str) -> bool:
    return _SEMVER_PATTERN.fullmatch(value) is not None


def _validate_written_name(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")


def _validate_unique_property_names(
    properties: tuple[ContractProperty, ...], field_name: str
) -> None:
    names = tuple(prop.name for prop in properties)
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} names must be unique")


@dataclass(frozen=True, slots=True)
class ContractProperty:
    """One bronze column, including recursive struct or array declarations."""

    name: str
    physical_type: str
    logical_type: str
    business_name: str | None = None
    description: str | None = None
    required: bool = False
    unique: bool = False
    primary_key: bool = False
    classification: str | None = None
    source_field: str | None = None
    source_format: str | None = None
    properties: tuple[ContractProperty, ...] = ()
    items: ContractProperty | None = None

    def __post_init__(self) -> None:
        _validate_written_name(self.name, "property name")
        if not self.physical_type.strip():
            raise ValueError("physical_type must not be empty")
        if not self.logical_type.strip():
            raise ValueError("logical_type must not be empty")
        _validate_unique_property_names(self.properties, self.name)

        if self.physical_type == "struct":
            if not self.properties:
                raise ValueError("struct properties must contain at least one property")
            if self.items is not None:
                raise ValueError("struct properties must not declare items")
        elif self.physical_type == "array":
            if self.items is None:
                raise ValueError("array properties must declare items")
            if self.properties:
                raise ValueError("array properties must not declare nested properties")
        elif self.properties or self.items is not None:
            raise ValueError(
                f"{self.physical_type} properties must not declare properties or items"
            )


@dataclass(frozen=True, slots=True)
class ContractSchema:
    """The single bronze table described by a JANUS contract."""

    name: str
    physical_type: str
    properties: tuple[ContractProperty, ...]

    def __post_init__(self) -> None:
        _validate_written_name(self.name, "schema name")
        if self.physical_type != "table":
            raise ValueError("schema physical_type must be 'table'")
        if not self.properties:
            raise ValueError("schema must contain at least one property")
        _validate_unique_property_names(self.properties, self.name)


@dataclass(frozen=True, slots=True)
class JanusContractOptions:
    """JANUS-specific options carried in ODCS custom properties."""

    compatibility: str
    enforcement: str
    drafted_from: str | None = None

    def __post_init__(self) -> None:
        if self.compatibility not in SUPPORTED_COMPATIBILITY_MODES:
            allowed = ", ".join(sorted(SUPPORTED_COMPATIBILITY_MODES))
            raise ValueError(f"compatibility must be one of: {allowed}")
        if self.enforcement not in SUPPORTED_ENFORCEMENT_MODES:
            allowed = ", ".join(sorted(SUPPORTED_ENFORCEMENT_MODES))
            raise ValueError(f"enforcement must be one of: {allowed}")
        if self.drafted_from is not None and not self.drafted_from.strip():
            raise ValueError("drafted_from must not be empty")


@dataclass(frozen=True, slots=True)
class DataContract:
    """A validated, immutable declaration for one bronze table."""

    contract_path: Path
    api_version: str
    id: str
    name: str
    version: str
    status: str
    domain: str
    purpose: str
    owners: tuple[str, ...]
    tags: tuple[str, ...]
    schema: ContractSchema
    janus: JanusContractOptions
    schema_version: str

    def __post_init__(self) -> None:
        if not _is_semver(self.version):
            raise ValueError("version must be MAJOR.MINOR.PATCH using digits only")
        if self.status not in SUPPORTED_CONTRACT_STATUSES:
            allowed = ", ".join(sorted(SUPPORTED_CONTRACT_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        if not self.owners:
            raise ValueError("owners must contain at least one owner")
        if _SHA256_PATTERN.fullmatch(self.schema_version) is None:
            raise ValueError("schema_version must be 64 lowercase hexadecimal characters")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.schema.properties)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.schema.properties if prop.required)

    @property
    def primary_key(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.schema.properties if prop.primary_key)
