from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from janus.models.config.issues import ValidationIssue
from janus.models.data_contracts.errors import ContractValidationError
from janus.models.data_contracts.loader import PINNED_ODCS_API_VERSION, compute_schema_version
from janus.models.data_contracts.model import (
    ContractProperty,
    ContractSchema,
    DataContract,
    JanusContractOptions,
)
from janus.models.data_contracts.vocabulary import (
    VocabularyError,
    contract_properties_from_spark_json,
    odcs_logical_type_for,
)


LEGACY_CONTRACT_STATUS = "draft"
LEGACY_CONTRACT_VERSION = "0.0.0"
LEGACY_CONTRACT_OWNER = "unassigned"
LEGACY_CONTRACT_PURPOSE = "Converted from a legacy schema file; not reviewed."
LEGACY_COMPATIBILITY = "additive"
LEGACY_ENFORCEMENT = "lenient"
LEGACY_COLUMN_TYPE = "string"

_ROOT_PATH = "<root>"
_COLUMN_KEYS = ("fields", "columns")


def contract_from_legacy_schema_file(
    path: Path,
    *,
    source_id: str,
    bronze_table: str,
    domain: str,
    project_root: Path | None = None,
) -> DataContract:
    """Read one legacy schema file and return the contract it describes.

    ``bronze_table`` names the schema, so the two entries that share one legacy file get
    one contract each, differing only in the table they describe while carrying the same
    ``schema_version`` — the digest is of the file, not of the entry.
    """
    issues: list[ValidationIssue] = []
    payload = _read_json(path, issues)
    properties = () if issues else _read_properties(payload, issues)
    if not properties and not issues:
        issues.append(ValidationIssue(_ROOT_PATH, "must declare at least one field"))
    if issues:
        raise ContractValidationError(path, issues)

    return DataContract(
        contract_path=path,
        api_version=PINNED_ODCS_API_VERSION,
        id=legacy_contract_id(path, project_root),
        name=f"Legacy schema for {source_id}",
        version=LEGACY_CONTRACT_VERSION,
        status=LEGACY_CONTRACT_STATUS,
        domain=domain,
        purpose=LEGACY_CONTRACT_PURPOSE,
        owners=(LEGACY_CONTRACT_OWNER,),
        tags=(),
        schema=ContractSchema(
            name=bronze_table, physical_type="table", properties=properties
        ),
        janus=JanusContractOptions(
            compatibility=LEGACY_COMPATIBILITY, enforcement=LEGACY_ENFORCEMENT
        ),
        schema_version=compute_schema_version(path),
    )


def legacy_contract_id(path: Path, project_root: Path | None = None) -> str:
    """Identify a converted file by where it sits, so lineage can name the source of it."""
    if project_root is not None:
        try:
            return f"legacy:{path.resolve().relative_to(project_root.resolve()).as_posix()}"
        except ValueError:
            pass
    return f"legacy:{path.as_posix()}"


def _read_json(path: Path, issues: list[ValidationIssue]) -> Any:
    try:
        return json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        issues.append(ValidationIssue(_ROOT_PATH, f"must be valid JSON: {exc.msg}"))
        return None


def _read_properties(
    payload: Any, issues: list[ValidationIssue]
) -> tuple[ContractProperty, ...]:
    """Dispatch on the two shapes the legacy loader accepted, in its own order."""
    if _is_spark_struct(payload):
        return _properties_from_spark_struct(payload, issues)
    return _properties_from_field_names(payload, issues)


def _is_spark_struct(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("type") == "struct"
        and isinstance(payload.get("fields"), list)
    )


def _properties_from_spark_struct(
    payload: Mapping[str, Any], issues: list[ValidationIssue]
) -> tuple[ContractProperty, ...]:
    """Preserve a ``StructType`` verbatim: same types, same nesting, same nullability."""
    try:
        return contract_properties_from_spark_json(payload)
    except VocabularyError as exc:
        issues.append(ValidationIssue("fields", str(exc)))
    except ValueError as exc:
        issues.append(ValidationIssue("fields", str(exc)))
    return ()


def _properties_from_field_names(
    payload: Any, issues: list[ValidationIssue]
) -> tuple[ContractProperty, ...]:
    """A columns-only file declares names and nothing else, so every column is a string."""
    located = _locate_field_entries(payload, issues)
    if located is None:
        return ()

    prefix, entries = located
    if not isinstance(entries, Sequence) or isinstance(entries, str | bytes | bytearray):
        issues.append(ValidationIssue(prefix, "fields/columns must be an array"))
        return ()

    names = _field_names(prefix, entries, issues)
    return tuple(
        ContractProperty(
            name=name,
            physical_type=LEGACY_COLUMN_TYPE,
            logical_type=odcs_logical_type_for(LEGACY_COLUMN_TYPE),
            required=False,
        )
        for name in names
    )


def _locate_field_entries(
    payload: Any, issues: list[ValidationIssue]
) -> tuple[str, Any] | None:
    """Find the declared entries in any of the four shapes the legacy loader accepted."""
    if isinstance(payload, list):
        return (_ROOT_PATH, payload)
    if isinstance(payload, Mapping):
        for key in _COLUMN_KEYS:
            if key in payload:
                return (key, payload[key])
        nested = payload.get("schema")
        if isinstance(nested, Mapping):
            for key in _COLUMN_KEYS:
                if key in nested:
                    return (f"schema.{key}", nested[key])

    issues.append(
        ValidationIssue(
            _ROOT_PATH,
            "must be a JSON array of field names or a mapping with fields/columns",
        )
    )
    return None


def _field_names(
    prefix: str, entries: Sequence[Any], issues: list[ValidationIssue]
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        entry_path = f"{prefix}[{index}]"
        name = _field_name(entry, entry_path, issues)
        if name is None:
            continue
        if not name:
            issues.append(ValidationIssue(f"{entry_path}.name", "must not be empty"))
            continue
        if name in seen:
            issues.append(
                ValidationIssue(f"{entry_path}.name", f"must be unique; duplicate {name!r}")
            )
            continue
        names.append(name)
        seen.add(name)
    return tuple(names)


def _field_name(entry: Any, entry_path: str, issues: list[ValidationIssue]) -> str | None:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
        return str(entry["name"]).strip()
    issues.append(
        ValidationIssue(
            entry_path, "entries must be strings or objects containing a 'name' field"
        )
    )
    return None


__all__ = [
    "LEGACY_COLUMN_TYPE",
    "LEGACY_COMPATIBILITY",
    "LEGACY_CONTRACT_OWNER",
    "LEGACY_CONTRACT_PURPOSE",
    "LEGACY_CONTRACT_STATUS",
    "LEGACY_CONTRACT_VERSION",
    "LEGACY_ENFORCEMENT",
    "contract_from_legacy_schema_file",
    "legacy_contract_id",
]
