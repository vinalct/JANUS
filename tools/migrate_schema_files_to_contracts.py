"""Render one legacy source schema as an ODCS contract migration draft.

The script deliberately leaves business names, descriptions, and classification as
TODO so the migration reviewer must supply those values before activating the
contract. It is a repository tool and is not part of the JANUS runtime package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from janus.models.data_contracts.legacy import contract_from_legacy_schema_file  # noqa: E402
from janus.registry.loader import SourceNotFoundError, load_registry  # noqa: E402


def _property_document(
    prop: Any,
    *,
    source_format: str,
    required_fields: frozenset[str],
    unique_fields: frozenset[str],
) -> dict[str, Any]:
    required = prop.required or prop.name in required_fields
    unique = prop.unique or prop.name in unique_fields
    custom_properties = [
        {"property": "sourceField", "value": prop.name},
        {"property": "sourceFormat", "value": source_format},
    ]
    if prop.source_nullable is not None:
        custom_properties.append(
            {
                "property": "sourceNullable",
                "value": "true" if prop.source_nullable else "false",
            }
        )
    document: dict[str, Any] = {
        "name": prop.name,
        "businessName": "TODO",
        "description": "TODO",
        "logicalType": prop.logical_type,
        "physicalType": prop.physical_type,
        "required": required,
        "unique": unique,
        "primaryKey": prop.primary_key or prop.name in unique_fields,
        "classification": "TODO",
        "customProperties": custom_properties,
    }
    if prop.properties:
        document["properties"] = [
            _property_document(
                child,
                source_format=source_format,
                required_fields=frozenset(),
                unique_fields=frozenset(),
            )
            for child in prop.properties
        ]
    if prop.items is not None:
        document["items"] = _property_document(
            prop.items,
            source_format=source_format,
            required_fields=frozenset(),
            unique_fields=frozenset(),
        )
    if prop.keys is not None and prop.values is not None:
        document["map"] = {
            "key": _property_document(
                prop.keys,
                source_format=source_format,
                required_fields=frozenset(),
                unique_fields=frozenset(),
            ),
            "value": _property_document(
                prop.values,
                source_format=source_format,
                required_fields=frozenset(),
                unique_fields=frozenset(),
            ),
        }
    return document


def _contract_document(source: Any, legacy_path: Path, project_root: Path) -> dict[str, Any]:
    table_name = source.outputs.bronze.table_name
    contract = contract_from_legacy_schema_file(
        legacy_path,
        source_id=source.source_id,
        bronze_table=table_name,
        domain=source.domain,
        project_root=project_root,
    )
    required_fields = frozenset(source.quality.required_fields)
    unique_fields = frozenset(source.quality.unique_fields)

    return {
        "apiVersion": contract.api_version,
        "kind": "DataContract",
        "id": f"{source.domain}.{table_name}",
        "name": source.name,
        "version": "1.0.0",
        "status": "draft",
        "domain": source.domain,
        "description": {"purpose": (source.description or source.name).strip()},
        "team": [{"username": source.owner, "role": "owner"}],
        "tags": list(source.tags),
        "schema": [
            {
                "name": table_name,
                "physicalType": "table",
                "properties": [
                    _property_document(
                        prop,
                        source_format=source.spark.input_format,
                        required_fields=required_fields,
                        unique_fields=unique_fields,
                    )
                    for prop in contract.schema.properties
                ],
            }
        ],
        "customProperties": [
            {"property": "janus.compatibility", "value": "additive"},
            {"property": "janus.enforcement", "value": "lenient"},
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="Source registry identifier")
    parser.add_argument(
        "--out",
        type=Path,
        help="Contract destination (defaults to conf/contracts/<domain>/<table>.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an active or deprecated contract",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    registry = load_registry(project_root)
    try:
        source = registry.get_source(args.source_id, include_disabled=True)
    except SourceNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    schema_path = source.schema.path
    if not schema_path:
        golden_path = (
            project_root
            / "tests"
            / "fixtures"
            / "contracts"
            / "baseline"
            / "spark_schema"
            / f"{source.source_id}.json"
        )
        if not golden_path.is_file():
            raise SystemExit(
                f"Source {source.source_id!r} has no legacy schema.path or M0 schema golden"
            )
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        schema_path = golden.get("schema_path")
        if not isinstance(schema_path, str):
            raise SystemExit(f"M0 schema golden has no schema_path: {golden_path}")

    configured_path = Path(schema_path)
    legacy_path = (
        configured_path if configured_path.is_absolute() else project_root / configured_path
    )
    if not legacy_path.is_file():
        raise SystemExit(f"Legacy schema file does not exist: {legacy_path}")

    table_name = source.outputs.bronze.table_name
    destination = args.out or Path("conf/contracts") / source.domain / f"{table_name}.yaml"
    if not destination.is_absolute():
        destination = project_root / destination
    if destination.is_file() and not args.force:
        existing = yaml.safe_load(destination.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing.get("status") in {"active", "deprecated"}:
            raise SystemExit(
                f"Refusing to replace {existing['status']} contract {destination}; pass --force"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)

    contract = _contract_document(source, legacy_path, project_root)
    destination.write_text(
        yaml.safe_dump(
            contract,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
    )
    try:
        display_path = destination.relative_to(project_root)
    except ValueError:
        display_path = destination
    print(f"{source.source_id}: {schema_path} -> {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
