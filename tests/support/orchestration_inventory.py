"""Read-only YAML inventory, using the writer's identifier derivation."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from janus.models import SourceConfig
from janus.utils.storage import bronze_table_identifier


def iceberg_leaves(value: dict[str, Any], path: str = "access.request_inputs"):
    if value.get("type") == "iceberg_rows":
        yield path, value
    for index, child in enumerate(value.get("inputs", [])):
        yield from iceberg_leaves(child, f"{path}.inputs[{index}]")


def inventory(project_root: Path, source_root: str = "conf/sources") -> dict[str, Any]:
    outputs = []
    references = []
    producers = defaultdict(list)
    for path in sorted((project_root / source_root).rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for index, payload in enumerate(document.get("sources", [document])):
            prefix = f"sources[{index}]" if "sources" in document else "$"
            source = SourceConfig.from_mapping(payload, path)
            bronze = source.outputs.bronze
            identifier = (
                bronze_table_identifier(
                    bronze.path,
                    fallback_name=source.source_id,
                    namespace=bronze.namespace,
                    table_name=bronze.table_name,
                )
                if bronze.format == "iceberg"
                else None
            )
            row = {
                "source_id": source.source_id,
                "enabled": source.enabled,
                "yaml": str(path.relative_to(project_root)),
                "entry": prefix,
                "bronze_path": bronze.path,
                "format": bronze.format,
                "namespace": bronze.namespace,
                "table_name": bronze.table_name,
                "physical_table": identifier,
            }
            outputs.append(row)
            if identifier:
                producers[identifier].append(row)
            for leaf_path, leaf in iceberg_leaves(
                payload.get("access", {}).get("request_inputs", {})
            ):
                references.append(
                    {
                        "consumer": source.source_id,
                        "consumer_enabled": source.enabled,
                        "yaml": row["yaml"],
                        "leaf": f"{prefix}.{leaf_path}",
                        "table": f"{leaf['namespace']}.{leaf['table_name']}",
                    }
                )
    for reference in references:
        reference["producers"] = [
            {"source_id": row["source_id"], "enabled": row["enabled"]}
            for row in producers[reference["table"]]
        ]
    return {
        "outputs": outputs,
        "references": references,
        "duplicates": {
            table: [row["source_id"] for row in rows]
            for table, rows in sorted(producers.items())
            if len(rows) > 1
        },
    }


if __name__ == "__main__":
    print(json.dumps(inventory(Path(sys.argv[1]).resolve()), indent=2, sort_keys=True))
