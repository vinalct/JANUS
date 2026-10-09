"""Pure mappings and SQL rendering for the Iceberg properties JANUS manages."""

from __future__ import annotations

from collections.abc import Mapping

from janus.models.config.types import SparkWriteConfig

# Other properties belong to the operator or to a separate maintenance policy.
MANAGED_TABLE_PROPERTIES = ("write.distribution-mode", "write.target-file-size-bytes")


def desired_table_properties(declaration: SparkWriteConfig) -> dict[str, str]:
    """Translate a write declaration into the two managed Iceberg properties."""
    return {
        "write.distribution-mode": declaration.distribution_mode,
        "write.target-file-size-bytes": str(declaration.target_file_size_bytes),
    }


def render_tblproperties_clause(properties: Mapping[str, str]) -> str:
    """Render sorted, escaped property literals, or an empty string for no properties."""
    if not properties:
        return ""
    entries = ", ".join(
        f"{_quote_literal(key)}={_quote_literal(properties[key])}" for key in sorted(properties)
    )
    return f"TBLPROPERTIES ({entries})"


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
