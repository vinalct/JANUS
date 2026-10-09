"""Plan managed Iceberg property changes; read catalog properties without failing writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from logging import getLogger
from types import MappingProxyType
from typing import Any

from janus.models.config.types import SparkWriteConfig
from janus.writers.identifiers import quote_identifier

logger = getLogger(__name__)

# Other properties belong to the operator or to a separate maintenance policy.
MANAGED_TABLE_PROPERTIES = ("write.distribution-mode", "write.target-file-size-bytes")


@dataclass(frozen=True, slots=True)
class TablePropertyReconciliation:
    """Managed changes and the keys left alone, in deterministic order."""

    to_set: Mapping[str, str]
    unchanged: tuple[str, ...]
    unmanaged: tuple[str, ...]

    @property
    def needs_alter(self) -> bool:
        return bool(self.to_set)


class _FailedPropertyRead(dict[str, str]):
    """An empty mapping distinguishable from a successful read of a legacy table."""

    read_failed = True


def desired_table_properties(declaration: SparkWriteConfig) -> dict[str, str]:
    """Translate a write declaration into the two managed Iceberg properties."""
    return {
        "write.distribution-mode": declaration.distribution_mode,
        "write.target-file-size-bytes": str(declaration.target_file_size_bytes),
    }


def plan_table_property_reconciliation(
    *, current: Mapping[str, str], desired: Mapping[str, str],
) -> TablePropertyReconciliation:
    """Compare rendered values of managed keys only, without catalog or Spark access."""
    to_set = {}
    unchanged = []
    for key in MANAGED_TABLE_PROPERTIES:
        if key not in desired:
            continue  # Omitted declarations never remove an existing property.
        value = str(desired[key])
        if key in current and str(current[key]) == value:
            unchanged.append(key)
        else:
            to_set[key] = value
    return TablePropertyReconciliation(
        to_set=MappingProxyType(to_set),
        unchanged=tuple(unchanged),
        unmanaged=tuple(sorted(set(current) - set(MANAGED_TABLE_PROPERTIES))),
    )


def build_set_tblproperties_sql(
    *, table_identifier: str, properties: Mapping[str, str],
) -> str:
    """Render one ALTER statement; refuse an empty property change."""
    if not properties:
        raise ValueError("SET TBLPROPERTIES requires at least one property")
    return (
        f"ALTER TABLE {quote_identifier(table_identifier)} SET "
        f"{render_tblproperties_clause(properties)}"
    )


def build_show_tblproperties_sql(*, table_identifier: str) -> str:
    """Quote the catalog identifier for the property lookup."""
    return f"SHOW TBLPROPERTIES {quote_identifier(table_identifier)}"


def read_table_properties(spark: Any, table_identifier: str) -> dict[str, str]:
    """Read catalog properties, logging and returning a marked empty map on failure."""
    try:
        rows = spark.sql(build_show_tblproperties_sql(table_identifier=table_identifier)).collect()
        return {row["key"]: str(row["value"]) for row in rows}
    except Exception:
        logger.warning("Could not read table properties for %s", table_identifier, exc_info=True)
        return _FailedPropertyRead()


def render_tblproperties_clause(properties: Mapping[str, str]) -> str:
    """Render sorted, escaped property literals, or an empty string for no properties."""
    if not properties:
        return ""
    entries = ", ".join(
        f"{_quote_literal(key)}={_quote_literal(properties[key])}" for key in sorted(properties)
    )
    return f"TBLPROPERTIES ({entries})"


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"
