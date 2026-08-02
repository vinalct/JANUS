"""Identifier quoting and staging-view naming for the Spark/Iceberg writers.

Deliberately free of any Spark import: :mod:`janus.writers.overwrite` builds SQL and must
be able to quote identifiers without importing :mod:`janus.writers.spark`, which imports
*this* module. Both helpers were moved here verbatim from ``spark.py`` — same logic, public
names, no behaviour change.
"""

from __future__ import annotations

from collections.abc import Sequence


def quote_identifier(identifier: str) -> str:
    """Backtick-quote each dot-separated segment of an identifier.

    An embedded backtick is doubled, so a hostile-looking segment cannot terminate the
    quoting and alter the surrounding statement.
    """
    return ".".join(f"`{part.replace('`', '``')}`" for part in identifier.split("."))


def partition_clause(partition_columns: Sequence[str]) -> str:
    """Render ``PARTITIONED BY (...)``, or an empty string when unpartitioned."""
    if not partition_columns:
        return ""
    rendered_columns = ", ".join(quote_identifier(column) for column in partition_columns)
    return f"PARTITIONED BY ({rendered_columns})"
