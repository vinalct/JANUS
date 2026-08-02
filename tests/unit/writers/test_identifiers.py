"""Spark-free coverage of the shared identifier quoting helpers.

These two functions are the only defence between a configured identifier and a rendered SQL
string, and every bronze statement builder routes through them, so they are pinned exactly:
per-segment quoting for dotted identifiers, a doubled backtick for an embedded one, and an
empty (not absent) partition clause for an unpartitioned table.
"""

from __future__ import annotations

from janus.writers.identifiers import partition_clause, quote_identifier


def test_quote_identifier_wraps_a_bare_name():
    assert quote_identifier("events") == "`events`"


def test_quote_identifier_quotes_each_dotted_segment_separately():
    assert quote_identifier("bronze_test.events") == "`bronze_test`.`events`"
    assert quote_identifier("catalog.bronze.events") == "`catalog`.`bronze`.`events`"


def test_quote_identifier_doubles_an_embedded_backtick():
    quoted = quote_identifier("weird`name")

    assert quoted == "`weird``name`"
    # The escaped backtick must not terminate the quoted identifier.
    assert "`weird`name`" not in quoted


def test_quote_identifier_neutralizes_a_hostile_looking_segment():
    quoted = quote_identifier("t` ; DROP TABLE x; --")

    assert quoted == "`t`` ; DROP TABLE x; --`"
    assert quoted.startswith("`") and quoted.endswith("`")


def test_partition_clause_renders_every_column_quoted():
    assert partition_clause(("ingestion_date",)) == "PARTITIONED BY (`ingestion_date`)"
    assert (
        partition_clause(("ingestion_date", "janus_source_id"))
        == "PARTITIONED BY (`ingestion_date`, `janus_source_id`)"
    )


def test_partition_clause_is_empty_for_an_unpartitioned_table():
    assert partition_clause(()) == ""
    assert partition_clause([]) == ""


def test_partition_clause_escapes_a_backtick_in_a_column_name():
    assert partition_clause(("weird`col",)) == "PARTITIONED BY (`weird``col`)"
