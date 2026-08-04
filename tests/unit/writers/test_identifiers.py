"""Spark-free coverage of the shared identifier quoting and staging-view helpers.

These functions are the only defence between a configured identifier and a rendered SQL
string, and every bronze statement builder routes through them, so they are pinned exactly:
per-segment quoting for dotted identifiers, a doubled backtick for an embedded one, and an
empty (not absent) partition clause for an unpartitioned table.

The temp-view builder is pinned to the same standard for a sharper reason than injection:
``source_id`` carries no character class, and :func:`quote_identifier` splits on dots, so an
unsanitized name would be rendered as a namespace-qualified reference to a view registered
under the whole unquoted string.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

from janus.writers import spark as spark_writer
from janus.writers.identifiers import (
    BRONZE_TEMP_VIEW_PREFIX,
    build_bronze_temp_view_name,
    partition_clause,
    quote_identifier,
)

SAFE_TEMP_VIEW_SHAPE = re.compile(r"^janus_bronze_[a-z0-9_]+_[0-9a-f]{32}$")

# Every one of these either breaks the reference or produces a name `dropTempView`
# cannot match, leaking the view for the session's lifetime.
HOSTILE_SOURCE_IDS = (
    "x`; DROP TABLE t --",
    'a" OR 1=1',
    "a b",
    "../etc",
    "órgão",
    "UPPER",
    "1st_source",
    "   ",
    "!!!",
)


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


def test_temp_view_name_matches_the_safe_shape():
    name = build_bronze_temp_view_name("events_api")

    assert SAFE_TEMP_VIEW_SHAPE.match(name)
    assert name.startswith(f"{BRONZE_TEMP_VIEW_PREFIX}_events_api_")


def test_temp_view_name_has_no_dot_for_a_dotted_source_id():
    name = build_bronze_temp_view_name("receita.federal.cnpj")

    assert "." not in name
    # The whole name must resolve as one temp view, not as `namespace`.`table`.
    assert quote_identifier(name) == f"`{name}`"


@pytest.mark.parametrize("source_id", HOSTILE_SOURCE_IDS)
def test_temp_view_name_neutralizes_hostile_ids(source_id):
    name = build_bronze_temp_view_name(source_id)

    assert SAFE_TEMP_VIEW_SHAPE.match(name), name


def test_temp_view_name_is_unique_per_call():
    names = {build_bronze_temp_view_name("events_api") for _ in range(100)}

    # The uuid4 suffix, not the id, is what keeps concurrent writes from colliding.
    assert len(names) == 100


def test_empty_sanitization_falls_back_to_source_stem():
    name = build_bronze_temp_view_name("!!!")

    # Not `janus_bronze__<hex>`: an all-punctuation id sanitizes to "" and would
    # otherwise double the separator.
    assert name.startswith("janus_bronze_source_")
    assert SAFE_TEMP_VIEW_SHAPE.match(name)


@pytest.mark.parametrize("source_id", ("events_api", "receita.federal.cnpj", *HOSTILE_SOURCE_IDS))
def test_quote_identifier_round_trips_the_temp_view_name(source_id):
    quoted = quote_identifier(build_bronze_temp_view_name(source_id))

    assert quoted.count("`") == 2


def test_writer_never_builds_a_raw_temp_view_name():
    """Guardrail: the staging-view name has exactly one construction site.

    Rebuilding it inline in ``spark.py`` is how the dotted-``source_id`` failure got in
    originally, and an f-string there would bypass the sanitizer silently.
    """
    tree = ast.parse(inspect.getsource(spark_writer))
    offending = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and BRONZE_TEMP_VIEW_PREFIX in node.value
    ]

    assert not offending, (
        f"{spark_writer.__name__} builds a temp-view name inline at line(s) "
        f"{offending}; use build_bronze_temp_view_name() instead "
        "(TASK-05 — temp-view identifier sanitization)"
    )
