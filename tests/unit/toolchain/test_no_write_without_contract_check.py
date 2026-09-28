"""No bronze write without the pre-write check: a package-scoped sweep with proven detectors."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

import janus

pytestmark = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PACKAGE_ROOT = Path(inspect.getfile(janus)).parent

WRITER_PACKAGE = "writers"
BUILDER_MODULE_ANCHORS = {"writers/overwrite.py", "writers/schema_ddl.py", "writers/spark.py"}
CALLER_PACKAGES = ("runtime", "scripts", "strategies", "adapters", "cli")
CALLER_FILES = ("main.py",)
MATERIALIZER = "runtime/materialize.py"

NO_DIRECT_BRONZE_WRITE = {
    "runtime/executor.py",
    "runtime/batch.py",
    "scripts/raw_to_bronze.py",
    "adapters/dagster/runtime.py",
}

STATEMENT_PREFIXES = ("INSERT", "MERGE", "REPLACE TABLE", "CREATE TABLE", "ALTER TABLE")
PRE_WRITE_PASS = "run_pre_write_pass"
STRUCTURAL_CHECK = "check_frame_against_contract"
STRUCTURAL_CHECK_CALLERS = {"quality/pre_write.py", "quality/validators.py"}

SELECT_STAR_ALLOWLIST = {
    "build_create_table_as_select_sql": (
        "A first write creates the table from the staged frame: there is no target yet whose "
        "column order a projection could follow, and the order CTAS takes is the one every "
        "later by-name insert resolves against."
    ),
    "build_replace_table_as_select_sql": (
        "A declared breaking change (a contract MAJOR bump on a full refresh, D-10) re-creates "
        "the table from the frame; like CTAS, it defines the target order rather than "
        "resolving against one."
    ),
}

INLINE_SQL_ALLOWLIST = {
    "CREATE NAMESPACE IF NOT EXISTS": (
        "The writer's idempotent namespace bootstrap: it creates no table and "
        "writes no row, so there is nothing for the contract check to guard."
    ),
}

Site = tuple[str, int] 


# ── the detectors: declared here ─────────────────────────


def _statement_literals(source: str) -> tuple[tuple[str, int, str], ...]:
    raise NotImplementedError("the statement detector")


def _positional_inserts(source: str) -> tuple[Site, ...]:
    raise NotImplementedError("the statement detector")


def _inline_sql_calls(source: str) -> tuple[Site, ...]:
    raise NotImplementedError("the statement detector")


def _bronze_writes(source: str) -> tuple[Site, ...]:
    raise NotImplementedError("the caller detector")


def _unguarded_bronze_writes(source: str) -> tuple[Site, ...]:
    raise NotImplementedError("the caller detector")


def _calls_to(source: str, name: str) -> tuple[int, ...]:
    raise NotImplementedError("the single-caller detector")


# ── the sweep ────────────────────────────────────────────────────────────────


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _modules(packages: Iterable[str], files: Iterable[str] = ()) -> dict[str, str]:
    paths = [PACKAGE_ROOT / name for name in files]
    for package in packages:
        paths.extend((PACKAGE_ROOT / package).rglob("*.py"))
    return {
        _relative(path): path.read_text(encoding="utf-8")
        for path in sorted(set(paths))
        if "__pycache__" not in path.parts
    }


def _writer_modules() -> dict[str, str]:
    return _modules((WRITER_PACKAGE,))


def _caller_modules() -> dict[str, str]:
    return _modules(CALLER_PACKAGES, CALLER_FILES)


def _findings(
    modules: dict[str, str], detector: Callable[[str], tuple]
) -> dict[str, tuple]:
    return {name: found for name, source in modules.items() if (found := detector(source))}


def test_the_sweep_covers_its_packages_and_finds_every_anchor():
    writers = set(_writer_modules())
    callers = set(_caller_modules())

    assert writers >= BUILDER_MODULE_ANCHORS
    assert callers >= {MATERIALIZER, "main.py", *NO_DIRECT_BRONZE_WRITE}
    assert len(callers) > len(NO_DIRECT_BRONZE_WRITE) + 2


def test_bronze_statements_live_only_in_the_builder_modules():
    assert set(_findings(_writer_modules(), _statement_literals)) == BUILDER_MODULE_ANCHORS


def test_no_positional_insert_remains_in_the_writers():
    assert _findings(_writer_modules(), _positional_inserts) == {}


def test_every_spark_sql_call_in_the_writers_passes_a_builder():
    assert _findings(_writer_modules(), _inline_sql_calls) == {}


def test_every_bronze_write_follows_the_pre_write_pass():
    assert _findings(_caller_modules(), _unguarded_bronze_writes) == {}


def test_only_the_materializer_writes_bronze():
    writers = set(_findings(_caller_modules(), _bronze_writes))

    assert writers == {MATERIALIZER}
    assert not writers & NO_DIRECT_BRONZE_WRITE


def test_the_pre_write_pass_has_exactly_one_caller():
    callers = {
        name
        for name, source in _modules(("",)).items()
        if _calls_to(source, PRE_WRITE_PASS)
    }

    assert callers == {MATERIALIZER}


def test_the_structural_check_has_exactly_the_pass_and_the_report_as_callers():
    callers = {
        name
        for name, source in _modules(("",)).items()
        if _calls_to(source, STRUCTURAL_CHECK)
    }

    assert callers == STRUCTURAL_CHECK_CALLERS


def test_every_allowlist_entry_is_reasoned_and_still_needed():
    selecting_star = {
        function
        for source in _writer_modules().values()
        for function, _, text in _statement_literals(source)
        if "SELECT *" in text
    }

    assert all(reason.strip() for reason in SELECT_STAR_ALLOWLIST.values())
    assert all(reason.strip() for reason in INLINE_SQL_ALLOWLIST.values())
    assert selecting_star == set(SELECT_STAR_ALLOWLIST), "stale or missing allowlist entry"


# ── detector meta-tests: each detector must detect before it is trusted ──────

POSITIONAL_LITERAL = """
def append(spark):
    spark.sql("INSERT INTO t SELECT * FROM v")
"""
POSITIONAL_F_STRING = """
def append(spark, table, view):
    spark.sql(f"INSERT INTO {table} SELECT * FROM {view}")
"""
ALLOWLISTED_CREATION = """
def build_create_table_as_select_sql(table, view):
    return f"CREATE TABLE {table} USING iceberg AS SELECT * FROM {view}"
"""
BUILDER_CALL = """
def overwrite(spark, table, view, p):
    spark.sql(build_insert_overwrite_sql(table_identifier=table, source_view=view, projection=p))
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {table}")
"""
INLINE_MERGE = """
def upsert(spark):
    spark.sql("MERGE INTO t USING v ON t.id = v.id WHEN MATCHED THEN UPDATE SET *")
"""
UNGUARDED_WRITE = """
def materialize(writer, frame, plan):
    writer.write(frame, plan, "bronze", intent=None)
"""
UNGUARDED_KEYWORD_WRITE = """
def materialize(writer, frame, plan):
    writer.write(frame, plan, zone="bronze")
"""
CHECK_AFTER_WRITE = """
def materialize(writer, frame, contract, plan):
    writer.write(frame, plan, "bronze")
    run_pre_write_pass(frame, contract, enforcement="strict")
"""
GUARDED_WRITE = """
def materialize(writer, frame, contract, plan):
    checked, evidence = run_pre_write_pass(frame, contract, enforcement="strict")
    writer.write(checked, plan, "bronze")
"""
RAW_WRITE = """
def persist(writer, frame, plan):
    writer.write(frame, plan, "raw")
"""
MENTIONS_NOT_CALLS = """
def run_pre_write_pass(frame):
    return "run_pre_write_pass"
"""
CALLS = """
def materialize(frame, contract):
    run_pre_write_pass(frame, contract)
    pre_write.run_pre_write_pass(frame, contract)
"""


def test_the_statement_detector_flags_a_positional_insert_literal():
    assert _positional_inserts(POSITIONAL_LITERAL) == (("append", 3),)


def test_the_statement_detector_flags_a_positional_insert_f_string():
    assert _positional_inserts(POSITIONAL_F_STRING) == (("append", 3),)


def test_the_statement_detector_spares_the_allowlisted_creation_builder():
    assert _statement_literals(ALLOWLISTED_CREATION)
    assert _positional_inserts(ALLOWLISTED_CREATION) == ()


def test_the_statement_detector_accepts_builder_calls_and_the_namespace_bootstrap():
    assert _inline_sql_calls(BUILDER_CALL) == ()


def test_the_statement_detector_flags_an_inline_statement():
    assert _inline_sql_calls(INLINE_MERGE) == (("upsert", 3),)


@pytest.mark.parametrize(
    "source",
    [UNGUARDED_WRITE, UNGUARDED_KEYWORD_WRITE, CHECK_AFTER_WRITE],
    ids=["positional", "keyword", "check-after"],
)
def test_the_caller_detector_flags_a_bronze_write_without_the_pass_before_it(source):
    assert len(_unguarded_bronze_writes(source)) == 1


def test_the_caller_detector_accepts_a_guarded_bronze_write():
    assert _bronze_writes(GUARDED_WRITE) == (("materialize", 4),)
    assert _unguarded_bronze_writes(GUARDED_WRITE) == ()


def test_the_caller_detector_ignores_a_raw_write():
    assert _bronze_writes(RAW_WRITE) == ()
    assert _unguarded_bronze_writes(RAW_WRITE) == ()


def test_the_single_caller_detector_counts_calls_not_mentions():
    assert _calls_to(CALLS, PRE_WRITE_PASS) == (3, 4)
    assert _calls_to(MENTIONS_NOT_CALLS, PRE_WRITE_PASS) == ()
