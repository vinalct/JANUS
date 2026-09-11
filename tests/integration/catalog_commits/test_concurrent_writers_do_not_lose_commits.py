"""AC-5: two writers committing to one table at the same time lose nothing."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

NAMESPACE = "bronze_catalog_commits"
SCHEMA = "id BIGINT, writer STRING"

SEED_ROWS = [(0, "seed")]
FIRST_WRITER_ROWS = [(1, "a"), (2, "a"), (3, "a")]
SECOND_WRITER_ROWS = [(4, "b"), (5, "b"), (6, "b")]

#: Iceberg's own retry knob, pinned so the dependency is stated rather than inherited.
COMMIT_RETRIES_PROPERTY = "commit.retry.num-retries"
COMMIT_RETRIES = 4

#: Generous: it bounds a hang, it does not pace the race. The barrier releases both writers
#: the instant the second arrives.
BARRIER_TIMEOUT_SECONDS = 120
RESULT_TIMEOUT_SECONDS = 300


@pytest.fixture
def seeded_table(shared_catalog_session, request):
    """A table with one committed row, owned by this test alone."""

    session = shared_catalog_session
    table = f"{NAMESPACE}.{_table_name(request)}"

    session.sql(f"CREATE NAMESPACE IF NOT EXISTS {NAMESPACE}")
    session.sql(f"DROP TABLE IF EXISTS {table}")
    (
        session.createDataFrame(SEED_ROWS, SCHEMA)
        .writeTo(table)
        .tableProperty(COMMIT_RETRIES_PROPERTY, str(COMMIT_RETRIES))
        .create()
    )
    yield table
    session.sql(f"DROP TABLE IF EXISTS {table}")


def test_two_racing_appends_both_land(shared_catalog_session, seeded_table):
    """The AC-5 claim itself: row count is the sum, and no row went missing."""

    session, table = shared_catalog_session, seeded_table
    before = _snapshots(session, table)

    race_appends(session, table)

    assert _ids(session, table) == _expected_ids()
    assert session.table(table).count() == len(_expected_ids())
    assert len(_snapshots(session, table)) == len(before) + 2


def test_neither_racing_commit_raised(shared_catalog_session, seeded_table):
    """Iceberg's retryable conflict path must absorb the race without surfacing it."""

    race_appends(shared_catalog_session, seeded_table)


def test_the_losing_commit_is_rebased_onto_the_winner_never_over_it(
    shared_catalog_session, seeded_table
):
    """The snapshot log must stay a single chain."""

    session, table = shared_catalog_session, seeded_table

    race_appends(session, table)

    snapshots = _snapshots(session, table)
    assert _chain_length(snapshots) == len(snapshots), (
        "the snapshot log forked: a commit was overwritten rather than rebased onto the "
        f"one that beat it — {snapshots}"
    )


def test_a_stale_read_cannot_silently_overwrite_the_commit_that_beat_it(
    shared_catalog_session, seeded_table
):
    """The lost-update variant: both writers read the table *first*, then both commit."""

    session, table = shared_catalog_session, seeded_table

    observed = race_appends(session, table, read_before_commit=True)

    assert observed == [len(SEED_ROWS), len(SEED_ROWS)], (
        "both writers were supposed to read the same pre-race state; they saw "
        f"{observed} — the barrier is not between the read and the commit"
    )
    assert _ids(session, table) == _expected_ids()
    assert _chain_length(_snapshots(session, table)) == len(_snapshots(session, table))


def race_appends(session, table: str, *, read_before_commit: bool = False) -> list[int | None]:
    """Append both row sets from two threads, synchronized so the commits overlap."""

    row_sets = (FIRST_WRITER_ROWS, SECOND_WRITER_ROWS)
    barrier = threading.Barrier(len(row_sets), timeout=BARRIER_TIMEOUT_SECONDS)

    def append(rows: list[tuple[int, str]]) -> int | None:
        frame = session.createDataFrame(rows, SCHEMA)
        observed = session.table(table).count() if read_before_commit else None
        barrier.wait()
        frame.writeTo(table).append()
        return observed

    with ThreadPoolExecutor(max_workers=len(row_sets)) as pool:
        futures = [pool.submit(append, rows) for rows in row_sets]
        return [future.result(timeout=RESULT_TIMEOUT_SECONDS) for future in futures]


def _expected_ids() -> set[int]:
    return {row[0] for row in (*SEED_ROWS, *FIRST_WRITER_ROWS, *SECOND_WRITER_ROWS)}


def _ids(session, table: str) -> set[int]:
    return {row["id"] for row in session.table(table).select("id").collect()}


def _snapshots(session, table: str) -> list[tuple[int, int | None]]:
    return [
        (row["snapshot_id"], row["parent_id"])
        for row in session.sql(
            f"SELECT snapshot_id, parent_id FROM {table}.snapshots"
        ).collect()
    ]


def _chain_length(snapshots: list[tuple[int, int | None]]) -> int:
    """How many snapshots a single walk from the root reaches."""

    children = {parent: snapshot for snapshot, parent in snapshots if parent is not None}
    roots = [snapshot for snapshot, parent in snapshots if parent is None]
    if len(roots) != 1:
        return 0

    chain = [roots[0]]
    while chain[-1] in children:
        chain.append(children[chain[-1]])
    return len(chain)


def _table_name(request) -> str:
    """A table per test, named after it, so a failure says which race left what behind."""

    return re.sub(r"[^a-z0-9_]+", "_", request.node.name.lower()).strip("_")


def test_the_fork_detector_accepts_a_single_chain():
    """Seed → first commit → second commit rebased onto it: three snapshots, one walk."""

    assert _chain_length([(10, None), (11, 10), (12, 11)]) == 3


def test_the_fork_detector_flags_a_fork():
    """Two commits claiming the same parent — a lost update, after the fact."""

    assert _chain_length([(10, None), (11, 10), (12, 10)]) < 3


def test_the_fork_detector_flags_a_history_with_no_root():
    """A log whose every snapshot has a parent is not a history this test can vouch for."""

    assert _chain_length([(11, 10), (12, 11)]) == 0


def test_the_fork_detector_flags_two_roots():
    """Two independent histories in one table is the other shape of a lost commit."""

    assert _chain_length([(10, None), (20, None), (11, 10)]) == 0


def test_the_row_sets_the_writers_append_are_disjoint():
    """The whole count assertion is meaningless if the two writers could collide."""

    first = {row[0] for row in FIRST_WRITER_ROWS}
    second = {row[0] for row in SECOND_WRITER_ROWS}

    assert first and second
    assert not first & second
    assert len(_expected_ids()) == len(SEED_ROWS) + len(first) + len(second)
