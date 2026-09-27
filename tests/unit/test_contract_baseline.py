from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from tests.support import contract_baseline


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="process timezone control is unavailable")
def test_row_digest_timestamp_is_independent_of_the_driver_timezone():
    project_root = Path("/tmp/project")
    canonical_timestamps = []

    for timezone, local_hour in (("America/Sao_Paulo", 9), ("UTC", 12)):
        with _process_timezone(timezone):
            row = {
                "ingestion_timestamp": datetime(2026, 9, 16, local_hour),
                "artifact_path": "/tmp/project/data/raw/page.json",
            }
            portable = contract_baseline._portable_row(row, project_root)
            canonical_timestamps.append(portable["ingestion_timestamp"])
            assert portable["artifact_path"] == "<PROJECT>/data/raw/page.json"

    assert canonical_timestamps == [
        datetime(2026, 9, 16, 12),
        datetime(2026, 9, 16, 12),
    ]


@contextmanager
def _process_timezone(value: str):
    previous = os.environ.get("TZ")
    os.environ["TZ"] = value
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()
