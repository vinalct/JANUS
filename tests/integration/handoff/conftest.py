"""Use the shared Spark runtime and the repository's fixture-time skip idiom."""

import pytest

from tests.support.spark_sessions import build_iceberg_session, require_iceberg_runtime


@pytest.fixture(scope="module", autouse=True)
def handoff_runtime():
    require_iceberg_runtime()


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    session = build_iceberg_session(
        "janus-handoff-baseline", tmp_path_factory.mktemp("janus-handoff-catalog")
    )
    yield session
    session.stop()
