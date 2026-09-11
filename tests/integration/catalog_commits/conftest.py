"""One catalog, two writers — the one suite where a *shared* catalog is the point."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.spark_sessions import (
    CATALOG_TARGET_FACTORIES,
    CatalogTarget,
    catalog_target_ids,
    require_iceberg_runtime,
    start_session,
)

#: Two cores, not one: the writers must be able to run their Spark jobs at the same time,
#: or the barrier would release them into a queue and the commits would never overlap.
COMMIT_SUITE_MASTER = "local[2]"

CATALOG_CACHE_DISABLED = "cache-enabled"

SUITE_SESSION_CONFIG = {
    "spark.sql.shuffle.partitions": "1",
}


@pytest.fixture(scope="module", params=CATALOG_TARGET_FACTORIES, ids=catalog_target_ids())
def catalog_target(request, tmp_path_factory) -> CatalogTarget:
    """The catalog both writers share, built from the profile by the shared helper."""

    require_iceberg_runtime()

    root: Path = tmp_path_factory.mktemp("janus-catalog-commits")
    target = request.param(root)
    target.prepare()
    return target


@pytest.fixture(scope="module")
def shared_catalog_session(catalog_target: CatalogTarget):
    """One live session over the shared catalog."""

    options = catalog_target.session_options()
    options.update(SUITE_SESSION_CONFIG)
    options[f"spark.sql.catalog.{_catalog_name(options)}.{CATALOG_CACHE_DISABLED}"] = "false"

    session = start_session(
        f"janus-catalog-commits-{catalog_target.id}", options, master=COMMIT_SUITE_MASTER
    )
    yield session
    session.stop()


def _catalog_name(options: dict[str, str]) -> str:
    """The catalog the emitted options actually name, rather than a restated literal."""

    return options["spark.sql.defaultCatalog"]
