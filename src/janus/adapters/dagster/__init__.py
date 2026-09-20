"""Optional Dagster adapter.

Install it with ``pip install 'janus[dagster]'``. Importing JANUS core never imports
Dagster; only requesting this package crosses the optional dependency boundary.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("dagster") is None:
    raise ModuleNotFoundError(
        "The JANUS Dagster adapter is optional. Install it with "
        "`pip install 'janus[dagster]'` and retry."
    ) from None

from .collector import collect_dagster_run
from .definitions import (
    DEFAULT_JOB_NAME,
    SOURCE_EXECUTION_POOL,
    DagsterAdapter,
    DagsterAdapterServices,
    build_dagster_adapter,
    build_definitions,
    build_job,
)
from .errors import (
    DagsterAdapterError,
    DagsterConfigurationDriftError,
    DagsterRunCollectionError,
)
from .names import source_op_name

__all__ = [
    "DEFAULT_JOB_NAME",
    "SOURCE_EXECUTION_POOL",
    "DagsterAdapter",
    "DagsterAdapterError",
    "DagsterAdapterServices",
    "DagsterConfigurationDriftError",
    "DagsterRunCollectionError",
    "build_dagster_adapter",
    "build_definitions",
    "build_job",
    "collect_dagster_run",
    "source_op_name",
]
