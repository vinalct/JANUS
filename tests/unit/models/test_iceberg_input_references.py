from __future__ import annotations

import ast
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

import janus.models.dependencies as dependencies_module
from janus.models import (
    ROOT_REQUEST_INPUT_PATH,
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    IcebergInputReference,
    IcebergRowsRequestInputsConfig,
    RequestInputsConfig,
    iter_iceberg_input_references,
)
from janus.utils.storage import bronze_table_identifier

FORBIDDEN_IMPORT_ROOTS = (
    "pyspark",
    "pyiceberg",
    "dagster",
    "airflow",
    "janus.strategies",
    "janus.registry",
    "janus.runtime",
    "janus.planner",
    "janus.writers",
    "janus.readers",
)

_MODULE_PATH = Path(dependencies_module.__file__).resolve()
_SRC_ROOT = _MODULE_PATH.parents[2]


def _iceberg_leaf(upstream: str, namespace: str, table: str, field: str = "entity_id"):
    return IcebergRowsRequestInputsConfig(
        type="iceberg_rows",
        upstream_source_id=upstream,
        namespace=namespace,
        table_name=table,
        columns={field: "id"},
    )


ORGAOS = _iceberg_leaf("orgaos_source", "bronze__transparencia", "orgaos__siafi", "orgao_codigo")
EMENDAS = _iceberg_leaf(
    "emendas_source", "bronze__transparencia", "emendas_parlamentares__emendas", "emenda_codigo"
)
WINDOW = DateWindowRequestInputsConfig(
    type="date_window", start=date(2025, 1, 1), end=date(2025, 3, 31), step="month"
)


def test_an_atomic_leaf_yields_one_reference_at_the_root_path():
    (reference,) = iter_iceberg_input_references(ORGAOS)

    assert reference == IcebergInputReference(
        upstream_source_id="orgaos_source",
        namespace="bronze__transparencia",
        table_name="orgaos__siafi",
        input_path="access.request_inputs",
    )
    assert ROOT_REQUEST_INPUT_PATH == "access.request_inputs"


def test_a_combined_config_yields_one_reference_per_iceberg_leaf_in_order():
    combined = CombinedRequestInputsConfig(type="combined", inputs=(EMENDAS, WINDOW, ORGAOS))

    references = list(iter_iceberg_input_references(combined))

    assert [(reference.upstream_source_id, reference.input_path) for reference in references] == [
        ("emendas_source", "access.request_inputs.inputs[0]"),
        ("orgaos_source", "access.request_inputs.inputs[2]"),
    ]


def test_repeated_reads_of_one_producer_are_both_kept():
    """Two leaves are two references: the graph needs both paths, not a deduplicated set."""
    combined = CombinedRequestInputsConfig(
        type="combined",
        inputs=(ORGAOS, _iceberg_leaf("orgaos_source", "bronze__transparencia", "orgaos__siafi")),
    )

    references = list(iter_iceberg_input_references(combined))

    assert [reference.upstream_source_id for reference in references] == [
        "orgaos_source",
        "orgaos_source",
    ]
    assert [reference.input_path for reference in references] == [
        "access.request_inputs.inputs[0]",
        "access.request_inputs.inputs[1]",
    ]


@pytest.mark.parametrize(
    "config",
    (
        pytest.param(RequestInputsConfig(type="none"), id="none"),
        pytest.param(WINDOW, id="date_window"),
        pytest.param(
            CombinedRequestInputsConfig(type="combined", inputs=(WINDOW, WINDOW)),
            id="combined_without_iceberg",
        ),
    ),
)
def test_an_input_that_declares_no_producer_yields_nothing(config):
    assert list(iter_iceberg_input_references(config)) == []


def test_a_caller_may_prefix_the_path_of_a_grouped_document():
    combined = CombinedRequestInputsConfig(type="combined", inputs=(WINDOW, ORGAOS))

    (reference,) = iter_iceberg_input_references(
        combined, input_path="sources[3].access.request_inputs"
    )

    assert reference.input_path == "sources[3].access.request_inputs.inputs[1]"


def test_the_table_reference_is_rendered_the_way_a_producer_output_is():
    """One naming rule: the consumer's reference and the producer's identifier must meet."""
    producer_identifier = bronze_table_identifier(
        "data/bronze/transparencia/orgaos__siafi",
        fallback_name="orgaos_source",
        namespace="bronze__transparencia",
        table_name="orgaos__siafi",
    )

    (reference,) = iter_iceberg_input_references(ORGAOS)

    assert reference.table_reference == "bronze__transparencia.orgaos__siafi"
    assert producer_identifier == reference.table_reference


def test_the_declaration_never_overwrites_the_table_reference():
    """A leaf naming a producer whose table it does not read stays inconsistent, not fixed."""
    inconsistent = _iceberg_leaf("orgaos_source", "bronze__transparencia", "some_other_table")

    (reference,) = iter_iceberg_input_references(inconsistent)

    assert reference.upstream_source_id == "orgaos_source"
    assert reference.table_reference == "bronze__transparencia.some_other_table"


def test_a_reference_is_frozen():
    (reference,) = iter_iceberg_input_references(ORGAOS)

    with pytest.raises(AttributeError):
        reference.upstream_source_id = "somebody_else"  # type: ignore[misc]


def test_the_module_imports_nothing_from_above_its_layer():
    """Static half of the guardrail: the import is visible before anything runs."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))

    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    offenders = sorted(
        name
        for name in imported
        for root in FORBIDDEN_IMPORT_ROOTS
        if name == root or name.startswith(f"{root}.")
    )

    assert not offenders, (
        f"janus.models.dependencies imports {offenders}. The dependency declarations are "
        "read below the registry boundary — resolving them against a registry, a strategy "
        "or a live table belongs to the layer above, which may import this one."
    )


def test_importing_the_module_pulls_in_no_engine_or_strategy():
    """Runtime half: a transitive import would be just as costly, and just as wrong."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_SRC_ROOT), env.get("PYTHONPATH")) if part
    )
    program = (
        "import sys, janus.models.dependencies\n"
        f"roots = {FORBIDDEN_IMPORT_ROOTS!r}\n"
        "print(sorted({m for m in sys.modules for r in roots "
        "if m == r or m.startswith(r + '.')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing janus.models.dependencies loaded {result.stdout.strip()}. It must stay "
        "importable by a planner or an adapter without starting an engine."
    )
