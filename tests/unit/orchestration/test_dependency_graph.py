"""FR-1/FR-4/AC-2: loading a registry returns a validated source graph, or refuses.

Every case here goes through ``load_registry``. The graph is not a thing a batch runner
computes later out of the same configs — if it were, an invalid graph would be discovered
by whoever happened to look, which is how "run A before B" stayed tribal knowledge.
"""

from __future__ import annotations

import copy
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import janus
from janus.models import (
    SourceDependencyEdge,
    SourceDependencyNode,
    iter_iceberg_input_references,
)
from janus.models.source_config import SourceConfigValidationError
from janus.planner import Planner, PlanningRequest
from janus.registry import (
    SourceGraphValidationError,
    SourceRegistry,
    load_app_config,
    load_registry,
)
from janus.registry.dependencies import (
    build_source_dependency_graph,
    producer_table_identifier,
)
from janus.strategies.api.request_inputs import _iceberg_table_identifier
from janus.utils.storage import bronze_table_identifier
from tests.support.contracts import (
    KEYED_CONTRACT_PATH,
    PRODUCER_CONTRACT_PATH,
    write_keyed_contract,
    write_producer_contract,
)
from tests.support.orchestration import (
    GRAPH_CASES,
    GraphCase,
    SourceSpec,
    source_documents,
    write_project,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MIXED_PROBLEMS = GraphCase(
    (
        SourceSpec("E", table_name="Shared-Target"),
        SourceSpec("F", table_name="shared target"),
        SourceSpec("G", ("nobody",)),
    )
)

MIXED_SHAPES = GraphCase(
    (
        SourceSpec("E", table_name="Shared-Target"),
        SourceSpec("F", table_name="shared target"),
        SourceSpec("A", ("D",)),
        SourceSpec("D", ("A",)),
    )
)

FORBIDDEN_IMPORT_ROOTS = ("pyspark", "pyiceberg", "dagster", "airflow")


def _project(tmp_path: Path, documents: list[dict[str, Any]], *, grouped: bool = True) -> Path:
    """Write one isolated registry per call, so no two cases share a source directory."""
    root = tmp_path / f"registry-{len(list(tmp_path.iterdir()))}"
    root.mkdir()
    return write_project(root, documents, grouped=grouped)


def _documents(name: str) -> list[dict[str, Any]]:
    """A deep copy of one shared graph fixture, safe to mutate into a negative case."""
    return copy.deepcopy(source_documents(GRAPH_CASES[name]))


def _document(documents: list[dict[str, Any]], source_id: str) -> dict[str, Any]:
    return next(document for document in documents if document["source_id"] == source_id)


def _leaf(documents: list[dict[str, Any]], source_id: str) -> dict[str, Any]:
    """The one Iceberg leaf of a consumer built by the shared fixture."""
    return _document(documents, source_id)["access"]["request_inputs"]


def _messages(error: SourceGraphValidationError) -> list[str]:
    return [issue.render() for issue in error.issues]


def _load_documents(tmp_path: Path, documents: list[dict[str, Any]], **kwargs: Any):
    return load_registry(_project(tmp_path, documents, **kwargs))


def _expect_graph_error(
    tmp_path: Path, documents: list[dict[str, Any]], **kwargs: Any
) -> SourceGraphValidationError:
    with pytest.raises(SourceGraphValidationError) as exc_info:
        _load_documents(tmp_path, documents, **kwargs)
    return exc_info.value


# --------------------------------------------------------------------------------------
# Shapes that must resolve
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_edges", "expected_order"),
    [
        ("independent", (), ("A", "C")),
        ("chain", (("A", "B"),), ("A", "B")),
        ("chain_and_peer", (("A", "B"), ("B", "D")), ("A", "B", "C", "D")),
        ("fan_in", (("A", "D"), ("B", "D")), ("A", "B", "D")),
        ("diamond", (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")), ("A", "B", "C", "D")),
    ],
)
def test_every_declared_shape_resolves_to_its_edges_and_a_stable_order(
    tmp_path, name, expected_edges, expected_order
):
    graph = _load_documents(tmp_path, _documents(name)).graph

    assert graph.source_ids == expected_order
    assert tuple((edge.producer_id, edge.consumer_id) for edge in graph.edges) == expected_edges
    assert graph.topological_order() == expected_order


def test_a_fan_out_producer_releases_both_of_its_consumers(tmp_path):
    """One producer, two independent consumers — the shape a batch runner parallelises."""
    documents = _documents("diamond")
    documents.remove(_document(documents, "D"))

    graph = _load_documents(tmp_path, documents).graph

    assert graph.downstreams("A") == ("B", "C")
    assert graph.upstreams("B") == ("A",)
    assert graph.upstreams("C") == ("A",)


def test_a_catalog_consumer_produces_the_same_graph_as_an_api_consumer(tmp_path):
    """AC: both families reach the graph through one code path, not two."""
    catalog = _load_documents(tmp_path, _documents("catalog_consumer")).graph
    api = _load_documents(tmp_path, _documents("chain")).graph

    assert catalog.edges == api.edges
    assert catalog.nodes == api.nodes


def test_a_combined_input_contributes_one_edge_per_declared_leaf(tmp_path):
    """Fan-in is expressed as ``combined``; each sub-input keeps its own nested path."""
    graph = _load_documents(tmp_path, _documents("fan_in")).graph

    assert graph.edges == (
        SourceDependencyEdge("A", "D", "bronze.a", ("access.request_inputs.inputs[0]",)),
        SourceDependencyEdge("B", "D", "bronze.b", ("access.request_inputs.inputs[1]",)),
    )


def test_reading_one_producer_twice_is_one_dependency_with_two_explanations(tmp_path):
    """Scheduling asks "after whom"; a diagnostic asks "because of which leaf"."""
    documents = _documents("chain")
    leaf = _leaf(documents, "B")
    _document(documents, "B")["access"]["request_inputs"] = {
        "type": "combined",
        "inputs": [leaf, {**leaf, "columns": {"a_code": "code"}}],
    }
    _document(documents, "B")["access"]["parameter_bindings"] = {
        "a_id": {"from": "request_input.a_id"},
        "a_code": {"from": "request_input.a_code"},
    }
    _document(documents, "A")["schema"] = {"contract": PRODUCER_CONTRACT_PATH}
    root = _project(tmp_path, documents)
    write_producer_contract(root, ("code",))

    graph = load_registry(root).graph

    assert graph.edges == (
        SourceDependencyEdge(
            "A",
            "B",
            "bronze.a",
            ("access.request_inputs.inputs[0]", "access.request_inputs.inputs[1]"),
        ),
    )
    assert graph.upstreams("B") == ("A",)


def test_a_producer_with_an_inferred_table_name_still_satisfies_a_dependency(tmp_path):
    """Nothing in the fixture declares ``table_name``; the identity comes from the writer."""
    graph = _load_documents(tmp_path, _documents("chain")).graph

    assert graph.node("A") == SourceDependencyNode("A", enabled=True, bronze_table="bronze.a")
    assert graph.edges[0].table == "bronze.a"


def test_a_source_that_writes_no_iceberg_table_is_a_node_without_a_table(tmp_path):
    """A consumer-only source still schedules; it just cannot be depended upon."""
    documents = _documents("chain")
    _document(documents, "B")["outputs"]["bronze"]["format"] = "parquet"

    graph = _load_documents(tmp_path, documents).graph

    assert graph.node("B").bronze_table is None
    assert graph.upstreams("B") == ("A",)


def test_a_dormant_pair_is_configuration_rather_than_an_error(tmp_path):
    """A disabled consumer of a disabled producer is a relationship, not a broken run."""
    graph = _load_documents(tmp_path, _documents("disabled_subgraph")).graph

    assert graph.edges == (
        SourceDependencyEdge("A", "B", "bronze.a", ("access.request_inputs",)),
    )
    assert [node.enabled for node in graph.nodes] == [False, False, True]


def _share_table(documents: list[dict[str, Any]], *source_ids: str) -> list[dict[str, Any]]:
    """Point several sources at one bronze table and have them declare each other."""
    target = _document(documents, source_ids[0])["outputs"]["bronze"]
    for source_id in source_ids:
        bronze = _document(documents, source_id)["outputs"]["bronze"]
        bronze["path"] = target["path"]
        bronze["namespace"] = "bronze__shared"
        bronze["table_name"] = "one_dataset"
        bronze["shared_with"] = [peer for peer in source_ids if peer != source_id]
    return documents


def test_a_full_refresh_and_an_incremental_pipeline_may_write_one_table(tmp_path):
    """Two operations over one dataset is a pipeline design, not a registry mistake."""
    documents = _share_table(_documents("independent"), "A", "C")
    _document(documents, "A")["extraction"]["mode"] = "full_refresh"
    _document(documents, "C")["extraction"]["mode"] = "incremental"
    _document(documents, "C")["extraction"]["checkpoint_field"] = "updated_at"
    _document(documents, "C")["extraction"]["checkpoint_strategy"] = "max_value"
    _document(documents, "C")["schema"] = {"contract": KEYED_CONTRACT_PATH}
    root = _project(tmp_path, documents)
    write_keyed_contract(root)

    graph = load_registry(root).graph

    assert [node.bronze_table for node in graph.nodes] == [
        "bronze__shared.one_dataset",
        "bronze__shared.one_dataset",
    ]
    assert graph.edges == ()


def test_a_consumer_of_a_shared_table_waits_for_the_pipeline_it_declares(tmp_path):
    """The co-writers are two; the consumer's own declaration says which one releases it."""
    documents = _share_table(_documents("chain_and_peer"), "A", "C")
    _leaf(documents, "B")["namespace"] = "bronze__shared"
    _leaf(documents, "B")["table_name"] = "one_dataset"

    graph = _load_documents(tmp_path, documents).graph

    assert graph.upstreams("B") == ("A",)
    assert graph.downstreams("C") == ()
    assert graph.edges[0].table == "bronze__shared.one_dataset"


def test_a_one_sided_declaration_names_the_side_that_forgot(tmp_path):
    """Mutual is the point: a half-declared pair is indistinguishable from an accident."""
    documents = _share_table(_documents("independent"), "A", "C")
    _document(documents, "C")["outputs"]["bronze"]["shared_with"] = []

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert message.startswith("C (")
    assert "an undeclared collision is an ambiguous producer target" in message


def test_declaring_a_co_writer_of_a_table_nobody_else_writes_is_rejected(tmp_path):
    """A stale declaration would otherwise survive a rename in silence."""
    documents = _documents("independent")
    _document(documents, "A")["outputs"]["bronze"]["shared_with"] = ["C"]

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert message.endswith(
        ".outputs.bronze.shared_with: declares co-writers ['C'] for 'bronze.a', but no "
        "other configured source writes that table; remove the declaration or fix the "
        "table identity"
    )


def test_a_declaration_that_names_the_wrong_co_writer_is_rejected(tmp_path):
    """Naming somebody is not the same as naming everybody who writes the table."""
    documents = _share_table(_documents("independent"), "A", "C")
    _document(documents, "A")["outputs"]["bronze"]["shared_with"] = ["D"]

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert message.startswith("A (")
    assert message.endswith(
        ".outputs.bronze.shared_with: declares ['D'] for 'bronze__shared.one_dataset', but "
        "its actual co-writers are ['C']; the declaration must name every other source "
        "writing the table, and only those"
    )


def test_the_checked_in_gastos_cartoes_pipelines_share_one_table():
    """The real pair: one dataset, one table, a rebuild job and a delta job over it."""
    registry = load_registry(PROJECT_ROOT)
    full_refresh = registry.get_source(
        "transparencia__gastos_cartoes__cartoes__full_refresh", include_disabled=True
    )
    incremental = registry.get_source(
        "transparencia__gastos_cartoes__cartoes__incremental", include_disabled=True
    )

    assert full_refresh.extraction.mode == "full_refresh"
    assert incremental.extraction.mode == "incremental"
    assert (
        registry.graph.node(full_refresh.source_id).bronze_table
        == registry.graph.node(incremental.source_id).bronze_table
        == "bronze__transparencia.gastos_cartoes__cartoes"
    )
    assert full_refresh.outputs.bronze.shared_with == (incremental.source_id,)
    assert incremental.outputs.bronze.shared_with == (full_refresh.source_id,)


def test_the_checked_in_registry_exposes_its_declared_edges():
    """The real configs are a graph, not a pile: four declared edges across two families."""
    graph = load_registry(PROJECT_ROOT).graph

    assert tuple((edge.producer_id, edge.consumer_id) for edge in graph.edges) == (
        (
            "dados_abertos_catalog__conjunto_dados__full_refresh",
            "dados_abertos_catalog__conjunto_dados_details__full_refresh",
        ),
        (
            "transparencia__emendas_parlamentares__emendas__full_refresh",
            "transparencia__emendas_parlamentares__documentos__full_refresh",
        ),
        (
            "transparencia__orgaos__siafi__full_refresh",
            "transparencia__contratos__contratos__full_refresh",
        ),
        (
            "transparencia__orgaos__siafi__full_refresh",
            "transparencia__licitacoes__licitacoes__full_refresh",
        ),
    )
    assert graph.source_ids == tuple(
        sorted(source.source_id for source in load_registry(PROJECT_ROOT).sources)
    )
    assert graph.topological_order().index(
        "transparencia__orgaos__siafi__full_refresh"
    ) < graph.topological_order().index("transparencia__licitacoes__licitacoes__full_refresh")


def test_the_matched_identifier_is_the_one_the_reader_will_ask_for(tmp_path):
    """A second identifier regex would let the graph and the extraction read two tables."""
    registry = _load_documents(tmp_path, _documents("chain"))
    consumer = registry.get_source("B")

    (reference,) = iter_iceberg_input_references(consumer.access.request_inputs)

    assert _iceberg_table_identifier(consumer.access.request_inputs) == reference.table_reference
    assert registry.graph.edges[0].table == reference.table_reference


def test_producer_identity_is_the_writer_s_identity_for_every_checked_in_source():
    """AC: the graph names the table that will actually be committed, defaults included."""
    for source in load_registry(PROJECT_ROOT).sources:
        bronze = source.outputs.bronze
        expected = (
            bronze_table_identifier(
                bronze.path,
                fallback_name=source.source_id,
                namespace=bronze.namespace,
                table_name=bronze.table_name,
            )
            if bronze.format == "iceberg"
            else None
        )
        assert producer_table_identifier(source) == expected


# --------------------------------------------------------------------------------------
# Shapes that must be refused
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_fragment"),
    [
        ("self_cycle", "cycle"),
        ("cycle", "cycle"),
        ("missing_producer", "missing"),
        ("disabled_producer", "disabled"),
        ("duplicate_producer", "ambiguous"),
    ],
)
def test_every_invalid_shape_is_named_before_anything_runs(tmp_path, name, expected_fragment):
    error = _expect_graph_error(tmp_path, _documents(name))

    assert expected_fragment in str(error).lower()
    assert GRAPH_CASES[name].expected_error == expected_fragment


def test_a_self_cycle_reports_the_source_reading_its_own_table(tmp_path):
    error = _expect_graph_error(tmp_path, _documents("self_cycle"))

    assert _messages(error) == [
        "A → A: is a source dependency cycle and can never be scheduled "
        "(A.access.request_inputs reads bronze.a)"
    ]


def test_a_longer_cycle_reports_a_reproducible_path_and_its_leaves(tmp_path):
    error = _expect_graph_error(tmp_path, _documents("cycle"))

    assert _messages(error) == [
        "A → B → D → A: is a source dependency cycle and can never be scheduled "
        "(B.access.request_inputs reads bronze.a; D.access.request_inputs reads bronze.b; "
        "A.access.request_inputs reads bronze.d)"
    ]


def test_an_absent_declared_id_names_the_declaration_not_the_table(tmp_path):
    error = _expect_graph_error(tmp_path, _documents("missing_producer"))

    (message,) = _messages(error)
    assert message.startswith("B (")
    assert message.endswith(
        ").access.request_inputs: declares upstream_source_id 'missing', which is missing "
        "from the registry: no configured source has that id"
    )


def test_a_declared_producer_that_writes_another_table_is_rejected(tmp_path):
    """The declaration never overrides the table reference — it is checked against it."""
    documents = _documents("diamond")
    _leaf(documents, "B")["upstream_source_id"] = "C"

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert _entry(message) == [
        _entry_for(documents, "B"),
        _entry_for(documents, "C"),
        _entry_for(documents, "A"),
    ]
    assert ".access.request_inputs: reads 'bronze.a', but its declared upstream source 'C'" in (
        message
    )
    assert "produces 'bronze.c'; the referenced table is produced by A (" in message


def test_a_reference_to_a_table_nobody_produces_says_so(tmp_path):
    documents = _documents("chain")
    _leaf(documents, "B")["table_name"] = "somewhere_else"

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert message.endswith(
        "produces 'bronze.a'; no configured source produces the referenced table"
    )


def test_a_non_iceberg_producer_cannot_satisfy_an_iceberg_dependency(tmp_path):
    """A Parquet directory is not a table; the warehouse must not be asked to disagree."""
    documents = _documents("chain")
    _document(documents, "A")["outputs"]["bronze"]["format"] = "parquet"

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert message.endswith(
        ".access.request_inputs: declares upstream source 'A', whose bronze output format "
        "is 'parquet'; an iceberg_rows input can only depend on a source that writes an "
        "iceberg bronze table"
    )


@pytest.mark.parametrize("field_name", ["namespace", "table_name"])
def test_a_qualified_table_reference_is_refused_rather_than_silently_matched(
    tmp_path, field_name
):
    """Cross-catalog syntax has no contract yet, so it must not become a false edge."""
    documents = _documents("chain")
    _leaf(documents, "B")[field_name] = f"other_catalog.{_leaf(documents, 'B')[field_name]}"

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert "is not a supported table reference" in message
    assert "must each be one unqualified identifier" in message


def test_an_enabled_consumer_may_not_depend_on_a_disabled_producer(tmp_path):
    documents = _documents("disabled_producer")

    error = _expect_graph_error(tmp_path, documents)

    (message,) = _messages(error)
    assert _entry(message) == [_entry_for(documents, "B"), _entry_for(documents, "A")]
    assert message.endswith("is disabled; a run never enables an upstream on its behalf")
    assert "is enabled, but its upstream source 'A' (" in message


def test_an_undeclared_collision_is_ambiguous_even_when_nothing_reads_it(tmp_path):
    """``Shared-Target`` and ``shared target`` are one physical table after sanitization."""
    error = _expect_graph_error(tmp_path, _documents("duplicate_producer"))

    first, second = _messages(error)
    assert first.startswith("A (")
    assert second.startswith("B (")
    for message, other in ((first, "B"), (second, "A")):
        assert ".outputs.bronze.shared_with: writes 'bronze.shared_target', which " in message
        assert f"{other} (" in message
        assert "an undeclared collision is an ambiguous producer target" in message
        assert "each must then name the other in shared_with" in message


def test_a_grouped_entry_is_named_in_the_diagnostic(tmp_path):
    """Six sources in one file: the file alone would not say which entry to edit."""
    grouped = _expect_graph_error(tmp_path, _documents("missing_producer"), grouped=True)
    flat = _expect_graph_error(tmp_path, _documents("missing_producer"), grouped=False)

    assert "sources.yaml:sources[0]" in _messages(grouped)[0]
    assert "sources.yaml:sources" not in _messages(flat)[0]
    assert "00.yaml" in _messages(flat)[0]


def test_independent_graph_problems_are_reported_together(tmp_path):
    """One load, every actionable problem — the same contract config validation has."""
    error = _expect_graph_error(tmp_path, source_documents(MIXED_PROBLEMS))

    assert len(error.issues) == 3
    assert "ambiguous producer target" in _messages(error)[0]
    assert "ambiguous producer target" in _messages(error)[1]
    assert "missing from the registry" in _messages(error)[2]


# --------------------------------------------------------------------------------------
# Determinism, construction, and what must not have run
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["diamond", "chain_and_peer", "fan_in"])
def test_discovery_order_cannot_reach_the_graph(tmp_path, name):
    """NFR-2: the same configs in any order are the same graph, edges and ordering."""
    documents = _documents(name)
    grouped = _load_documents(tmp_path, documents).graph
    shuffled = _load_documents(tmp_path, list(reversed(documents)), grouped=False).graph

    assert grouped.nodes == shuffled.nodes
    assert grouped.edges == shuffled.edges
    assert grouped.topological_order() == shuffled.topological_order()


def test_discovery_order_cannot_reach_the_diagnostics(tmp_path):
    """A reordered registry must not produce a reordered — or differently worded — error."""
    documents = source_documents(MIXED_SHAPES)

    forward = _expect_graph_error(tmp_path, documents)
    backward = _expect_graph_error(tmp_path, list(reversed(documents)), grouped=False)

    assert _normalise(_messages(forward)) == _normalise(_messages(backward))


def test_a_directly_constructed_registry_is_held_to_the_same_invariants(tmp_path):
    """Batch use must not be a second door into an unvalidated registry."""
    project_root = _project(tmp_path, _documents("chain"))
    registry = load_registry(project_root)
    broken = tuple(
        source for source in registry.sources if source.source_id != "A"
    )

    assert "graph" not in inspect.signature(SourceRegistry).parameters
    with pytest.raises(SourceGraphValidationError):
        SourceRegistry(
            project_root=project_root,
            app_config=load_app_config(project_root),
            sources=broken,
        )


def test_a_registry_built_without_locations_validates_identically(tmp_path):
    """Provenance sharpens a message; it is never what makes the graph valid."""
    registry = _load_documents(tmp_path, _documents("diamond"))

    assert build_source_dependency_graph(registry.sources) == registry.graph


def test_the_graph_error_arrives_before_any_planner_engine_or_network_work(
    tmp_path, monkeypatch
):
    """AC-2: an unrunnable graph costs a parse, never a session, a request or a catalog."""

    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid registry reached runtime work")

    monkeypatch.setattr("socket.socket.connect", reject)
    monkeypatch.setattr("janus.runtime.spark_lifecycle.SparkSessionProvider.get", reject)
    monkeypatch.setattr("janus.planner.core.StrategyCatalog.resolve", reject)
    monkeypatch.setattr("janus.strategies.http.transport.UrllibApiTransport.send", reject)
    project_root = _project(tmp_path, _documents("cycle"))

    with pytest.raises(SourceGraphValidationError):
        Planner().plan(
            PlanningRequest.create(
                source_id="B",
                environment="local",
                project_root=project_root,
            )
        )


def test_a_leftover_warehouse_table_cannot_stand_in_for_a_configured_upstream(tmp_path):
    """Managed upstreams only: producer identity is configured, never discovered on disk."""
    documents = _documents("chain")
    documents.remove(_document(documents, "A"))
    project_root = _project(tmp_path, documents)
    (project_root / "data" / "bronze" / "bronze" / "a" / "metadata").mkdir(parents=True)

    with pytest.raises(SourceGraphValidationError) as exc_info:
        load_registry(project_root)

    assert "missing from the registry" in str(exc_info.value)


def test_a_config_error_still_precedes_the_graph(tmp_path):
    """Graph validation reads typed configs, so it cannot be what reports a broken one."""
    documents = _documents("chain")
    _document(documents, "A")["strategy_variant"] = "not_a_variant"

    with pytest.raises(SourceConfigValidationError):
        _load_documents(tmp_path, documents)


def test_graph_validation_runs_without_spark_iceberg_or_an_orchestrator():
    """The registry is importable on a machine that has none of them installed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(janus.__file__).parents[1]), env.get("PYTHONPATH")) if part
    )
    program = (
        "import sys, janus.registry.dependencies\n"
        f"roots = {FORBIDDEN_IMPORT_ROOTS!r}\n"
        "print(sorted({m for m in sys.modules for r in roots "
        "if m == r or m.startswith(r + '.')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing janus.registry.dependencies loaded {result.stdout.strip()}. Graph "
        "validation runs before any engine exists, and must not need one to be installed."
    )


def _entry_for(documents: list[dict[str, Any]], source_id: str) -> str:
    """The grouped entry the loader attaches to ``source_id`` in a fixture project."""
    return f"sources[{[document['source_id'] for document in documents].index(source_id)}]"


def _entry(message: str) -> list[str]:
    """Every ``sources[index]`` provenance a diagnostic names, in order."""
    return [f"sources[{part.split(']')[0]}]" for part in message.split("sources[")[1:]]


def _normalise(messages: list[str]) -> list[str]:
    """Replace each parenthesised file location, which is where the source happens to live.

    What must not move is the set, the wording and the order of the diagnostics. Which
    file a source sits in is the one thing a reordered registry legitimately changes.
    """
    return [re.sub(r"\([^()]*/[^()]*\)", "(<location>)", message) for message in messages]
