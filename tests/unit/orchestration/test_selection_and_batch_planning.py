"""FR-2/AC-1/NFR-1/NFR-2: a selection becomes one ordered, fully planned batch — or none.

Everything a batch decides before it runs is decided here: who is in it, in what order,
under which identity, and whether it may run at all. The distinction the suite keeps
returning to is between a problem with *one source*, which is recorded and leaves its
independent peers runnable, and a problem with *the batch*, which stops it while it is
still a document.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import janus
from janus.lineage import compute_config_version
from janus.models import SourceDependencyGraph, SourceDependencyNode
from janus.orchestration import (
    BatchPlanner,
    BatchPlanRequest,
    BatchSelection,
    DisabledUpstreamError,
    EmptySelectionError,
    GraphDriftError,
    PipelineIdentityError,
    SelectionFilterError,
    select_sources,
    source_attempt_run_id,
    validate_pipeline_run_id,
)
from janus.planner import HookCatalog, Planner, PlanningRequest
from janus.registry import load_registry
from janus.strategies.base import SourceHook
from tests.support.orchestration import (
    GRAPH_CASES,
    SELECTION_CASES,
    build_graph_project,
    source_documents,
    write_project,
)

PLANNED_AT = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
PIPELINE_RUN_ID = "order14-batch"
ORCHESTRATION_ROOT = Path(inspect.getfile(janus)).parent / "orchestration"
FORBIDDEN_IMPORT_ROOTS = ("pyspark", "pyiceberg", "dagster", "airflow")


def _request(project_root: Path, **kwargs: Any) -> BatchPlanRequest:
    """A batch request pinned to a fixed pipeline id and planning instant."""
    kwargs.setdefault("pipeline_run_id", PIPELINE_RUN_ID)
    kwargs.setdefault("planned_at", PLANNED_AT)
    return BatchPlanRequest.create(
        environment="local",
        project_root=project_root,
        **kwargs,
    )


def _plan(project_root: Path, planner: BatchPlanner | None = None, **kwargs: Any):
    return (planner or BatchPlanner()).plan(_request(project_root, **kwargs))


def _project(tmp_path: Path, documents: list[dict[str, Any]], *, grouped: bool = True) -> Path:
    """Write one isolated registry per call, so no two cases share a source directory."""
    root = tmp_path / f"registry-{len(list(tmp_path.iterdir()))}"
    root.mkdir()
    return write_project(root, documents, grouped=grouped)


def _documents(name: str) -> list[dict[str, Any]]:
    return copy.deepcopy(source_documents(GRAPH_CASES[name]))


def _document(documents: list[dict[str, Any]], source_id: str) -> dict[str, Any]:
    return next(document for document in documents if document["source_id"] == source_id)


# --------------------------------------------------------------------------------------
# Selection: which sources, and why each of them is in the batch
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_order"),
    [(name, case.expected_order) for name, case in GRAPH_CASES.items() if case.expected_order],
)
def test_every_runnable_shape_selects_its_enabled_closure_in_dependency_order(
    tmp_path, name, expected_order
):
    """Chain, diamond, fan-in, peers and a dormant subgraph each resolve to one order."""
    plan = _plan(build_graph_project(tmp_path, name))

    assert plan.source_ids == expected_order
    assert plan.root_ids == expected_order
    assert plan.included_upstream_ids == ()


@pytest.mark.parametrize(("kind", "values", "roots", "expanded"), SELECTION_CASES)
def test_a_filter_selects_roots_and_brings_their_upstreams_with_them(
    tmp_path, kind, values, roots, expanded
):
    """FR-2: filters choose roots; the graph chooses what those roots cannot run without."""
    project_root = build_graph_project(tmp_path, "chain_and_peer")
    selection = BatchSelection.create(**{f"{kind}s": values})

    if not roots:
        with pytest.raises(EmptySelectionError):
            _plan(project_root, selection=selection)
        return

    plan = _plan(project_root, selection=selection)

    assert plan.root_ids == roots
    assert plan.source_ids == expanded
    assert plan.included_upstream_ids == tuple(sorted(set(expanded) - set(roots)))
    assert all(plan.source(source_id).selected_directly for source_id in roots)


def test_an_upstream_outside_the_requested_domain_still_joins_the_batch(tmp_path):
    """A is in domain 'reference'; selecting 'reporting' still cannot run B without it."""
    plan = _plan(
        build_graph_project(tmp_path, "chain_and_peer"),
        selection=BatchSelection.create(domains=("reporting",)),
    )

    included = plan.source("A")
    assert included.selected_directly is False
    assert plan.source_ids.index("A") < plan.source_ids.index("B")
    assert plan.upstreams_of("B") == ("A",)
    assert plan.downstreams_of("A") == ("B",)


def test_a_shared_producer_is_included_once(tmp_path):
    """The diamond's A feeds B and C; the batch holds one A, not one per consumer."""
    plan = _plan(build_graph_project(tmp_path, "diamond"))

    assert plan.source_ids.count("A") == 1
    assert plan.upstreams_of("D") == ("B", "C")


def test_repeating_or_reordering_a_filter_is_the_same_request():
    """NFR-2: ``--tag a --tag b --tag a`` and ``--tag b --tag a`` are one selection."""
    assert BatchSelection.create(tags=("report", "terminal", "report")) == BatchSelection.create(
        tags=(" terminal ", "report")
    )


def test_selecting_by_tag_and_domain_together_is_refused():
    """Neither 'both filters' nor 'their intersection' is obviously what was meant."""
    with pytest.raises(SelectionFilterError) as exc_info:
        BatchSelection.create(tags=("report",), domains=("reporting",))

    assert "not by both" in str(exc_info.value)


def test_an_empty_selection_is_a_configuration_error_naming_what_exists(tmp_path):
    """A filter that matches nothing asked for work; doing none of it is not success."""
    with pytest.raises(EmptySelectionError) as exc_info:
        _plan(
            build_graph_project(tmp_path, "chain_and_peer"),
            selection=BatchSelection.create(tags=("absent",)),
        )

    message = str(exc_info.value)
    assert "tag in (absent)" in message
    assert "['A', 'B', 'C', 'D']" in message


def test_an_all_disabled_registry_is_refused_rather_than_run_empty(tmp_path):
    documents = _documents("chain")
    for document in documents:
        document["enabled"] = False

    with pytest.raises(EmptySelectionError) as exc_info:
        _plan(_project(tmp_path, documents))

    assert "No source is enabled" in str(exc_info.value)


def test_a_disabled_upstream_is_refused_before_anything_is_planned(tmp_path):
    """The loader already rejects this pairing; selection refuses it on its own terms too."""
    registry = load_registry(build_graph_project(tmp_path, "disabled_subgraph"))
    graph = SourceDependencyGraph(
        nodes=tuple(
            SourceDependencyNode(
                source_id=node.source_id,
                enabled=node.enabled or node.source_id == "B",
                bronze_table=node.bronze_table,
            )
            for node in registry.graph.nodes
        ),
        edges=registry.graph.edges,
    )

    with pytest.raises(DisabledUpstreamError) as exc_info:
        select_sources(graph, registry.sources)

    assert "['A']" in str(exc_info.value)
    assert "never enables a source on your behalf" in str(exc_info.value)


def test_a_dormant_subgraph_is_skipped_without_being_enabled(tmp_path):
    """Disabled A→B stay configured and out of the batch; enabled C runs alone."""
    plan = _plan(build_graph_project(tmp_path, "disabled_subgraph"))

    assert plan.source_ids == ("C",)
    assert plan.edges == ()


# --------------------------------------------------------------------------------------
# Determinism: the same configs and the same request produce the same batch
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("grouped", (True, False))
def test_permuted_configs_yield_the_same_order_roots_and_edges(tmp_path, grouped):
    """NFR-2: discovery order, file layout and YAML order do not reach the plan."""
    documents = _documents("chain_and_peer")
    permutations = [
        documents,
        list(reversed(documents)),
        sorted(documents, key=lambda document: document["source_id"]),
    ]

    summaries = [
        _plan(_project(tmp_path, copy.deepcopy(permutation), grouped=grouped)).to_summary()
        for permutation in permutations
    ]

    for summary in summaries[1:]:
        assert _without_config_evidence(summary) == _without_config_evidence(summaries[0])
    assert summaries[0]["selection"]["source_ids"] == ["A", "B", "C", "D"]


def test_the_same_request_over_the_same_configs_plans_identically(tmp_path):
    """Fixed pipeline id, attempt and planning instant make a batch reproducible."""
    project_root = build_graph_project(tmp_path, "diamond")

    first = _plan(project_root)
    second = _plan(project_root)

    assert first.to_summary() == second.to_summary()
    assert first.selection == second.selection
    assert [source.run_id for source in first.sources] == [
        source.run_id for source in second.sources
    ]


# --------------------------------------------------------------------------------------
# Identity: one pipeline, one run directory per source attempt
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_id", ("a.b", "a-b", "a b", "A.B"))
def test_source_run_ids_are_deterministic(source_id):
    assert source_attempt_run_id(
        pipeline_run_id=PIPELINE_RUN_ID, source_id=source_id, attempt=1
    ) == source_attempt_run_id(pipeline_run_id=PIPELINE_RUN_ID, source_id=source_id, attempt=1)


def test_source_ids_that_normalize_alike_still_get_separate_run_directories():
    """Without the digest, three configured sources would share one run directory."""
    run_ids = {
        source_attempt_run_id(pipeline_run_id=PIPELINE_RUN_ID, source_id=source_id, attempt=1)
        for source_id in ("a.b", "a-b", "a b", "A.B")
    }

    assert len(run_ids) == 4
    assert all(run_id.startswith(f"{PIPELINE_RUN_ID}-a-b-a1-") for run_id in run_ids)


def test_a_retry_keeps_its_correlation_and_still_gets_a_new_run_id():
    first = source_attempt_run_id(pipeline_run_id=PIPELINE_RUN_ID, source_id="A", attempt=1)
    second = source_attempt_run_id(pipeline_run_id=PIPELINE_RUN_ID, source_id="A", attempt=2)

    assert first != second
    assert first.startswith(f"{PIPELINE_RUN_ID}-a-a1-")
    assert second.startswith(f"{PIPELINE_RUN_ID}-a-a2-")
    assert first.rsplit("-", 1)[-1] != second.rsplit("-", 1)[-1]


@pytest.mark.parametrize(
    "pipeline_run_id",
    ("", " ", "..", "../escape", "runs/nightly", "runs\\nightly", "-leading", "night ly", "a" * 97),
)
def test_a_pipeline_id_that_cannot_be_a_path_component_is_refused(pipeline_run_id):
    with pytest.raises(PipelineIdentityError):
        validate_pipeline_run_id(pipeline_run_id)


def test_a_valid_pipeline_id_is_returned_unchanged():
    """An id an operator typed is also the id they will look for afterwards."""
    assert validate_pipeline_run_id("nightly.2026-09-13_full") == "nightly.2026-09-13_full"


def test_an_unusable_pipeline_id_is_refused_before_any_source_is_planned(tmp_path):
    with pytest.raises(PipelineIdentityError):
        _plan(build_graph_project(tmp_path, "chain"), pipeline_run_id="../escape")


def test_a_default_pipeline_id_is_derived_from_the_environment_and_planning_instant(tmp_path):
    plan = _plan(build_graph_project(tmp_path, "chain"), pipeline_run_id=None)

    assert plan.request.pipeline_run_id == "pipeline-local-20260913T120000Z"


def test_every_source_run_id_is_derived_from_the_pipeline_identity(tmp_path):
    plan = _plan(build_graph_project(tmp_path, "chain"), attempt=3)

    for source in plan.sources:
        assert source.attempt == 3
        assert source.run_id == source_attempt_run_id(
            pipeline_run_id=PIPELINE_RUN_ID, source_id=source.source_id, attempt=3
        )
        assert source.require_planned_run().plan.run_context.run_id == source.run_id


# --------------------------------------------------------------------------------------
# The planner seam: one implementation, one snapshot
# --------------------------------------------------------------------------------------


def test_the_batch_plans_every_source_through_the_one_planner_and_one_snapshot(
    tmp_path, monkeypatch
):
    """NFR-1: the batch adds identity and order, never a second way to plan a source."""
    project_root = build_graph_project(tmp_path, "diamond")
    calls: list[tuple[str, object]] = []
    original = Planner.plan

    def recording(self, request, *, registry=None):
        calls.append((request.source_id, registry))
        return original(self, request, registry=registry)

    monkeypatch.setattr(Planner, "plan", recording)
    loads: list[Path] = []

    def counting_loader(root: Path):
        loads.append(root)
        return load_registry(root)

    plan = BatchPlanner(registry_loader=counting_loader).plan(_request(project_root))

    assert [source_id for source_id, _ in calls] == list(plan.source_ids)
    assert len(loads) == 1
    assert all(registry is not None for _, registry in calls)
    assert len({id(registry) for _, registry in calls}) == 1


def test_every_node_is_planned_against_the_one_contract_the_snapshot_holds(tmp_path):
    project_root = build_graph_project(tmp_path, "diamond")
    registry = load_registry(project_root)

    plan = BatchPlanner().plan(_request(project_root), registry=registry)

    carried = [source.planned_run.plan.data_contract for source in plan.sources]
    assert carried == [registry.contract_for(source.source_id) for source in plan.sources]
    assert len({id(contract) for contract in carried}) == 1
    assert all(
        source.planned_run.to_summary()["contract"]["id"] == "example.minimal"
        for source in plan.sources
    )


def test_a_caller_can_hand_the_batch_a_registry_it_already_validated(tmp_path):
    """An adapter that rendered the graph should not have to load the registry twice."""
    project_root = build_graph_project(tmp_path, "chain")
    registry = load_registry(project_root)

    def refuse(root: Path):
        raise AssertionError("the batch reloaded a registry it was handed")

    plan = BatchPlanner(registry_loader=refuse).plan(_request(project_root), registry=registry)

    assert plan.source_ids == ("A", "B")


def test_single_source_planning_keeps_its_call_shape(tmp_path):
    """The seam is an optional argument: one request in, one load, one plan (NFR-1)."""
    project_root = build_graph_project(tmp_path, "chain")
    request = PlanningRequest.create(
        source_id="B",
        environment="local",
        project_root=project_root,
        run_id="fixed-run",
        started_at=PLANNED_AT,
    )

    loaded = Planner().plan(request)
    injected = Planner().plan(request, registry=load_registry(project_root))

    assert inspect.signature(Planner.plan).parameters["registry"].default is None
    assert loaded.to_summary() == injected.to_summary()


def test_a_registry_from_another_project_is_refused(tmp_path):
    """A snapshot must be the one the request's configs were validated with."""
    other = load_registry(build_graph_project(tmp_path, "chain"))
    request = PlanningRequest.create(
        source_id="B",
        environment="local",
        project_root=build_graph_project(tmp_path, "chain"),
    )

    with pytest.raises(Exception, match="was loaded from"):
        Planner().plan(request, registry=other)


# --------------------------------------------------------------------------------------
# Failures: one source's problem versus the batch's
# --------------------------------------------------------------------------------------


class _RaisingHook(SourceHook):
    """A hook whose planning fails, the way a source-local extension can."""

    def on_plan(self, plan):
        raise RuntimeError(f"hook refused to plan {plan.source.source_id}")


class _TableMovingHook(SourceHook):
    """A hook that moves the bronze table its source produces."""

    def on_plan(self, plan):
        return replace(plan, bronze_output=replace(plan.bronze_output, table_name="elsewhere"))


class _UpstreamSwappingHook(SourceHook):
    """A hook that repoints a source's declared upstream behind the graph."""

    def on_plan(self, plan):
        request_inputs = plan.source_config.access.request_inputs
        repointed = replace(request_inputs, upstream_source_id="C")
        return replace(
            plan,
            source_config=replace(
                plan.source_config,
                access=replace(plan.source_config.access, request_inputs=repointed),
            ),
        )


def _hooked_project(tmp_path, name: str, source_id: str, hook_id: str = "test-hook") -> Path:
    documents = _documents(name)
    _document(documents, source_id)["source_hook"] = hook_id
    return _project(tmp_path, documents)


def _planner_with(hook: SourceHook, hook_id: str = "test-hook") -> BatchPlanner:
    """A batch whose planner resolves ``hook_id``, over a registry validated against it."""
    return BatchPlanner(
        planner=Planner(hook_catalog=HookCatalog(hooks=((hook_id, hook),))),
        registry_loader=partial(load_registry, hook_ids=frozenset({hook_id})),
    )


def test_one_sources_planning_failure_stays_that_sources_failure(tmp_path):
    """FR-2/AC-4: an unrelated valid plan is still prepared and still runnable."""
    project_root = _hooked_project(tmp_path, "chain_and_peer", "C")

    plan = _plan(project_root, planner=_planner_with(_RaisingHook()))

    failed = plan.source("C")
    assert failed.is_planned is False
    assert failed.failure is not None
    assert failed.failure.error_type == "RuntimeError"
    assert "hook refused to plan C" in failed.failure.reason
    assert failed.failure.phase == "planning"
    assert [source.source_id for source in plan.planning_failures()] == ["C"]
    assert all(plan.source(source_id).is_planned for source_id in ("A", "B", "D"))


def test_a_failed_node_still_carries_its_identity_and_config_version(tmp_path):
    project_root = _hooked_project(tmp_path, "chain", "B")

    failed = _plan(project_root, planner=_planner_with(_RaisingHook())).source("B")

    assert failed.run_id == source_attempt_run_id(
        pipeline_run_id=PIPELINE_RUN_ID, source_id="B", attempt=1
    )
    assert failed.config_version
    assert failed.upstream_ids == ("A",)
    with pytest.raises(Exception, match="failed to plan"):
        failed.require_planned_run()


@pytest.mark.parametrize(
    ("hook", "source_id", "expected"),
    [
        (_TableMovingHook(), "A", "bronze table 'bronze.a'"),
        (_UpstreamSwappingHook(), "B", "upstream references"),
    ],
)
def test_a_hook_that_changes_the_graph_stops_the_batch_before_execution(
    tmp_path, monkeypatch, hook, source_id, expected
):
    """The order was derived before the hook ran; a changed graph invalidates all of it."""
    from janus.runtime.executor import SourceExecutor

    monkeypatch.setattr(
        SourceExecutor,
        "execute",
        lambda *args, **kwargs: pytest.fail("a drifting batch reached the executor"),
    )
    project_root = _hooked_project(tmp_path, "chain_and_peer", source_id)

    with pytest.raises(GraphDriftError) as exc_info:
        _plan(project_root, planner=_planner_with(hook))

    message = str(exc_info.value)
    assert expected in message
    assert "Nothing was executed." in message


def test_an_ordinary_hook_that_only_annotates_a_plan_is_left_alone(tmp_path):
    """Drift detection guards the graph, not hooks in general."""

    class _NotingHook(SourceHook):
        def on_plan(self, plan):
            return plan.with_note("hook:annotated")

    project_root = _hooked_project(tmp_path, "chain", "B")

    plan = _plan(project_root, planner=_planner_with(_NotingHook()))

    assert "hook:annotated" in plan.source("B").require_planned_run().plan.notes


def test_an_invalid_graph_is_refused_by_the_registry_before_selection(tmp_path):
    """A globally invalid configuration aborts; it is never a per-source failure."""
    from janus.registry import SourceGraphValidationError

    with pytest.raises(SourceGraphValidationError):
        _plan(build_graph_project(tmp_path, "cycle"))


def test_config_versions_are_the_existing_lineage_calculation(tmp_path):
    project_root = build_graph_project(tmp_path, "chain", grouped=False)

    plan = _plan(project_root)

    assert plan.config_versions() == {
        source.source_id: compute_config_version(source.config_path) for source in plan.sources
    }
    assert len(set(plan.config_versions().values())) == 2


def test_sources_sharing_a_grouped_document_share_its_version(tmp_path):
    """The hash pins a config file, and a grouped file is one file."""
    plan = _plan(build_graph_project(tmp_path, "chain", grouped=True))

    assert len(set(plan.config_versions().values())) == 1


def test_every_run_context_carries_the_pipeline_correlation(tmp_path):
    """Correlation travels through the attributes a run context already had."""
    plan = _plan(build_graph_project(tmp_path, "chain"), attempt=2, trigger="dagster")

    for source in plan.sources:
        attributes = source.require_planned_run().plan.run_context.attributes_as_dict()
        assert attributes["pipeline_run_id"] == PIPELINE_RUN_ID
        assert attributes["pipeline_attempt"] == "2"
        assert attributes["trigger"] == "dagster"
        assert attributes["source_id"] == source.source_id


def test_a_caller_cannot_supply_the_attributes_the_batch_derives(tmp_path):
    with pytest.raises(Exception, match="cannot be supplied"):
        _request(build_graph_project(tmp_path, "chain"), attributes={"pipeline_run_id": "mine"})


def test_the_logical_planning_instant_is_what_every_plan_records(tmp_path):
    """Durations are measured later, against a real clock; this one only pins the plan."""
    plan = _plan(build_graph_project(tmp_path, "chain"))

    assert plan.request.planned_at == PLANNED_AT
    assert all(
        source.require_planned_run().plan.run_context.started_at == PLANNED_AT
        for source in plan.sources
    )


def test_the_plan_summary_is_json_serializable_evidence(tmp_path):
    plan = _plan(build_graph_project(tmp_path, "fan_in"))

    summary = json.loads(json.dumps(plan.to_summary()))

    assert summary["pipeline"]["pipeline_run_id"] == PIPELINE_RUN_ID
    assert summary["selection"]["source_ids"] == ["A", "B", "D"]
    assert [(edge["producer_id"], edge["consumer_id"]) for edge in summary["graph"]["edges"]] == [
        ("A", "D"),
        ("B", "D"),
    ]
    assert set(summary["config_versions"]) == {"A", "B", "D"}
    assert all(source["planned"] for source in summary["sources"])


def test_planning_a_batch_starts_no_compute_and_touches_no_network(tmp_path, monkeypatch):
    """AC-1 boundary: a plan is a document describing work that has not happened yet."""

    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("batch planning reached runtime work")

    monkeypatch.setattr("socket.socket.connect", reject)
    monkeypatch.setattr("janus.runtime.spark_lifecycle.SparkSessionProvider.get", reject)
    monkeypatch.setattr("janus.strategies.http.transport.UrllibApiTransport.send", reject)

    assert _plan(build_graph_project(tmp_path, "diamond")).source_ids == ("A", "B", "C", "D")


def test_importing_the_orchestration_package_needs_no_engine_or_orchestrator():
    """NFR-3: batch planning runs on a machine with no Spark and no Dagster installed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(janus.__file__).parents[1]), env.get("PYTHONPATH")) if part
    )
    program = (
        "import sys, janus.orchestration\n"
        f"roots = {FORBIDDEN_IMPORT_ROOTS!r}\n"
        "print(sorted({m for m in sys.modules for r in roots "
        "if m == r or m.startswith(r + '.')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing janus.orchestration loaded {result.stdout.strip()}. The batch planner "
        "is orchestrator- and engine-neutral, and must not need either to be installed."
    )


def test_the_orchestration_package_holds_no_execution_or_scheduler_code():
    """The package plans; it does not run, materialize, or schedule anything."""
    modules = sorted(ORCHESTRATION_ROOT.glob("*.py"))
    assert {path.name for path in modules} >= {
        "identity.py",
        "planning.py",
        "plans.py",
        "selection.py",
    }, f"the sweep did not find the package modules (swept {[p.name for p in modules]})"

    sources = {path.name: path.read_text(encoding="utf-8") for path in modules}
    forbidden_names = ("SparkSession", "BronzeMaterializer", "SourceExecutor", "saveAsTable")
    named = {
        name: sorted(token for token in forbidden_names if token in source)
        for name, source in sources.items()
    }
    assert not {name: found for name, found in named.items() if found}, (
        f"janus/orchestration must stay a planning package: {named}"
    )

    imported = sorted(
        {
            (alias.name if isinstance(node, ast.Import) else node.module) or ""
            for source in sources.values()
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        }
    )
    assert "janus.planner" in imported, f"the import sweep found nothing to check: {imported}"
    forbidden_modules = (
        "janus.runtime",
        "janus.writers",
        "janus.readers",
        "janus.quality",
        "sched",
        "asyncio",
        "threading",
        "crontab",
        "apscheduler",
    )
    assert not [
        module for module in imported if module.startswith(forbidden_modules)
    ], (
        "janus/orchestration imported a runtime, write or scheduling module; planning "
        f"stops before execution and owns no scheduler: {imported}"
    )


def _without_config_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop what a permutation legitimately changes: where a config lives, and its bytes."""
    normalized = copy.deepcopy(summary)
    normalized["pipeline"].pop("project_root")
    normalized.pop("config_versions")
    for source in normalized["sources"]:
        source.pop("config_path")
        source.pop("config_version")
    return normalized
