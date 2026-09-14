"""Measured pre-migration behavior, not acceptance tests for an unimplemented runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from janus.models import SourceConfigValidationError
from janus.registry import load_registry
from janus.utils.storage import bronze_table_identifier
from tests.support.orchestration import (
    GRAPH_CASES,
    build_graph_project,
    source_documents,
)
from tests.support.orchestration_capture import CAPTURE_CASES, capture_case
from tests.support.orchestration_inventory import iceberg_leaves, inventory

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDENS = PROJECT_ROOT / "tests" / "fixtures" / "orchestration" / "baseline"


@pytest.mark.parametrize("case", CAPTURE_CASES)
def test_single_source_compatibility_capture(tmp_path, monkeypatch, case):
    def reject_network(*args, **kwargs):
        raise AssertionError("baseline capture attempted network I/O")

    monkeypatch.setattr("socket.socket.connect", reject_network)
    captured = capture_case(tmp_path / case, case)
    assert captured == json.loads((GOLDENS / f"{case}.json").read_text())
    summary = captured["summary"]
    assert summary["executed_run"]["status"] == ("failed" if case == "failed" else "succeeded")
    if case in {"empty", "failed"}:
        assert "spark_session" not in summary
    if case == "replay":
        assert captured["requests"] == []


def test_checked_in_source_inventory_matches_implementation_base():
    assert inventory(PROJECT_ROOT) == json.loads((GOLDENS / "source-inventory.json").read_text())


@pytest.mark.parametrize("name", GRAPH_CASES)
def test_fixture_documents_declare_every_leaf_and_match_physical_producers(name):
    case = GRAPH_CASES[name]
    documents = source_documents(case)
    tables = {
        source["source_id"]: bronze_table_identifier(
            source["outputs"]["bronze"]["path"],
            fallback_name=source["source_id"],
            table_name=source["outputs"]["bronze"].get("table_name"),
        )
        for source in documents
    }
    actual_edges = set()
    for source in documents:
        for _, leaf in iceberg_leaves(source["access"].get("request_inputs", {})):
            upstream = leaf["upstream_source_id"]
            actual_edges.add((upstream, source["source_id"]))
            if upstream in tables:
                assert f"{leaf['namespace']}.{leaf['table_name']}" == tables[upstream]
            else:
                assert name == "missing_producer"
    assert actual_edges == {
        (upstream, spec.source_id) for spec in case.sources for upstream in spec.upstreams
    }
    if name == "duplicate_producer":
        assert tables["A"] == tables["B"] == "bronze.shared_target"


@pytest.mark.parametrize("grouped", (True, False))
def test_registry_builders_are_isolated_and_support_both_document_shapes(tmp_path, grouped):
    roots = [build_graph_project(tmp_path, "fan_in", grouped=grouped) for _ in range(2)]
    first, second = (load_registry(root) for root in roots)
    assert roots[0] != roots[1]
    assert [source.source_id for source in first.sources] == ["D", "B", "A"]
    assert [source.source_id for source in second.sources] == ["D", "B", "A"]
    for root in roots:
        for zone in ("raw", "bronze", "metadata"):
            assert (root / "data" / zone).is_dir()
    first_documents = source_documents(GRAPH_CASES["chain"])
    first_documents[0]["enabled"] = False
    assert source_documents(GRAPH_CASES["chain"])[0]["enabled"] is True


@pytest.mark.parametrize("grouped", (True, False))
def test_the_pre_migration_document_shape_no_longer_loads(tmp_path, grouped):
    """The declaration is required, so the fixtures' legacy mode is now a rejected shape."""
    root = build_graph_project(tmp_path, "chain", grouped=grouped, declare_upstreams=False)

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(root)

    message = str(exc_info.value)
    assert "access.request_inputs.upstream_source_id: is required" in message
    assert ("sources[0]." in message) is grouped
