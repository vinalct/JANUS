"""Contract coverage for the runnable orchestration example."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from examples.orchestration.bootstrap import PINNED_JARS, prepare_example_runtime
from examples.orchestration.fixture_service import build_server

from janus.registry import load_registry
from janus.utils.environment import load_environment_config, materialize_runtime_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3] / "examples" / "orchestration"


def test_example_registry_is_exactly_a_to_b_plus_independent_c() -> None:
    registry = load_registry(PROJECT_ROOT)

    assert tuple(source.source_id for source in registry.sources) == ("A", "B", "C")
    assert registry.graph.topological_order() == ("A", "B", "C")
    assert {
        (edge.producer_id, edge.consumer_id, edge.table, edge.input_paths)
        for edge in registry.graph.edges
    } == {
        (
            "A",
            "B",
            "orchestration_example.reference_a",
            ("access.request_inputs",),
        )
    }

    source_b = registry.get_source("B")
    assert source_b.access.request_inputs.upstream_source_id == "A"
    assert source_b.access.request_inputs.columns == {"reference_id": "reference_id"}


def test_every_example_runtime_location_is_isolated() -> None:
    config = load_environment_config("example", PROJECT_ROOT)
    locations = materialize_runtime_paths(config, PROJECT_ROOT)
    runtime_root = (PROJECT_ROOT / "runtime").resolve()

    for location in locations.values():
        assert isinstance(location, Path)
        assert location.resolve().is_relative_to(runtime_root)


def test_bootstrap_seeds_pinned_jars_and_dagster_concurrency_config(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    project_root = repository_root / "examples" / "orchestration"
    jar_root = repository_root / "deps"
    dagster_config = project_root / "conf" / "dagster"
    jar_root.mkdir(parents=True)
    dagster_config.mkdir(parents=True)
    for filename in PINNED_JARS:
        (jar_root / filename).write_bytes(filename.encode("utf-8"))
    (dagster_config / "dagster.yaml").write_text(
        "concurrency:\n  pools:\n    default_limit: 1\n    granularity: run\n",
        encoding="utf-8",
    )

    summary = prepare_example_runtime(project_root, repository_root)

    assert Path(summary["runtime_root"]) == project_root / "runtime"
    assert Path(summary["dagster_home"]) == project_root / "runtime" / "dagster"
    assert {Path(path).name for path in summary["ivy_jars"]} == set(PINNED_JARS)
    copied_config = project_root / "runtime" / "dagster" / "dagster.yaml"
    assert "granularity: run" in copied_config.read_text(encoding="utf-8")


@pytest.mark.parametrize("fail_a", (False, True))
def test_fixture_service_is_deterministic_and_failure_scoped(fail_a: bool) -> None:
    server = build_server(host="127.0.0.1", port=0, fail_a=fail_a)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        assert _get_json(f"http://{host}:{port}/health") == {"status": "ready"}
        assert _get_json(
            f"http://{host}:{port}/independent?window_start=2026-09-01"
            "&window_end=2026-09-01&page=1&page_size=100"
        ) == [
            {
                "event_id": "independent-2026-09-01-2026-09-01",
                "window_end": "2026-09-01",
                "window_start": "2026-09-01",
            }
        ]
        if fail_a:
            with pytest.raises(HTTPError) as exc_info:
                _get_json(f"http://{host}:{port}/reference?page=1&page_size=100")
            assert exc_info.value.code == 503
        else:
            assert (
                _get_json(f"http://{host}:{port}/reference?page=1&page_size=100")
                == server.reference
            )
            assert (
                _get_json(f"http://{host}:{port}/details?reference_id=north&page=1&page_size=100")
                == server.details["north"]
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dagster_definitions_have_exact_edge_and_disabled_schedule() -> None:
    dagster = pytest.importorskip("dagster")
    from examples.orchestration.definitions import (
        SCHEDULE_TIMEZONE,
        dagster_adapter,
        daily_example_schedule,
        defs,
    )

    dagster.Definitions.validate_loadable(defs)
    assert set(dagster_adapter.source_op_names) == {"A", "B", "C"}
    assert _definition_edges(dagster_adapter) == {("A", "B")}
    assert daily_example_schedule.default_status == dagster.DefaultScheduleStatus.STOPPED
    assert daily_example_schedule.execution_timezone == SCHEDULE_TIMEZONE
    assert SCHEDULE_TIMEZONE == "America/Sao_Paulo"


def _definition_edges(adapter: object) -> set[tuple[str, str]]:
    job = adapter.job
    op_names = adapter.source_op_names
    structure = job.graph.dependency_structure
    by_op = {op_name: source_id for source_id, op_name in op_names.items()}
    edges: set[tuple[str, str]] = set()
    for consumer_id, consumer_op in op_names.items():
        upstreams = structure.input_to_upstream_outputs_for_node(consumer_op)
        edges.update(
            (by_op[output.node_name], consumer_id)
            for outputs in upstreams.values()
            for output in outputs
        )
    return edges


def _get_json(url: str) -> object:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))
