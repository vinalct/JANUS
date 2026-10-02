from __future__ import annotations

import json
from pathlib import Path

import pytest

from janus.models.config.strategy_registry import STRATEGY_REGISTRY
from janus.orchestration import BatchSelection, SelectionFilterError
from janus.registry import load_registry
from tests.support.cli_golden import GOLDENS_DIR, INVOCATIONS
from tests.support.operator_cli import arm_spark_tripwire, run_janus
from tests.support.semantics_fixtures import CLEAN, CLEAN_CONSUMER, CLEAN_PRODUCER, materialize

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="janus.cli.dispatch registers no `list` verb yet",
)

TABLE_COLUMNS = ["SOURCE_ID", "FAMILY", "VARIANT", "MODE", "EN", "HOOK", "UPSTREAMS", "TAGS"]
JSON_SOURCE_KEYS = {
    "source_id",
    "name",
    "family",
    "variant",
    "mode",
    "enabled",
    "domain",
    "hook",
    "tags",
    "upstreams",
    "downstreams",
    "bronze_table",
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return materialize(CLEAN, tmp_path / "project")


def _list(root: Path, *extra: str):
    return run_janus(("list", "--project-root", str(root), *extra))


def _listed_ids(root: Path, *extra: str) -> list[str]:
    result = _list(root, "--format", "json", *extra)
    assert result.exit_code == 0, result.output
    return [source["source_id"] for source in json.loads(result.stdout)["sources"]]


def test_table_carries_every_ac5_attribute_sorted_by_source_id(root: Path) -> None:
    result = _list(root)
    lines = result.stdout.splitlines()

    assert result.exit_code == 0, result.output
    assert lines[0].split() == TABLE_COLUMNS
    consumer, producer = lines[1].split(), lines[2].split()
    assert consumer == [
        CLEAN_CONSUMER,
        "api",
        "page_number_api",
        "full_refresh",
        "no",
        "ibge.sidra_flat",
        CLEAN_PRODUCER,
        "consumer,semantics",
    ]
    assert producer == [
        CLEAN_PRODUCER,
        "api",
        "page_number_api",
        "incremental",
        "yes",
        "-",
        "-",
        "producer,semantics",
    ]
    assert lines[-1] == "2 source(s); 1 enabled; 1 edge(s)"


def test_json_carries_the_graph_view_of_each_source(root: Path) -> None:
    registry = load_registry(root)
    result = _list(root, "--format", "json")
    payload = json.loads(result.stdout)
    by_id = {source["source_id"]: source for source in payload["sources"]}

    assert result.exit_code == 0, result.output
    assert payload["counts"] == {"sources": 2, "enabled": 1, "edges": 1}
    assert list(by_id) == [CLEAN_CONSUMER, CLEAN_PRODUCER]
    assert all(source.keys() == JSON_SOURCE_KEYS for source in payload["sources"])
    assert by_id[CLEAN_CONSUMER]["upstreams"] == [CLEAN_PRODUCER]
    assert by_id[CLEAN_PRODUCER]["downstreams"] == [CLEAN_CONSUMER]
    assert by_id[CLEAN_CONSUMER]["tags"] == ["consumer", "semantics"]
    assert by_id[CLEAN_CONSUMER]["hook"] == "ibge.sidra_flat"
    assert by_id[CLEAN_PRODUCER]["hook"] is None
    for source_id, source in by_id.items():
        assert source["bronze_table"] == registry.graph.node(source_id).bronze_table
    assert result.stdout == json.dumps(payload, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (("--tag", "producer"), [CLEAN_PRODUCER]),
        (("--tag", "producer", "--tag", "consumer"), [CLEAN_CONSUMER, CLEAN_PRODUCER]),
        (("--domain", "semantics_reporting"), [CLEAN_CONSUMER]),
        (("--family", "api"), [CLEAN_CONSUMER, CLEAN_PRODUCER]),
        (("--enabled-only",), [CLEAN_PRODUCER]),
        (("--tag", "semantics", "--enabled-only"), [CLEAN_PRODUCER]),
    ],
    ids=["tag", "repeated-tag", "domain", "family", "enabled-only", "tag-and-enabled-only"],
)
def test_filters_select_exactly_what_matches(root: Path, flags: tuple[str, ...], expected) -> None:
    assert _listed_ids(root, *flags) == expected


def test_a_consumer_is_listed_without_its_upstream(root: Path) -> None:
    """Unlike `run-all`, `list` never closes over upstreams: it shows what was asked for."""
    assert _listed_ids(root, "--tag", "consumer") == [CLEAN_CONSUMER]


def test_tag_and_domain_together_fail_the_way_run_all_fails(root: Path) -> None:
    """One selection type (`BatchSelection`), so one meaning and one message."""
    with pytest.raises(SelectionFilterError) as refused:
        BatchSelection.create(tags=("producer",), domains=("semantics_reference",))

    result = _list(root, "--tag", "producer", "--domain", "semantics_reference")

    assert result.exit_code == 2
    assert str(refused.value) in result.stderr


def test_family_choices_come_from_the_strategy_registry(root: Path) -> None:
    unregistered = "not_a_family"
    assert unregistered not in STRATEGY_REGISTRY.families

    result = _list(root, "--family", unregistered)

    assert result.exit_code == 2
    for family in sorted(STRATEGY_REGISTRY.families):
        assert family in result.stderr


def test_an_empty_selection_is_a_true_answer_not_an_error(root: Path) -> None:
    result = _list(root, "--family", "catalog")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == "0 source(s); 0 enabled; 0 edge(s)"


def test_graph_renders_the_graphs_own_topological_order_and_every_edge(root: Path) -> None:
    """Dependency order (producer first) is the reverse of id order here: a second `sorted()`
    over `topological_order()` would be visible."""
    registry = load_registry(root)
    order = registry.graph.topological_order()
    assert order == (CLEAN_PRODUCER, CLEAN_CONSUMER)

    result = _list(root, "--graph")
    text = result.stdout

    assert result.exit_code == 0, result.output
    assert text.index(CLEAN_PRODUCER) < text.index(CLEAN_CONSUMER)
    assert f"{CLEAN_PRODUCER}  ->  {CLEAN_CONSUMER}" in text
    assert "semantics.clean_producer" in text
    assert "access.request_inputs" in text


@pytest.mark.parametrize("flags", [(), ("--format", "json"), ("--graph",)], ids=str)
def test_list_is_byte_identical_across_runs(root: Path, flags: tuple[str, ...]) -> None:
    first = _list(root, *flags)

    assert first.exit_code == 0, first.output
    assert _list(root, *flags) == first


def test_list_never_acquires_a_spark_session(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm_spark_tripwire(monkeypatch)

    assert _list(root, "--graph").exit_code == 0


def test_the_checked_in_registry_listing_is_golden() -> None:
    """AC-5's golden lives in the AC-1 corpus, so it is captured, normalized and asserted by
    the same runner as every other documented invocation — never by a second mechanism."""
    by_name = {invocation.name: invocation for invocation in INVOCATIONS}

    assert by_name["list_table"].argv == ("list",)
    assert by_name["list_json"].argv == ("list", "--format", "json")
    for name in ("list_table", "list_json"):
        assert (GOLDENS_DIR / f"{name}.code").read_text(encoding="utf-8") == "0\n"
