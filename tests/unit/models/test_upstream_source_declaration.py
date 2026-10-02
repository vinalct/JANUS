"""FR-4: every ``iceberg_rows`` leaf names the source that produces the table it reads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from janus.models.config.policy import DEFAULT_VALIDATION_POLICY
from janus.models.source_config import (
    CombinedRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    SourceConfig,
    SourceConfigValidationError,
    _parse_request_input_entry,
)
from janus.registry import load_registry
from tests.support.contracts import (
    CONTRACT_SCHEMA_BLOCK,
    PRODUCER_CONTRACT_PATH,
    write_minimal_contract,
    write_producer_contract,
)

CONFIG_PATH = Path("conf/sources/example/upstream_declaration.yaml")

EVERYTHING_POLICY_CAN_RELAX = replace(
    DEFAULT_VALIDATION_POLICY,
    require_public_access=False,
    require_strategy_matches_source_type=False,
    federation_levels=frozenset({"federal", "state", "municipal"}),
    source_types=frozenset({"api", "catalog", "file", "graphql"}),
    strategies=frozenset({"api", "catalog", "file", "graphql"}),
)

ORGAOS_LEAF: dict[str, Any] = {
    "type": "iceberg_rows",
    "upstream_source_id": "transparencia__orgaos__siafi__full_refresh",
    "namespace": "bronze__transparencia",
    "table_name": "orgaos__siafi",
    "columns": {"orgao_codigo": "codigo"},
    "distinct": True,
}
EMENDAS_LEAF: dict[str, Any] = {
    "type": "iceberg_rows",
    "upstream_source_id": "transparencia__emendas_parlamentares__emendas__full_refresh",
    "namespace": "bronze__transparencia",
    "table_name": "emendas_parlamentares__emendas",
    "columns": {"emenda_codigo": "codigoEmenda"},
}
DATE_WINDOW_LEAF: dict[str, Any] = {
    "type": "date_window",
    "start": "2025-01-01",
    "end": "2025-03-31",
    "step": "month",
}


def test_a_top_level_leaf_carries_its_producer_and_its_table_separately():
    request_inputs = _load_request_inputs(ORGAOS_LEAF)

    assert isinstance(request_inputs, IcebergRowsRequestInputsConfig)
    assert request_inputs.upstream_source_id == "transparencia__orgaos__siafi__full_refresh"
    assert (request_inputs.namespace, request_inputs.table_name) == (
        "bronze__transparencia",
        "orgaos__siafi",
    )


def test_combined_declares_a_producer_per_leaf_including_two_different_ones():
    request_inputs = _load_request_inputs(
        {"type": "combined", "inputs": [ORGAOS_LEAF, EMENDAS_LEAF]}
    )

    assert isinstance(request_inputs, CombinedRequestInputsConfig)
    assert [leaf.upstream_source_id for leaf in request_inputs.inputs] == [
        "transparencia__orgaos__siafi__full_refresh",
        "transparencia__emendas_parlamentares__emendas__full_refresh",
    ]


def test_two_leaves_may_read_the_same_producer_twice():
    """Fan-in on one producer is legal; the second read is a leaf, not a duplicate."""
    second_read = {
        **ORGAOS_LEAF,
        "columns": {"orgao_superior_codigo": "codigoSuperior"},
        "distinct": False,
    }

    request_inputs = _load_request_inputs(
        {"type": "combined", "inputs": [ORGAOS_LEAF, second_read]}
    )

    assert isinstance(request_inputs, CombinedRequestInputsConfig)
    assert {leaf.upstream_source_id for leaf in request_inputs.inputs} == {
        "transparencia__orgaos__siafi__full_refresh"
    }
    assert [sorted(leaf.columns) for leaf in request_inputs.inputs] == [
        ["orgao_codigo"],
        ["orgao_superior_codigo"],
    ]


def test_a_date_window_leaf_needs_no_declaration():
    request_inputs = _load_request_inputs(
        {"type": "combined", "inputs": [ORGAOS_LEAF, DATE_WINDOW_LEAF]}
    )

    assert isinstance(request_inputs, CombinedRequestInputsConfig)
    assert [leaf.type for leaf in request_inputs.inputs] == ["iceberg_rows", "date_window"]


def test_a_source_without_request_inputs_is_untouched():
    config = SourceConfig.from_mapping(_base_mapping(), CONFIG_PATH)

    assert config.access.request_inputs.type == "none"
    assert config.access.request_inputs.requires_spark is False


@pytest.mark.parametrize(
    ("value", "expected_message"),
    (
        pytest.param(None, "is required", id="missing"),
        pytest.param("", "must not be empty", id="empty"),
        pytest.param("   ", "must not be empty", id="whitespace_only"),
        pytest.param(17, "must be a string", id="integer"),
        pytest.param(["orgaos"], "must be a string", id="list"),
        pytest.param({"id": "orgaos"}, "must be a string", id="mapping"),
    ),
)
def test_an_unusable_declaration_is_rejected_with_its_field_path(value, expected_message):
    leaf = dict(ORGAOS_LEAF)
    if value is None:
        del leaf["upstream_source_id"]
    else:
        leaf["upstream_source_id"] = value

    with pytest.raises(SourceConfigValidationError) as exc_info:
        _load_request_inputs(leaf)

    assert f"access.request_inputs.upstream_source_id: {expected_message}" in str(exc_info.value)


def test_a_malformed_leaf_reports_its_exact_nested_path():
    broken = {key: value for key, value in EMENDAS_LEAF.items() if key != "upstream_source_id"}

    with pytest.raises(SourceConfigValidationError) as exc_info:
        _load_request_inputs({"type": "combined", "inputs": [ORGAOS_LEAF, broken]})

    message = str(exc_info.value)
    assert "access.request_inputs.inputs[1].upstream_source_id: is required" in message
    assert "access.request_inputs.inputs[0]" not in message


def test_an_undeclared_leaf_never_becomes_a_config_object():
    """Fail closed: no placeholder producer, so no edge the graph would have to guess."""
    issues: list[Any] = []

    entry = _parse_request_input_entry(
        {key: value for key, value in ORGAOS_LEAF.items() if key != "upstream_source_id"},
        "iceberg_rows",
        "access.request_inputs",
        issues,
    )

    assert entry is None
    assert [issue.render() for issue in issues] == [
        "access.request_inputs.upstream_source_id: is required"
    ]


def test_a_directly_constructed_config_cannot_omit_the_producer():
    with pytest.raises(TypeError, match="upstream_source_id"):
        IcebergRowsRequestInputsConfig(  
            type="iceberg_rows",
            namespace="bronze__transparencia",
            table_name="orgaos__siafi",
            columns={"orgao_codigo": "codigo"},
        )


@pytest.mark.parametrize("declaration", ("", "   ", None))
def test_a_directly_constructed_config_cannot_name_nobody(declaration):
    with pytest.raises(ValueError, match="upstream_source_id must be a non-empty string"):
        IcebergRowsRequestInputsConfig(
            type="iceberg_rows",
            upstream_source_id=declaration,  
            namespace="bronze__transparencia",
            table_name="orgaos__siafi",
            columns={"orgao_codigo": "codigo"},
        )


def test_independent_problems_in_one_load_are_all_reported_together():
    """One load, five problems: the declaration rule must not short-circuit the rest."""
    mapping = _base_mapping()
    mapping["access"]["request_inputs"] = {
        "type": "combined",
        "inputs": [
            {key: value for key, value in ORGAOS_LEAF.items() if key != "upstream_source_id"},
            {"type": "date_window", "start": "not-a-date", "end": "2025-03-31", "step": "year"},
        ],
    }
    mapping["access"]["parameter_bindings"] = {"codigo": {"from": "request_input.orgao_codigo"}}
    mapping["strategy_variant"] = "not_a_variant"

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(mapping, CONFIG_PATH)

    message = str(exc_info.value)
    assert "access.request_inputs.inputs[0].upstream_source_id: is required" in message
    assert "access.request_inputs.inputs[1].start: must be a YYYY-MM-DD date" in message
    assert "access.request_inputs.inputs[1].step: must be one of: day, month" in message
    assert "access.parameter_bindings.codigo.from: must reference one of the combined" in message
    assert "strategy_variant: must be one of" in message


def test_no_validation_policy_can_relax_the_declaration():
    """Structural, not phase-scope: relaxing every policy rule leaves this one enforced."""
    mapping = _base_mapping(federation_level="state", public_access=False, strategy="catalog")
    mapping["access"]["request_inputs"] = {
        key: value for key, value in ORGAOS_LEAF.items() if key != "upstream_source_id"
    }

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(mapping, CONFIG_PATH, policy=EVERYTHING_POLICY_CAN_RELAX)

    message = str(exc_info.value)
    assert "access.request_inputs.upstream_source_id: is required" in message
    assert "federation_level" not in message
    assert "public_access" not in message


def test_a_grouped_document_keeps_the_sources_index_prefix(tmp_path):
    first = _base_mapping(source_id="grouped_producer", name="grouped_producer")
    second = _base_mapping(source_id="grouped_consumer", name="grouped_consumer")
    second["access"]["request_inputs"] = {
        key: value for key, value in ORGAOS_LEAF.items() if key != "upstream_source_id"
    }
    project_root = _write_grouped_project(tmp_path, [first, second])

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert "sources[1].access.request_inputs.upstream_source_id: is required" in str(
        exc_info.value
    )


def test_a_declared_leaf_still_loads_through_the_registry(tmp_path):
    consumer = _base_mapping(source_id="grouped_consumer", name="grouped_consumer")
    consumer["access"]["request_inputs"] = dict(ORGAOS_LEAF)
    consumer["access"]["parameter_bindings"] = {
        "codigoOrgao": {"from": "request_input.orgao_codigo"}
    }
    project_root = _write_grouped_project(tmp_path, [_producer_mapping(ORGAOS_LEAF), consumer])
    write_producer_contract(project_root, tuple(ORGAOS_LEAF["columns"].values()))

    registry = load_registry(project_root)
    request_inputs = registry.get_source("grouped_consumer").access.request_inputs

    assert isinstance(request_inputs, IcebergRowsRequestInputsConfig)
    assert request_inputs.upstream_source_id == "transparencia__orgaos__siafi__full_refresh"
    assert request_inputs.distinct is True
    assert request_inputs.requires_spark is True


def _load_request_inputs(raw: dict[str, Any]):
    """Build one source whose only interesting part is its request-input block."""
    mapping = _base_mapping()
    mapping["access"]["request_inputs"] = raw
    return SourceConfig.from_mapping(mapping, CONFIG_PATH).access.request_inputs


def _producer_mapping(leaf: dict[str, Any]) -> dict[str, Any]:
    """The source that writes the table ``leaf`` reads, under the id the leaf declares.

    The declaration is only half the contract: the registry also checks that the named
    source really produces that table, and that its contract declares every column the
    leaf reads, so a fixture proving a leaf loads needs its producer in the same project,
    under the contract ``write_producer_contract`` writes.
    """
    source_id = leaf["upstream_source_id"]
    mapping = _base_mapping(source_id=source_id, name=source_id)
    mapping["schema"] = {"contract": PRODUCER_CONTRACT_PATH}
    mapping["outputs"] = {
        "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
        "bronze": {
            "path": f"data/bronze/example/{source_id}",
            "format": "iceberg",
            "namespace": leaf["namespace"],
            "table_name": leaf["table_name"],
        },
        "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
    }
    return mapping


def _write_grouped_project(tmp_path: Path, documents: list[dict[str, Any]]) -> Path:
    """Write a ``sources:`` document so errors carry the loader's ``sources[index]`` prefix."""
    sources_dir = tmp_path / "conf" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "conf" / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: '*.yaml'\n", encoding="utf-8"
    )
    write_minimal_contract(tmp_path)
    (sources_dir / "grouped.yaml").write_text(
        yaml.safe_dump({"sources": documents}, sort_keys=False), encoding="utf-8"
    )
    return tmp_path


def _base_mapping(**overrides: Any) -> dict[str, Any]:
    """A source that loads cleanly, so every failure below belongs to the override."""
    mapping: dict[str, Any] = {
        "source_id": "upstream_declaration_source",
        "name": "upstream_declaration_source",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/records",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 100,
            },
            "rate_limit": {"requests_per_minute": 10, "concurrency": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "retry": {"max_attempts": 3, "backoff_strategy": "fixed", "backoff_seconds": 1},
        },
        "schema": dict(CONTRACT_SCHEMA_BLOCK),
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/upstream_declaration", "format": "json"},
            "bronze": {"path": "data/bronze/example/upstream_declaration", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/upstream_declaration", "format": "json"},
        },
        "quality": {},
    }
    mapping.update(overrides)
    return mapping
