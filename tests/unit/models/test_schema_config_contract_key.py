from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from janus.models import SourceConfig
from janus.models.source_config import (
    DEPRECATED_SCHEMA_MODES,
    SCHEMA_DECLARATION_DEPRECATION_MESSAGE,
    SUPPORTED_SCHEMA_MODES,
    SourceConfigValidationError,
)

CONFIG_PATH = Path("conf/sources/example/contract_key.yaml")
CONTRACT_PATH = "conf/contracts/example/contract_key.yaml"

VERBATIM_DEPRECATION = (
    "`schema.mode`/`schema.path` are deprecated and will be removed; "
    "declare `schema.contract: conf/contracts/<domain>/<table>.yaml` instead "
    "(see docs/data-contracts.md)."
)


def _payload(schema_block: dict[str, Any]) -> dict[str, Any]:
    """A minimal valid API source whose only variable is its schema declaration."""
    return {
        "source_id": "contract_key_example",
        "name": "Contract Key Example",
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
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {"concurrency": 1, "backoff_seconds": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "checkpoint_strategy": "none",
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": copy.deepcopy(schema_block),
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/contract_key", "format": "json"},
            "bronze": {"path": "data/bronze/example/contract_key", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/contract_key", "format": "json"},
        },
        "quality": {"allow_schema_evolution": True},
    }


def _issues(schema_block: dict[str, Any]) -> dict[str, str]:
    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(_payload(schema_block), CONFIG_PATH)
    return {issue.path: issue.message for issue in exc_info.value.issues}


# ── the new declaration ───────────────────────────────────────────────────────


def test_contract_alone_parses_and_implies_the_contract_mode() -> None:
    config = SourceConfig.from_mapping(_payload({"contract": CONTRACT_PATH}), CONFIG_PATH)

    assert config.schema.mode == "contract"
    assert config.schema.contract == CONTRACT_PATH
    assert config.schema.path is None
    assert config.schema.declares_contract is True
    assert config.schema.declares_legacy_file is False
    assert config.schema.is_declared is True
    assert config.deprecations == ()


def test_contract_mode_may_also_be_written_out() -> None:
    config = SourceConfig.from_mapping(
        _payload({"mode": "contract", "contract": CONTRACT_PATH}), CONFIG_PATH
    )

    assert config.schema.mode == "contract"
    assert config.schema.contract == CONTRACT_PATH
    assert config.deprecations == ()


def test_contract_mode_is_part_of_the_supported_set() -> None:
    assert "contract" in SUPPORTED_SCHEMA_MODES
    assert set(DEPRECATED_SCHEMA_MODES) == {"explicit", "infer"}
    assert "contract" not in DEPRECATED_SCHEMA_MODES


def test_contract_with_a_conflicting_mode_is_rejected() -> None:
    issues = _issues({"mode": "explicit", "contract": CONTRACT_PATH})

    assert issues["schema.mode"] == (
        "must be 'contract' or omitted when schema.contract is set"
    )


def test_contract_with_a_legacy_path_is_rejected() -> None:
    issues = _issues(
        {"contract": CONTRACT_PATH, "path": "conf/schemas/example/source_schema.json"}
    )

    assert issues["schema.path"] == "must not be set together with schema.contract"


def test_contract_mode_without_a_contract_is_rejected() -> None:
    issues = _issues({"mode": "contract"})

    assert issues["schema.contract"] == "is required when schema.mode is 'contract'"


def test_mode_stays_required_for_the_legacy_forms() -> None:
    issues = _issues({"path": "conf/schemas/example/source_schema.json"})

    assert issues["schema.mode"] == "is required"


# ── the legacy declaration, still loading ─────────────────────────────────────


def test_legacy_explicit_path_loads_with_the_verbatim_deprecation() -> None:
    config = SourceConfig.from_mapping(
        _payload({"mode": "explicit", "path": "conf/schemas/example/source_schema.json"}),
        CONFIG_PATH,
    )

    assert config.schema.mode == "explicit"
    assert config.schema.declares_legacy_file is True
    assert config.schema.is_declared is True
    assert [issue.path for issue in config.deprecations] == ["schema"]
    assert config.deprecations[0].message == VERBATIM_DEPRECATION


def test_legacy_infer_loads_with_one_deprecation() -> None:
    config = SourceConfig.from_mapping(_payload({"mode": "infer"}), CONFIG_PATH)

    assert config.schema.mode == "infer"
    assert config.schema.is_declared is False
    assert len(config.deprecations) == 1
    assert config.deprecations[0].message == VERBATIM_DEPRECATION


def test_the_pinned_message_is_the_one_the_config_layer_exports() -> None:
    assert SCHEMA_DECLARATION_DEPRECATION_MESSAGE == VERBATIM_DEPRECATION


def test_explicit_without_a_path_still_fails_and_still_deprecates() -> None:
    issues = _issues({"mode": "explicit"})

    assert issues["schema.path"] == "is required when schema.mode is 'explicit'"


def test_a_deprecation_never_masks_an_unrelated_issue() -> None:
    payload = _payload({"mode": "infer"})
    payload["outputs"]["raw"].pop("path")

    with pytest.raises(SourceConfigValidationError) as exc_info:
        SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert "outputs.raw.path" in {issue.path for issue in exc_info.value.issues}


def test_deprecations_are_not_serialised_into_any_run_artifact() -> None:
    """A deprecation describes the config file, never the run it configured."""
    config = SourceConfig.from_mapping(
        _payload({"mode": "explicit", "path": "conf/schemas/example/source_schema.json"}),
        CONFIG_PATH,
    )

    assert config.deprecations  # the fixture must actually carry one
    rendered = json.dumps(
        {
            "schema": {
                "mode": config.schema.mode,
                "path": config.schema.path,
                "contract": config.schema.contract,
            }
        }
    )
    assert "deprecat" not in rendered
