"""One declaration for required and unique: the contract's ``required`` and ``primaryKey``."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from janus.models import ExecutionPlan, RunContext, SourceConfig, resolve_bronze_write_intent
from janus.models.data_contracts import load_data_contract
from janus.models.source_config import SourceConfigValidationError
from janus.registry import load_registry
from tests.support.contracts import DECLARED_CONTRACT_PATH

RED_TASK = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
SOURCE_ID = "quality_agreement_example"
CONFIG_PATH = Path("conf/sources/example/quality_agreement.yaml")


def _payload(*, quality: dict[str, Any], incremental: bool = False) -> dict[str, Any]:
    extraction: dict[str, Any] = {
        "mode": "full_refresh",
        "checkpoint_strategy": "none",
        "retry": {"max_attempts": 3, "backoff_strategy": "fixed", "backoff_seconds": 1},
    }
    if incremental:
        extraction.update(
            {"mode": "incremental", "checkpoint_field": "when", "checkpoint_strategy": "max_value"}
        )
    return {
        "source_id": SOURCE_ID,
        "name": SOURCE_ID,
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
        "extraction": extraction,
        "schema": {"contract": DECLARED_CONTRACT_PATH},
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/quality_agreement", "format": "json"},
            "bronze": {"path": "data/bronze/example/quality_agreement", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/quality_agreement", "format": "json"},
        },
        "quality": copy.deepcopy(quality),
    }


def _project(tmp_path: Path, *, contract: str, quality: dict[str, Any]) -> Path:
    """A one-source project whose contract is the named hostile fixture."""
    (tmp_path / "conf").mkdir(parents=True)
    (tmp_path / "conf" / "app.yaml").write_text(
        'registry:\n  sources_dir: conf/sources\n  file_pattern: "*.yaml"\n', encoding="utf-8"
    )
    declared = tmp_path / DECLARED_CONTRACT_PATH
    declared.parent.mkdir(parents=True)
    declared.write_bytes((HOSTILE / f"{contract}.yaml").read_bytes())
    source = tmp_path / "conf" / "sources" / "example" / "source.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump(_payload(quality=quality), sort_keys=False), encoding="utf-8")
    return tmp_path


def _issues(project: Path) -> dict[str, str]:
    with pytest.raises(SourceConfigValidationError) as raised:
        load_registry(project)
    return {
        issue.path.removeprefix(f"{SOURCE_ID}."): issue.message for issue in raised.value.issues
    }


# ── pins: agreement and absence stay silent ──────────────────────────────────


@pytest.mark.parametrize(
    "quality",
    [
        {"required_fields": ["id"], "unique_fields": ["id"]},
        {"unique_fields": ["id"], "required_fields": ["id"]},
        {},
    ],
    ids=["equal", "equal-reordered-keys", "absent"],
)
def test_agreeing_or_absent_quality_keys_load(tmp_path, quality):
    """Green on arrival: the cross-check must never fire on a source that agrees."""
    registry = load_registry(_project(tmp_path, contract="base", quality=quality))

    assert registry.get_source(SOURCE_ID).source_id == SOURCE_ID


# ── disagreement is an issue at the key's own path ───────────────────────────


@RED_TASK
def test_required_fields_naming_more_than_the_contract_is_an_issue(tmp_path):
    issues = _issues(
        _project(tmp_path, contract="base", quality={"required_fields": ["id", "label"]})
    )

    assert "label" in issues["quality.required_fields"]
    assert "contract" in issues["quality.required_fields"]


@RED_TASK
def test_required_fields_naming_less_than_the_contract_is_an_issue(tmp_path):
    issues = _issues(
        _project(tmp_path, contract="base_plus_required", quality={"required_fields": ["id"]})
    )

    assert "code" in issues["quality.required_fields"]


@RED_TASK
def test_unique_fields_must_equal_the_primary_key(tmp_path):
    quality = {"required_fields": ["id"], "unique_fields": ["id", "label"]}

    issues = _issues(_project(tmp_path, contract="base", quality=quality))

    assert list(issues) == ["quality.unique_fields"]
    assert "label" in issues["quality.unique_fields"]


@RED_TASK
def test_each_disagreement_is_its_own_issue(tmp_path):
    quality = {"required_fields": ["label"], "unique_fields": ["label"]}

    issues = _issues(_project(tmp_path, contract="base", quality=quality))

    assert {"quality.required_fields", "quality.unique_fields"} <= set(issues)


# ── the write intent reads the contract ──────────────────────────────────────


def _plan(payload: dict[str, Any], contract: str) -> ExecutionPlan:
    source_config = SourceConfig.from_mapping(payload, CONFIG_PATH)
    run_context = RunContext.create(
        run_id="run-quality-agreement",
        environment="local",
        project_root=PROJECT_ROOT,
        started_at=datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(
        source_config,
        run_context,
        data_contract=load_data_contract(HOSTILE / f"{contract}.yaml"),
    )


@RED_TASK
def test_merge_keys_come_from_the_primary_key_when_unique_fields_is_absent():
    plan = _plan(_payload(quality={}, incremental=True), "base")

    intent = resolve_bronze_write_intent(plan)

    assert intent.is_upsert
    assert intent.merge_keys == ("id",)


def test_merge_keys_equal_the_primary_key_when_both_are_declared():
    """Green on arrival: an agreeing source upserts on the same key before and after."""
    agreeing = {"required_fields": ["id"], "unique_fields": ["id"]}
    plan = _plan(_payload(quality=agreeing, incremental=True), "base")

    assert resolve_bronze_write_intent(plan).merge_keys == ("id",)
