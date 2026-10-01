"""The loader's structural contract rules: an incremental source needs a ``primaryKey``."""

from __future__ import annotations

from pathlib import Path

import pytest

from janus.models.source_config import SourceConfigValidationError
from janus.registry import load_registry
from tests.support.contracts import DECLARED_CONTRACT_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
MINIMAL = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "minimal_contract.yaml"
SOURCE_ID = "structural_rules_example"


def _source_yaml(*, mode: str, quality: str = "quality: {}\n") -> str:
    checkpoint = (
        "  checkpoint_field: id\n  checkpoint_strategy: max_value\n"
        if mode == "incremental"
        else "  checkpoint_strategy: none\n"
    )
    return f"""
source_id: {SOURCE_ID}
name: {SOURCE_ID}
owner: janus
enabled: true
source_type: api
strategy: api
strategy_variant: page_number_api
federation_level: federal
domain: example
public_access: true
access:
  base_url: https://example.invalid
  path: /records
  method: GET
  format: json
  auth:
    type: none
  pagination:
    type: page_number
    page_param: page
    size_param: page_size
    page_size: 10
  rate_limit:
    concurrency: 1
extraction:
  mode: {mode}
{checkpoint}  retry:
    max_attempts: 3
    backoff_strategy: fixed
    backoff_seconds: 1
schema:
  contract: {DECLARED_CONTRACT_PATH}
spark:
  input_format: json
  write_mode: append
outputs:
  raw:
    path: data/raw/example/{SOURCE_ID}
    format: json
  bronze:
    path: data/bronze/example/{SOURCE_ID}
    format: iceberg
  metadata:
    path: data/metadata/example/{SOURCE_ID}
    format: json
{quality}""".lstrip()


def _project(
    tmp_path: Path, *, contract: Path, mode: str, quality: str = "quality: {}\n"
) -> Path:
    (tmp_path / "conf").mkdir(parents=True)
    (tmp_path / "conf" / "app.yaml").write_text(
        'registry:\n  sources_dir: conf/sources\n  file_pattern: "*.yaml"\n', encoding="utf-8"
    )
    declared = tmp_path / DECLARED_CONTRACT_PATH
    declared.parent.mkdir(parents=True)
    declared.write_bytes(contract.read_bytes())
    source = tmp_path / "conf" / "sources" / "example" / "source.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(_source_yaml(mode=mode, quality=quality), encoding="utf-8")
    return tmp_path


def _issues(project: Path) -> dict[str, str]:
    with pytest.raises(SourceConfigValidationError) as raised:
        load_registry(project)
    return {issue.path: issue.message for issue in raised.value.issues}


def test_an_incremental_source_whose_contract_has_no_primary_key_is_refused_at_load(tmp_path):
    issues = _issues(_project(tmp_path, contract=MINIMAL, mode="incremental"))

    [path] = [path for path in issues if path.endswith("schema.contract")]
    assert "primaryKey" in issues[path]
    assert "incremental" in issues[path]
    assert not any(path.endswith("quality.unique_fields") for path in issues)


def test_an_incremental_source_with_a_primary_key_needs_no_unique_fields(tmp_path):
    registry = load_registry(_project(tmp_path, contract=HOSTILE / "base.yaml", mode="incremental"))

    source = registry.get_source(SOURCE_ID)
    assert source.extraction.mode == "incremental"
    assert source.quality.unique_fields == ()


def test_the_rule_is_collected_with_the_loaders_other_contract_rules(tmp_path):
    """Both rules run where the contract is known, so one load reports both."""
    disagreeing = "quality:\n  required_fields:\n    - id\n"

    issues = _issues(_project(tmp_path, contract=MINIMAL, mode="incremental", quality=disagreeing))

    assert any(path.endswith("schema.contract") for path in issues)
    assert any(path.endswith("quality.required_fields") for path in issues)


@pytest.mark.parametrize("mode", ["full_refresh", "snapshot"])
def test_non_incremental_sources_need_no_primary_key(tmp_path, mode):
    """Green on arrival: only an upsert needs a key."""
    registry = load_registry(_project(tmp_path, contract=MINIMAL, mode=mode))

    assert registry.get_source(SOURCE_ID).extraction.mode == mode


# ── order-07's incremental key rule, moved here from ``SourceConfig.from_mapping`` ──


def test_the_refusal_names_the_idempotency_the_key_protects(tmp_path):
    issues = _issues(_project(tmp_path, contract=MINIMAL, mode="incremental"))

    assert issues == {
        "schema.contract": "an incremental source needs a primaryKey in its contract to derive "
        "an idempotent bronze write."
    }


def test_an_incremental_source_whose_unique_fields_equal_the_primary_key_loads(tmp_path):
    agreeing = "quality:\n  required_fields:\n    - id\n  unique_fields:\n    - id\n"

    registry = load_registry(
        _project(tmp_path, contract=HOSTILE / "base.yaml", mode="incremental", quality=agreeing)
    )

    source = registry.get_source(SOURCE_ID)
    assert source.extraction.mode == "incremental"
    assert source.quality.unique_fields == ("id",)


def test_every_contract_rule_of_one_source_is_reported_in_one_raise(tmp_path):
    """The rule appends to the collected issues rather than short-circuiting the load."""
    draft = tmp_path / "draft_minimal.yaml"
    draft.write_text(
        MINIMAL.read_text(encoding="utf-8").replace("status: active", "status: draft"),
        encoding="utf-8",
    )

    with pytest.raises(SourceConfigValidationError) as raised:
        load_registry(_project(tmp_path, contract=draft, mode="incremental"))

    assert [issue.path for issue in raised.value.issues] == ["schema.contract"] * 2
    active, keyed = (issue.message for issue in raised.value.issues)
    assert "status 'active'" in active
    assert "primaryKey" in keyed


def test_a_mapping_problem_is_reported_before_the_contract_rules_run(tmp_path):
    """A keyless incremental source that also lacks its checkpoint field reports the field."""
    project = _project(tmp_path, contract=MINIMAL, mode="incremental")
    source = project / "conf" / "sources" / "example" / "source.yaml"
    source.write_text(
        source.read_text(encoding="utf-8").replace("  checkpoint_field: id\n", ""),
        encoding="utf-8",
    )

    issues = _issues(project)

    assert "extraction.checkpoint_field" in issues
    assert "schema.contract" not in issues
