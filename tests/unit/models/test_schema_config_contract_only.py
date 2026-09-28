"""``schema.contract`` is the only schema key; every retired key is a named error (AC-6, FR-6)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from janus.models import SourceConfig
from janus.models.config.policy import PhaseValidationPolicy, ValidationPolicy
from janus.models.source_config import SchemaConfig, SourceConfigValidationError, ValidationIssue
from janus.registry import load_registry
from tests.support.contracts import DECLARED_CONTRACT_PATH, write_minimal_contract

pytestmark = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

CONFIG_PATH = Path("conf/sources/example/contract_only.yaml")
LEGACY_SCHEMA = "conf/schemas/example/source_schema.json"
SCHEMA_REPLACEMENT = ("schema.contract", "janus contract draft")
EVOLUTION_REPLACEMENT = "janus.compatibility"


def _payload(schema: dict[str, Any], *, quality: dict[str, Any] | None = None) -> dict[str, Any]:
    """A minimal valid api source whose variables are its schema and quality blocks."""
    return {
        "source_id": "contract_only_example",
        "name": "Contract Only Example",
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
            "retry": {"max_attempts": 3, "backoff_strategy": "fixed", "backoff_seconds": 1},
        },
        "schema": copy.deepcopy(schema),
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/contract_only", "format": "json"},
            "bronze": {"path": "data/bronze/example/contract_only", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/contract_only", "format": "json"},
        },
        "quality": copy.deepcopy(quality or {}),
    }


def _issues(payload: dict[str, Any]) -> dict[str, list[str]]:
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(payload, CONFIG_PATH)
    collected: dict[str, list[str]] = {}
    for issue in raised.value.issues:
        collected.setdefault(issue.path, []).append(issue.message)
    return collected


def _assert_names(message: str, *replacements: str) -> None:
    for replacement in replacements:
        assert replacement in message, f"{message!r} does not name {replacement!r}"


# ── the model ────────────────────────────────────────────────────────────────


def test_a_contract_is_the_whole_schema_declaration():
    config = SourceConfig.from_mapping(_payload({"contract": DECLARED_CONTRACT_PATH}), CONFIG_PATH)

    assert config.schema.contract == DECLARED_CONTRACT_PATH
    assert [schema_field.name for schema_field in fields(SchemaConfig)] == ["contract"]


def test_schema_mode_infer_is_one_named_error():
    issues = _issues(_payload({"mode": "infer"}))

    assert len(issues["schema.mode"]) == 1
    _assert_names(issues["schema.mode"][0], *SCHEMA_REPLACEMENT)


def test_schema_mode_explicit_with_a_path_is_two_named_errors():
    issues = _issues(_payload({"mode": "explicit", "path": LEGACY_SCHEMA}))

    assert len(issues["schema.mode"]) == 1
    assert len(issues["schema.path"]) == 1
    _assert_names(issues["schema.mode"][0], *SCHEMA_REPLACEMENT)
    _assert_names(issues["schema.path"][0], *SCHEMA_REPLACEMENT)


def test_a_retired_key_beside_a_contract_is_still_an_error():
    issues = _issues(_payload({"contract": DECLARED_CONTRACT_PATH, "mode": "contract"}))

    assert list(issues) == ["schema.mode"]


def test_allow_schema_evolution_is_one_named_error():
    payload = _payload(
        {"contract": DECLARED_CONTRACT_PATH}, quality={"allow_schema_evolution": True}
    )

    issues = _issues(payload)

    assert list(issues) == ["quality.allow_schema_evolution"]
    assert len(issues["quality.allow_schema_evolution"]) == 1
    _assert_names(issues["quality.allow_schema_evolution"][0], EVOLUTION_REPLACEMENT)


def test_every_retired_key_is_reported_in_one_error():
    payload = _payload({"mode": "infer"}, quality={"allow_schema_evolution": False})

    issues = _issues(payload)

    assert {"schema.mode", "quality.allow_schema_evolution"} <= set(issues)


def test_incremental_no_longer_needs_unique_fields_to_parse():
    """The order-07 key rule moved to the loader, where the contract's primaryKey is known."""
    payload = _payload({"contract": DECLARED_CONTRACT_PATH})
    payload["extraction"].update(
        {"mode": "incremental", "checkpoint_field": "id", "checkpoint_strategy": "max_value"}
    )

    config = SourceConfig.from_mapping(payload, CONFIG_PATH)

    assert config.extraction.mode == "incremental"
    assert config.quality.unique_fields == ()


# ── the active-contract rule is structural, not policy (D-15) ────────────────


def test_the_policy_no_longer_decides_contract_status():
    assert not hasattr(ValidationPolicy, "validate_schema_declaration")
    assert not hasattr(PhaseValidationPolicy, "validate_schema_declaration")
    assert "require_active_contract" not in {
        policy_field.name for policy_field in fields(PhaseValidationPolicy)
    }


@dataclass
class EverythingAllowedPolicy:
    """Relaxes every rule the policy still owns, and records each consultation."""

    calls: list[str] = field(default_factory=list)

    @property
    def allowed_source_types(self) -> frozenset[str]:
        self.calls.append("allowed_source_types")
        return frozenset({"api", "catalog", "file"})

    @property
    def allowed_strategies(self) -> frozenset[str]:
        self.calls.append("allowed_strategies")
        return frozenset({"api", "catalog", "file"})

    @property
    def allowed_federation_levels(self) -> frozenset[str]:
        self.calls.append("allowed_federation_levels")
        return frozenset({"federal", "state", "municipal"})

    def validate_strategy_pairing(
        self, source_type: str, strategy: str, issues: list[ValidationIssue]
    ) -> None:
        del source_type, strategy, issues
        self.calls.append("validate_strategy_pairing")

    def validate_public_access(self, public_access: bool, issues: list[ValidationIssue]) -> None:
        del public_access, issues
        self.calls.append("validate_public_access")


def _project_with_draft_contract(tmp_path: Path, *, enabled: bool) -> Path:
    (tmp_path / "conf").mkdir(parents=True)
    (tmp_path / "conf" / "app.yaml").write_text(
        'registry:\n  sources_dir: conf/sources\n  file_pattern: "*.yaml"\n', encoding="utf-8"
    )
    contract = write_minimal_contract(tmp_path)
    contract.write_text(
        contract.read_text(encoding="utf-8").replace("status: active", "status: draft"),
        encoding="utf-8",
    )
    payload = _payload({"contract": DECLARED_CONTRACT_PATH})
    payload["enabled"] = enabled
    source = tmp_path / "conf" / "sources" / "example" / "source.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return tmp_path


def test_an_enabled_draft_contract_is_refused_even_by_a_policy_that_relaxes_everything(tmp_path):
    policy = EverythingAllowedPolicy()

    with pytest.raises(SourceConfigValidationError) as raised:
        load_registry(_project_with_draft_contract(tmp_path, enabled=True), policy=policy)

    contract_issues = [
        issue for issue in raised.value.issues if issue.path.endswith("schema.contract")
    ]
    assert len(contract_issues) == 1
    _assert_names(contract_issues[0].message, "'active'", "'draft'")
    assert isinstance(policy, ValidationPolicy)


def test_a_disabled_source_may_carry_a_draft_contract(tmp_path):
    registry = load_registry(
        _project_with_draft_contract(tmp_path, enabled=False), policy=EverythingAllowedPolicy()
    )

    assert registry.contract_for("contract_only_example").status == "draft"
