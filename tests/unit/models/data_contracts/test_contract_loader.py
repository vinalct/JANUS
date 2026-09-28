"""ODCS pin and schema-validation guardrails for JANUS data contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import FormatChecker
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ODCS_SCHEMA_DIR = PROJECT_ROOT / "docs" / "schemas" / "odcs"
ODCS_SCHEMA = ODCS_SCHEMA_DIR / "odcs-json-schema-v3.2.0.json"
ODCS_SCHEMA_SHA256 = ODCS_SCHEMA_DIR / "odcs-json-schema-v3.2.0.sha256"
CONTRACT_FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "contracts"

FIXTURES = tuple(
    sorted(
        path.relative_to(CONTRACT_FIXTURES).as_posix()
        for directory in ("hostile", "baseline")
        for path in (CONTRACT_FIXTURES / directory).glob("*.yaml")
    )
)
FIXTURE_ANCHORS = {
    "hostile/base.yaml",
    "hostile/base_v2.yaml",
    "hostile/corrupt_name.yaml",
    "baseline/concurrency_contract.yaml",
    "baseline/incremental_upsert_fixture.yaml",
    "baseline/multi_batch_run_keys.yaml",
}


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _odcs_validator() -> Validator:
    schema = json.loads(ODCS_SCHEMA.read_text(encoding="utf-8"))
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema, format_checker=FormatChecker())


def _render_errors(validator: Validator, instance: Any) -> list[str]:
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in errors
    ]


def test_pinned_odcs_schema_sha256_matches_the_sidecar():
    sidecar_parts = ODCS_SCHEMA_SHA256.read_text(encoding="utf-8").split()

    assert sidecar_parts == [
        "edb41f33ec46e84780e99872ab2bd67f074959d2bf3e9c9fc54e61f8982b0d93",
        "docs/schemas/odcs/odcs-json-schema-v3.2.0.json",
    ]
    pinned_path = PROJECT_ROOT / sidecar_parts[1]

    assert pinned_path == ODCS_SCHEMA
    assert hashlib.sha256(pinned_path.read_bytes()).hexdigest() == sidecar_parts[0]


def test_every_checked_in_contract_validates_against_the_pinned_odcs_schema():
    contract_paths = sorted((PROJECT_ROOT / "conf" / "contracts").glob("**/*.yaml"))
    validator = _odcs_validator()

    assert contract_paths, "conf/contracts/**/*.yaml must contain at least one contract"
    failures = {
        str(path.relative_to(PROJECT_ROOT)): errors
        for path in contract_paths
        if (errors := _render_errors(validator, _load_yaml(path)))
    }

    assert failures == {}


@pytest.mark.parametrize(
    ("fixture_name", "expected_valid"),
    [
        ("odcs_invalid_missing_kind.yaml", False),
        ("odcs_invalid_customproperties_mapping.yaml", False),
        ("odcs_invalid_two_schemas.yaml", True),
        *((fixture_name, True) for fixture_name in FIXTURES),
    ],
)
def test_odcs_validator_rejects_the_hostile_fixtures(fixture_name, expected_valid):
    errors = _render_errors(_odcs_validator(), _load_yaml(CONTRACT_FIXTURES / fixture_name))

    assert (errors == []) is expected_valid


def test_the_order_19_fixture_sweep_matches_its_anchors():
    """A glob that matched nothing would drop every order-19 case above without a failure."""
    assert set(FIXTURES) >= FIXTURE_ANCHORS
