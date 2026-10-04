from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.maintenance_zone import PROJECT_ROOT, build_maintenance_zone

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


@pytest.fixture
def now():
    return NOW


@pytest.fixture
def policy_config():
    return json.loads((PROJECT_ROOT / "tests/fixtures/maintenance/policy.json").read_text())


@pytest.fixture
def planted(tmp_path: Path, now):
    return build_maintenance_zone(tmp_path / "metadata", now)
