"""The exact runtime set, including order-15's deliberate second-engine decision."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
EXPECTED_RUNTIME_DEPENDENCIES = ("PyYAML", "certifi", "pyspark", "pyiceberg")
EXPECTED_DEV_DEPENDENCIES = (
    "jsonschema",
    "mypy",
    "pip-audit",
    "pytest",
    "pytest-cov",
    "ruff",
)
CI_ONLY_DEV_DEPENDENCIES = {"pip-audit"}

PYICEBERG_RUNTIME_REQUIREMENT = (
    "pyiceberg[pyarrow,pyiceberg-core,sql-postgres,sql-sqlite,s3fs]==0.12.0"
)

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _pyproject() -> dict:
    if not PYPROJECT.exists():  # pragma: no cover - the container mounts it read-only
        pytest.skip(f"{PYPROJECT.name} is not readable from here")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _distribution(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    assert match is not None, f"unparsable requirement: {requirement!r}"
    return match.group(1).lower()


def _runtime() -> list[str]:
    return list(_pyproject()["project"]["dependencies"])


def _dev() -> list[str]:
    return list(_pyproject()["project"]["optional-dependencies"]["dev"])


def test_the_sweep_actually_read_the_declarations():
    """Empty lists would make every assertion below vacuously true."""

    assert _runtime(), "no runtime dependencies were parsed — pyproject.toml moved"
    assert _dev(), "no dev dependencies were parsed — the optional-dependencies table moved"


def test_the_runtime_dependencies_are_the_four_the_project_ships():
    assert tuple(_distribution(item) for item in _runtime()) == tuple(
        name.lower() for name in EXPECTED_RUNTIME_DEPENDENCIES
    )


def test_the_dev_dependencies_include_the_ci_auditor():
    assert tuple(_distribution(item) for item in _dev()) == EXPECTED_DEV_DEPENDENCIES


def test_order_15_deliberately_promotes_pyiceberg_to_runtime():
    """Overrule the old prohibition in the order authorized to pay its runtime cost."""

    assert PYICEBERG_RUNTIME_REQUIREMENT in _runtime()
    assert all(_distribution(item) != "pyiceberg" for item in _dev()), (
        "pyiceberg must have one declaration: moved it from dev to runtime"
    )


def test_every_runtime_dependency_is_declared_once_and_pinned_exactly():
    runtime = _runtime()
    distributions = [_distribution(item) for item in runtime]

    assert len(distributions) == len(set(distributions)), (
        "each runtime distribution must have one declaration"
    )
    assert all("==" in item for item in runtime), (
        f"every runtime dependency must be pinned exactly: {runtime!r}"
    )


def test_the_declared_pins_and_the_image_requirements_agree():
    """The image installs all declared pins except tooling that exists only for CI."""

    if not REQUIREMENTS.exists():
        pytest.skip("requirements.txt is not readable from here")

    installed = {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    image_requirements = {
        item for item in _runtime() + _dev() if _distribution(item) not in CI_ONLY_DEV_DEPENDENCIES
    }
    missing = sorted(image_requirements - installed)
    ci_only_installed = sorted(
        item for item in installed if _distribution(item) in CI_ONLY_DEV_DEPENDENCIES
    )

    assert not missing, (
        "requirements.txt does not install every image pin pyproject.toml declares: "
        f"{missing}. Add each to requirements.txt too — the image installs that file, so a "
        "pin missing from it is a dependency the container never gets."
    )
    assert not ci_only_installed, (
        f"CI-only dependencies leaked into the runtime image: {ci_only_installed}"
    )
