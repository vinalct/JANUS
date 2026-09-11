"""The runtime dependency list is three entries, and a second engine is not one of them."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
EXPECTED_RUNTIME_DEPENDENCIES = ("PyYAML", "certifi", "pyspark")
DEV_ONLY_DISTRIBUTIONS = ("pyiceberg",)

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


def test_the_runtime_dependencies_are_the_three_the_project_ships():
    assert tuple(_distribution(item) for item in _runtime()) == tuple(
        name.lower() for name in EXPECTED_RUNTIME_DEPENDENCIES
    )


@pytest.mark.parametrize("distribution", DEV_ONLY_DISTRIBUTIONS)
def test_a_dev_only_distribution_never_reaches_the_runtime_list(distribution: str):
    runtime = [_distribution(item) for item in _runtime()]

    assert distribution not in runtime, (
        f"{distribution} is a dev/test dependency: it exists to prove the catalog is "
        "engine-neutral, not to be imported by the product. Promoting it to a runtime "
        "dependency is decision and needs its own order."
    )


@pytest.mark.parametrize("distribution", DEV_ONLY_DISTRIBUTIONS)
def test_a_dev_only_distribution_is_declared_and_pinned(distribution: str):
    """Declared, and pinned exactly — an unpinned engine makes the evidence unreproducible."""

    declared = [item for item in _dev() if _distribution(item) == distribution]

    assert len(declared) == 1, f"{distribution} must be declared once in the dev extras"
    assert "==" in declared[0], f"{distribution} must be pinned exactly: {declared[0]!r}"


def test_the_declared_pins_and_the_images_requirements_agree():
    """Two files declare the same pins, and a dependency must be added to both."""

    if not REQUIREMENTS.exists():  
        pytest.skip("requirements.txt is not readable from here")

    installed = {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(set(_runtime() + _dev()) - installed)

    assert not missing, (
        "requirements.txt does not install every pin pyproject.toml declares: "
        f"{missing}. Add each to requirements.txt too — the image installs that file, so a "
        "pin missing from it is a dependency the container never gets."
    )
