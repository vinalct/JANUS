"""The offline-jar rule, made executable."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from janus.utils.environment import ENV_PATTERN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPS_DIR = PROJECT_ROOT / "deps"
PROFILES_DIR = PROJECT_ROOT / "conf" / "environments"
MAKEFILE = PROJECT_ROOT / "Makefile"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"

# The `spark.iceberg` keys whose value is a Maven coordinate rather than a path or a name.
PACKAGE_KEYS = ("runtime_package", "driver_package")
SHA256_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(PROJECT_ROOT)} is not visible here — the dev container mounts "
            "only src/, tests/, conf/ and data/. CI's fast gate runs this over a full checkout."
        )


def _checked_in_default(value: object) -> str | None:
    """The value a profile ships, ignoring whatever the environment happens to override it to.

    Reads the literal file, so this asserts what a fresh clone gets rather than what this
    machine's exported variables produce. `${VAR:-default}` yields the default; `${VAR}` with
    no default yields nothing to vendor.
    """

    if not isinstance(value, str):
        return None
    match = ENV_PATTERN.fullmatch(value)
    text = value if match is None else (match.group("default") or "")
    return text.strip() or None


def _jar_name(package: str) -> str:
    """`group:artifact:version` → the `group_artifact-version.jar` Ivy retrieves it as."""

    group, artifact, version = package.split(":")
    return f"{group}_{artifact}-{version}.jar"


def _pinned_packages() -> dict[str, Path]:
    """Every Maven coordinate the checked-in profiles pin, mapped to its profile."""

    packages: dict[str, Path] = {}
    for profile in sorted(PROFILES_DIR.glob("*.yaml")):
        document = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
        iceberg = document.get("spark", {}).get("iceberg", {})
        for key in PACKAGE_KEYS:
            package = _checked_in_default(iceberg.get(key))
            if package is not None:
                packages.setdefault(package, profile)
    return packages


def _vendored_jars() -> list[Path]:
    return sorted(DEPS_DIR.glob("*.jar"))


def test_every_maven_package_a_profile_pins_is_vendored_under_deps():
    """A profile that names a jar nobody committed sends the test gate to Maven."""

    _require(DEPS_DIR)

    missing = {
        package: str(profile.relative_to(PROJECT_ROOT))
        for package, profile in _pinned_packages().items()
        if not (DEPS_DIR / _jar_name(package)).is_file()
    }

    assert not missing, (
        "these pinned packages have no committed jar under deps/, so resolving them would "
        f"reach Maven: {missing}"
    )


def test_every_vendored_jar_is_seeded_by_the_makefile():
    """A jar under deps/ that `seed-ivy` never copies is invisible to a Spark session.

    The *expanded* recipe is what is read, not the Makefile text: the jar names are assembled
    from variables, so a source-text search would pass on a target that seeds nothing.
    """

    _require(DEPS_DIR)
    _require(MAKEFILE)
    if shutil.which("make") is None:
        pytest.skip("make is not installed, so the seed-ivy recipe cannot be expanded")

    recipe = subprocess.run(
        ["make", "-n", "seed-ivy"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    unseeded = [jar.name for jar in _vendored_jars() if jar.name not in recipe]

    assert not unseeded, f"deps/ jars `make seed-ivy` never copies: {unseeded}"


def test_every_vendored_jar_is_sha256_verified_by_ci_at_its_current_digest():
    """A jar swapped without updating its recorded digest is an unverified binary."""

    _require(DEPS_DIR)
    _require(CI_WORKFLOW)

    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    recorded = set(SHA256_PATTERN.findall(workflow))

    unverified = {
        jar.name: hashlib.sha256(jar.read_bytes()).hexdigest()
        for jar in _vendored_jars()
        if hashlib.sha256(jar.read_bytes()).hexdigest() not in recorded
    }

    assert not unverified, (
        "CI records no matching sha256 for these jars — either they were replaced without "
        f"updating the digest, or they are never verified: {unverified}"
    )


def test_the_sweeps_actually_found_profiles_jars_and_packages():
    """A glob that stopped matching would pass every assertion above while enforcing nothing."""

    _require(DEPS_DIR)

    assert list(PROFILES_DIR.glob("*.yaml")), "no environment profiles were swept"
    assert _vendored_jars(), "no jars were found under deps/"

    packages = _pinned_packages()
    assert packages, "no profile pins a Maven coordinate — the package sweep read nothing"
    # Both halves of the rule must be represented: the engine and the catalog driver.
    assert len(packages) >= 2, f"expected the runtime and driver packages, found {packages}"
