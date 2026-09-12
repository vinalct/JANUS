"""The offline-jar rule, made executable."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from janus.utils.environment import (
    ENV_PATTERN,
    OBJECT_STORE_KEY,
    OBJECT_STORE_PACKAGE_KEY,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPS_DIR = PROJECT_ROOT / "deps"
PROFILES_DIR = PROJECT_ROOT / "conf" / "environments"
MAKEFILE = PROJECT_ROOT / "Makefile"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"

# The `spark.iceberg` keys whose value is a Maven coordinate rather than a path or a name.
PACKAGE_KEYS = ("runtime_package", "driver_package")
#: The same kind of value, one level down in the object-store block.
NESTED_PACKAGE_KEYS = ((OBJECT_STORE_KEY, OBJECT_STORE_PACKAGE_KEY),)
SHA256_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")

UNVENDORED_PACKAGES: dict[str, str] = {
    "org.apache.iceberg:iceberg-aws-bundle:1.10.1": (
        "62,673,230 bytes (59.8 MiB) — on its own larger than everything else under deps/ "
        "put together, and needed only by the opt-in `cluster` stack, which no CI job "
        "starts. `make seed-cluster-jars` fetches it once at `up-cluster` time and "
        "sha256-verifies it against AWS_BUNDLE_SHA256 in the Makefile. The rule this "
        "protects is the CI **test gate**, which resolves nothing from Maven either way: "
        "the suites attach jars by path (`spark.jars`), never by coordinate, and no CI job "
        "runs the cluster profile at all. Vendoring it would double the size of a clone "
        "for a profile most developers never start."
    ),
}


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
        candidates = [iceberg.get(key) for key in PACKAGE_KEYS]
        for block_key, package_key in NESTED_PACKAGE_KEYS:
            block = iceberg.get(block_key)
            candidates.append(block.get(package_key) if isinstance(block, dict) else None)

        for candidate in candidates:
            package = _checked_in_default(candidate)
            if package is not None:
                packages.setdefault(package, profile)
    return packages


def _vendored_jars() -> list[Path]:
    return sorted(DEPS_DIR.glob("*.jar"))


def _expanded_recipe(target: str) -> str:
    """What `make <target>` would actually run.

    The *expanded* recipe is what is read, never the Makefile's text: jar names are
    assembled from variables, so a source-text search would pass on a target that does
    nothing with them.
    """

    return subprocess.run(
        ["make", "-n", target],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_every_maven_package_a_profile_pins_is_vendored_under_deps():
    """A profile that names a jar nobody committed sends the test gate to Maven."""

    _require(DEPS_DIR)

    missing = {
        package: str(profile.relative_to(PROJECT_ROOT))
        for package, profile in _pinned_packages().items()
        if package not in UNVENDORED_PACKAGES
        and not (DEPS_DIR / _jar_name(package)).is_file()
    }

    assert not missing, (
        "these pinned packages have no committed jar under deps/, so resolving them would "
        f"reach Maven: {missing}. Vendor the jar, or record it in UNVENDORED_PACKAGES with "
        "its measured size and how the offline rule is met without it."
    )


def test_every_unvendored_package_is_still_pinned_by_a_profile():
    """An exemption for a package nobody pins any more is fiction, not a decision."""

    _require(DEPS_DIR)

    pinned = _pinned_packages()
    unpinned = sorted(package for package in UNVENDORED_PACKAGES if package not in pinned)

    assert not unpinned, (
        "UNVENDORED_PACKAGES names package(s) no profile pins any more; remove the "
        f"entr(ies): {unpinned}"
    )


def test_no_unvendored_package_has_quietly_been_vendored():
    """If the jar is committed after all, the exemption must go — deps/ is then the rule."""

    _require(DEPS_DIR)

    vendored = sorted(
        package
        for package in UNVENDORED_PACKAGES
        if (DEPS_DIR / _jar_name(package)).is_file()
    )

    assert not vendored, (
        "these packages are committed under deps/ *and* listed as deliberately "
        f"unvendored: {vendored}. Drop the UNVENDORED_PACKAGES entry."
    )


def test_every_unvendored_package_is_fetched_and_verified_by_the_makefile():
    """The other half of the exemption: something must still pin the bytes.

    A package outside deps/ is only acceptable because a named target fetches it against a
    recorded sha256. If the Makefile stops naming it, the exemption stops holding.
    """

    _require(MAKEFILE)
    if shutil.which("make") is None:
        pytest.skip("make is not installed, so the seed-cluster-jars recipe cannot be expanded")

    recipe = _expanded_recipe("seed-cluster-jars")
    unfetched = sorted(
        package for package in UNVENDORED_PACKAGES if _jar_name(package) not in recipe
    )

    assert not unfetched, (
        f"`make seed-cluster-jars` never fetches these unvendored packages: {unfetched}"
    )
    assert SHA256_PATTERN.search(recipe), (
        "the fetch recipe records no sha256, so nothing verifies the bytes it downloads"
    )


def test_every_unvendored_package_carries_a_written_reason():
    """An exemption with no reason is a shortcut wearing a decision's clothes."""

    unreasoned = sorted(
        package for package, reason in UNVENDORED_PACKAGES.items() if not reason.strip()
    )

    assert not unreasoned, f"UNVENDORED_PACKAGES entr(ies) with no reason: {unreasoned}"


def test_every_vendored_jar_is_seeded_by_the_makefile():
    """A jar under deps/ that `seed-ivy` never copies is invisible to a Spark session.

    The *expanded* recipe is what is read, not the Makefile text: the jar names are assembled
    from variables, so a source-text search would pass on a target that seeds nothing.
    """

    _require(DEPS_DIR)
    _require(MAKEFILE)
    if shutil.which("make") is None:
        pytest.skip("make is not installed, so the seed-ivy recipe cannot be expanded")

    recipe = _expanded_recipe("seed-ivy")
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
    
    assert len(packages) >= 3, f"expected the runtime and driver packages, found {packages}"
    assert [package for package in packages if package in UNVENDORED_PACKAGES], (
        "the nested package sweep found no object-store package — either the profile "
        "stopped pinning one or the sweep stopped reading the block"
    )
