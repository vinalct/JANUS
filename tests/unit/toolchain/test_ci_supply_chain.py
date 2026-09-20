"""The workflow's own supply chain, swept."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
DEPENDABOT = PROJECT_ROOT / ".github" / "dependabot.yml"
AUDIT_IGNORE = PROJECT_ROOT / ".github" / "pip-audit-ignore.txt"
DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"


RED_UNTIL_18 = pytest.mark.xfail(
    strict=True,
    reason="red until: SHA-pinned actions, pip-audit, Dependabot, image digest",
)

PINNED_USES = re.compile(
    r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}\s*#\s*v\S+$"
)
USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>.+?)\s*$")

#: ``<ID> <YYYY-MM-DD> <reason…>`` — an entry without a date is a lint failure, not a debate.
IGNORE_ENTRY = re.compile(r"^(?P<id>\S+)\s+(?P<expiry>\d{4}-\d{2}-\d{2})\s+(?P<reason>\S.*)$")

DIGEST_PATTERN = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}\s*$", re.MULTILINE)

#: Non-emptiness floor: the workflow referenced eight actions when this was written.
MINIMUM_USES = 8

REQUIRED_ECOSYSTEMS = {"pip", "github-actions", "docker"}


def _workflow_paths() -> list[Path]:
    if not WORKFLOWS_DIR.is_dir():
        pytest.skip(".github/workflows is not visible here (run this module on the host)")
    return sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))


def _uses_references() -> list[tuple[str, int, str]]:
    """``(file, line number, reference)`` for every ``uses:`` in every workflow.

    Read from the raw text rather than the parsed YAML: ``yaml.safe_load`` drops the trailing
    comment that carries the version, which is half of what this sweep checks.
    """
    references: list[tuple[str, int, str]] = []
    for path in _workflow_paths():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = USES_LINE.match(line)
            if match:
                references.append((path.name, number, match.group("ref")))
    return references


def _jobs() -> dict:
    jobs: dict = {}
    for path in _workflow_paths():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        jobs.update(workflow.get("jobs") or {})
    return jobs


def test_the_sweep_actually_reads_the_workflows():
    """A glob that matched nothing would pass every assertion below it."""
    references = _uses_references()

    assert _workflow_paths(), "no workflow files found"
    assert len(references) >= MINIMUM_USES, (
        f"only {len(references)} `uses:` references found; the sweep is reading the wrong "
        "files and the pinning assertion is vacuous"
    )


@RED_UNTIL_18
def test_every_action_is_pinned_to_a_full_commit_sha():
    """A tag is a moving target that runs with the repository's token."""
    unpinned = [
        f"{name}:{number}: {reference}"
        for name, number, reference in _uses_references()
        if not PINNED_USES.match(reference)
    ]

    assert unpinned == [], (
        "mutable action reference(s):\n"
        + "\n".join(f"  {line}" for line in unpinned)
        + "\nPin each to a full commit SHA with the version in a trailing comment, e.g. "
        "`actions/checkout@<40 hex>  # v4.2.2`."
    )


@RED_UNTIL_18
def test_the_fast_job_audits_its_dependencies():
    """The gate that would have named a CVE before a release note did."""
    fast = _jobs().get("fast")

    assert fast is not None, f"no `fast` job found; jobs are {sorted(_jobs())}"
    audit_steps = [
        step
        for step in fast.get("steps") or []
        if isinstance(step, dict) and "pip-audit" in str(step.get("run", ""))
    ]
    assert audit_steps, "the fast job runs no pip-audit step"


@RED_UNTIL_18
def test_dependabot_covers_pip_actions_and_docker():
    """Pinning without a bot to move the pins is how a pin becomes a stale dependency."""
    if not DEPENDABOT.is_file():
        raise AssertionError(
            ".github/dependabot.yml does not exist; the SHA pins have nothing keeping them "
            "current"
        )
    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8")) or {}

    ecosystems = {
        entry.get("package-ecosystem")
        for entry in config.get("updates") or []
        if isinstance(entry, dict)
    }
    assert ecosystems >= REQUIRED_ECOSYSTEMS, (
        f"dependabot covers {sorted(ecosystems)}; missing "
        f"{sorted(REQUIRED_ECOSYSTEMS - ecosystems)}"
    )


@RED_UNTIL_18
def test_every_audit_allowlist_entry_carries_a_reason_and_an_unexpired_date():
    """risk 6: an advisory with no fix is acceptable for a while, never forever."""
    if not AUDIT_IGNORE.is_file():
        raise AssertionError(
            ".github/pip-audit-ignore.txt does not exist; the audit step has no allowlist "
            "contract and an accepted advisory would have to be silenced in the workflow"
        )

    today = datetime.now(tz=UTC).date()
    malformed: list[str] = []
    expired: list[str] = []
    for number, raw in enumerate(AUDIT_IGNORE.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = IGNORE_ENTRY.match(line)
        if match is None:
            malformed.append(f"{number}: {line}")
            continue
        if date.fromisoformat(match.group("expiry")) < today:
            expired.append(f"{number}: {match.group('id')} expired {match.group('expiry')}")

    assert malformed == [], f"allowlist entries must be `<ID> <YYYY-MM-DD> <reason>`: {malformed}"
    assert expired == [], f"allowlist entries have outlived their reason: {expired}"


@RED_UNTIL_18
def test_the_base_image_is_pinned_by_digest():
    """Asserted from the CI side too: Dependabot's ``docker`` ecosystem is what moves it."""
    if not DOCKERFILE.is_file():
        pytest.skip("docker/Dockerfile is not visible here (run this module on the host)")

    assert DIGEST_PATTERN.search(DOCKERFILE.read_text(encoding="utf-8"))


@RED_UNTIL_18
def test_the_allowlist_helper_prints_flags_and_refuses_an_expired_entry(tmp_path):
    """The helper is what makes the expiry rule fail CI by itself rather than by review."""
    from tests.support.pip_audit_ignores import render_ignore_flags

    good = tmp_path / "good.txt"
    good.write_text(
        "# accepted advisories\nGHSA-aaaa-bbbb-cccc 2099-01-01 no fix upstream yet\n",
        encoding="utf-8",
    )
    assert render_ignore_flags(good) == ["--ignore-vuln", "GHSA-aaaa-bbbb-cccc"]

    expired = tmp_path / "expired.txt"
    expired.write_text("GHSA-dddd-eeee-ffff 2020-01-01 stale\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        render_ignore_flags(expired)

    malformed = tmp_path / "malformed.txt"
    malformed.write_text("GHSA-gggg-hhhh-iiii no-date-here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        render_ignore_flags(malformed)
