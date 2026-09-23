"""The image, the compose stack and the secrets file, swept."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"
ENTRYPOINT = PROJECT_ROOT / "docker" / "entrypoint.sh"
COMPOSE = PROJECT_ROOT / "docker" / "docker-compose.yml"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
GITIGNORE = PROJECT_ROOT / ".gitignore"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
SOURCES_DIR = PROJECT_ROOT / "conf" / "sources"

CLUSTER_SERVICES = ("minio", "postgres", "nessie")

BIND_PREFIX = "${JANUS_CLUSTER_BIND_ADDRESS:-127.0.0.1}:"

MINIMUM_PUBLISHED_PORTS = 4
MINIMUM_ENV_VAR_NAMES = 2

DIGEST_PATTERN = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}\s*$", re.MULTILINE)
PASSWD_CHMOD_PATTERN = re.compile(r"chmod.*?/etc/passwd")
PASSWD_WRITE_PATTERN = re.compile(r"(>>?\s*/etc/passwd)|(tee\s+(-a\s+)?/etc/passwd)")

#: Every auth key under ``conf/sources/**`` that names an environment variable.
ENV_VAR_KEYS = ("env_var", "username_env_var", "password_env_var")


def _read(path: Path, *, why: str) -> str:
    if not path.is_file():
        pytest.skip(f"{path.relative_to(PROJECT_ROOT)} is not visible here ({why})")
    return path.read_text(encoding="utf-8")


def _source_documents() -> list[dict]:
    """Every source entry under ``conf/sources``, whether the file holds one or a list."""
    documents: list[dict] = []
    for path in sorted(SOURCES_DIR.rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        entries = payload.get("sources")
        if isinstance(entries, list):
            documents.extend(entry for entry in entries if isinstance(entry, dict))
        else:
            documents.append(payload)
    return documents


def _declared_env_var_names() -> set[str]:
    names: set[str] = set()
    for document in _source_documents():
        auth = document.get("access", {}).get("auth")
        if not isinstance(auth, dict):
            continue
        for key in ENV_VAR_KEYS:
            value = auth.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    return names


# ---------------------------------------------------------------------------
# FR-8 — the image


def test_the_image_declares_a_non_root_user_after_its_last_build_step():
    """``USER`` must come after the last ``RUN``, or the build steps would run unprivileged."""
    text = _read(DOCKERFILE, why="the dev container does not mount docker/")
    lines = [line.rstrip() for line in text.splitlines()]

    user_indexes = [index for index, line in enumerate(lines) if line.startswith("USER ")]
    run_indexes = [index for index, line in enumerate(lines) if line.startswith("RUN ")]

    assert user_indexes, "the image carries no USER directive and therefore runs as root"
    assert lines[user_indexes[-1]].split(maxsplit=1)[1].strip() == "1000:1000"
    assert user_indexes[-1] > max(run_indexes), (
        "USER must follow the last RUN so the build itself is not degraded"
    )


def test_the_image_never_makes_etc_passwd_world_writable():
    """Any process in the container could otherwise add or alter an account entry."""
    text = _read(DOCKERFILE, why="the dev container does not mount docker/")

    offenders = [line.strip() for line in text.splitlines() if PASSWD_CHMOD_PATTERN.search(line)]

    assert offenders == [], f"the image relaxes permissions on /etc/passwd: {offenders}"


def test_the_entrypoint_never_writes_to_etc_passwd():
    """FR-8 replaces the append with ``nss_wrapper``: a per-process view, not a system file edit."""
    text = _read(ENTRYPOINT, why="the dev container does not mount docker/")

    offenders = [line.strip() for line in text.splitlines() if PASSWD_WRITE_PATTERN.search(line)]

    assert offenders == [], f"the entrypoint still edits /etc/passwd: {offenders}"


def test_the_image_resolves_an_arbitrary_uid_without_editing_system_accounts():
    """Docker uid overrides use a per-process NSS view that is inherited by child processes."""
    dockerfile = _read(DOCKERFILE, why="the dev container does not mount docker/")
    entrypoint = _read(ENTRYPOINT, why="the dev container does not mount docker/")

    assert "libnss-wrapper" in dockerfile
    for marker in ("NSS_WRAPPER_PASSWD", "NSS_WRAPPER_GROUP", "LD_PRELOAD"):
        assert marker in entrypoint, f"the entrypoint does not configure {marker}"


def test_the_base_image_is_pinned_by_digest():
    """A tag moves. Shared with ``test_ci_supply_chain.py``, which pins it from the CI side."""
    text = _read(DOCKERFILE, why="the dev container does not mount docker/")

    assert DIGEST_PATTERN.search(text), (
        "the FROM line carries no @sha256: digest, so the base image is whatever the tag "
        "points at on build day"
    )


# ---------------------------------------------------------------------------
# FR-9 — the compose stack


def _published_port_entries() -> list[tuple[str, str]]:
    text = _read(COMPOSE, why="the dev container does not mount docker/")
    compose = yaml.safe_load(text)
    entries: list[tuple[str, str]] = []
    for name, service in (compose.get("services") or {}).items():
        if not isinstance(service, dict):
            continue
        for entry in service.get("ports") or []:
            entries.append((name, str(entry)))
    return entries


def test_every_published_cluster_port_binds_to_loopback_by_default():
    """A dev laptop on a shared network must not expose an object store and a database."""
    entries = _published_port_entries()
    cluster_entries = [
        (service, entry) for service, entry in entries if service in CLUSTER_SERVICES
    ]

    assert len(cluster_entries) >= MINIMUM_PUBLISHED_PORTS, (
        f"only {len(cluster_entries)} published cluster ports found in {entries}; the sweep "
        "is reading the wrong file or the wrong services"
    )
    assert {service for service, _ in cluster_entries} >= set(CLUSTER_SERVICES)

    unbound = [entry for _, entry in cluster_entries if not entry.startswith(BIND_PREFIX)]
    assert unbound == [], (
        f"published on every interface: {unbound}. Prefix each with {BIND_PREFIX!r} so one "
        "variable widens them deliberately."
    )


def test_the_janus_service_publishes_nothing():
    """Green on arrival: a pin, not a red test.

    The ``janus`` service publishes nothing today, and FR-9 must not "fix" it by giving it a
    loopback-bound port. The driver is bound to loopback and the Spark UI is disabled.
    """
    entries = _published_port_entries()

    assert [entry for service, entry in entries if service == "janus"] == []


# ---------------------------------------------------------------------------
# FR-12 — secrets on disk


def test_env_example_lists_every_token_the_checked_in_sources_need():
    """A contributor should learn a token exists from the example file, not from a failed run."""
    text = _read(ENV_EXAMPLE, why="the dev container does not mount .env.example")
    declared = _declared_env_var_names()

    assert len(declared) >= MINIMUM_ENV_VAR_NAMES, (
        f"only {sorted(declared)} collected from conf/sources/**; the sweep found nothing to "
        "check against"
    )
    missing = sorted(name for name in declared if f"{name}=" not in text)
    assert missing == [], f".env.example does not list: {missing}"


def test_env_example_says_how_to_protect_the_real_file_and_ends_with_a_newline():
    """SEC-10 is a developer's ``.env`` at mode 0644; the remedy belongs where they copy from."""
    text = _read(ENV_EXAMPLE, why="the dev container does not mount .env.example")

    assert "chmod 600" in text
    assert text.endswith("\n")


def test_the_secrets_files_are_ignored_by_both_git_and_docker():
    """Green-looking today, pinned here because FR-12 edits both files' neighbourhood."""
    gitignore = _read(GITIGNORE, why="not visible here")
    dockerignore = _read(DOCKERIGNORE, why="not visible here")

    for text, name in ((gitignore, ".gitignore"), (dockerignore, ".dockerignore")):
        lines = {line.strip() for line in text.splitlines()}
        assert ".env" in lines, f"{name} does not ignore .env"
        assert "conf/environments/*.env" in lines, (
            f"{name} does not ignore the generated cluster secrets"
        )


# ---------------------------------------------------------------------------
# The sweep must actually read the repository


def test_the_sweep_reads_the_files_it_claims_to_check():
    """Non-emptiness for the whole module: a wrong root would skip every case above silently."""
    visible = {
        path.name: path.is_file()
        for path in (DOCKERFILE, ENTRYPOINT, COMPOSE, ENV_EXAMPLE, GITIGNORE, DOCKERIGNORE)
    }

    assert SOURCES_DIR.is_dir(), "conf/sources is not visible; the env-var sweep is vacuous"
    assert len(_source_documents()) >= 20, (
        f"only {len(_source_documents())} source entries parsed from conf/sources/**"
    )
    assert any(visible.values()), (
        f"none of the swept infrastructure files is visible here: {visible}. Run this module "
        "on the host."
    )
