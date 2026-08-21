"""The policy seam reaches the loader, which is the only entry point anything real uses."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

# The registry suite is not a package (no ``__init__.py``), so pytest puts this directory
# on ``sys.path`` and the sibling module is imported by plain name.
from test_source_registry import (
    _create_project,
    _grouped_sources_yaml,
    _valid_source_yaml,
)

from janus.models.source_config import (
    DEFAULT_VALIDATION_POLICY,
    STRATEGY_REGISTRY,
    SourceConfigValidationError,
)
from janus.registry import SourceRegistry, load_registry
from janus.registry.loader import _load_grouped_source_configs, _load_source_configs

PUBLIC_ACCESS_MESSAGE = (
    "must be true because JANUS only supports public federal sources in phase 1"
)

PUBLIC_SOURCES_ALLOWED = replace(DEFAULT_VALIDATION_POLICY, require_public_access=False)

STATE_SOURCES_ALLOWED = replace(
    DEFAULT_VALIDATION_POLICY,
    federation_levels=frozenset({"federal", "state"}),
)


def _project_with_one_source(tmp_path: Path, **overrides: object) -> Path:
    """A one-source project tree whose single config carries `overrides`."""
    return _create_project(
        tmp_path,
        {
            "example/source.yaml": _valid_source_yaml(
                "policy_seam_source",
                enabled=True,
                **overrides,  # type: ignore[arg-type]
            )
        },
    )


def test_load_registry_defaults_to_the_strict_policy(tmp_path):
    """AC-1 at the loader: omitting the policy is the phase-1 posture, unchanged."""
    project_root = _project_with_one_source(tmp_path, public_access=False)

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert [(issue.path, issue.message) for issue in exc_info.value.issues] == [
        ("public_access", PUBLIC_ACCESS_MESSAGE)
    ]


def test_load_registry_accepts_a_permissive_policy(tmp_path):
    """AC-2 at registry level: the tree that just failed loads under a relaxed policy.

    Same bytes on disk, same loader, no edit to ``source_config.py`` — only the policy
    object differs, which is the whole claim this order makes.
    """
    project_root = _project_with_one_source(tmp_path, public_access=False)

    registry = load_registry(project_root, policy=PUBLIC_SOURCES_ALLOWED)

    assert registry.get_source("policy_seam_source").public_access is False


def test_permissive_policy_relaxes_only_what_it_names(tmp_path):
    """A policy is a scalpel, not an off switch.

    The config breaks two rules at once. The policy names one of them, so the other must
    still be reported — and reported alone, with no trace of the relaxed rule.
    """
    project_root = _project_with_one_source(
        tmp_path,
        public_access=False,
        strategy_variant="telepathy_api",
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root, policy=PUBLIC_SOURCES_ALLOWED)

    assert [issue.path for issue in exc_info.value.issues] == ["strategy_variant"]
    assert exc_info.value.issues[0].message == (
        f"must be one of: {STRATEGY_REGISTRY.describe_variants('api')}"
    )


def test_grouped_file_reports_prefixed_issues_under_a_custom_policy(tmp_path):
    """FR-3 ergonomics survive the new parameter: entry index still prefixes the path.

    Both entries would fail the default policy, so a clean report on the first also proves
    the custom policy reached every entry rather than only the first.
    """
    project_root = _create_project(
        tmp_path,
        {
            "example/group.yaml": _grouped_sources_yaml(
                _valid_source_yaml("group_ok", enabled=True, public_access=False),
                _valid_source_yaml(
                    "group_bad",
                    enabled=True,
                    public_access=False,
                    strategy_variant="telepathy_api",
                ),
            )
        },
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root, policy=PUBLIC_SOURCES_ALLOWED)

    assert [issue.path for issue in exc_info.value.issues] == ["sources[1].strategy_variant"]


def test_policy_does_not_leak_between_loads(tmp_path):
    """NFR-2: no global state. The relaxed load must not colour the one after it.

    Both loads are in one test, in this order, on purpose — a module-level ``_ACTIVE_POLICY``
    would keep this green only while someone remembered to reset the setter, which is the
    fragility being ruled out rather than merely tested for.
    """
    project_root = _project_with_one_source(tmp_path, public_access=False)

    permissive = load_registry(project_root, policy=PUBLIC_SOURCES_ALLOWED)
    assert permissive.get_source("policy_seam_source").public_access is False

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert [issue.path for issue in exc_info.value.issues] == ["public_access"]


def test_federation_policy_relaxation_end_to_end(tmp_path):
    """PRD Q2's payoff, proven where it pays off: a state source onboarded by policy alone."""
    project_root = _project_with_one_source(tmp_path, federation_level="state")

    with pytest.raises(SourceConfigValidationError):
        load_registry(project_root)

    registry = load_registry(project_root, policy=STATE_SOURCES_ALLOWED)

    assert registry.get_source("policy_seam_source").federation_level == "state"


@pytest.mark.parametrize(
    "loader_callable",
    [
        load_registry,
        SourceRegistry.load,
        _load_source_configs,
        _load_grouped_source_configs,
    ],
    ids=[
        "load_registry",
        "SourceRegistry.load",
        "_load_source_configs",
        "_load_grouped_source_configs",
    ],
)
def test_loader_signatures_are_keyword_only(loader_callable):
    """Every threaded parameter is keyword-only with its canonical default.

    This is what makes AC-1 structural rather than hopeful: no positional call site in the
    tree can be silently handed a policy it never asked for.
    """
    parameters = inspect.signature(loader_callable).parameters

    assert parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["strategy_registry"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["policy"].default is DEFAULT_VALIDATION_POLICY
    assert parameters["strategy_registry"].default is STRATEGY_REGISTRY


def test_the_loader_does_not_name_the_source_registry_twice(tmp_path):
    """The naming asymmetry is load-bearing: ``registry`` here would mean two things.

    ``from_mapping`` takes ``registry``; the loader takes ``strategy_registry`` because
    ``AppConfig.registry``, ``RegistrySettings`` and ``SourceRegistry`` already own that
    word in this module. Passing it under the loader's name must reach ``from_mapping``
    under its own.
    """
    project_root = _project_with_one_source(tmp_path, strategy_variant="telepathy_api")
    extended = STRATEGY_REGISTRY.with_family(
        "api",
        STRATEGY_REGISTRY.variants_for("api") | {"telepathy_api"},
    )

    with pytest.raises(SourceConfigValidationError):
        load_registry(project_root)

    registry = load_registry(project_root, strategy_registry=extended)

    assert registry.get_source("policy_seam_source").strategy_variant == "telepathy_api"
