"""Two structural keys, validated for shape and never for scope."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from janus.models import SourceConfig
from janus.models.config.constants import (
    DEFAULT_MAX_ARCHIVE_RATIO,
    DEFAULT_MAX_ARCHIVE_TOTAL_BYTES,
    DEFAULT_MAX_PAYLOAD_BYTES,
    DEFAULT_MAX_REDIRECTS,
    MAX_REDIRECTS_CEILING,
)
from janus.models.config.issues import SourceConfigValidationError, ValidationIssue
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GIB = 1024**3

EXPECTED_DEFAULTS = {
    "max_payload_bytes": DEFAULT_MAX_PAYLOAD_BYTES,
    "max_redirects": DEFAULT_MAX_REDIRECTS,
    "max_archive_member_bytes": DEFAULT_MAX_PAYLOAD_BYTES,
    "max_archive_total_bytes": DEFAULT_MAX_ARCHIVE_TOTAL_BYTES,
    "max_archive_ratio": DEFAULT_MAX_ARCHIVE_RATIO,
}


class SpyPolicy:
    """The strict policy, plus a record of every method the model consulted."""

    def __init__(self) -> None:
        from janus.models.config.policy import PhaseValidationPolicy

        self._inner = PhaseValidationPolicy()
        self.calls: list[str] = []

    @property
    def allowed_source_types(self) -> frozenset[str]:
        self.calls.append("allowed_source_types")
        return self._inner.allowed_source_types

    @property
    def allowed_strategies(self) -> frozenset[str]:
        self.calls.append("allowed_strategies")
        return self._inner.allowed_strategies

    @property
    def allowed_federation_levels(self) -> frozenset[str]:
        self.calls.append("allowed_federation_levels")
        return self._inner.allowed_federation_levels

    def validate_strategy_pairing(self, source_type, strategy, issues) -> None:
        self.calls.append("validate_strategy_pairing")
        self._inner.validate_strategy_pairing(source_type, strategy, issues)

    def validate_public_access(self, public_access, issues) -> None:
        self.calls.append("validate_public_access")
        self._inner.validate_public_access(public_access, issues)

    def validate_schema_declaration(
        self, *, enabled, contract_status, issues
    ) -> None:
        self.calls.append("validate_schema_declaration")
        self._inner.validate_schema_declaration(
            enabled=enabled, contract_status=contract_status, issues=issues
        )


def _source_mapping(**access_overrides: Any) -> dict[str, Any]:
    access: dict[str, Any] = {
        "url": "https://example.gov.br/dados/",
        "method": "GET",
        "format": "binary",
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": {"type": "none"},
        "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 5},
    }
    access.update(access_overrides)
    return {
        "source_id": "limits_fixture",
        "name": "limits fixture",
        "owner": "janus",
        "enabled": True,
        "source_type": "file",
        "strategy": "file",
        "strategy_variant": "static_file",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": access,
        "extraction": {
            "mode": "full_refresh",
            "checkpoint_strategy": "none",
            "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
        },
        "schema": {"mode": "infer"},
        "spark": {"input_format": "csv", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/limits_fixture", "format": "binary"},
            "bronze": {"path": "data/bronze/example/limits_fixture", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/limits_fixture", "format": "json"},
        },
        "quality": {"allow_schema_evolution": True},
    }


def _build(tmp_path: Path, *, policy=None, **access_overrides: Any) -> SourceConfig:
    kwargs = {"policy": policy} if policy is not None else {}
    return SourceConfig.from_mapping(
        _source_mapping(**access_overrides),
        tmp_path / "conf" / "sources" / "limits_fixture.yaml",
        **kwargs,
    )


def _issues(tmp_path: Path, **access_overrides: Any) -> tuple[ValidationIssue, ...]:
    with pytest.raises(SourceConfigValidationError) as excinfo:
        _build(tmp_path, **access_overrides)
    return excinfo.value.issues


def _issues_for(issues: tuple[ValidationIssue, ...], prefix: str) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.path.startswith(prefix)]


# ---------------------------------------------------------------------------
# access.allowed_hosts


def test_allowed_hosts_defaults_to_the_empty_tuple(tmp_path):
    """Absent means same origin — the default is a rule, not a missing value."""
    assert _build(tmp_path).access.allowed_hosts == ()


def test_allowed_hosts_normalizes_case_and_whitespace_and_keeps_the_wildcard(tmp_path):
    """One spelling reaches the policy, so the resolver never lower-cases at match time."""
    config = _build(
        tmp_path, allowed_hosts=[" CDN.Example.GOV.br ", "*.example.gov.br"]
    )

    assert config.access.allowed_hosts == ("cdn.example.gov.br", "*.example.gov.br")


ALLOWED_HOSTS_REJECTIONS = (
    ("not_a_list", "cdn.example.gov.br", "access.allowed_hosts"),
    ("empty_entry", ["  "], "access.allowed_hosts[0]"),
    ("non_string_entry", [7], "access.allowed_hosts[0]"),
    ("carries_a_scheme", ["https://cdn.example.gov.br"], "access.allowed_hosts[0]"),
    ("carries_a_port", ["cdn.example.gov.br:8443"], "access.allowed_hosts[0]"),
    ("carries_a_path", ["cdn.example.gov.br/dados"], "access.allowed_hosts[0]"),
    ("carries_userinfo", ["user@cdn.example.gov.br"], "access.allowed_hosts[0]"),
    ("carries_whitespace", ["cdn example"], "access.allowed_hosts[0]"),
    ("bare_wildcard", ["*"], "access.allowed_hosts[0]"),
    ("bare_wildcard_dot", ["*."], "access.allowed_hosts[0]"),
    ("duplicate", ["cdn.example.gov.br", "CDN.example.gov.br"], "access.allowed_hosts[1]"),
)


@pytest.mark.parametrize(
    ("label", "value", "expected_path"),
    ALLOWED_HOSTS_REJECTIONS,
    ids=[row[0] for row in ALLOWED_HOSTS_REJECTIONS],
)
def test_each_malformed_allowed_hosts_entry_is_one_issue_at_its_own_path(
    tmp_path, label, value, expected_path
):
    """An allow-all wildcard is not an allow-list, and a URL is not a hostname."""
    del label
    paths = [issue.path for issue in _issues(tmp_path, allowed_hosts=value)]

    assert paths == [expected_path], f"expected exactly one issue at {expected_path}, got {paths}"


def test_three_bad_entries_report_three_issues(tmp_path):
    """Issue collection, unchanged: one raise site, every problem reported (order-11)."""
    issues = _issues(
        tmp_path, allowed_hosts=["https://a.example", "*", "b example"]
    )

    assert [issue.path for issue in issues] == [
        "access.allowed_hosts[0]",
        "access.allowed_hosts[1]",
        "access.allowed_hosts[2]",
    ]


@pytest.mark.parametrize(
    ("field_name", "url"),
    (
        ("url", "file:///etc/passwd"),
        ("url", "ftp://example.gov.br/x.zip"),
        ("base_url", "file:///etc/passwd"),
        ("base_url", "ftp://example.gov.br/x.zip"),
    ),
)
def test_a_non_http_access_url_is_refused_at_load_time(tmp_path, field_name, url):
    """Structural, not policy: TASK-03 makes such a URL unexecutable, so it fails at load.

    Without this rule a ``file:`` ``access.url`` costs ``max_attempts`` transport attempts
    before its dead letter, which reads like a network problem rather than a typo.
    """
    issues = _issues(tmp_path, **{field_name: url})

    assert ValidationIssue("access.url", "must use the http or https scheme") in issues


def test_allowed_hosts_issues_follow_the_link_resolver_issue(tmp_path):
    """The builder's append order remains deterministic when adjacent keys are bad."""
    issues = _issues(
        tmp_path,
        link_resolver="telepathy",
        allowed_hosts=["https://cdn.example.gov.br"],
    )

    assert [issue.path for issue in issues] == [
        "access.link_resolver",
        "access.allowed_hosts[0]",
    ]


# ---------------------------------------------------------------------------
# access.limits


def test_limits_default_to_the_measured_ceilings(tmp_path):
    """Sized from the largest checked-in artifact family so a default cannot dead-letter it."""
    limits = _build(tmp_path).access.limits

    assert {name: getattr(limits, name) for name in EXPECTED_DEFAULTS} == EXPECTED_DEFAULTS


def test_the_member_cap_follows_a_configured_payload_cap(tmp_path):
    """FR-5's default relationship: one knob moves both unless the operator splits them."""
    limits = _build(tmp_path, limits={"max_payload_bytes": 4 * GIB}).access.limits

    assert limits.max_payload_bytes == 4 * GIB
    assert limits.max_archive_member_bytes == 4 * GIB
    assert limits.max_archive_total_bytes == EXPECTED_DEFAULTS["max_archive_total_bytes"]


def test_every_limit_round_trips(tmp_path):
    overrides = {
        "max_payload_bytes": 1024,
        "max_redirects": 2,
        "max_archive_member_bytes": 512,
        "max_archive_total_bytes": 2048,
        "max_archive_ratio": 10,
    }

    limits = _build(tmp_path, limits=dict(overrides)).access.limits

    assert {name: getattr(limits, name) for name in overrides} == overrides


LIMIT_REJECTIONS = (
    ("zero", {"max_payload_bytes": 0}, "access.limits.max_payload_bytes"),
    ("negative", {"max_payload_bytes": -1}, "access.limits.max_payload_bytes"),
    ("human_readable", {"max_payload_bytes": "8GiB"}, "access.limits.max_payload_bytes"),
    ("float", {"max_payload_bytes": 1.5}, "access.limits.max_payload_bytes"),
    ("null", {"max_payload_bytes": None}, "access.limits.max_payload_bytes"),
    ("negative_redirects", {"max_redirects": -1}, "access.limits.max_redirects"),
    (
        "over_the_redirect_ceiling",
        {"max_redirects": MAX_REDIRECTS_CEILING + 1},
        "access.limits.max_redirects",
    ),
    ("zero_ratio", {"max_archive_ratio": 0}, "access.limits.max_archive_ratio"),
    (
        "member_over_total",
        {"max_archive_member_bytes": 4096, "max_archive_total_bytes": 2048},
        "access.limits.max_archive_member_bytes",
    ),
    ("not_a_mapping", "8GiB", "access.limits"),
)


@pytest.mark.parametrize(
    ("label", "value", "expected_path"),
    LIMIT_REJECTIONS,
    ids=[row[0] for row in LIMIT_REJECTIONS],
)
def test_each_malformed_limit_is_one_issue_at_its_own_path(
    tmp_path, label, value, expected_path
):
    """Bytes are plain integers; ``"8GiB"`` would be the config package's first string coercion."""
    del label
    paths = [issue.path for issue in _issues(tmp_path, limits=value)]

    assert paths == [expected_path], f"expected exactly one issue at {expected_path}, got {paths}"


def test_multiple_malformed_limits_are_collected_in_field_order(tmp_path):
    """One invalid block reports every independent field problem at the load boundary."""
    issues = _issues(
        tmp_path,
        limits={"max_payload_bytes": 0, "max_archive_ratio": "many"},
    )

    assert [issue.path for issue in issues] == [
        "access.limits.max_payload_bytes",
        "access.limits.max_archive_ratio",
    ]


def test_zero_redirects_is_a_valid_choice(tmp_path):
    """"Never follow a Location" is a legitimate contract for an API that must not redirect."""
    assert _build(tmp_path, limits={"max_redirects": 0}).access.limits.max_redirects == 0


def test_limits_config_fails_closed_at_construction(tmp_path):
    """Like ``DateWindowRequestInputsConfig``: no object rather than one carrying a bad cap."""
    del tmp_path
    from janus.models.source_config import LimitsConfig

    with pytest.raises(ValueError):
        LimitsConfig(max_payload_bytes=0)


def test_limits_issues_sit_between_the_rate_limit_and_link_resolver_issues(tmp_path):
    """Issue *ordering* is part of the contract (order-11), so the block's position is pinned."""
    issues = _issues(
        tmp_path,
        rate_limit={"requests_per_minute": "often", "concurrency": 1, "backoff_seconds": 5},
        limits={"max_payload_bytes": 0},
        link_resolver="telepathy",
    )
    paths = [issue.path for issue in issues]

    assert _issues_for(issues, "access.rate_limit"), f"no rate_limit issue collected: {paths}"
    assert _issues_for(issues, "access.limits"), f"no limits issue collected: {paths}"
    assert _issues_for(issues, "access.link_resolver"), f"no link_resolver issue: {paths}"

    rate_limit_index = next(i for i, p in enumerate(paths) if p.startswith("access.rate_limit"))
    limits_index = next(i for i, p in enumerate(paths) if p.startswith("access.limits"))
    resolver_index = next(i for i, p in enumerate(paths) if p.startswith("access.link_resolver"))

    assert rate_limit_index < limits_index < resolver_index, paths


# ---------------------------------------------------------------------------
# NFR-5 — structural, never policy


def test_allowed_hosts_does_not_consult_the_validation_policy(tmp_path):
    """A hostname allow-list is structural and adds no policy calls."""
    policy = SpyPolicy()

    config = _build(
        tmp_path,
        policy=policy,
        allowed_hosts=["cdn.example.gov.br"],
    )

    assert config.access.allowed_hosts == ("cdn.example.gov.br",)
    assert policy.calls == [
        "allowed_source_types",
        "allowed_strategies",
        "allowed_federation_levels",
        "validate_strategy_pairing",
        "validate_public_access",
    ]


def test_limits_does_not_consult_the_validation_policy(tmp_path):
    """The policy owns product scope; payload limits are structural."""
    policy = SpyPolicy()

    config = _build(
        tmp_path,
        policy=policy,
        limits={"max_payload_bytes": 4096, "max_archive_member_bytes": 4096},
    )

    assert config.access.limits.max_payload_bytes == 4096

    assert policy.calls == [
        "allowed_source_types",
        "allowed_strategies",
        "allowed_federation_levels",
        "validate_strategy_pairing",
        "validate_public_access",
    ], (
        "the policy was consulted a different number of times for a config whose only "
        f"unusual keys are structural: {policy.calls}"
    )


def test_every_checked_in_source_still_loads_and_takes_the_allowed_hosts_default():
    """Adding the structural key does not alter any checked-in source contract."""
    registry = load_registry(PROJECT_ROOT)
    sources = registry.list_sources(enabled_only=False)

    assert len(sources) >= 20, f"the registry looks truncated: {len(sources)} sources"
    for source in sources:
        assert source.access.allowed_hosts == (), source.source_id


def test_every_checked_in_source_still_loads_and_takes_the_limits_defaults():
    """AC-4's config half: no tracked YAML changes meaning when limits are added."""
    registry = load_registry(PROJECT_ROOT)
    sources = registry.list_sources(enabled_only=False)

    assert len(sources) >= 20, f"the registry looks truncated: {len(sources)} sources"
    for source in sources:
        limits = source.access.limits
        assert {name: getattr(limits, name) for name in EXPECTED_DEFAULTS} == EXPECTED_DEFAULTS, (
            source.source_id
        )
