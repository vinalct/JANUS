"""Defaults and fail-closed profile validation precede planning."""

from hashlib import sha256

import pytest
import yaml

from tests.support.operator_cli import run_janus
from tests.support.parallel_execution import REPO_ROOT, graph_project


def test_absent_block_uses_sequential_fail_fast_defaults():
    from janus.orchestration.settings import resolve_orchestration_settings

    settings = resolve_orchestration_settings({})
    assert settings.max_parallel_default == 1
    assert settings.lock_wait_seconds == 0


@pytest.mark.parametrize("block", [{}, {"max_parallel_default": 4}, {"lock_wait_seconds": 5}])
def test_supported_partial_blocks_use_defaults_for_omitted_keys(block):
    from janus.orchestration.settings import resolve_orchestration_settings

    settings = resolve_orchestration_settings({"orchestration": block})
    assert settings.max_parallel_default == block.get("max_parallel_default", 1)
    assert settings.lock_wait_seconds == block.get("lock_wait_seconds", 0)


@pytest.mark.parametrize("block", [None, False, 4, "private-value", []])
def test_present_non_mapping_block_is_refused_without_echoing_value(block):
    from janus.orchestration.settings import resolve_orchestration_settings

    with pytest.raises(ValueError, match="^orchestration must be a mapping$"):
        resolve_orchestration_settings({"orchestration": block})


@pytest.mark.parametrize("value", [0, -1, "2", 2.0, True])
def test_parallel_default_requires_positive_integer(value):
    from janus.orchestration.settings import resolve_orchestration_settings

    with pytest.raises(ValueError) as error:
        resolve_orchestration_settings({"orchestration": {"max_parallel_default": value}})
    assert str(error.value) == (
        f"orchestration.max_parallel_default must be an integer >= 1, got {type(value).__name__}"
    )


@pytest.mark.parametrize("value", [-1, "2", 2.0, True, None])
def test_lock_wait_requires_nonnegative_integer(value):
    from janus.orchestration.settings import resolve_orchestration_settings

    with pytest.raises(ValueError) as error:
        resolve_orchestration_settings({"orchestration": {"lock_wait_seconds": value}})
    assert str(error.value) == (
        f"orchestration.lock_wait_seconds must be an integer >= 0, got {type(value).__name__}"
    )


def test_unknown_key_error_names_key_without_echoing_value():
    from janus.orchestration.settings import resolve_orchestration_settings

    with pytest.raises(ValueError, match="unknown_key") as error:
        resolve_orchestration_settings({"orchestration": {"unknown_key": "private-value"}})
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("profile", ["local", "cluster"])
def test_shipped_profile_explicitly_declares_defaults(profile):
    from janus.orchestration.settings import resolve_orchestration_settings

    config = yaml.safe_load((REPO_ROOT / f"conf/environments/{profile}.yaml").read_text())
    assert config["orchestration"] == {"max_parallel_default": 1, "lock_wait_seconds": 0}
    settings = resolve_orchestration_settings(config)
    assert settings.max_parallel_default == 1 and settings.lock_wait_seconds == 0


@pytest.mark.parametrize(
    "block,key",
    [
        *[
            ({"max_parallel_default": value}, "max_parallel_default")
            for value in (0, -1, "2", 2.0, True)
        ],
        *[({"lock_wait_seconds": value}, "lock_wait_seconds") for value in (-1, "2", 2.0, True)],
        ({"unknown_key": "private-value"}, "unknown_key"),
        ("private-value", "orchestration"),
        (None, "orchestration"),
    ],
)
def test_invalid_profile_refuses_before_planning(tmp_path, monkeypatch, block, key):
    from janus.orchestration import BatchPlanner

    root = graph_project(tmp_path)
    profile = yaml.safe_load((root / "conf/environments/local.yaml").read_text())
    profile["orchestration"] = block
    (root / "conf/environments/broken.yaml").write_text(yaml.safe_dump(profile))

    def forbidden(*args, **kwargs):
        pytest.fail("broken profile reached planning")

    monkeypatch.setattr(BatchPlanner, "plan", forbidden)
    result = run_janus(
        ["run-all", "--environment", "broken", "--project-root", str(root), "--max-parallel", "1"]
    )
    assert result.exit_code == 2 and result.stdout == ""
    assert key in result.stderr
    assert "private-value" not in result.stderr


def test_other_shipped_profiles_are_unchanged():
    from janus.orchestration.settings import resolve_orchestration_settings

    paths = [
        path
        for path in (REPO_ROOT / "conf/environments").glob("*.yaml")
        if path.stem not in {"local", "cluster"}
    ]
    assert paths
    assert {sha256(path.read_bytes()).hexdigest() for path in paths} == {
        "8a6d057e816255dbb662cbd6714168d462337e3570817049281af1b0138776e5"
    }
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert "orchestration" not in config
        assert resolve_orchestration_settings(config).max_parallel_default == 1
