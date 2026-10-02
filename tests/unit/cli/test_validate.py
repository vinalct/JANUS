from __future__ import annotations

import json
from pathlib import Path

import pytest

from janus.observability.openlineage import resolve_openlineage_settings
from janus.registry import load_registry
from janus.utils.environment import (
    build_spark_options,
    load_environment_config,
    materialize_runtime_paths,
)
from tests.support.operator_cli import arm_spark_tripwire, engine_modules_loaded, run_janus
from tests.support.semantics_fixtures import (
    CLEAN,
    CLEAN_CONSUMER,
    CLEAN_PRODUCER,
    ENVIRONMENT_FIXTURES,
    REFUSED_AT_LOAD,
    REPO_ROOT,
    RULE_CASES,
    install_profile,
    materialize,
    tree_snapshot,
)

RED_REGISTRY = pytest.mark.xfail(
    strict=True,
    reason="janus.cli.dispatch registers no `validate` verb yet",
)
RED_ENVIRONMENT = pytest.mark.xfail(
    strict=True,
    reason="`janus validate --environment/--prepare` does not exist yet",
)

REST_OVERLAY = REPO_ROOT / "conf" / "environments" / "cluster-rest.env.example"
SECRET = "do-not-print-me"


def _project(tmp_path: Path, name: str = CLEAN) -> Path:
    return materialize(name, tmp_path / "project")


def _validate(root: Path, *extra: str, env: dict[str, str] | None = None):
    return run_janus(("validate", "--project-root", str(root), *extra), env=env)


def _overlay(path: Path) -> dict[str, str]:
    """The `KEY=value` lines of an env overlay, the way `make` would export them."""
    exported: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key and not key.startswith("#"):
            exported[key] = value
    return exported


# ---------------------------------------------------------------------------------------
# The registry half


@RED_REGISTRY
def test_validate_exits_zero_on_the_clean_registry(tmp_path: Path) -> None:
    result = _validate(_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert CLEAN_PRODUCER in result.stdout and CLEAN_CONSUMER in result.stdout


@RED_REGISTRY
@pytest.mark.parametrize("rule", sorted(RULE_CASES), ids=lambda rule: f"rule_{rule}")
def test_validate_names_the_entry_and_the_rule_for_each_semantic_violation(
    rule: str, tmp_path: Path
) -> None:
    case = RULE_CASES[rule]

    result = _validate(_project(tmp_path, case.fixture))

    assert result.exit_code == 2
    assert case.rendered in result.stderr.splitlines()


@RED_REGISTRY
@pytest.mark.parametrize("name", sorted(REFUSED_AT_LOAD))
def test_validate_prints_the_existing_load_refusal_verbatim(name: str, tmp_path: Path) -> None:
    root = _project(tmp_path, name)
    with pytest.raises(ValueError) as refused:
        load_registry(root)
    assert REFUSED_AT_LOAD[name] in str(refused.value)

    result = _validate(root)

    assert result.exit_code == 2
    assert str(refused.value) in result.stderr


@RED_REGISTRY
def test_validate_json_report_has_the_documented_shape(tmp_path: Path) -> None:
    result = _validate(_project(tmp_path), "--format", "json")
    payload = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert {"sources", "issues", "counts"} <= payload.keys()
    assert [source["source_id"] for source in payload["sources"]] == [
        CLEAN_CONSUMER,
        CLEAN_PRODUCER,
    ]
    assert {source["dispatch_path"] for source in payload["sources"]} == {"api.page_number_api"}
    assert payload["issues"] == []
    assert payload["counts"]["sources"] == 2
    assert payload["counts"]["issues"] == 0
    assert result.stdout == json.dumps(payload, indent=2, sort_keys=True) + "\n"


@RED_REGISTRY
@pytest.mark.parametrize("output_format", ["text", "json"])
def test_validate_is_byte_identical_across_runs(output_format: str, tmp_path: Path) -> None:
    """D-18: no run id, timestamp or duration, so a golden can hold it."""
    root = _project(tmp_path)

    first = _validate(root, "--format", output_format)
    second = _validate(root, "--format", output_format)

    assert first.exit_code == 0, first.output
    assert first == second


@RED_REGISTRY
def test_source_id_narrows_the_plan_step_but_never_the_semantic_pass(tmp_path: Path) -> None:
    """FR-3: the bystander plans clean, and its neighbour's rule (c) issue still exits 2."""
    case = RULE_CASES["c"]

    result = _validate(
        _project(tmp_path, case.fixture), "--source-id", "semantics_rule_c_unrelated"
    )

    assert result.exit_code == 2
    assert case.rendered in result.stderr.splitlines()


@RED_REGISTRY
def test_validate_never_acquires_a_spark_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm_spark_tripwire(monkeypatch)
    before = engine_modules_loaded()

    result = _validate(_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert engine_modules_loaded() == before


# ---------------------------------------------------------------------------------------
# The profile half 


@RED_ENVIRONMENT
@pytest.mark.parametrize(
    ("profile", "overlay", "catalog_type"),
    [("local", None, "jdbc"), ("cluster", None, "jdbc"), ("cluster", REST_OVERLAY, "rest")],
    ids=["local", "cluster", "cluster-rest"],
)
def test_shipped_profiles_validate_clean(
    profile: str, overlay: Path | None, catalog_type: str, tmp_path: Path
) -> None:
    """`cluster-rest` is the overlay of `cluster.yaml`, exported the way `make` exports it;
    its unset credentials resolve to nothing rather than to an empty value."""
    root = _project(tmp_path)
    install_profile(root, profile)

    result = _validate(
        root, "--environment", profile, "--format", "json",
        env=_overlay(overlay) if overlay else None,
    )
    environment = json.loads(result.stdout)["environment"]

    assert result.exit_code == 0, result.output
    assert environment["name"] == profile
    assert environment["catalog_type"] == catalog_type
    assert environment["openlineage_transport"] == "file"
    assert environment["prepared"] is False


@RED_ENVIRONMENT
def test_a_profile_without_catalog_type_exits_2_with_the_existing_message(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    install_profile(root, "broken-catalog", source=ENVIRONMENT_FIXTURES)
    config = load_environment_config("broken-catalog", root)
    with pytest.raises(ValueError) as refused:
        build_spark_options(config, materialize_runtime_paths(config, root))

    result = _validate(root, "--environment", "broken-catalog")

    assert result.exit_code == 2
    assert str(refused.value) in result.stderr


@RED_ENVIRONMENT
def test_an_unrecognised_openlineage_transport_exits_2_with_the_existing_message(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    install_profile(root, "broken-openlineage", source=ENVIRONMENT_FIXTURES)
    with pytest.raises(ValueError) as refused:
        resolve_openlineage_settings(load_environment_config("broken-openlineage", root))

    result = _validate(root, "--environment", "broken-openlineage")

    assert result.exit_code == 2
    assert str(refused.value) in result.stderr


@RED_ENVIRONMENT
def test_a_profile_without_an_openlineage_block_reports_disabled(tmp_path: Path) -> None:
    """absent means disabled, which is legitimate; only a wrong value is an error."""
    root = _project(tmp_path)
    install_profile(root, "no-openlineage", source=ENVIRONMENT_FIXTURES)

    result = _validate(root, "--environment", "no-openlineage", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["environment"]["openlineage_transport"] == "disabled"


@RED_ENVIRONMENT
def test_validate_creates_no_directory_without_prepare(tmp_path: Path) -> None:
    """D-19: the dry run resolves every runtime location and creates none of them."""
    root = _project(tmp_path)
    install_profile(root, "local")
    before = tree_snapshot(root)

    result = _validate(root, "--environment", "local")

    assert result.exit_code == 0, result.output
    assert tree_snapshot(root) == before


@RED_ENVIRONMENT
def test_validate_prepare_materializes_the_runtime_paths(tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")

    result = _validate(root, "--environment", "local", "--prepare", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["environment"]["prepared"] is True
    for zone in ("raw", "bronze", "metadata"):
        assert (root / "data" / zone).is_dir()


@RED_ENVIRONMENT
@pytest.mark.parametrize("output_format", ["text", "json"])
def test_no_credential_value_reaches_either_output_format(
    output_format: str, tmp_path: Path
) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")
    secrets = {
        "JANUS_ICEBERG_CATALOG_USER": f"{SECRET}-user",
        "JANUS_ICEBERG_CATALOG_PASSWORD": f"{SECRET}-password",
        "JANUS_OPENLINEAGE_API_KEY": f"{SECRET}-token",
    }

    result = _validate(root, "--environment", "local", "--format", output_format, env=secrets)

    assert result.exit_code == 0, result.output
    assert SECRET not in result.output


@RED_ENVIRONMENT
def test_validate_with_environment_never_acquires_a_spark_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")
    arm_spark_tripwire(monkeypatch)

    result = _validate(root, "--environment", "local")

    assert result.exit_code == 0, result.output
