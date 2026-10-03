from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from janus.cli.common import format_runtime_permission_error
from janus.observability.openlineage import resolve_openlineage_settings
from janus.registry import load_registry
from janus.utils.environment import (
    ICEBERG_CATALOG_DB_PATH_KEY,
    build_spark_options,
    load_environment_config,
    materialize_runtime_paths,
)
from janus.utils.logging import REDACTED_VALUE, redact_url
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

REST_OVERLAY = REPO_ROOT / "conf" / "environments" / "cluster-rest.env.example"
SECRET = "do-not-print-me"
# Both report formats name the profile they read. Every verb parses --environment through
# the parent parser, so without this a profile-half claim also holds for the registry half.
LOCAL_PROFILE = "conf/environments/local.yaml"
# The options a catalog credential would render as; an unset `${VAR:-}` renders none of them.
CREDENTIAL_OPTIONS = (".jdbc.user", ".jdbc.password", ".token", ".credential", ".scope")


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


def test_validate_exits_zero_on_the_clean_registry(tmp_path: Path) -> None:
    result = _validate(_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert CLEAN_PRODUCER in result.stdout and CLEAN_CONSUMER in result.stdout


@pytest.mark.parametrize("rule", sorted(RULE_CASES), ids=lambda rule: f"rule_{rule}")
def test_validate_names_the_entry_and_the_rule_for_each_semantic_violation(
    rule: str, tmp_path: Path
) -> None:
    case = RULE_CASES[rule]

    result = _validate(_project(tmp_path, case.fixture))

    assert result.exit_code == 2
    assert case.rendered in result.stderr.splitlines()


@pytest.mark.parametrize("name", sorted(REFUSED_AT_LOAD))
def test_validate_prints_the_existing_load_refusal_verbatim(name: str, tmp_path: Path) -> None:
    root = _project(tmp_path, name)
    with pytest.raises(ValueError) as refused:
        load_registry(root)
    assert REFUSED_AT_LOAD[name] in str(refused.value)

    result = _validate(root)

    assert result.exit_code == 2
    assert str(refused.value) in result.stderr


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


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_validate_is_byte_identical_across_runs(output_format: str, tmp_path: Path) -> None:
    """D-18: no run id, timestamp or duration, so a golden can hold it."""
    root = _project(tmp_path)

    first = _validate(root, "--format", output_format)
    second = _validate(root, "--format", output_format)

    assert first.exit_code == 0, first.output
    assert first == second


def test_source_id_narrows_the_plan_step_but_never_the_semantic_pass(tmp_path: Path) -> None:
    """FR-3: the bystander plans clean, and its neighbour's rule (c) issue still exits 2."""
    case = RULE_CASES["c"]

    result = _validate(
        _project(tmp_path, case.fixture), "--source-id", "semantics_rule_c_unrelated"
    )

    assert result.exit_code == 2
    assert case.rendered in result.stderr.splitlines()


def test_validate_never_acquires_a_spark_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm_spark_tripwire(monkeypatch)
    before = engine_modules_loaded()

    result = _validate(_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert engine_modules_loaded() == before


def test_the_text_report_lists_every_source_with_its_dispatch_path(tmp_path: Path) -> None:
    lines = _validate(_project(tmp_path)).stdout.splitlines()

    assert lines[:3] == [
        "JANUS registry validation - conf/sources",
        "2 sources (1 enabled, 1 disabled), 2 nodes, 1 edges",
        "",
    ]
    assert [line.split() for line in lines if line.startswith("  ok")] == [
        ["ok", CLEAN_CONSUMER, "api.page_number_api"],
        ["ok", CLEAN_PRODUCER, "api.page_number_api"],
    ]
    assert lines[-1] == "0 issue(s) in 0 source(s); 0 unverified required-field declaration(s)"


def test_a_disabled_source_is_planned_and_its_hook_resolved(tmp_path: Path) -> None:
    root = _project(tmp_path)

    payload = json.loads(_validate(root, "--format", "json").stdout)
    consumer = {source["source_id"]: source for source in payload["sources"]}[CLEAN_CONSUMER]

    assert consumer == {
        "source_id": CLEAN_CONSUMER,
        "family": "api",
        "variant": "page_number_api",
        "dispatch_path": "api.page_number_api",
        "enabled": False,
        "hook": "ibge.sidra_flat",
        "hook_implementation": "IbgeSidraFlatHook",
        "planned": True,
        "issues": [],
    }
    assert payload["project_root"] == str(root.resolve())
    assert payload["sources_dir"] == "conf/sources"
    assert payload["counts"] == {
        "sources": 2,
        "enabled": 1,
        "disabled": 1,
        "nodes": 2,
        "edges": 1,
        "issues": 0,
        "unverified_required_fields": 0,
    }


def test_validate_loads_once_and_plans_through_one_planner_and_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from janus.cli import validate
    from janus.planner import Planner

    loaded: list[object] = []
    planners: list[object] = []
    calls: list[tuple[object, object]] = []

    def counting_load(project_root: Path):
        loaded.append(load_registry(project_root))
        return loaded[-1]

    class RecordingPlanner:
        def __init__(self) -> None:
            self._planner = Planner()
            planners.append(self)

        def plan(self, request, *, registry=None):
            calls.append((request, registry))
            return self._planner.plan(request, registry=registry)

    monkeypatch.setattr(validate, "load_registry", counting_load)
    monkeypatch.setattr(validate, "_build_planner", RecordingPlanner)

    result = _validate(_project(tmp_path))

    assert result.exit_code == 0, result.output
    assert (len(loaded), len(planners), len(calls)) == (1, 1, 2)
    assert all(snapshot is loaded[0] for _request, snapshot in calls)
    requests = [request for request, _snapshot in calls]
    assert all(request.include_disabled for request in requests)
    assert {request.started_at for request in requests} == {validate.VALIDATE_INSTANT}
    assert all(("trigger", "validate") in request.attributes for request in requests)


def test_a_source_the_planner_refuses_is_one_issue_and_its_peers_still_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planner whose hook catalog lacks the consumer's hook: the producer still plans."""
    from janus.cli import validate
    from janus.planner import HookCatalog, HookResolutionError, Planner

    with pytest.raises(HookResolutionError) as refused:
        HookCatalog().resolve("ibge.sidra_flat", source_id=CLEAN_CONSUMER)
    monkeypatch.setattr(validate, "_build_planner", lambda: Planner(hook_catalog=HookCatalog()))
    root = _project(tmp_path)

    data = _validate(root, "--format", "json")
    text = _validate(root)
    payload = json.loads(data.stdout)
    by_id = {source["source_id"]: source for source in payload["sources"]}
    issue = {"source_id": CLEAN_CONSUMER, "path": "plan", "message": str(refused.value)}

    assert (data.exit_code, text.exit_code) == (2, 2)
    assert data.stderr == text.stderr == ""
    assert payload["issues"] == by_id[CLEAN_CONSUMER]["issues"] == [issue]
    assert (by_id[CLEAN_CONSUMER]["planned"], by_id[CLEAN_CONSUMER]["dispatch_path"]) == (
        False,
        None,
    )
    assert by_id[CLEAN_PRODUCER]["planned"] is True
    assert by_id[CLEAN_PRODUCER]["issues"] == []
    lines = text.stdout.splitlines()
    failing = next(index for index, line in enumerate(lines) if line.startswith("  FAIL"))
    assert lines[failing + 1] == f"          plan: {refused.value}"
    assert [line.split() for line in lines if line.startswith(("  FAIL", "  ok"))] == [
        ["FAIL", CLEAN_CONSUMER, "(not", "planned)"],
        ["ok", CLEAN_PRODUCER, "api.page_number_api"],
    ]
    assert lines[-1] == "1 issue(s) in 1 source(s); 0 unverified required-field declaration(s)"


def test_a_strategy_refusal_the_loader_accepts_is_reported_on_its_source(tmp_path: Path) -> None:
    """A cursor variant paging by number parses and loads; only the strategy's plan refuses it,
    with the plain `ValueError` `janus run` already reports as exit 2."""
    root = _project(tmp_path)
    producer = root / "conf" / "sources" / "01_producer.yaml"
    producer.write_text(
        producer.read_text(encoding="utf-8").replace(
            "strategy_variant: page_number_api", "strategy_variant: cursor_api"
        ),
        encoding="utf-8",
    )
    load_registry(root)

    result = _validate(root, "--format", "json")
    payload = json.loads(result.stdout)

    assert result.exit_code == 2
    assert payload["issues"] == [
        {
            "source_id": CLEAN_PRODUCER,
            "path": "plan",
            "message": "cursor_api requires access.pagination.type='cursor'",
        }
    ]
    assert [source["planned"] for source in payload["sources"]] == [True, False]


def test_an_unknown_source_id_exits_2_with_the_registry_message(tmp_path: Path) -> None:
    result = _validate(_project(tmp_path), "--source-id", "not_a_source")

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == "Source 'not_a_source' was not found in the registry\n"


def test_source_id_lists_only_that_source_and_counts_the_whole_registry(tmp_path: Path) -> None:
    result = _validate(_project(tmp_path), "--source-id", CLEAN_CONSUMER, "--format", "json")
    payload = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert [source["source_id"] for source in payload["sources"]] == [CLEAN_CONSUMER]
    assert payload["counts"]["sources"] == 2


def test_unverified_required_fields_are_notes_that_never_change_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-3. Every source of a loaded registry has a contract since order-18, so the
    population is empty for any registry that loads; the engine's answer is injected."""
    from janus.cli import validate

    monkeypatch.setattr(
        validate,
        "unverified_required_fields",
        lambda sources, *, contracts: ((CLEAN_CONSUMER, ("detail_id", "code")),),
    )
    root = _project(tmp_path)

    text = _validate(root)
    data = _validate(root, "--format", "json")
    payload = json.loads(data.stdout)

    assert (text.exit_code, data.exit_code) == (0, 0)
    assert text.stdout.splitlines()[-6:] == [
        "",
        "notes",
        f"  note  {CLEAN_CONSUMER}: required_fields declared with no contract to verify "
        "them against",
        "          (detail_id, code)",
        "",
        "0 issue(s) in 0 source(s); 1 unverified required-field declaration(s)",
    ]
    assert payload["notes"] == [
        {
            "kind": "unverified_required_fields",
            "source_id": CLEAN_CONSUMER,
            "fields": ["detail_id", "code"],
        }
    ]
    assert (payload["counts"]["unverified_required_fields"], payload["issues"]) == (1, [])


def test_the_registry_half_reads_no_profile_and_writes_nothing(tmp_path: Path) -> None:
    """It is about conf/sources: a checkout with no conf/environments validates."""
    root = _project(tmp_path)
    assert not (root / "conf" / "environments").exists()
    before = tree_snapshot(root)

    result = _validate(root)

    assert result.exit_code == 0, result.output
    assert tree_snapshot(root) == before


# ---------------------------------------------------------------------------------------
# The profile half


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
        root,
        "--environment",
        profile,
        "--format",
        "json",
        env=_overlay(overlay) if overlay else None,
    )
    environment = json.loads(result.stdout)["environment"]

    assert result.exit_code == 0, result.output
    assert environment["name"] == profile
    assert environment["catalog_type"] == catalog_type
    assert environment["openlineage_transport"] == "file"
    assert environment["prepared"] is False
    assert [
        key for key in environment["spark_option_keys"] if key.endswith(CREDENTIAL_OPTIONS)
    ] == []


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


def test_a_profile_without_an_openlineage_block_reports_disabled(tmp_path: Path) -> None:
    """absent means disabled, which is legitimate; only a wrong value is an error."""
    root = _project(tmp_path)
    install_profile(root, "no-openlineage", source=ENVIRONMENT_FIXTURES)

    result = _validate(root, "--environment", "no-openlineage", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["environment"]["openlineage_transport"] == "disabled"


def test_validate_creates_no_directory_without_prepare(tmp_path: Path) -> None:
    """D-19: the dry run resolves every runtime location and creates none of them."""
    root = _project(tmp_path)
    install_profile(root, "local")
    before = tree_snapshot(root)

    result = _validate(root, "--environment", "local")

    assert result.exit_code == 0, result.output
    assert LOCAL_PROFILE in result.stdout
    assert tree_snapshot(root) == before


def test_validate_prepare_materializes_the_runtime_paths(tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")

    result = _validate(root, "--environment", "local", "--prepare", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["environment"]["prepared"] is True
    for zone in ("raw", "bronze", "metadata"):
        assert (root / "data" / zone).is_dir()


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
    assert LOCAL_PROFILE in result.stdout
    assert SECRET not in result.output


def test_validate_with_environment_never_acquires_a_spark_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")
    arm_spark_tripwire(monkeypatch)

    result = _validate(root, "--environment", "local")

    assert result.exit_code == 0, result.output
    assert LOCAL_PROFILE in result.stdout


def _environment(result) -> dict:
    return json.loads(result.stdout)["environment"]


def test_the_text_report_gains_the_environment_section_before_the_verdict(tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")
    config = load_environment_config("local", root)
    paths = materialize_runtime_paths(config, root)
    options = build_spark_options(config, paths)
    rendered = {key: Path(value).relative_to(root).as_posix() for key, value in paths.items()}
    events = f"{rendered['metadata_dir']}/{config['observability']['openlineage']['path']}"
    width = max(len(key) for key in rendered)

    lines = _validate(root, "--environment", "local").stdout.splitlines()
    start = lines.index(f"environment: local  ({LOCAL_PROFILE})")

    assert lines[start - 1] == ""
    assert lines[start + 1 : start + 5] == [
        f"  catalog        jdbc  (database file: {rendered[ICEBERG_CATALOG_DB_PATH_KEY]})",
        f"  openlineage    file  ({events})",
        f"  spark options  {len(options)} keys",
        "  paths          [not created: dry run]",
    ]
    assert lines[start + 5 : -2] == [
        f"    {key:<{width}}  {value}" for key, value in rendered.items()
    ]
    assert lines[-2:] == [
        "",
        "0 issue(s) in 0 source(s); 0 unverified required-field declaration(s)",
    ]


def test_the_environment_object_has_the_documented_shape_and_leaves_the_rest_alone(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")
    config = load_environment_config("local", root)
    options = build_spark_options(config, materialize_runtime_paths(config, root))

    with_profile = json.loads(_validate(root, "--environment", "local", "--format", "json").stdout)
    without = json.loads(_validate(root, "--format", "json").stdout)
    environment = with_profile.pop("environment")

    assert environment.keys() == {
        "name",
        "config_path",
        "catalog_type",
        "openlineage_transport",
        "openlineage_target",
        "spark_option_keys",
        "paths",
        "prepared",
    }
    assert (environment["name"], environment["config_path"]) == ("local", LOCAL_PROFILE)
    assert environment["spark_option_keys"] == sorted(options)
    assert "environment" not in without
    assert with_profile == without


@pytest.mark.parametrize(
    ("profile", "overlay", "warehouse"),
    [
        ("local", None, "data/bronze/iceberg"),
        ("cluster", None, "s3://janus-bronze/warehouse"),
        ("cluster", REST_OVERLAY, "janus"),
    ],
    ids=["local", "cluster", "cluster-rest"],
)
def test_paths_are_project_relative_and_locations_verbatim(
    profile: str, overlay: Path | None, warehouse: str, tmp_path: Path
) -> None:
    """A location URI or a catalog-managed warehouse name is never resolved as a path."""
    root = _project(tmp_path)
    install_profile(root, profile)

    result = _validate(
        root,
        "--environment",
        profile,
        "--format",
        "json",
        env=_overlay(overlay) if overlay else None,
    )
    environment = _environment(result)
    paths = environment["paths"]

    assert result.exit_code == 0, result.output
    assert paths["iceberg_warehouse_dir"] == warehouse
    assert (paths["raw_dir"], paths["metadata_dir"]) == ("data/raw", "data/metadata")
    assert (ICEBERG_CATALOG_DB_PATH_KEY in paths) == (profile == "local")
    assert str(root.resolve()) not in json.dumps(environment)


def test_catalog_auth_is_reported_by_key_and_never_by_value(tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "cluster")
    exported = {
        **_overlay(REST_OVERLAY),
        "JANUS_ICEBERG_CATALOG_TOKEN": f"{SECRET}-token",
        "JANUS_ICEBERG_CATALOG_CREDENTIAL": f"{SECRET}-credential",
    }

    data = _validate(root, "--environment", "cluster", "--format", "json", env=exported)
    text = _validate(root, "--environment", "cluster", env=exported)

    assert (data.exit_code, text.exit_code) == (0, 0)
    assert {"spark.sql.catalog.janus.token", "spark.sql.catalog.janus.credential"} <= set(
        _environment(data)["spark_option_keys"]
    )
    assert SECRET not in data.output + text.output


def test_an_http_openlineage_target_is_reported_redacted(tmp_path: Path) -> None:
    """The transport's own `redact_url` rendering; the bearer token is a header, never shown."""
    root = _project(tmp_path)
    install_profile(root, "local")
    exported = {
        "JANUS_OPENLINEAGE_TRANSPORT": "http",
        "JANUS_OPENLINEAGE_URL": "https://lineage.example.invalid",
        "JANUS_OPENLINEAGE_ENDPOINT": f"api/v1/lineage?api_key={SECRET}-query",
        "JANUS_OPENLINEAGE_API_KEY": f"{SECRET}-token",
    }
    target = redact_url(f"https://lineage.example.invalid/api/v1/lineage?api_key={SECRET}-query")

    data = _validate(root, "--environment", "local", "--format", "json", env=exported)
    text = _validate(root, "--environment", "local", env=exported)
    environment = _environment(data)

    assert REDACTED_VALUE in target
    assert (environment["openlineage_transport"], environment["openlineage_target"]) == (
        "http",
        target,
    )
    assert f"  openlineage    http  ({target})" in text.stdout.splitlines()
    assert SECRET not in data.output + text.output


def test_an_events_path_outside_the_metadata_zone_is_refused_not_disabled(tmp_path: Path) -> None:
    """Inside a run this degrades to no lineage at all; validate is where it is caught."""
    root = _project(tmp_path)
    install_profile(root, "local")

    result = _validate(
        root, "--environment", "local", env={"JANUS_OPENLINEAGE_EVENTS_DIR": "../../escaped"}
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith(
        "The OpenLineage events path must stay inside the metadata zone: "
    )


def test_a_profile_missing_a_required_key_exits_2_naming_it(tmp_path: Path) -> None:
    root = _project(tmp_path)
    profile = yaml.safe_load(install_profile(root, "local").read_text(encoding="utf-8"))
    del profile["storage"]
    incomplete = root / "conf" / "environments" / "incomplete.yaml"
    incomplete.write_text(yaml.safe_dump(profile), encoding="utf-8")

    result = _validate(root, "--environment", "incomplete")

    assert (result.exit_code, result.stdout) == (2, "")
    assert result.stderr == "Environment config is incomplete: 'root_dir'\n"


def test_both_refusals_are_reported_profile_first_and_no_report_is_printed(
    tmp_path: Path,
) -> None:
    """The halves are independent: a broken profile does not hide a broken registry."""
    root = _project(tmp_path, "graph_cycle")
    install_profile(root, "broken-openlineage", source=ENVIRONMENT_FIXTURES)
    with pytest.raises(ValueError) as profile_refused:
        resolve_openlineage_settings(load_environment_config("broken-openlineage", root))
    with pytest.raises(ValueError) as registry_refused:
        load_registry(root)

    result = _validate(root, "--environment", "broken-openlineage")

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == f"{profile_refused.value}\n{registry_refused.value}\n"


def test_a_refused_profile_prints_no_report_even_over_a_clean_registry(tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "broken-catalog", source=ENVIRONMENT_FIXTURES)

    result = _validate(root, "--environment", "broken-catalog", "--format", "json")

    assert (result.exit_code, result.stdout) == (2, "")
    assert result.stderr.count("\n") == 1


def test_prepare_without_environment_exits_2_and_creates_nothing(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before = tree_snapshot(root)

    result = _validate(root, "--prepare")

    assert (result.exit_code, result.stdout) == (2, "")
    assert result.stderr == "--prepare requires --environment\n"
    assert tree_snapshot(root) == before


def test_prepare_creates_every_directory_the_dry_run_resolves(tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")

    dry = _environment(_validate(root, "--environment", "local", "--format", "json"))
    expected = {
        root / (Path(value).parent if key == ICEBERG_CATALOG_DB_PATH_KEY else Path(value))
        for key, value in dry["paths"].items()
    }
    assert expected and not any(path.exists() for path in expected)

    prepared = _validate(root, "--environment", "local", "--prepare", "--format", "json")

    assert prepared.exit_code == 0, prepared.output
    assert _environment(prepared) == {**dry, "prepared": True}
    assert all(path.is_dir() for path in expected)


def test_an_unwritable_path_under_prepare_is_explained_as_run_explains_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from janus.cli import validate

    denied = PermissionError(13, "Permission denied", "/workspace/data/raw")

    def refusing(config: dict, project_root: Path) -> dict:
        raise denied

    monkeypatch.setattr(validate, "prepare_runtime", refusing)
    root = _project(tmp_path)
    install_profile(root, "local")

    result = _validate(root, "--environment", "local", "--prepare")

    assert (result.exit_code, result.stdout) == (2, "")
    assert result.stderr == f"{format_runtime_permission_error(denied)}\n"


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_the_profile_half_is_byte_identical_across_runs(output_format: str, tmp_path: Path) -> None:
    root = _project(tmp_path)
    install_profile(root, "local")

    first = _validate(root, "--environment", "local", "--format", output_format)
    second = _validate(root, "--environment", "local", "--format", output_format)

    assert first.exit_code == 0, first.output
    assert first == second


@pytest.mark.parametrize(
    ("argv", "planned_in"),
    [((), "local"), (("--environment", "cluster"), "cluster")],
    ids=["registry-half", "with-profile"],
)
def test_the_plan_step_keeps_the_parent_default_environment(
    argv: tuple[str, ...], planned_in: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent --environment skips the profile half without changing what is planned."""
    from janus.cli import validate
    from janus.planner import Planner

    environments: list[str] = []

    class RecordingPlanner:
        def __init__(self) -> None:
            self._planner = Planner()

        def plan(self, request, *, registry=None):
            environments.append(request.environment)
            return self._planner.plan(request, registry=registry)

    monkeypatch.setattr(validate, "_build_planner", RecordingPlanner)
    root = _project(tmp_path)
    install_profile(root, "cluster")

    result = _validate(root, *argv)

    assert result.exit_code == 0, result.output
    assert environments and set(environments) == {planned_in}
