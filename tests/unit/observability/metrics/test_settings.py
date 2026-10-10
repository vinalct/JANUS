"""Fail-closed profile reads and metadata-zone resolution for metrics."""

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from janus.cli.run_all import main as run_all_main
from janus.main import main
from janus.observability import IcebergAppendOutcome, IcebergAppendResult, build_run_event_emitter
from janus.observability.metrics import (
    DISABLED_METRICS_SETTINGS,
    FileMetricsSettings,
    MetricsProfileError,
    MetricsSettings,
    MetricsTransportKind,
    StatsdMetricsSettings,
    TextfileMetricsSettings,
    resolve_metrics_directory,
    resolve_metrics_settings,
)
from janus.utils.environment import load_environment_config
from janus.utils.storage import ResolvedOutputTarget
from tests.support import observability_baseline as baseline
from tests.support.metrics_order23 import WarningSpy, exit_code, record, sink, write_profile
from tests.unit.observability.test_runs_table_contract import _load_clean_profile

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SHIPPED_PROFILES = tuple(
    path.stem for path in sorted((PROJECT_ROOT / "conf/environments").glob("*.yaml"))
)


@pytest.mark.parametrize("profile", ["local", "cluster", "cluster-rest"])
def test_absent_block_disables_emission_and_writes_no_metadata_file(profile, tmp_path):
    from janus.observability.metrics.settings import MetricsTransportKind, resolve_metrics_settings
    from janus.observability.metrics.sink import build_metrics_sink
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    overrides = {"JANUS_ICEBERG_CATALOG_TYPE": "rest"} if profile == "cluster-rest" else {}
    config = deepcopy(
        _load_clean_profile("cluster" if profile == "cluster-rest" else profile, overrides)
    )
    config.setdefault("observability", {}).pop("metrics", None)
    assert resolve_metrics_settings(config).kind is MetricsTransportKind.DISABLED
    destination = build_metrics_sink(config, {"metadata_dir": tmp_path})
    assert (
        destination.emit(record(tmp_path), budget_seconds=1).outcome
        is MetricsEmissionOutcome.SKIPPED
    )
    assert not list(tmp_path.rglob("*"))


def test_explicit_disabled_is_a_real_silent_transport(tmp_path):
    from janus.observability.metrics.transport import (
        DisabledMetricsTransport,
        MetricsEmissionOutcome,
    )

    destination = sink(tmp_path, "disabled")
    assert isinstance(destination.transport, DisabledMetricsTransport)
    assert (
        destination.emit(record(tmp_path), budget_seconds=1).outcome
        is MetricsEmissionOutcome.SKIPPED
    )
    assert not list(tmp_path.rglob("*"))


def test_bad_transport_raises_at_profile_read_time_and_lists_all_choices(tmp_path):
    from janus.observability.metrics.settings import MetricsProfileError

    write_profile(tmp_path, {"transport": "htpp"})
    config = load_environment_config("local", tmp_path)
    with pytest.raises(MetricsProfileError) as raised:
        resolve_metrics_settings(config)
    message = str(raised.value)
    assert "observability.metrics.transport" in message
    assert "htpp" in message
    assert message.endswith("supported values: disabled, file, statsd, textfile")


@pytest.mark.parametrize("batch", [False, True], ids=["single-source", "run-all"])
def test_bad_transport_makes_both_entry_points_exit_two_before_registry_load(
    batch, tmp_path, capsys
):
    write_profile(tmp_path, {"transport": "htpp"})
    entry = run_all_main if batch else main
    arguments = ["--project-root", str(tmp_path)]
    if not batch:
        arguments.extend(["--source-id", "must-not-be-loaded", "--execute"])
    with patch(
        "janus.cli.run_all.load_registry" if batch else "janus.cli.run.load_registry"
    ) as load:
        code = exit_code(lambda: entry(arguments))
    stderr = capsys.readouterr().err
    assert code == 2
    assert "observability.metrics.transport" in stderr
    assert all(kind in stderr for kind in ("disabled", "file", "textfile", "statsd"))
    load.assert_not_called()


def test_bad_in_run_block_warns_once_and_preserves_status_exit_and_authoritative_bytes(
    tmp_path, monkeypatch
):
    from janus.observability.metrics.transport import DisabledMetricsTransport

    root = tmp_path / "project"
    baseline_manifest = baseline.capture_case(root, "api_success")
    files = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.json")}
    warnings = WarningSpy()
    emitter = build_run_event_emitter(
        {"observability": {"metrics": {"transport": "htpp"}}},
        {"metadata_dir": root / "data/metadata"},
        logger=warnings,
        runs_table_sink=lambda *args, **kwargs: IcebergAppendResult(
            IcebergAppendOutcome.EMITTED, "metadata.runs"
        ),
    )
    assert isinstance(emitter.metrics_sink.transport, DisabledMetricsTransport)
    assert len(warnings.warnings) == 1
    fixed = baseline.FixedObserver

    class EmittingObserver(fixed):
        def __init__(self):
            super().__init__(emitter=emitter)

    monkeypatch.setattr(baseline, "FixedObserver", EmittingObserver)
    shutil.rmtree(root)
    actual = baseline.capture_case(root, "api_success")
    assert actual["status"] == baseline_manifest["status"] == "succeeded"
    assert actual["summary"]["executed_run"]["status"] == "succeeded"
    assert actual["events"] == baseline_manifest["events"]
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.json")} == files
    assert len(warnings.warnings) == 1


def test_unknown_key_error_names_the_key_without_disclosing_its_value():
    from janus.observability.metrics.settings import MetricsProfileError, resolve_metrics_settings

    secret = "credential-must-not-escape"
    with pytest.raises(MetricsProfileError) as raised:
        resolve_metrics_settings(
            {"observability": {"metrics": {"transport": "disabled", "api_token": secret}}}
        )
    assert "api_token" in str(raised.value)
    assert secret not in str(raised.value)


@pytest.mark.parametrize("kind", ["file", "textfile"])
def test_relative_file_path_is_resolved_under_metadata_zone(kind, tmp_path):
    settings = resolve_metrics_settings(
        {"observability": {"metrics": {"transport": kind, "path": "nested/metrics"}}}
    )
    block = settings.file if kind == "file" else settings.textfile
    assert block is not None
    assert resolve_metrics_directory(block.directory, {"metadata_dir": tmp_path}) == (
        tmp_path / "nested/metrics"
    )
    assert not list(tmp_path.rglob("*")), "directory resolution must not create outputs"


@pytest.mark.parametrize("kind", ["file", "textfile"])
@pytest.mark.parametrize("path", ["/etc", "../../etc", "nested/../metrics"])
def test_absolute_and_parent_escape_paths_are_refused_at_directory_resolution(
    kind, path, tmp_path
):
    settings = resolve_metrics_settings(
        {"observability": {"metrics": {"transport": kind, "path": path}}}
    )
    block = settings.file if kind == "file" else settings.textfile
    assert block is not None and block.directory == path
    with pytest.raises(MetricsProfileError, match="observability.metrics.path.*metadata zone"):
        resolve_metrics_directory(block.directory, {"metadata_dir": tmp_path})


@pytest.mark.xfail(
    strict=True, reason="shipped profiles lack explicit disabled metrics"
)
@pytest.mark.parametrize("profile", ["local", "cluster"])
def test_every_shipped_profile_declares_metrics_disabled_by_default(profile):
    root = Path(__file__).resolve().parents[4]
    document = yaml.safe_load((root / "conf/environments" / f"{profile}.yaml").read_text())
    assert (
        document["observability"]["metrics"]["transport"] == "${JANUS_METRICS_TRANSPORT:-disabled}"
    )
    assert _load_clean_profile(profile)["observability"]["metrics"]["transport"] == "disabled"


@pytest.mark.parametrize("profile", [*SHIPPED_PROFILES, "cluster-rest"])
def test_shipped_profiles_resolve_to_disabled(profile, tmp_path):
    assert SHIPPED_PROFILES and {"local", "cluster"} <= set(SHIPPED_PROFILES)
    overrides = {"JANUS_ICEBERG_CATALOG_TYPE": "rest"} if profile == "cluster-rest" else {}
    config = _load_clean_profile("cluster" if profile == "cluster-rest" else profile, overrides)
    assert resolve_metrics_settings(config) is DISABLED_METRICS_SETTINGS
    assert not list(tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"observability": None},
        {"observability": {}},
        {"observability": {"runs_table": "audit.runs"}},
        {"observability": {"metrics": None}},
        {"observability": {"metrics": {}}},
        *[
            {"observability": {"metrics": {"transport": transport}}}
            for transport in (None, "", "  ", "disabled")
        ],
    ],
)
def test_absent_empty_and_disabled_return_the_disabled_singleton(config):
    settings = resolve_metrics_settings(config)
    assert settings is DISABLED_METRICS_SETTINGS
    assert not settings.enabled
    assert settings.file is settings.textfile is settings.statsd is None


def test_unexported_transport_expansion_is_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("JANUS_METRICS_TRANSPORT", raising=False)
    write_profile(tmp_path, {"transport": "${JANUS_METRICS_TRANSPORT:-}"})
    config = load_environment_config("local", tmp_path)
    assert config["observability"]["metrics"]["transport"] == ""
    assert resolve_metrics_settings(config) is DISABLED_METRICS_SETTINGS


@pytest.mark.parametrize("kind", ["file", "textfile"])
@pytest.mark.parametrize("path", [None, "", "  ", "nested/metrics"])
def test_file_kinds_carry_their_directory_defaults_and_overrides(kind, path):
    settings = resolve_metrics_settings(
        {"observability": {"metrics": {"transport": kind, "path": path}}}
    )
    expected = {
        "file": MetricsSettings(MetricsTransportKind.FILE, file=FileMetricsSettings()),
        "textfile": MetricsSettings(
            MetricsTransportKind.TEXTFILE, textfile=TextfileMetricsSettings()
        ),
    }[kind]
    assert settings.enabled
    if path == "nested/metrics":
        block = settings.file if kind == "file" else settings.textfile
        assert block is not None and block.directory == path
    else:
        assert settings == expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, StatsdMetricsSettings("localhost")),
        ({"port": "", "prefix": ""}, StatsdMetricsSettings("localhost")),
        ({"port": 1}, StatsdMetricsSettings("localhost", port=1)),
        ({"port": "65535"}, StatsdMetricsSettings("localhost", port=65535)),
        (
            {"port": "9125", "prefix": "JANUS"},
            StatsdMetricsSettings("localhost", port=9125, prefix="JANUS"),
        ),
    ],
)
def test_statsd_defaults_and_valid_overrides(extra, expected):
    settings = resolve_metrics_settings(
        {"observability": {"metrics": {"transport": "statsd", "host": "localhost", **extra}}}
    )
    assert settings == MetricsSettings(MetricsTransportKind.STATSD, statsd=expected)
    assert settings.enabled


@pytest.mark.parametrize("value", ["broken", [], 0, False])
@pytest.mark.parametrize("key", ["observability", "observability.metrics"])
def test_non_mapping_blocks_are_profile_errors(key, value):
    config = {"observability": value if key == "observability" else {"metrics": value}}
    with pytest.raises(MetricsProfileError) as raised:
        resolve_metrics_settings(config)
    assert str(raised.value) == f"{key} must be a mapping"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        *[("port", value) for value in (0, -1, 70000, "eight", 1.5, True)],
        *[("host", value) for value in (None, "", "  ")],
        *[("prefix", value) for value in ("janus-run", "janus.run", "two names", "métries")],
    ],
)
def test_malformed_statsd_scalars_name_the_qualified_key(key, value):
    with pytest.raises(MetricsProfileError, match=rf"observability\.metrics\.{key}"):
        resolve_metrics_settings(
            {
                "observability": {
                    "metrics": {"transport": "statsd", "host": "localhost", key: value}
                }
            }
        )


@pytest.mark.parametrize("transport", [None, "", "disabled", "file", "textfile", "statsd"])
def test_unknown_keys_are_rejected_before_transport_selection(transport):
    secret = "credential-must-not-escape"
    with pytest.raises(MetricsProfileError) as raised:
        resolve_metrics_settings(
            {"observability": {"metrics": {"transport": transport, "transpot": secret}}}
        )
    assert "transpot" in str(raised.value)
    assert secret not in str(raised.value)


@pytest.mark.parametrize(
    ("kind", "blocks"),
    [
        (MetricsTransportKind.FILE, {}),
        (MetricsTransportKind.TEXTFILE, {}),
        (MetricsTransportKind.STATSD, {}),
        (MetricsTransportKind.DISABLED, {"file": FileMetricsSettings()}),
        (MetricsTransportKind.FILE, {"textfile": TextfileMetricsSettings()}),
        (
            MetricsTransportKind.FILE,
            {"file": FileMetricsSettings(), "textfile": TextfileMetricsSettings()},
        ),
        (
            MetricsTransportKind.STATSD,
            {"file": FileMetricsSettings(), "statsd": StatsdMetricsSettings("localhost")},
        ),
    ],
)
def test_transport_kind_and_exactly_one_matching_block_travel_together(kind, blocks):
    with pytest.raises(MetricsProfileError, match="exactly that transport's block"):
        MetricsSettings(kind=kind, **blocks)


def test_settings_are_frozen():
    settings = MetricsSettings(MetricsTransportKind.FILE, file=FileMetricsSettings())
    with pytest.raises(FrozenInstanceError):
        settings.file.directory = "changed"
    with pytest.raises(FrozenInstanceError):
        settings.kind = MetricsTransportKind.DISABLED


@pytest.mark.parametrize(
    "paths", [{}, {"metadata_dir": None}, {"metadata_dir": "relative/metadata"}]
)
def test_directory_needs_a_resolved_absolute_metadata_zone(paths):
    with pytest.raises(MetricsProfileError, match="metadata zone"):
        resolve_metrics_directory("metrics", paths)


def test_metrics_and_openlineage_borrow_the_same_zone_containment_rule(tmp_path):
    from janus.observability.openlineage.transport import _resolve_events_directory

    paths = {"metadata_dir": tmp_path}
    with patch.object(ResolvedOutputTarget, "child", return_value=tmp_path / "shared") as child:
        assert resolve_metrics_directory("nested/metrics", paths) == tmp_path / "shared"
        assert _resolve_events_directory("nested/events", paths) == tmp_path / "shared"
    assert [call.args for call in child.call_args_list] == [("nested/metrics",), ("nested/events",)]


@pytest.mark.parametrize("batch", [False, True], ids=["single-source", "run-all"])
@pytest.mark.parametrize("error_type", [MetricsProfileError, ValueError])
def test_existing_value_error_handler_receives_metrics_profile_errors(
    batch, error_type, tmp_path, capsys
):
    module = "janus.cli.run_all" if batch else "janus.cli.run"
    entry = run_all_main if batch else main
    order = []

    def reject(config):
        order.append("metrics")
        raise error_type("metrics profile refused by existing ValueError handler")

    with (
        patch(f"{module}.load_environment_config", return_value={}),
        patch(f"{module}.prepare_runtime", side_effect=lambda *args: order.append("runtime") or {}),
        patch(
            f"{module}.resolve_openlineage_settings",
            side_effect=lambda config: order.append("openlineage"),
        ),
        patch(f"{module}.resolve_metrics_settings", side_effect=reject),
        patch(f"{module}.load_registry") as registry,
    ):
        assert entry(["--project-root", str(tmp_path)]) == 2
    assert issubclass(MetricsProfileError, ValueError)
    assert order == (["metrics"] if batch else ["runtime", "openlineage", "metrics"])
    registry.assert_not_called()
    assert capsys.readouterr().err == (
        "metrics profile refused by existing ValueError handler\n"
    )


def test_batch_checks_metrics_before_registry_and_again_after_runtime_preparation(tmp_path):
    module = "janus.cli.run_all"
    order = []
    with (
        patch(f"{module}.load_environment_config", return_value={}),
        patch(
            f"{module}.resolve_metrics_settings",
            side_effect=lambda config: order.append("metrics"),
        ),
        patch(
            f"{module}.load_registry",
            side_effect=lambda *args, **kwargs: order.append("registry"),
        ),
        patch(f"{module}.BatchPlanner") as planner,
        patch(f"{module}.prepare_runtime", side_effect=lambda *args: order.append("runtime") or {}),
        patch(
            f"{module}.resolve_openlineage_settings",
            side_effect=lambda config: order.append("openlineage"),
        ),
        patch(f"{module}._execute_batch", return_value=0),
    ):
        planner.return_value.plan.side_effect = lambda *args, **kwargs: order.append("plan")
        assert run_all_main(["--project-root", str(tmp_path)]) == 0
    assert order == ["metrics", "registry", "plan", "runtime", "openlineage", "metrics"]


def test_metrics_public_package_loads_no_compute_or_catalog_engine():
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import janus.observability.metrics, json, sys; "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.split('.', 1)[0] in {'pyspark', 'pyiceberg', 'pyarrow'})))",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
