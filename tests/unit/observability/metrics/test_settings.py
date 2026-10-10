from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from janus.cli.run_all import main as run_all_main
from janus.main import main
from janus.observability import IcebergAppendOutcome, IcebergAppendResult, build_run_event_emitter
from janus.utils.environment import load_environment_config
from tests.support import observability_baseline as baseline
from tests.support.metrics_order23 import WarningSpy, exit_code, record, sink, write_profile
from tests.unit.observability.test_runs_table_contract import _load_clean_profile

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="metrics settings and fail-closed profile reads are absent",
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
    with pytest.raises(MetricsProfileError) as raised:
        load_environment_config("local", tmp_path)
    message = str(raised.value)
    assert "observability.metrics.transport" in message
    assert "htpp" in message
    assert all(kind in message for kind in ("disabled", "file", "textfile", "statsd"))


@pytest.mark.parametrize("batch", [False, True], ids=["single-source", "run-all"])
def test_bad_transport_makes_both_entry_points_exit_two_before_registry_load(
    batch, tmp_path, capsys
):
    write_profile(tmp_path, {"transport": "htpp"})
    entry = run_all_main if batch else main
    with patch(
        "janus.cli.run_all.load_registry" if batch else "janus.cli.run.load_registry"
    ) as load:
        code = exit_code(lambda: entry(["--project-root", str(tmp_path)]))
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


def test_relative_file_path_is_resolved_under_metadata_zone(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = sink(tmp_path, "file", path="nested/metrics")
    assert (
        destination.emit(record(tmp_path), budget_seconds=1).outcome
        is MetricsEmissionOutcome.EMITTED
    )
    files = list((tmp_path / "nested/metrics").glob("*.ndjson"))
    assert len(files) == 1
    assert files[0].resolve().is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("path", ["/tmp/escaped-order23", "../escaped-order23"])
def test_absolute_and_parent_escape_paths_are_refused(path):
    from janus.observability.metrics.settings import MetricsProfileError, resolve_metrics_settings

    with pytest.raises(MetricsProfileError, match="path"):
        resolve_metrics_settings(
            {"observability": {"metrics": {"transport": "file", "path": path}}}
        )


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
