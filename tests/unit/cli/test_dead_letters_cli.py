from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from janus.checkpoints import DeadLetterStore, ExtractionProgressStore
from janus.lineage import MetadataZonePaths
from janus.planner import PlannedRun, Planner, PlanningRequest
from janus.runtime.executor import SourceExecutor
from tests.support.operator_cli import OPERATOR_ENV, arm_spark_tripwire, run_janus
from tests.support.semantics_fixtures import (
    CLEAN,
    CLEAN_CONSUMER,
    CLEAN_PRODUCER,
    install_profile,
    materialize,
    tree_snapshot,
)

KEYS = ("window_start=2026-09-01", "window_start=2026-09-02")
OPERATOR = {OPERATOR_ENV: "ops-tester"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return materialize(CLEAN, tmp_path / "project")


def _plan(root: Path, source_id: str = CLEAN_PRODUCER):
    return (
        Planner()
        .plan(
            PlanningRequest.create(
                source_id=source_id,
                environment="local",
                project_root=root,
                run_id="run-seed",
                started_at=datetime(2026, 9, 18, 6, 0, tzinfo=UTC),
                include_disabled=True,
            )
        )
        .plan
    )


def _seed(root: Path, source_id: str = CLEAN_PRODUCER, keys: tuple[str, ...] = KEYS):
    plan = _plan(root, source_id)
    store = DeadLetterStore()
    for minute, key in enumerate(keys, start=1):
        store.record(
            plan,
            item_key=key,
            item_type="request_input",
            error=RuntimeError(f"API request failed with status 400: body excerpt for {key}"),
            metadata={"request_url": f"https://example.invalid/reference?{key}"},
            recorded_at=datetime(2026, 9, 18, 6, minute, tzinfo=UTC),
        )
    return plan, store


def _dead_letters(root: Path, action: str, *extra: str, source_id: str = CLEAN_PRODUCER):
    return run_janus(
        ("dead-letters", action, "--project-root", str(root), "--source-id", source_id, *extra),
        env=OPERATOR,
    )


class _ExecutedRun:
    def __init__(self, planned_run: PlannedRun) -> None:
        self.planned_run = planned_run
        self.is_successful = True

    def to_summary(self) -> dict[str, Any]:
        return {"status": "succeeded"}


@pytest.fixture
def executor_calls(monkeypatch: pytest.MonkeyPatch) -> list[PlannedRun]:
    """Replace the one execution seam; both `run --execute` and `replay --execute` hit it."""
    calls: list[PlannedRun] = []

    def execute(self, planned_run, spark_provider, environment_config):
        calls.append(planned_run)
        return _ExecutedRun(planned_run)

    monkeypatch.setattr(SourceExecutor, "execute", execute)
    return calls


@pytest.fixture
def planning_requests(monkeypatch: pytest.MonkeyPatch) -> list[PlanningRequest]:
    requests: list[PlanningRequest] = []
    plan = Planner.plan

    def recording(self, request, *, registry=None):
        requests.append(request)
        return plan(self, request, registry=registry)

    monkeypatch.setattr(Planner, "plan", recording)
    return requests


# ---------------------------------------------------------------------------------------
# list


def test_list_prints_every_entry_with_its_error_and_metadata(root: Path) -> None:
    """The error message is printed whole: it carries the bounded response excerpt, which is
    the only record of *why* the item was given up on."""
    _seed(root)

    result = _dead_letters(root, "list")

    assert result.exit_code == 0, result.output
    for key in KEYS:
        assert key in result.stdout
        assert f"API request failed with status 400: body excerpt for {key}" in result.stdout
        assert f"https://example.invalid/reference?{key}" in result.stdout
    assert "RuntimeError" in result.stdout


def test_list_json_is_the_recorded_state(root: Path) -> None:
    plan, store = _seed(root)

    result = _dead_letters(root, "list", "--format", "json")
    payload = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert payload["source_id"] == CLEAN_PRODUCER
    assert payload["entries"] == store.load(plan).to_dict()["entries"]
    assert result.stdout == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_list_narrows_to_one_item_key(root: Path) -> None:
    _seed(root)

    result = _dead_letters(root, "list", "--item-key", KEYS[1], "--format", "json")

    assert result.exit_code == 0, result.output
    assert [entry["item_key"] for entry in json.loads(result.stdout)["entries"]] == [KEYS[1]]


def test_list_with_no_state_is_a_healthy_answer(root: Path) -> None:
    result = _dead_letters(root, "list")

    assert result.exit_code == 0, result.output
    assert f"no dead letters recorded for {CLEAN_PRODUCER}" in result.stdout


def test_list_names_a_requested_key_that_is_not_recorded(root: Path) -> None:
    """A read does not fail on content: absence is an answer, shown beside the recorded keys
    so a typo is visible."""
    _seed(root)

    result = _dead_letters(root, "list", "--item-key", "window_start=1999-01-01")

    assert result.exit_code == 0, result.output
    assert "0 of 2 dead letter(s)" in result.stdout
    assert "'window_start=1999-01-01'" in result.stdout
    assert all(repr(key) in result.stdout for key in KEYS)


def test_parent_options_before_the_action_are_not_reset_by_its_parser(root: Path) -> None:
    """argparse copies an action parser's defaults over what the verb parser parsed. The
    action parsers suppress theirs, so `--project-root` may come before the action too."""
    _seed(root)

    result = run_janus(
        ("dead-letters", "--project-root", str(root), "list", "--source-id", CLEAN_PRODUCER),
        env=OPERATOR,
    )

    assert result.exit_code == 0, result.output
    assert all(key in result.stdout for key in KEYS)


# ---------------------------------------------------------------------------------------
# release


def test_release_removes_exactly_the_named_key_and_writes_history(root: Path) -> None:
    plan, store = _seed(root)

    result = _dead_letters(root, "release", "--item-key", KEYS[0], "--reason", "upstream fixed")

    assert result.exit_code == 0, result.output
    assert store.load(plan).item_keys == frozenset({KEYS[1]})
    (history,) = sorted(
        (MetadataZonePaths.from_plan(plan).dead_letters_dir / "history").glob("*.json")
    )
    payload = json.loads(history.read_text(encoding="utf-8"))
    assert payload["operator"] == "ops-tester"
    assert payload["reason"] == "upstream fixed"
    assert [entry["item_key"] for entry in payload["released_entries"]] == [KEYS[0]]
    assert str(history.name) in result.stdout


def test_release_all_deletes_the_state_file(root: Path) -> None:
    plan, store = _seed(root)

    result = _dead_letters(root, "release", "--all", "--reason", "retry everything")

    assert result.exit_code == 0, result.output
    assert store.path(plan).exists() is False


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (("--item-key", KEYS[0]), "--reason"),
        (("--reason", "no selector"), "--item-key"),
        (("--item-key", KEYS[0], "--all", "--reason", "both"), "--all"),
    ],
    ids=["no-reason", "no-selector", "key-and-all"],
)
def test_release_argument_errors_exit_2_and_touch_nothing(
    root: Path, flags: tuple[str, ...], named: str
) -> None:
    """Goal 5: no state change without a reason; and never a bare `release` that empties."""
    plan, store = _seed(root)
    before = store.path(plan).read_bytes()

    result = _dead_letters(root, "release", *flags)

    assert result.exit_code == 2
    assert named in result.stderr
    assert store.path(plan).read_bytes() == before


def test_releasing_an_unknown_key_names_the_available_ones_and_changes_nothing(root: Path) -> None:
    plan, store = _seed(root)
    before = tree_snapshot(root)

    result = _dead_letters(
        root, "release", "--item-key", "window_start=1999-01-01", "--reason", "typo"
    )

    assert result.exit_code == 2
    assert "window_start=1999-01-01" in result.stderr
    assert all(key in result.stderr for key in KEYS)
    assert tree_snapshot(root) == before


def test_release_with_no_state_exits_2(root: Path) -> None:
    result = _dead_letters(root, "release", "--all", "--reason", "nothing to release")

    assert result.exit_code == 2
    assert CLEAN_PRODUCER in result.stderr


def test_release_json_is_the_history_record_and_where_it_was_written(root: Path) -> None:
    _seed(root)

    result = _dead_letters(
        root, "release", "--all", "--reason", "upstream fixed", "--format", "json"
    )
    payload = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert result.stdout == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert payload["metadata"] == {"source": "cli"}
    assert payload["state_deleted"] is True
    history = root.resolve() / payload["history_path"]
    written = {
        key: value
        for key, value in payload.items()
        if key not in {"history_path", "state_path", "state_deleted"}
    }
    assert json.loads(history.read_text(encoding="utf-8")) == written


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (("--item-key", KEYS[0], "--reason", "   "), "--reason"),
        (("--item-key", " ", "--reason", "blank key"), "--item-key"),
    ],
    ids=["blank-reason", "blank-key"],
)
def test_a_blank_reason_or_item_key_is_an_argument_error(
    root: Path, flags: tuple[str, ...], named: str
) -> None:
    """A blank key must never act as a wildcard, and a blank reason explains nothing."""
    _seed(root)
    before = tree_snapshot(root)

    for action in ("release", "replay"):
        result = _dead_letters(root, action, *flags)
        assert result.exit_code == 2, (action, result.output)
        assert named in result.stderr

    assert tree_snapshot(root) == before


# ---------------------------------------------------------------------------------------
# replay


def test_replay_without_execute_is_a_dry_run_that_writes_nothing(root: Path) -> None:
    """Q4: the safe default prints what a resume would retry, and what stays skipped."""
    _seed(root)
    before = tree_snapshot(root)

    result = _dead_letters(root, "replay", "--item-key", KEYS[0], "--reason", "dry run")

    assert result.exit_code == 0, result.output
    assert KEYS[0] in result.stdout and KEYS[1] in result.stdout
    assert tree_snapshot(root) == before


def test_replay_execute_releases_then_plans_with_resume(
    root: Path,
    executor_calls: list[PlannedRun],
    planning_requests: list[PlanningRequest],
) -> None:
    """Without `resume=true`, `ResumeState.load` clears both stores and the other dead
    letters are forgotten: the attribute is asserted on the request itself."""
    install_profile(root, "local")
    plan, store = _seed(root)

    result = run_janus(
        (
            "dead-letters", "replay", "--project-root", str(root), "--environment", "local",
            "--source-id", CLEAN_PRODUCER, "--item-key", KEYS[0], "--reason", "upstream fixed",
            "--execute",
        ),
        env=OPERATOR,
    )

    assert result.exit_code == 0, result.output
    assert store.load(plan).item_keys == frozenset({KEYS[1]})
    (executed,) = executor_calls
    assert executed.plan.source.source_id == CLEAN_PRODUCER
    executed_requests = [r for r in planning_requests if r.attributes_as_dict().get("resume")]
    assert [r.attributes_as_dict()["resume"] for r in executed_requests] == ["true"]
    assert executed.plan.run_context.attributes_as_dict()["resume"] == "true"


def test_replay_execute_refuses_a_disabled_source_without_include_disabled(
    root: Path, executor_calls: list[PlannedRun]
) -> None:
    """D-9: state can be inspected on a disabled source; *executing* one needs the flag."""
    install_profile(root, "local")
    _seed(root, CLEAN_CONSUMER)
    before = tree_snapshot(root)

    result = run_janus(
        (
            "dead-letters", "replay", "--project-root", str(root), "--environment", "local",
            "--source-id", CLEAN_CONSUMER, "--all", "--reason", "retry", "--execute",
        ),
        env=OPERATOR,
    )

    assert result.exit_code == 2
    assert "--include-disabled" in result.stderr
    assert executor_calls == []
    assert tree_snapshot(root) == before


def test_replay_execute_and_run_execute_share_one_executor_call_site(
    root: Path, executor_calls: list[PlannedRun]
) -> None:
    """Q4's "sugar over the same executor" holds only if both commands reach one seam."""
    install_profile(root, "local")
    _seed(root)
    common = ("--project-root", str(root), "--environment", "local", "--source-id", CLEAN_PRODUCER)

    ran = run_janus(("--execute", *common), env=OPERATOR)
    replayed = run_janus(
        ("dead-letters", "replay", *common, "--all", "--reason", "retry", "--execute"),
        env=OPERATOR,
    )

    assert (ran.exit_code, replayed.exit_code) == (0, 0), ran.output + replayed.output
    assert [call.plan.source.source_id for call in executor_calls] == [CLEAN_PRODUCER] * 2


@pytest.mark.parametrize(
    "argv",
    [("list",), ("release", "--item-key", KEYS[0], "--reason", "tripwire")],
    ids=["list", "release"],
)
def test_list_and_release_never_acquire_a_spark_session(
    root: Path, monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...]
) -> None:
    _seed(root)
    arm_spark_tripwire(monkeypatch)

    assert _dead_letters(root, *argv).exit_code == 0


def test_the_dry_run_names_what_it_would_retry_and_what_stays_skipped(root: Path) -> None:
    plan, _store = _seed(root)
    ExtractionProgressStore().save(
        plan,
        page_number=3,
        request_index=4,
        artifact_count=3,
        completed_inputs=[("window_start=2026-08-31", 1)],
        current_input_key=KEYS[1],
        current_input_index=2,
        request_input_count=3,
    )
    before = tree_snapshot(root)

    result = _dead_letters(root, "replay", "--item-key", KEYS[1], "--reason", "dry run")

    assert result.exit_code == 0, result.output
    assert re.search(rf"would release\s+{re.escape(KEYS[1])}", result.stdout), result.stdout
    assert re.search(rf"stays skipped\s+{re.escape(KEYS[0])}", result.stdout), result.stdout
    assert f"request input 2 of 3 ({KEYS[1]}); last page 3; 1 input(s) completed" in result.stdout
    assert f"janus --environment local --source-id {CLEAN_PRODUCER} --execute --resume" in (
        result.stdout
    )
    assert tree_snapshot(root) == before


def test_the_dry_run_refuses_what_the_release_would_refuse(root: Path) -> None:
    """Built on `preview_release`: a dry run never promises a release that would fail."""
    _seed(root)
    before = tree_snapshot(root)

    result = _dead_letters(
        root, "replay", "--item-key", "window_start=1999-01-01", "--reason", "typo"
    )

    assert result.exit_code == 2
    assert "window_start=1999-01-01" in result.stderr
    assert all(key in result.stderr for key in KEYS)
    assert tree_snapshot(root) == before


def test_replay_execute_with_an_unreadable_profile_releases_nothing(
    root: Path, executor_calls: list[PlannedRun]
) -> None:
    """The profile is read before the release, so a misspelt environment writes nothing."""
    _seed(root)
    before = tree_snapshot(root)

    result = _dead_letters(
        root, "replay", "--environment", "nowhere", "--all", "--reason", "retry", "--execute"
    )

    assert result.exit_code == 2
    assert "nowhere" in result.stderr
    assert executor_calls == []
    assert tree_snapshot(root) == before


def test_replay_execute_prints_the_run_summary_and_reports_the_release_on_stderr(
    root: Path, executor_calls: list[PlannedRun]
) -> None:
    """stdout is the run's summary as `run --execute` prints it, so one parser reads both."""
    install_profile(root, "local")
    plan, _store = _seed(root)

    flags = ("--environment", "local", "--item-key", KEYS[0], "--reason", "fixed", "--execute")

    result = _dead_letters(root, "replay", *flags)
    summary = json.loads(result.stdout)

    assert result.exit_code == 0, result.output
    assert summary["executed_run"] == {"status": "succeeded"}
    attributes = summary["planned_run"]["run"]["attributes"]
    assert (attributes["resume"], attributes["trigger"]) == ("true", "cli")
    assert "released 1 dead letter(s); 1 remain" in result.stderr
    (history,) = (MetadataZonePaths.from_plan(plan).dead_letters_dir / "history").glob("*.json")
    assert json.loads(history.read_text(encoding="utf-8"))["metadata"] == {
        "replay": "true",
        "source": "cli",
    }


def test_replay_execute_exits_1_when_the_run_fails(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule `run --execute` follows: the run happened, and it did not succeed."""
    install_profile(root, "local")
    _seed(root)

    def execute(self, planned_run, spark_provider, environment_config):
        executed = _ExecutedRun(planned_run)
        executed.is_successful = False
        return executed

    monkeypatch.setattr(SourceExecutor, "execute", execute)

    result = _dead_letters(
        root, "replay", "--environment", "local", "--all", "--reason", "retry", "--execute"
    )

    assert result.exit_code == 1, result.output


def test_replay_execute_runs_a_disabled_source_with_include_disabled(
    root: Path, executor_calls: list[PlannedRun]
) -> None:
    install_profile(root, "local")
    _seed(root, CLEAN_CONSUMER)

    flags = ("--environment", "local", "--all", "--reason", "retry", "--include-disabled")

    result = _dead_letters(root, "replay", *flags, "--execute", source_id=CLEAN_CONSUMER)

    assert result.exit_code == 0, result.output
    assert [call.plan.source.source_id for call in executor_calls] == [CLEAN_CONSUMER]


# ---------------------------------------------------------------------------------------
# operator identity 


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"JANUS_OPERATOR": "ops-tester", "USER": "unix-user"}, "ops-tester"),
        ({"USER": "unix-user"}, "unix-user"),
        ({}, "unknown"),
    ],
    ids=["janus-operator", "user", "neither"],
)
def test_the_operator_is_janus_operator_then_user_then_unknown(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], expected: str
) -> None:
    from janus.cli.operator import resolve_operator

    for key in ("JANUS_OPERATOR", "USER", "LOGNAME", "USERNAME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    assert resolve_operator() == expected


@pytest.mark.parametrize("operator", ["ops-tester", "../../etc/passwd", "Ana Maria / ops"])
def test_a_manual_run_id_is_path_safe_whatever_the_operator_is_called(operator: str) -> None:
    from janus.cli.operator import manual_run_id

    run_id = manual_run_id(operator, now=datetime(2026, 9, 21, 9, 15, 3, tzinfo=UTC))

    assert re.fullmatch(r"manual-20260921T091503Z-[a-z0-9-]+", run_id), run_id


@pytest.mark.parametrize("reason", [None, "", "   "], ids=["absent", "empty", "blank"])
def test_a_missing_or_blank_reason_is_an_argument_error(
    capsys: pytest.CaptureFixture[str], reason: str | None
) -> None:
    import argparse

    from janus.cli.operator import require_reason

    parser = argparse.ArgumentParser(prog="janus dead-letters release")

    with pytest.raises(SystemExit) as refused:
        require_reason(parser, reason)

    assert refused.value.code == 2
    assert "--reason" in capsys.readouterr().err


def test_a_reason_is_kept_without_its_surrounding_whitespace() -> None:
    import argparse

    from janus.cli.operator import require_reason

    assert require_reason(argparse.ArgumentParser(), "  upstream fixed \n") == "upstream fixed"
