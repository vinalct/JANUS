"""every documented invocation keeps its stdout, stderr and exit code."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.cli_golden import (
    FIXTURE_TREES,
    GOLDEN_SUFFIXES,
    GOLDENS_DIR,
    INVOCATIONS,
    REPO_ROOT,
    CapturedInvocation,
    Invocation,
    capture,
    engines_importable,
    normalize,
    read_golden,
)

HELP_INVOCATIONS = tuple(invocation for invocation in INVOCATIONS if "--help" in invocation.argv)


@pytest.mark.parametrize("invocation", INVOCATIONS, ids=lambda invocation: invocation.name)
def test_documented_invocation_matches_its_golden(invocation: Invocation, tmp_path: Path) -> None:
    present = engines_importable()
    if invocation.requires_spark and present:
        pytest.skip(
            f"{invocation.name} records the engine-free host, and {', '.join(present)} is "
            "importable here; the engine-present variant belongs to the container gate"
        )

    captured = capture(invocation, tmp_path / "capture")
    golden = read_golden(invocation)

    assert golden is not None, (
        f"no golden for {invocation.name}: capture with `python -m tests.support.cli_golden "
        "<dir>` and review the diff before copying it into tests/fixtures/cli_goldens"
    )
    assert captured == golden
    assert capture(invocation, tmp_path / "recapture") == golden


def test_the_corpus_names_each_invocation_once() -> None:
    names = [invocation.name for invocation in INVOCATIONS]

    assert len(names) >= 20, "the AC-1 corpus must cover every documented form"
    assert len(names) == len(set(names))
    assert all(name.isidentifier() and name == name.lower() for name in names)


def test_every_golden_file_belongs_to_an_invocation() -> None:
    expected = {
        f"{invocation.name}{suffix}" for invocation in INVOCATIONS for suffix in GOLDEN_SUFFIXES
    }
    present = {path.name for path in GOLDENS_DIR.iterdir() if path.is_file()}

    assert present, f"no goldens under {GOLDENS_DIR}"
    assert present <= expected, f"stale goldens: {sorted(present - expected)}"


def test_every_invocation_reads_a_profile_its_fixture_ships() -> None:
    for invocation in INVOCATIONS:
        trees = FIXTURE_TREES[invocation.project_root_fixture]
        profiles = [
            REPO_ROOT / tree / "environments" / f"{invocation.environment}.yaml" for tree in trees
        ]
        assert any(profile.is_file() for profile in profiles), invocation.name
        assert invocation.documented_at, invocation.name


@pytest.mark.parametrize("invocation", HELP_INVOCATIONS, ids=lambda invocation: invocation.name)
def test_help_goldens_list_every_declared_option(
    invocation: Invocation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed: list[argparse.ArgumentParser] = []
    format_help = argparse.ArgumentParser.format_help

    def recording_format_help(parser: argparse.ArgumentParser) -> str:
        printed.append(parser)
        return format_help(parser)

    monkeypatch.setattr(argparse.ArgumentParser, "format_help", recording_format_help)
    capture(invocation, tmp_path)
    declared = {
        option
        for parser in printed
        for action in parser._actions
        if action.help != argparse.SUPPRESS
        for option in action.option_strings
    }
    golden = read_golden(invocation)

    assert declared, f"{invocation.name} printed no parser help"
    assert golden is not None
    assert sorted(option for option in declared if option not in golden.stdout) == []


def _shared_options(parser: argparse.ArgumentParser) -> dict[str, tuple[object, ...]]:
    return {
        action.dest: (tuple(action.option_strings), action.default, action.type, action.help)
        for action in parser._actions
        if action.dest in {"environment", "project_root"}
    }


def test_the_parent_options_read_exactly_as_runs_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-1: a new verb takes --environment and --project-root the way `run` does."""
    from janus.cli.dispatch import build_parent_parser
    from janus.cli.run import parse_run_args

    built: list[argparse.ArgumentParser] = []
    parse_args = argparse.ArgumentParser.parse_args

    def recording_parse_args(parser: argparse.ArgumentParser, *args, **kwargs):
        built.append(parser)
        return parse_args(parser, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", recording_parse_args)
    parse_run_args([])
    (run_parser,) = built
    parent = _shared_options(build_parent_parser())

    assert parent.keys() == {"environment", "project_root"}
    assert parent == _shared_options(run_parser)


def test_a_configured_verb_parses_its_own_options_over_the_parent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registration path every new verb takes: `configure` plus a namespace handler."""
    from janus.cli import dispatch

    received: list[argparse.Namespace] = []

    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--format", choices=("text", "json"), default="text")

    def handler(args: argparse.Namespace) -> int:
        received.append(args)
        return 0

    probe = dispatch.Verb("probe", "Registered the way a new verb is.", configure, handler)
    registered = dispatch.verbs()
    monkeypatch.setattr(dispatch, "verbs", lambda: (*registered, probe))

    assert dispatch.main(["probe", "--environment", "cluster", "--format", "json"]) == 0
    (args,) = received
    assert (args.environment, args.format) == ("cluster", "json")
    assert isinstance(args.project_root, Path)

    with pytest.raises(SystemExit) as exc_info:
        dispatch.main(["probe", "--help"])
    usage = capsys.readouterr().out

    assert exc_info.value.code == 0
    assert usage.startswith("usage: janus probe")
    assert all(option in usage for option in ("--environment", "--project-root", "--format"))


def test_the_normalizer_replaces_clock_reads_and_keeps_what_the_argv_pinned(
    tmp_path: Path,
) -> None:
    clock = datetime(2031, 5, 6, 7, 8, 9, 123456, tzinfo=UTC)
    stamp = clock.strftime("%Y%m%dT%H%M%SZ")
    stdout = "\n".join(
        (
            f'"root": "{tmp_path}/data",',
            '"pinned": "2026-04-09T12:00:00+00:00",',
            '"pinned_run_id": "run-local-x-20260409T120000Z",',
            f'"clock": "{clock.isoformat()}",',
            f'"run_id": "pipeline-local-{stamp}-x-a1-0123456789",',
            '"pinned_attempt": "backfill-1-x-a1-0123456789",',
            f'"events": "events-{clock.date().isoformat()}.ndjson",',
            '"fixture_date": "2002-01-01",',
            '"duration_seconds": 0.000743,',
            '"rows": 3,   ',
        )
    )
    captured = CapturedInvocation(
        "synthetic", ("--started-at", "2026-04-09T12:00:00Z"), 0, stdout, ""
    )

    normalized = normalize(captured, project_root=tmp_path)

    assert normalized.stdout.splitlines() == [
        '"root": "<project-root>/data",',
        '"pinned": "2026-04-09T12:00:00+00:00",',
        '"pinned_run_id": "run-local-x-20260409T120000Z",',
        '"clock": "<timestamp>",',
        '"run_id": "pipeline-local-<stamp>-x-a1-<digest>",',
        '"pinned_attempt": "backfill-1-x-a1-0123456789",',
        '"events": "events-<date>.ndjson",',
        '"fixture_date": "2002-01-01",',
        '"duration_seconds": <duration>,',
        '"rows": 3,',
    ]
    assert normalized.stdout.endswith("\n")
    assert (normalized.exit_code, normalized.stderr) == (0, "")
