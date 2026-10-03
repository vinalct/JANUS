"""AC-1 goldens: every documented CLI invocation, captured in process."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

from janus.cli.common import parse_started_at
from janus.main import main as janus_main

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDENS_DIR = REPO_ROOT / "tests" / "fixtures" / "cli_goldens"
GOLDEN_SUFFIXES = (".out", ".err", ".code")

ENGINE_MODULES = ("pyspark", "pyiceberg", "pyarrow")

TERMINAL_COLUMNS = "80"

FIXTURE_TREES: dict[str, tuple[str, ...]] = {
    "repository": ("conf",),
}
_IGNORED = shutil.ignore_patterns("*.env", "__pycache__")

_PROJECT_ROOT_TOKEN = "<project-root>"
_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")
_STAMP = re.compile(r"(?<![0-9])\d{8}T\d{6}Z")
_DATE = re.compile(r"(?<![0-9])\d{4}-\d{2}-\d{2}(?![0-9])")
_DURATION = re.compile(r'("[a-z_]*duration_seconds": )-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

_CLOCK_DIGEST = re.compile(r"(<stamp>-(?:[a-z0-9]+-)*?a\d+-)[0-9a-f]+(?![0-9a-z])")
_TRACEBACK_HEADER = "Traceback (most recent call last):\n  <frames>\n"


@dataclass(frozen=True, slots=True)
class Invocation:
    """One documented CLI form, named so a golden file can be found by name."""

    name: str
    argv: tuple[str, ...]
    project_root_fixture: str
    environment: str = "local"
    requires_spark: bool = False
    documented_at: str = ""


@dataclass(frozen=True, slots=True)
class CapturedInvocation:
    name: str
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


_BRONZE_TABLE = "bronze_inep.censo_escolar_microdados"
_LIST_TASK = "docs/tasks/operator-command-surface/list_command.md"
_DEAD_LETTERS_TASK = "docs/tasks/operator-command-surface/dead_letters_command.md"

INVOCATIONS: tuple[Invocation, ...] = (
    Invocation(
        "run_help",
        ("--help",),
        "repository",
        documented_at=".github/workflows/ci.yml:44",
    ),
    Invocation(
        "run_verb_help",
        ("run", "--help"),
        "repository",
        documented_at="docs/tasks/dispatcher.md:113",
    ),
    Invocation(
        "run_all_help",
        ("run-all", "--help"),
        "repository",
        documented_at="docs/orchestration.md:46-54",
    ),
    Invocation(
        "contract_help",
        ("contract", "--help"),
        "repository",
        documented_at="README.md:118",
    ),
    Invocation(
        "contract_draft_help",
        ("contract", "draft", "--help"),
        "repository",
        documented_at="docs/data-contracts.md:99-100",
    ),
    Invocation(
        "no_arguments",
        (),
        "repository",
        documented_at="docs/tasks/dispatcher.md:136",
    ),
    Invocation(
        "profile_local",
        ("--environment", "local"),
        "repository",
        documented_at="Makefile:281; docs/reproducibility.md:166,247",
    ),
    Invocation(
        "profile_local_with_spark",
        ("--environment", "local", "--with-spark"),
        "repository",
        requires_spark=True,
        documented_at="Makefile:278; docs/reproducibility.md:180,248",
    ),
    Invocation(
        "profile_cluster",
        ("--environment", "cluster"),
        "repository",
        environment="cluster",
        documented_at="docs/reproducibility.md:267",
    ),
    Invocation(
        "profile_cluster_with_spark",
        ("--environment", "cluster", "--with-spark"),
        "repository",
        environment="cluster",
        requires_spark=True,
        documented_at="docs/reproducibility.md:273",
    ),
    Invocation(
        "plan_pinned",
        (
            "--environment",
            "local",
            "--source-id",
            "federal_open_data_example",
            "--run-id",
            "run-20260409-demo",
            "--started-at",
            "2026-04-09T12:00:00+00:00",
        ),
        "repository",
        documented_at="docs/reproducibility.md:213-217",
    ),
    Invocation(
        "plan_disabled_source",
        ("--environment", "local", "--source-id", "ibge_pib_brasil", "--include-disabled"),
        "repository",
        documented_at="README.md:110-114",
    ),

    Invocation(
        "execute_disabled_source_refused",
        ("--environment", "local", "--source-id", "ibge_pib_brasil", "--execute"),
        "repository",
        documented_at="README.md:124-129; docs/reproducibility.md:400-405",
    ),
    Invocation(
        "ingest_disabled_source_refused",
        (
            "--environment",
            "local",
            "--source-id",
            "inep_censo_escolar_microdados",
            "--ingest-raw-to-bronze",
            "--bronze-table",
            _BRONZE_TABLE,
        ),
        "repository",
        documented_at="docs/reproducibility.md:410-416",
    ),
    Invocation(
        "error_execute_without_source_id",
        ("--environment", "local", "--execute"),
        "repository",
        documented_at="src/janus/main.py:108-109",
    ),
    Invocation(
        "error_ingest_without_source_id",
        ("--environment", "local", "--ingest-raw-to-bronze", "--bronze-table", _BRONZE_TABLE),
        "repository",
        documented_at="src/janus/main.py:110-111",
    ),
    Invocation(
        "error_ingest_without_bronze_table",
        (
            "--environment",
            "local",
            "--source-id",
            "inep_censo_escolar_microdados",
            "--include-disabled",
            "--ingest-raw-to-bronze",
            "--with-spark",
        ),
        "repository",
        documented_at="docs/reproducibility.md:370-371; src/janus/main.py:112-113",
    ),
    Invocation(
        "error_execute_with_ingest",
        (
            "--environment",
            "local",
            "--source-id",
            "inep_censo_escolar_microdados",
            "--include-disabled",
            "--execute",
            "--ingest-raw-to-bronze",
            "--bronze-table",
            _BRONZE_TABLE,
        ),
        "repository",
        documented_at="src/janus/main.py:114-115",
    ),
    Invocation(
        "error_include_disabled_without_source_id",
        ("--environment", "local", "--include-disabled"),
        "repository",
        documented_at="src/janus/main.py:116-117",
    ),
    Invocation(
        "error_unknown_verb",
        ("frobnicate",),
        "repository",
        documented_at="docs/operator_command_surface.md:77",
    ),
    Invocation(
        "run_all_checked_in_registry",
        ("run-all", "--environment", "local"),
        "repository",
        requires_spark=True,
        documented_at="README.md:159; docs/orchestration.md:42",
    ),
    Invocation(
        "run_all_backfill_empty_selection",
        (
            "run-all",
            "--environment",
            "local",
            "--tag",
            "monthly",
            "--pipeline-run-id",
            "backfill-2026-08-attempt-1",
        ),
        "repository",
        documented_at="docs/orchestration.md:219-222",
    ),
    Invocation(
        "run_all_error_tag_with_domain",
        ("run-all", "--tag", "consumer", "--domain", "reporting"),
        "repository",
        documented_at="docs/orchestration.md:57-58",
    ),
    Invocation(
        "run_all_rejects_include_disabled",
        ("run-all", "--include-disabled"),
        "repository",
        documented_at="docs/orchestration.md:330-333",
    ),
    Invocation(
        "run_all_rejects_single_source_flags",
        (
            "run-all",
            "--source-id",
            "federal_open_data_example",
            "--execute",
            "--run-id",
            "run-20260409-demo",
            "--include-disabled",
            "--bronze-table",
            _BRONZE_TABLE,
            "--ingest-raw-to-bronze",
            "--with-spark",
        ),
        "repository",
        documented_at="docs/orchestration.md:335-336; src/janus/cli/run_all.py:47-55",
    ),
    Invocation(
        "contract_draft_missing_raw_run",
        (
            "contract",
            "draft",
            "--source-id",
            "ibge_pib_brasil",
            "--from-raw",
            "run-20260409-demo",
            "--include-disabled",
            "--out",
            "conf/contracts/estatisticas/pib_brasil_draft.yaml",
        ),
        "repository",
        documented_at="docs/data-contracts.md:99",
    ),
    Invocation(
        "validate_help",
        ("validate", "--help"),
        "repository",
        documented_at="docs/tasks/operator-command-surface/validate_registry_half.md:45-48",
    ),
    Invocation(
        "validate_checked_in_registry",
        ("validate", "--format", "json"),
        "repository",
        documented_at="docs/operator_command_surface_.md:79",
    ),
    Invocation(
        "list_help",
        ("list", "--help"),
        "repository",
        documented_at=f"{_LIST_TASK}:44-48",
    ),
    Invocation(
        "list_table",
        ("list",),
        "repository",
        documented_at=f"{_LIST_TASK}:60-73",
    ),
    Invocation(
        "list_json",
        ("list", "--format", "json"),
        "repository",
        documented_at=f"{_LIST_TASK}:75-90",
    ),
    Invocation(
        "list_graph",
        ("list", "--graph"),
        "repository",
        documented_at=f"{_LIST_TASK}:92-107",
    ),
    Invocation(
        "list_filtered_by_tag",
        ("list", "--tag", "ibge"),
        "repository",
        documented_at=f"{_LIST_TASK}:154-155",
    ),
    Invocation(
        "dead_letters_help",
        ("dead-letters", "--help"),
        "repository",
        documented_at=f"{_DEAD_LETTERS_TASK}:42-50",
    ),
    Invocation(
        "dead_letters_replay_help",
        ("dead-letters", "replay", "--help"),
        "repository",
        documented_at=f"{_DEAD_LETTERS_TASK}:78-86,154-155",
    ),
    Invocation(
        "dead_letters_list_no_state",
        ("dead-letters", "list", "--source-id", "ibge_pib_brasil"),
        "repository",
        documented_at=f"{_DEAD_LETTERS_TASK}:69-70",
    ),
)


def materialize_fixture(name: str, workspace: Path) -> Path:
    """Copy the named fixture tree into ``workspace`` and return it as the project root."""
    workspace.mkdir(parents=True, exist_ok=True)
    for relative in FIXTURE_TREES[name]:
        shutil.copytree(REPO_ROOT / relative, workspace / relative, ignore=_IGNORED)
    return workspace


def run_invocation(invocation: Invocation, *, project_root: Path) -> CapturedInvocation:
    """Run one invocation in process, the way the ``janus`` console script would."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        _process_state(project_root, invocation.argv),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            exit_code = _exit_status(janus_main(list(invocation.argv)), stderr)
        except SystemExit as exc:
            exit_code = _exit_status(exc.code, stderr)
        except Exception as exc:
            stderr.write(_TRACEBACK_HEADER + "".join(traceback.format_exception_only(exc)))
            exit_code = 1
    return CapturedInvocation(
        invocation.name, invocation.argv, exit_code, stdout.getvalue(), stderr.getvalue()
    )


def normalize(captured: CapturedInvocation, *, project_root: Path) -> CapturedInvocation:
    """Replace the workspace path, the derived program name and clock reads; nothing else."""
    pinned = _pinned_instants(captured.argv)
    texts = (captured.stdout, captured.stderr)
    clock_dates = {day for text in texts for day in _clock_dates(text, pinned)}
    stdout, stderr = (
        _normalize_text(text, project_root=project_root, pinned=pinned, clock_dates=clock_dates)
        for text in texts
    )
    return replace(captured, stdout=stdout, stderr=stderr)


def capture(invocation: Invocation, workspace: Path) -> CapturedInvocation:
    """Materialize the fixture in ``workspace``, run the invocation, and normalize it."""
    project_root = materialize_fixture(invocation.project_root_fixture, workspace)
    captured = run_invocation(invocation, project_root=project_root)
    return normalize(captured, project_root=project_root)


def engines_importable() -> tuple[str, ...]:
    return tuple(name for name in ENGINE_MODULES if importlib.util.find_spec(name) is not None)


def golden_paths(name: str, directory: Path = GOLDENS_DIR) -> tuple[Path, Path, Path]:
    out, err, code = (directory / f"{name}{suffix}" for suffix in GOLDEN_SUFFIXES)
    return out, err, code


def write_golden(captured: CapturedInvocation, directory: Path = GOLDENS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    out, err, code = golden_paths(captured.name, directory)
    contents = ((out, captured.stdout), (err, captured.stderr), (code, f"{captured.exit_code}\n"))
    for path, text in contents:
        path.write_text(text, encoding="utf-8", newline="\n")


def read_golden(invocation: Invocation, directory: Path = GOLDENS_DIR) -> CapturedInvocation | None:
    out, err, code = golden_paths(invocation.name, directory)
    if not all(path.is_file() for path in (out, err, code)):
        return None
    return CapturedInvocation(
        invocation.name,
        invocation.argv,
        int(code.read_text(encoding="utf-8")),
        out.read_text(encoding="utf-8"),
        err.read_text(encoding="utf-8"),
    )


@contextlib.contextmanager
def _process_state(working_directory: Path, argv: Sequence[str]) -> Iterator[None]:
    saved_cwd, saved_argv, saved_environ = Path.cwd(), sys.argv, dict(os.environ)
    try:
        os.chdir(working_directory)
        sys.argv = ["janus", *argv]
        for key in [key for key in os.environ if key.startswith("JANUS_")]:
            del os.environ[key]
        os.environ["COLUMNS"] = TERMINAL_COLUMNS
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def _exit_status(code: object, stderr: io.StringIO) -> int:
    """``sys.exit`` semantics: ``None`` is 0, an int is itself, anything else is printed."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    stderr.write(f"{code}\n")
    return 1


def _pinned_instants(argv: Sequence[str]) -> tuple[datetime, ...]:
    pinned = []
    for index, token in enumerate(argv):
        value = argv[index + 1] if token == "--started-at" and index + 1 < len(argv) else None
        if token.startswith("--started-at="):
            value = token.partition("=")[2]
        if value is not None:
            with contextlib.suppress(argparse.ArgumentTypeError):
                pinned.append(parse_started_at(value))
    return tuple(pinned)


def _is_pinned_instant(text: str, pinned: tuple[datetime, ...]) -> bool:
    return datetime.fromisoformat(text) in pinned


def _is_pinned_stamp(text: str, pinned: tuple[datetime, ...]) -> bool:
    return any(text == instant.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ") for instant in pinned)


def _clock_dates(text: str, pinned: tuple[datetime, ...]) -> set[str]:
    """The UTC dates of every clock read in ``text``: a date-named file shares them."""
    days: set[date] = set()
    for match in _INSTANT.finditer(text):
        if not _is_pinned_instant(match.group(), pinned):
            days.add(datetime.fromisoformat(match.group()).astimezone(UTC).date())
    for match in _STAMP.finditer(text):
        if not _is_pinned_stamp(match.group(), pinned):
            days.add(datetime.strptime(match.group(), "%Y%m%dT%H%M%SZ").date())
    return {day.isoformat() for day in days}


def _normalize_text(
    text: str, *, project_root: Path, pinned: tuple[datetime, ...], clock_dates: set[str]
) -> str:
    for root in sorted({str(project_root.resolve()), str(project_root)}, key=len, reverse=True):
        text = text.replace(root, _PROJECT_ROOT_TOKEN)
    text = _normalize_program_name(text)

    def instant(match: re.Match[str]) -> str:
        return match.group() if _is_pinned_instant(match.group(), pinned) else "<timestamp>"

    def stamp(match: re.Match[str]) -> str:
        return match.group() if _is_pinned_stamp(match.group(), pinned) else "<stamp>"

    def day(match: re.Match[str]) -> str:
        return "<date>" if match.group() in clock_dates else match.group()

    text = _DATE.sub(day, _STAMP.sub(stamp, _INSTANT.sub(instant, text)))
    text = _CLOCK_DIGEST.sub(r"\1<digest>", text)
    text = _DURATION.sub(r"\1<duration>", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines) + "\n" if lines else ""


def _normalize_program_name(text: str) -> str:
    """Map the default ``prog`` argparse derives from the launcher (``python -m ...``) to janus."""
    with _argv_zero("janus"):
        derived = argparse.ArgumentParser().prog
    if derived == "janus":
        return text
    text = re.sub(rf"^usage: {re.escape(derived)}\b", "usage: janus", text, flags=re.MULTILINE)
    return re.sub(rf"^{re.escape(derived)}: error:", "janus: error:", text, flags=re.MULTILINE)


@contextlib.contextmanager
def _argv_zero(value: str) -> Iterator[None]:
    saved = sys.argv
    sys.argv = [value, *saved[1:]]
    try:
        yield
    finally:
        sys.argv = saved


def _write_all(output_dir: Path) -> int:
    with tempfile.TemporaryDirectory() as scratch:
        for invocation in INVOCATIONS:
            captured = capture(invocation, Path(scratch) / invocation.name)
            write_golden(captured, output_dir)
            sizes = " ".join(
                f"{path.suffix[1:]}={path.stat().st_size}"
                for path in golden_paths(invocation.name, output_dir)
            )
            print(f"{invocation.name}  exit={captured.exit_code}  {sizes}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.support.cli_golden OUTPUT_DIR")
    raise SystemExit(_write_all(Path(sys.argv[1])))
