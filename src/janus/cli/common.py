"""Small CLI conventions shared by single-source and batch commands."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path


def default_project_root() -> Path:
    """Resolve the project root from the environment, cwd, or installed package."""
    env_project_root = os.getenv("JANUS_PROJECT_ROOT")
    if env_project_root:
        return Path(env_project_root)

    cwd = Path.cwd()
    if (cwd / "conf" / "environments").exists():
        return cwd

    return Path(__file__).resolve().parents[3]


def build_parent_parser(*, suppress_defaults: bool = False) -> argparse.ArgumentParser:
    """The parent: --environment and --project-root, shared by every verb.

    The help texts are `run`'s own, so a new verb takes both options the way `run` does.

    A verb with actions (`janus dead-letters list …`) declares the pair again on each action
    parser with ``suppress_defaults=True``, so the options may follow the action as well.
    argparse copies an action parser's defaults over what the verb parser already parsed, so
    a real default there would silently undo `--project-root X` given before the action. A
    suppressed one leaves the verb parser's value, given or defaulted, unless the option is
    repeated after the action.

    It lives here, not in the dispatcher, because a verb module that needs it cannot import
    the dispatcher that imports it.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--environment",
        default=argparse.SUPPRESS if suppress_defaults else "local",
        help="Environment profile name under conf/environments without the .yaml suffix.",
    )
    parent.add_argument(
        "--project-root",
        type=Path,
        default=argparse.SUPPRESS if suppress_defaults else default_project_root(),
        help="Project root used to resolve conf/ and data/ paths.",
    )
    return parent


def parse_started_at(value: str) -> datetime:
    """Parse the timezone-aware logical planning timestamp used by every CLI."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("started-at must be a valid ISO-8601 timestamp") from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("started-at must include a timezone offset")

    return parsed


def format_runtime_permission_error(exc: PermissionError) -> str:
    """Explain an unwritable runtime path, including the local-container remedy."""
    path = exc.filename or "<unknown>"
    message = (
        f"JANUS could not prepare the runtime path {path!r}. "
        "The active environment needs write access to the configured storage "
        "and Spark cache directories."
    )
    if str(path).startswith("/workspace/"):
        message += (
            " If you are running inside the local container, recreate it with "
            "`make down && make up` so the Docker/Podman user mapping is "
            "applied correctly."
        )
    if Path(str(path)).name in {"ivy", "ivy_dir"}:
        message += (
            " The Spark Ivy jar cache is never relocated automatically; set "
            "JANUS_SPARK_IVY_DIR to select a writable location."
        )
    return message
