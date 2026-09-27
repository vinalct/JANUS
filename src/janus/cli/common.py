"""Small CLI conventions shared by single-source and batch commands."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from janus.models.source_config import SourceConfig
from janus.utils.logging import StructuredLogger


def default_project_root() -> Path:
    """Resolve the project root from the environment, cwd, or installed package."""
    env_project_root = os.getenv("JANUS_PROJECT_ROOT")
    if env_project_root:
        return Path(env_project_root)

    cwd = Path.cwd()
    if (cwd / "conf" / "environments").exists():
        return cwd

    return Path(__file__).resolve().parents[3]


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


def log_source_config_deprecations(
    logger: StructuredLogger, sources: Iterable[SourceConfig]
) -> None:
    """Report every deprecated declaration a command planned against."""
    for source in sources:
        for issue in source.deprecations:
            logger.warning(
                "source_config_deprecated",
                source_id=source.source_id,
                config_path=str(source.config_path),
                field=issue.path,
                detail=issue.message,
            )
