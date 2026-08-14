"""What files exist, and which of them this run wants.

Discovery answers the two questions that come before any byte is fetched: *what candidates
does the configured source expose* (a remote URL resolved through the link-resolver chain, a
local path, a glob) and *which of those does this run take* (variant, version ordering,
checkpoint). Everything here is decided from configuration, filesystem metadata and link
resolution — no payload has been downloaded yet, which is why version inference is filename-
and mtime-based.

The dependency on ``resolvers.py`` runs one way, ``discovery → resolvers``: the
Direct/Redirect/Nextcloud/Html chain is a separate, already-clean abstraction that turns one
URL into candidates. It is imported inside :func:`_discover_files` because ``resolvers.py``
imports ``DiscoveredFile`` from ``core.py``.
"""

from __future__ import annotations

import fnmatch
import glob
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.checkpoints import CheckpointState
from janus.models import ExecutionPlan
from janus.strategies.common import _compare_checkpoint_values, _parse_datetime
from janus.strategies.http import ApiTransport
from janus.utils.environment import resolve_project_path

from .core import DiscoveredFile, FileDiscoveryError
from .formats import _filename_from_url, _infer_format_name

if TYPE_CHECKING:
    from .core import FileHook

#: A four-digit year, optionally followed by month and day, anywhere in a filename. The last
#: match wins, so ``dados_2026_04.csv`` resolves to the version ``2026-04``.
VERSION_TOKEN_PATTERN = re.compile(r"(\d{4}(?:[-_]\d{2}(?:[-_]\d{2})?)?)")


def _discover_files(
    plan: ExecutionPlan,
    file_hook: FileHook | None,
    transport: ApiTransport,
) -> tuple[DiscoveredFile, ...]:
    from janus.strategies.files.resolvers import build_resolver_chain, resolve_link

    access = plan.source_config.access
    discovered: list[DiscoveredFile] = []

    if access.url:
        if file_hook is not None:
            resolved_from_url: Sequence[DiscoveredFile] = file_hook.resolve_links(
                plan, access.url, access.format, transport
            )
        else:
            chain = build_resolver_chain(access.link_resolver)
            resolved_from_url = resolve_link(access.url, access.format, transport, chain)

        if resolved_from_url:
            discovered.extend(
                _filter_discovered_files(
                    resolved_from_url,
                    access.remote_file_pattern,
                )
            )
        else:
            # Fallback: treat URL as direct so the download loop can dead-letter it on failure
            discovered.extend(
                _filter_discovered_files(
                    (
                        DiscoveredFile(
                            source_kind="remote",
                            location=access.url,
                            filename=_filename_from_url(access.url),
                            format=_infer_format_name(
                                access.url,
                                fallback=access.format,
                            ),
                        ),
                    ),
                    access.remote_file_pattern,
                )
            )

    if access.path:
        discovered.extend(_discover_local_path(plan, access.path, access.file_pattern))

    if access.discovery_pattern:
        discovered.extend(_discover_local_pattern(plan, access.discovery_pattern))

    deduplicated = {
        (item.source_kind, item.location): item
        for item in sorted(discovered, key=lambda item: (item.source_kind, item.location))
    }
    resolved = tuple(deduplicated.values())
    if file_hook is not None:
        resolved = tuple(file_hook.discovered_files(plan, resolved))
    return resolved


def _discover_local_path(
    plan: ExecutionPlan,
    configured_path: str,
    file_pattern: str | None,
) -> list[DiscoveredFile]:
    path = resolve_project_path(plan.run_context.project_root, configured_path)
    if not path.exists():
        raise FileDiscoveryError(f"Configured file path does not exist: {path}")

    if path.is_file():
        return [_local_discovered_file(plan, path)]

    matched_paths = (
        sorted(file_path for file_path in path.glob(file_pattern) if file_path.is_file())
        if file_pattern
        else sorted(file_path for file_path in path.iterdir() if file_path.is_file())
    )
    return [_local_discovered_file(plan, file_path) for file_path in matched_paths]


def _discover_local_pattern(
    plan: ExecutionPlan,
    discovery_pattern: str,
) -> list[DiscoveredFile]:
    expanded_pattern = str(
        resolve_project_path(plan.run_context.project_root, discovery_pattern)
    )
    matched_paths = sorted(
        Path(match)
        for match in glob.glob(expanded_pattern, recursive=True)
        if Path(match).is_file()
    )
    return [_local_discovered_file(plan, file_path) for file_path in matched_paths]


def _local_discovered_file(plan: ExecutionPlan, file_path: Path) -> DiscoveredFile:
    stat = file_path.stat()
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    version = _default_discovered_version(
        plan,
        file_path.name,
        modified_at=modified_at,
    )
    return DiscoveredFile(
        source_kind="local",
        location=str(file_path.resolve()),
        filename=file_path.name,
        format=_infer_format_name(file_path.name, fallback=plan.source_config.access.format),
        version=version,
        size_bytes=stat.st_size,
        modified_at=modified_at,
    )


def _select_files(
    plan: ExecutionPlan,
    discovered_files: Sequence[DiscoveredFile],
    checkpoint_state: CheckpointState | None,
) -> tuple[DiscoveredFile, ...]:
    if not discovered_files:
        return ()

    if plan.source.strategy_variant == "static_file":
        return tuple(discovered_files)

    if len(discovered_files) > 1 and any(item.version is None for item in discovered_files):
        raise FileDiscoveryError(
            "Versioned file selection requires a deterministic version for each discovered "
            "candidate when more than one file is present"
        )

    ordered_files = tuple(sorted(discovered_files, key=_version_sort_key))
    if checkpoint_state is not None:
        ordered_files = tuple(
            item
            for item in ordered_files
            if item.version is None
            or _compare_checkpoint_values(item.version, checkpoint_state.checkpoint_value) > 0
        )

    if plan.extraction_mode == "incremental":
        return ordered_files
    if not ordered_files:
        return ()
    return (ordered_files[-1],)


def _filter_discovered_files(
    files: Sequence[DiscoveredFile],
    file_pattern: str | None,
) -> tuple[DiscoveredFile, ...]:
    if not file_pattern:
        return tuple(files)
    return tuple(
        item
        for item in files
        if fnmatch.fnmatch(item.location, file_pattern)
        or fnmatch.fnmatch(item.filename, file_pattern)
    )


def _default_discovered_version(
    plan: ExecutionPlan,
    filename: str,
    *,
    modified_at: datetime,
) -> str | None:
    if plan.source.strategy_variant == "static_file":
        return "current"

    version_match = VERSION_TOKEN_PATTERN.findall(filename)
    if version_match:
        return version_match[-1].replace("_", "-")
    return modified_at.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _version_sort_key(discovered_file: DiscoveredFile) -> tuple[str, Any]:
    version = discovered_file.version
    if version is None:
        return ("missing", discovered_file.location)

    normalized = version.strip()
    parsed_datetime = _parse_datetime(normalized)
    if parsed_datetime is not None:
        return ("datetime", parsed_datetime.astimezone(UTC))

    try:
        return ("decimal", Decimal(normalized))
    except InvalidOperation:
        return ("text", normalized)


def _should_skip_for_checkpoint(
    plan: ExecutionPlan,
    checkpoint_state: CheckpointState | None,
    candidate_version: str,
) -> bool:
    if checkpoint_state is None or plan.extraction_mode != "incremental":
        return False
    return _compare_checkpoint_values(candidate_version, checkpoint_state.checkpoint_value) <= 0
