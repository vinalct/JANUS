"""Where an API page lands in the raw zone, and how it is found again after a crash.

One responsibility, two directions. ``_raw_relative_path`` and ``_pages_dir`` decide the
layout on the way out; ``_rediscover_all_artifacts_for_input`` and ``_rediscover_raw_artifacts``
read that same layout back on the way in, so a resumed run can rebuild the artifact list for
pages a previous attempt already wrote. The two directions must agree — keeping them in one
module is what makes a divergence visible.
"""

from __future__ import annotations

import contextlib
from hashlib import sha256
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.api.pagination import PaginationState
from janus.strategies.common import _raw_page_path
from janus.utils.storage import StorageLayout

#: Filename suffix written for each supported raw payload format. Shared by the write path
#: (``_raw_relative_path``) and the rediscovery globs, which is the point.
RAW_FILE_SUFFIXES = {
    "binary": ".bin",
    "json": ".json",
    "jsonl": ".jsonl",
    "text": ".txt",
}


def _raw_relative_path(
    raw_format: str,
    pagination_state: PaginationState,
    *,
    request_input_index: int,
    request_input_count: int,
) -> Path:
    return _raw_page_path(
        pagination_state,
        RAW_FILE_SUFFIXES[raw_format],
        request_input_index=request_input_index,
        request_input_count=request_input_count,
    )


def _pages_dir(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    request_input_index: int,
    request_input_count: int,
) -> Path:
    """Return the raw subdirectory for one request input, mirroring _raw_relative_path."""
    raw_dir = storage_layout.resolve_output(plan, "raw").resolved_path
    if request_input_count > 1:
        return raw_dir / f"request-input-{request_input_index:06d}"
    return raw_dir / "pages"


def _rediscover_all_artifacts_for_input(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    request_input_index: int,
    request_input_count: int,
) -> list[ExtractedArtifact]:
    """Return all raw artifacts written for a fully completed request input."""
    raw_format = plan.source_config.outputs.raw.format
    suffix = RAW_FILE_SUFFIXES.get(raw_format, "")
    directory = _pages_dir(plan, storage_layout, request_input_index, request_input_count)

    if not directory.exists():
        return []

    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"*{suffix}"):
        stem = path.stem
        for prefix, start in (("page-", 5), ("offset-", 7), ("cursor-", 7), ("response-", 9)):
            if stem.startswith(prefix):
                with contextlib.suppress(ValueError):
                    candidates.append((int(stem[start:]), path))
                break

    artifacts = []
    for _, path in sorted(candidates):
        checksum = sha256(path.read_bytes()).hexdigest()
        artifacts.append(ExtractedArtifact(path=str(path), format=raw_format, checksum=checksum))
    return artifacts


def _rediscover_raw_artifacts(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    progress: dict[str, Any],
    request_input_index: int = 1,
    request_input_count: int = 1,
) -> list[ExtractedArtifact]:
    """Re-discover raw artifact files written by a previous partial run."""
    raw_format = plan.source_config.outputs.raw.format
    suffix = RAW_FILE_SUFFIXES.get(raw_format, "")
    directory = _pages_dir(plan, storage_layout, request_input_index, request_input_count)

    if not directory.exists():
        return []

    last_page = progress.get("last_page_number")
    last_offset = progress.get("last_offset")

    if last_page is not None:
        paths = _numbered_pages_up_to(
            directory, prefix="page-", suffix=suffix, last_index=last_page
        )
    elif last_offset is not None:
        paths = _numbered_pages_up_to(
            directory, prefix="offset-", suffix=suffix, last_index=last_offset
        )
    else:
        paths = []

    return [
        ExtractedArtifact(
            path=str(path),
            format=raw_format,
            checksum=sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    ]


def _numbered_pages_up_to(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    last_index: int,
) -> list[Path]:
    """Raw page files named ``<prefix><n><suffix>``, index-ordered, up to ``last_index``.

    One collector for both paginator flavours: page-number resume and offset resume differ
    only in the filename prefix and the progress key that bounds them.
    """
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"{prefix}*{suffix}"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        try:
            index = int(stem[len(prefix) :])
        except ValueError:
            continue
        if index <= last_index:
            candidates.append((index, path))
    return [path for _, path in sorted(candidates)]
