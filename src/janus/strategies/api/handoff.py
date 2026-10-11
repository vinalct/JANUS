"""Derive deterministic JSONL handoffs without changing the raw resume record.

Pages remain read-only. A mapping is one bronze row per page, including envelopes;
arrays contribute their elements, matching Spark's uncompacted JSON reader.
Custom record extraction needs
the original request/response and cannot safely be reconstructed from saved pages.
Serialization and streamed persistence both belong to the raw writer.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult
from janus.strategies.api.artifacts import RAW_FILE_SUFFIXES, _page_index
from janus.strategies.api.metadata import _handoff_metadata
from janus.strategies.base import SourceHook
from janus.strategies.common import _raw_run_path_prefix
from janus.writers.raw import RawArtifactWriter, iter_json_lines

if TYPE_CHECKING:
    from janus.strategies.api.core import ApiHook

# A sibling of pages/ and request-input-NNNNNN/, never a child of either.
HANDOFF_DIRECTORY_NAME = "handoff"

COMPACTION_JSONL = "jsonl"
COMPACTION_PAGES = "pages"
COMPACTION_NONE = "none"
COMPACTION_HOOK = "hook"
COMPACTION_HOOK_RECORDS = "hook_records"
COMPACTION_EMPTY = "empty"

_REQUEST_INPUT_DIRECTORY = re.compile(r"request-input-[0-9]{6,}")
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompactionOutcome:
    """Derived artifacts and the four extraction metadata values describing them."""

    artifacts: tuple[ExtractedArtifact, ...]
    compaction: str
    source_page_count: int
    handoff_bytes: int

    @property
    def metadata(self) -> dict[str, str]:
        return _handoff_metadata(self)


def _hook_replaces_handoff(hook: SourceHook | None) -> bool:
    """Ask the hook class whether it replaces the base identity handoff."""
    return (
        hook is not None
        and type(hook).on_normalization_handoff is not SourceHook.on_normalization_handoff
    )


def _hook_handoff_outcome(
    extraction_result: ExtractionResult, handoff: ExtractionResult
) -> CompactionOutcome:
    return _outcome(handoff.artifacts, COMPACTION_HOOK, len(extraction_result.artifacts))


def compact_api_handoff(
    plan: ExecutionPlan,
    extraction_result: ExtractionResult,
    *,
    raw_writer: RawArtifactWriter,
    hook: ApiHook | None = None,
) -> CompactionOutcome:
    """Stream a derived handoff, carrying extraction's decisions through untouched.

    Only the returned artifact list is replaced. The caller retains the original
    record count and checkpoint; a disagreement is logged rather than repaired.
    ``hook`` makes the record-extraction capability explicit without resolving a
    second hook from configuration at handoff time.
    """
    artifacts = extraction_result.artifacts
    if not artifacts:
        return _outcome(artifacts, COMPACTION_EMPTY, 0)
    if not plan.source_config.spark.handoff.is_compacting:
        return _outcome(artifacts, COMPACTION_PAGES, len(artifacts))
    raw_format = plan.source_config.outputs.raw.format
    if raw_format not in {"json", "jsonl"}:
        return _outcome(artifacts, COMPACTION_NONE, len(artifacts))
    if _hook_extracts_records(hook):
        return _outcome(artifacts, COMPACTION_HOOK_RECORDS, len(artifacts))

    prefix = _raw_run_path_prefix(plan) or Path()
    run_root = raw_writer.storage_layout.resolve_output(plan, "raw").resolved_path / prefix
    groups = _group_pages(artifacts, raw_format=raw_format, run_root=run_root.resolve())
    if groups is None:
        return _outcome(artifacts, COMPACTION_NONE, len(artifacts))

    # Paths below already include the run prefix, even when the supplied writer is scoped.
    writer = raw_writer
    if writer.raw_path_prefix is not None:
        writer = writer.with_raw_path_prefix(None)
    derived: list[ExtractedArtifact] = []
    record_count = 0
    for group, pages in sorted(groups.items()):
        lines = iter_json_lines(_records_from_pages(pages, raw_format=raw_format))
        split_index = 1
        while (first_line := next(lines, None)) is not None:
            relative_path = (
                prefix / HANDOFF_DIRECTORY_NAME / group / f"records-{split_index:04d}.jsonl"
            )
            artifact, count = _write_split(
                plan,
                writer,
                relative_path,
                first_line,
                lines,
                max_bytes=plan.source_config.spark.handoff.max_artifact_bytes,
            )
            derived.append(artifact)
            record_count += count
            split_index += 1

    expected_count = extraction_result.records_extracted
    if expected_count is not None and record_count != expected_count:
        _LOG.warning(
            "handoff_record_count_mismatch: records_extracted=%s handoff_records=%s",
            expected_count,
            record_count,
            extra={"records_extracted": expected_count, "handoff_records": record_count},
        )
    return _outcome(tuple(derived), COMPACTION_JSONL, len(artifacts))


def _hook_extracts_records(hook: ApiHook | None) -> bool:
    if hook is None:
        return False
    # The façade will import this module when wired; defer its hook-class import.
    from janus.strategies.api.core import ApiHook

    return type(hook).extract_records is not ApiHook.extract_records


def _outcome(
    artifacts: tuple[ExtractedArtifact, ...],
    compaction: str,
    source_page_count: int,
) -> CompactionOutcome:
    return CompactionOutcome(
        artifacts=artifacts,
        compaction=compaction,
        source_page_count=source_page_count,
        handoff_bytes=sum(
            Path(artifact.path).stat().st_size
            for artifact in artifacts
            if Path(artifact.path).is_file()
        ),
    )


def _group_pages(
    artifacts: tuple[ExtractedArtifact, ...],
    *,
    raw_format: str,
    run_root: Path,
) -> dict[str, list[Path]] | None:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for artifact in artifacts:
        path = Path(artifact.path).resolve()
        group = path.parent.name
        index = _page_index(path)
        if (
            artifact.format != raw_format
            or path.suffix != RAW_FILE_SUFFIXES[raw_format]
            or path.parent.parent != run_root
            or (group != "pages" and _REQUEST_INPUT_DIRECTORY.fullmatch(group) is None)
            or index is None
        ):
            return None
        groups.setdefault(group, []).append((index, path))
    return {group: [path for _, path in sorted(pages)] for group, pages in groups.items()}


def _records_from_pages(pages: list[Path], *, raw_format: str) -> Iterator[Any]:
    """Preserve Spark's page-read rows, including wrapper fields and nested arrays.

    Extraction counts may flatten envelopes, but the uncompacted Spark reader does
    not. Flattening here changes both rows and types under an explicit contract.
    """
    for path in pages:
        with path.open(encoding="utf-8") as stream:
            if raw_format == "json":
                yield from _page_rows(json.load(stream))
            else:
                for line in stream:
                    if line.strip():
                        yield from _page_rows(json.loads(line))


def _page_rows(payload: Any) -> Iterator[Any]:
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        yield from payload


def _write_split(
    plan: ExecutionPlan,
    writer: RawArtifactWriter,
    relative_path: Path,
    first_line: bytes,
    lines: Iterator[bytes],
    *,
    max_bytes: int,
) -> tuple[ExtractedArtifact, int]:
    record_count = 0

    def split_lines() -> Iterator[bytes]:
        nonlocal record_count
        byte_count = 0
        for line in chain((first_line,), lines):
            record_count += 1
            byte_count += len(line)
            yield line
            if byte_count >= max_bytes:
                break

    persisted = writer.write_stream(
        plan,
        relative_path,
        split_lines(),
        mode="overwrite",
        format_name="jsonl",
    )
    return persisted.artifact, record_count
