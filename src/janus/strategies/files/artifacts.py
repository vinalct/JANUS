"""Where a downloaded file lands in the raw zone, and how it is described afterwards."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from janus.utils.logging import redact_url

from .core import ArchiveExtractionError, DiscoveredFile, FileDiscoveryError
from .formats import (
    SUPPORTED_FILE_INPUT_FORMATS,
    _infer_format_name,
    _safe_path_segment,
)


def _raw_download_relative_path(version: str, filename: str) -> Path:
    return Path("downloads") / _safe_path_segment(version) / _safe_filename(filename)


def _raw_extracted_relative_path(
    version: str,
    archive_filename: str,
    member_location: str,
) -> Path:
    return (
        Path("extracted")
        / _safe_path_segment(version)
        / _safe_path_segment(Path(archive_filename).stem or archive_filename)
        / _safe_archive_member_path(member_location)
    )


def _safe_archive_member_path(member_location: str) -> Path:
    """Path-traversal guard for archive extraction — the Zip-Slip defence."""
    pure_path = PurePosixPath(member_location)
    if pure_path.is_absolute():
        raise ArchiveExtractionError("Archive members must not use absolute paths")
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ArchiveExtractionError("Archive members must not contain unsafe path segments")
    return Path(*pure_path.parts)


def _safe_filename(value: str) -> str:
    """Reduce a discovered or ``Content-Disposition`` name to a bare, non-empty filename."""
    filename = Path(value).name.strip()
    if not filename:
        raise FileDiscoveryError("Resolved filename must not be empty")
    return filename


def _infer_handoff_format(discovered_file: DiscoveredFile, *, fallback: str) -> str:
    format_name = _infer_format_name(discovered_file.filename, fallback=fallback)
    if format_name == "binary" and fallback in SUPPORTED_FILE_INPUT_FORMATS:
        return fallback
    return format_name


def _redacted_location(discovered_file: DiscoveredFile) -> str:
    if discovered_file.source_kind == "remote":
        return redact_url(discovered_file.location)
    return discovered_file.location


def _discovered_file_dead_letter_key(discovered_file: DiscoveredFile) -> str:
    return f"{discovered_file.source_kind}:{discovered_file.location}"


def _discovered_file_dead_letter_metadata(
    discovered_file: DiscoveredFile,
    *,
    file_index: int,
    selected_file_count: int,
) -> dict[str, str]:
    return {
        "file_index": str(file_index),
        "selected_file_count": str(selected_file_count),
        "source_kind": discovered_file.source_kind,
        "source_location": _redacted_location(discovered_file),
        "filename": discovered_file.filename,
    }
