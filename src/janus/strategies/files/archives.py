"""Expand one archive safely, and persist what came out of it.

ZIP and ``tar.gz`` payloads are read fully in memory, filtered against
``access.file_pattern`` (and then the hook's ``archive_members``), and written to the raw
zone as individual member artifacts.

Every member name passes through :func:`_safe_archive_member_path` before it is used as a
key or a path — that is the Zip-Slip guard, and it lives in ``artifacts.py`` because the
sanitized member path *is* the raw-zone path. It is a security control; see its docstring
before touching either module.
"""

from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from janus.models import ExecutionPlan, ExtractedArtifact
from janus.writers import RawArtifactWriter

from .artifacts import (
    _infer_handoff_format,
    _raw_extracted_relative_path,
    _safe_archive_member_path,
)
from .core import ArchiveExtractionError, DiscoveredFile
from .discovery import _filter_discovered_files
from .formats import ARCHIVE_FILE_SUFFIXES, _infer_format_name, _is_tarball_filename

if TYPE_CHECKING:
    from .core import FileHook


def _extract_archive(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    archive_file: DiscoveredFile,
    payload: bytes,
    *,
    version: str,
    file_hook: FileHook | None,
) -> tuple[ExtractedArtifact, ...]:
    member_payloads = _archive_member_payloads(payload, archive_file.filename)
    members = tuple(
        DiscoveredFile(
            source_kind="archive",
            location=member_name,
            filename=PurePosixPath(member_name).name,
            format=_infer_format_name(member_name, fallback="binary"),
            version=version,
        )
        for member_name in member_payloads
    )
    members = _filter_members(members, plan.source_config.access.file_pattern)
    if file_hook is not None:
        members = tuple(file_hook.archive_members(plan, archive_file, members))

    extracted_artifacts: list[ExtractedArtifact] = []
    for member in members:
        member_payload = member_payloads.get(member.location)
        if member_payload is None:
            raise ArchiveExtractionError(
                f"Archive member {member.location!r} was selected but is not available"
            )
        persisted = raw_writer.write_bytes(
            plan,
            _raw_extracted_relative_path(version, archive_file.filename, member.location),
            member_payload,
            metadata={
                "archive_filename": archive_file.filename,
                "archive_member": member.location,
                "resolved_version": version,
            },
        )
        extracted_artifacts.append(
            ExtractedArtifact(
                path=persisted.artifact.path,
                format=_infer_handoff_format(
                    member,
                    fallback=plan.source_config.spark.input_format,
                ),
                checksum=persisted.artifact.checksum,
            )
        )
    return tuple(extracted_artifacts)


def _archive_member_payloads(payload: bytes, filename: str = "") -> dict[str, bytes]:
    if _is_tarball_filename(filename):
        return _tarball_member_payloads(payload)
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members: dict[str, bytes] = {}
            for member_name in archive.namelist():
                if member_name.endswith("/"):
                    continue
                safe_member_path = _safe_archive_member_path(member_name)
                with archive.open(member_name) as stream:
                    members[str(PurePosixPath(*safe_member_path.parts))] = stream.read()
            return members
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ArchiveExtractionError("Could not extract ZIP archive payload") from exc


def _tarball_member_payloads(payload: bytes) -> dict[str, bytes]:
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
            members: dict[str, bytes] = {}
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                safe_member_path = _safe_archive_member_path(member.name)
                f = archive.extractfile(member)
                if f is not None:
                    members[str(PurePosixPath(*safe_member_path.parts))] = f.read()
            return members
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveExtractionError("Could not extract tar.gz archive payload") from exc


def _filter_members(
    members: Sequence[DiscoveredFile],
    file_pattern: str | None,
) -> tuple[DiscoveredFile, ...]:
    return _filter_discovered_files(members, file_pattern)


def _is_archive_file(
    plan: ExecutionPlan,
    discovered_file: DiscoveredFile,
    payload: bytes,
) -> bool:
    if plan.source.strategy_variant == "archive_package":
        return True
    if _is_tarball_filename(discovered_file.filename):
        try:
            return tarfile.is_tarfile(BytesIO(payload))
        except (OSError, tarfile.TarError):
            return False
    suffix = Path(discovered_file.filename).suffix.lower()
    return suffix in ARCHIVE_FILE_SUFFIXES and zipfile.is_zipfile(BytesIO(payload))
