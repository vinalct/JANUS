"""Expand one archive safely, and persist what came out of it.

Archive headers are listed before any member body is opened. Members are then filtered by
access.file_pattern and the hook's archive_members result, and the configured member, total
and compression-ratio caps are checked over that selection. This is deliberate: an unselected
member costs only its header read and is never decompressed.

Selected members stream through RawArtifactWriter.write_stream, which re-enforces the
per-member cap while hashing and atomically persisting each artifact. Every member name passes
through _safe_archive_member_path during listing, before it can be used as a key or a path.
That is the Zip-Slip guard, and it lives in artifacts.py because the sanitized member path is
the raw-zone path. It is a security control; see its docstring before touching either module.
"""

from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO, cast

from janus.models import ExecutionPlan, ExtractedArtifact, LimitsConfig
from janus.writers import RawArtifactWriter, RawWriteLimitError

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
    from .download import DownloadedPayload


@dataclass(frozen=True, slots=True)
class ArchiveMemberInfo:
    """Header metadata needed to select and bound one archive member."""

    name: str
    safe_path: Path
    declared_size: int
    compressed_size: int | None


def _extract_archive(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    archive_file: DiscoveredFile,
    payload: DownloadedPayload,
    *,
    version: str,
    file_hook: FileHook | None,
) -> tuple[ExtractedArtifact, ...]:
    member_infos = _list_archive_members(payload, archive_file.filename)
    info_by_location = {_member_location(info): info for info in member_infos}
    members = tuple(
        _discovered_archive_member(info, version=version)
        for info in info_by_location.values()
    )
    members = _filter_members(members, plan.source_config.access.file_pattern)
    if file_hook is not None:
        members = tuple(file_hook.archive_members(plan, archive_file, members))

    selected = _selected_members(members, info_by_location)
    _check_archive_limits(
        [info for _member, info in selected],
        archive_size_bytes=payload.size_bytes,
        limits=plan.source_config.access.limits,
    )

    if _is_tarball_filename(archive_file.filename):
        return _extract_tarball_members(
            plan,
            raw_writer,
            archive_file,
            payload,
            selected,
            version=version,
        )
    return _extract_zip_members(
        plan,
        raw_writer,
        archive_file,
        payload,
        selected,
        version=version,
    )


def _list_archive_members(
    payload: DownloadedPayload,
    filename: str,
) -> list[ArchiveMemberInfo]:
    """Read and validate member headers without opening a member body."""
    if _is_tarball_filename(filename):
        return _list_tarball_members(payload)
    try:
        with payload.open() as stream, zipfile.ZipFile(stream) as archive:
            return [
                ArchiveMemberInfo(
                    name=member.filename,
                    safe_path=_safe_archive_member_path(member.filename),
                    declared_size=member.file_size,
                    compressed_size=member.compress_size,
                )
                for member in archive.infolist()
                if not member.is_dir()
            ]
    except ArchiveExtractionError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ArchiveExtractionError("Could not extract ZIP archive payload") from exc


def _list_tarball_members(payload: DownloadedPayload) -> list[ArchiveMemberInfo]:
    try:
        with payload.open() as stream, tarfile.open(fileobj=stream, mode="r:gz") as archive:
            return [
                ArchiveMemberInfo(
                    name=member.name,
                    safe_path=_safe_archive_member_path(member.name),
                    declared_size=member.size,
                    compressed_size=None,
                )
                for member in archive.getmembers()
                if member.isfile()
            ]
    except ArchiveExtractionError:
        raise
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise ArchiveExtractionError("Could not extract tar.gz archive payload") from exc


def _check_archive_limits(
    selected: Sequence[ArchiveMemberInfo],
    *,
    archive_size_bytes: int,
    limits: LimitsConfig,
) -> None:
    """Reject declared sizes and compression ratios before decompression starts."""
    for member in selected:
        if member.declared_size > limits.max_archive_member_bytes:
            raise ArchiveExtractionError(
                f"Archive member {member.name!r} declares {member.declared_size} bytes, over "
                "access.limits.max_archive_member_bytes="
                f"{limits.max_archive_member_bytes}"
            )

    total_size = sum(member.declared_size for member in selected)
    if total_size > limits.max_archive_total_bytes:
        raise ArchiveExtractionError(
            f"Selected archive members declare {total_size} bytes, over "
            f"access.limits.max_archive_total_bytes={limits.max_archive_total_bytes}"
        )

    zip_members = [member for member in selected if member.compressed_size is not None]
    if zip_members:
        for member in zip_members:
            compressed_size = member.compressed_size
            if compressed_size is None:  # pragma: no cover - narrowed by zip_members
                continue
            ratio = member.declared_size / max(compressed_size, 1)
            _raise_if_ratio_exceeded(ratio, limits, member_name=member.name)

        total_compressed_size = sum(
            member.compressed_size or 0 for member in zip_members
        )
        overall_ratio = total_size / max(total_compressed_size, 1)
        _raise_if_ratio_exceeded(overall_ratio, limits)
        return

    overall_ratio = total_size / max(archive_size_bytes, 1)
    _raise_if_ratio_exceeded(overall_ratio, limits)


def _raise_if_ratio_exceeded(
    ratio: float,
    limits: LimitsConfig,
    *,
    member_name: str | None = None,
) -> None:
    if ratio <= limits.max_archive_ratio:
        return
    subject = (
        f"Archive member {member_name!r} has"
        if member_name is not None
        else "Selected archive members have"
    )
    raise ArchiveExtractionError(
        f"{subject} compression ratio {ratio:.2f}, over "
        f"access.limits.max_archive_ratio={limits.max_archive_ratio}"
    )


def _extract_zip_members(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    archive_file: DiscoveredFile,
    payload: DownloadedPayload,
    selected: Sequence[tuple[DiscoveredFile, ArchiveMemberInfo]],
    *,
    version: str,
) -> tuple[ExtractedArtifact, ...]:
    try:
        with payload.open() as stream, zipfile.ZipFile(stream) as archive:
            artifacts = []
            for member, info in selected:
                with archive.open(info.name) as member_stream:
                    artifacts.append(
                        _persist_archive_member(
                            plan,
                            raw_writer,
                            archive_file,
                            member,
                            info,
                            cast(BinaryIO, member_stream),
                            version=version,
                        )
                    )
            return tuple(artifacts)
    except ArchiveExtractionError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArchiveExtractionError("Could not extract ZIP archive payload") from exc


def _extract_tarball_members(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    archive_file: DiscoveredFile,
    payload: DownloadedPayload,
    selected: Sequence[tuple[DiscoveredFile, ArchiveMemberInfo]],
    *,
    version: str,
) -> tuple[ExtractedArtifact, ...]:
    try:
        with payload.open() as stream, tarfile.open(fileobj=stream, mode="r:gz") as archive:
            artifacts = []
            for member, info in selected:
                archive_member = archive.getmember(info.name)
                member_stream = archive.extractfile(archive_member)
                if member_stream is None:
                    raise ArchiveExtractionError(
                        f"Archive member {member.location!r} was selected but is not available"
                    )
                with member_stream:
                    artifacts.append(
                        _persist_archive_member(
                            plan,
                            raw_writer,
                            archive_file,
                            member,
                            info,
                            cast(BinaryIO, member_stream),
                            version=version,
                        )
                    )
            return tuple(artifacts)
    except ArchiveExtractionError:
        raise
    except (KeyError, OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        raise ArchiveExtractionError("Could not extract tar.gz archive payload") from exc


def _persist_archive_member(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    archive_file: DiscoveredFile,
    member: DiscoveredFile,
    info: ArchiveMemberInfo,
    stream: BinaryIO,
    *,
    version: str,
) -> ExtractedArtifact:
    limits = plan.source_config.access.limits
    try:
        persisted = raw_writer.write_stream(
            plan,
            _raw_extracted_relative_path(
                version,
                archive_file.filename,
                str(info.safe_path),
            ),
            stream,
            max_bytes=limits.max_archive_member_bytes,
            metadata={
                "archive_filename": archive_file.filename,
                "archive_member": member.location,
                "resolved_version": version,
            },
        )
    except RawWriteLimitError as exc:
        raise ArchiveExtractionError(
            f"Archive member {info.name!r} produced {exc.bytes_seen} bytes while extracting, "
            "over access.limits.max_archive_member_bytes="
            f"{exc.max_bytes}"
        ) from exc

    return ExtractedArtifact(
        path=persisted.artifact.path,
        format=_infer_handoff_format(
            member,
            fallback=plan.source_config.spark.input_format,
        ),
        checksum=persisted.artifact.checksum,
    )


def _archive_member_payloads(payload: bytes, filename: str = "") -> dict[str, bytes]:
    """Materialize archive members for the retained compatibility import surface."""
    from .download import DownloadedPayload

    downloaded = DownloadedPayload(
        size_bytes=len(payload),
        sha256_hex="",
        inline=payload,
    )
    member_infos = _list_archive_members(downloaded, filename)
    try:
        if _is_tarball_filename(filename):
            return _materialize_tarball_members(downloaded, member_infos)
        return _materialize_zip_members(downloaded, member_infos)
    except ArchiveExtractionError:
        raise
    except (OSError, ValueError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        archive_kind = "tar.gz" if _is_tarball_filename(filename) else "ZIP"
        raise ArchiveExtractionError(
            f"Could not extract {archive_kind} archive payload"
        ) from exc


def _materialize_zip_members(
    payload: DownloadedPayload,
    member_infos: Sequence[ArchiveMemberInfo],
) -> dict[str, bytes]:
    with payload.open() as stream, zipfile.ZipFile(stream) as archive:
        members = {}
        for info in member_infos:
            with archive.open(info.name) as member_stream:
                members[_member_location(info)] = member_stream.read()
        return members


def _materialize_tarball_members(
    payload: DownloadedPayload,
    member_infos: Sequence[ArchiveMemberInfo],
) -> dict[str, bytes]:
    with payload.open() as stream, tarfile.open(fileobj=stream, mode="r:gz") as archive:
        members = {}
        for info in member_infos:
            member_stream = archive.extractfile(archive.getmember(info.name))
            if member_stream is None:
                continue
            with member_stream:
                members[_member_location(info)] = member_stream.read()
        return members


def _selected_members(
    members: Sequence[DiscoveredFile],
    info_by_location: dict[str, ArchiveMemberInfo],
) -> tuple[tuple[DiscoveredFile, ArchiveMemberInfo], ...]:
    selected = []
    for member in members:
        info = info_by_location.get(member.location)
        if info is None:
            raise ArchiveExtractionError(
                f"Archive member {member.location!r} was selected but is not available"
            )
        selected.append((member, info))
    return tuple(selected)


def _discovered_archive_member(
    info: ArchiveMemberInfo,
    *,
    version: str,
) -> DiscoveredFile:
    location = _member_location(info)
    return DiscoveredFile(
        source_kind="archive",
        location=location,
        filename=PurePosixPath(location).name,
        format=_infer_format_name(location, fallback="binary"),
        version=version,
    )


def _member_location(info: ArchiveMemberInfo) -> str:
    return str(PurePosixPath(*info.safe_path.parts))


def _filter_members(
    members: Sequence[DiscoveredFile],
    file_pattern: str | None,
) -> tuple[DiscoveredFile, ...]:
    return _filter_discovered_files(members, file_pattern)


def _is_archive_file(
    plan: ExecutionPlan,
    discovered_file: DiscoveredFile,
    payload: DownloadedPayload,
) -> bool:
    if plan.source.strategy_variant == "archive_package":
        return True
    if _is_tarball_filename(discovered_file.filename):
        try:
            with payload.open() as stream:
                return tarfile.is_tarfile(stream)
        except (OSError, tarfile.TarError):
            return False
    suffix = Path(discovered_file.filename).suffix.lower()
    if suffix not in ARCHIVE_FILE_SUFFIXES:
        return False
    with payload.open() as stream:
        return zipfile.is_zipfile(stream)
