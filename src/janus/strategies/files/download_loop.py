"""One discovered file's journey: bytes in, persisted raw artifacts out.

The body of the candidate loop, lifted out of ``extraction.py`` so that module stays the
loop, the dead-letter policy and the result — this is what happens *inside* one iteration:
load → validate → hook → resolve version → checkpoint gate → checksum → persist → expand.

It composes the family's mechanics and adds none of its own: bytes come from
``download.py``, archive expansion from ``archives.py``, paths and redaction from
``artifacts.py``, the checkpoint comparison from ``discovery.py``. The one decision it owns
is the *checksum comparison* — it computes the digest and raises ``FileIntegrityError`` on a
mismatch. What that failure costs the run is not decided here: the caller in ``extraction.py``
owns the dead-letter policy, and a mismatch on a single-file run still propagates because one
dead letter exhausts that run's budget.

``FileExtractionContext`` is named under ``TYPE_CHECKING`` only. ``extraction.py`` defines it
and imports this module, so binding it at runtime would close a cycle — the same reason
``catalog/requests.py`` states its hook as a protocol rather than importing ``CatalogHook``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TYPE_CHECKING

from janus.models import ExtractedArtifact
from janus.strategies.http import ApiClient
from janus.writers import RawArtifactWriter

from .archives import _extract_archive, _is_archive_file
from .artifacts import _infer_handoff_format, _raw_download_relative_path, _redacted_location
from .core import FileIntegrityError
from .discovery import _should_skip_for_checkpoint
from .download import (
    _expected_checksum,
    _resolve_version,
    _resolved_filename,
    _validate_remote_payload,
)
from .formats import _infer_format_name

if TYPE_CHECKING:
    from .core import DiscoveredFile
    from .extraction import FileExtractionContext


@dataclass(frozen=True, slots=True)
class FileCandidateResult:
    """What one discovered file contributed, once it is known to be worth keeping."""

    artifacts: tuple[ExtractedArtifact, ...]
    version: str
    loaded_file_count: int
    total_attempts: int
    archive_member_count: int
    checksum_verified_count: int


def _extract_one_file(
    context: FileExtractionContext,
    discovered_file: DiscoveredFile,
    *,
    client: ApiClient,
    raw_writer: RawArtifactWriter,
    file_index: int,
) -> FileCandidateResult | None:
    """Take one candidate from bytes to persisted artifacts.

    Returns ``None`` when the checkpoint says this version is not newer than the last run's —
    a skip, not a failure: the attempt it cost is deliberately dropped rather than counted,
    exactly as the pre-split loop did. Anything raised here is the caller's dead-letter
    decision, including the ``FileIntegrityError`` a checksum mismatch produces.
    """
    plan = context.plan
    logger = context.logger
    file_hook = context.file_hook

    payload, response, attempts_used = context.downloader.load_payload(
        plan,
        discovered_file,
        client=client,
        throttle=context.throttle,
        logger=logger,
    )
    if logger is not None:
        logger.info(
            "file_payload_loaded",
            file_index=file_index,
            source_kind=discovered_file.source_kind,
            filename=discovered_file.filename,
            status_code=response.status_code if response is not None else None,
            attempts_used=attempts_used,
            payload_size_bytes=len(payload),
        )

    _validate_remote_payload(discovered_file, payload, response)

    if file_hook is not None:
        payload = file_hook.prepare_download(
            plan,
            discovered_file,
            payload,
            response=response,
        )

    resolved_filename = _resolved_filename(discovered_file.filename, response)
    resolved_file = replace(
        discovered_file,
        filename=resolved_filename,
        format=_infer_format_name(
            resolved_filename,
            fallback=plan.source_config.access.format,
        ),
    )
    version = _resolve_version(plan, resolved_file, payload, response, file_hook)

    if _should_skip_for_checkpoint(plan, context.checkpoint_state, version):
        if logger is not None:
            logger.info(
                "file_candidate_skipped",
                file_index=file_index,
                filename=resolved_filename,
                resolved_version=version,
                checkpoint_value=(
                    context.checkpoint_state.checkpoint_value
                    if context.checkpoint_state is not None
                    else None
                ),
                reason="checkpoint_not_newer",
            )
        return None

    expected_checksum = _expected_checksum(
        plan,
        resolved_file,
        response=response,
        file_hook=file_hook,
    )
    actual_checksum = sha256(payload).hexdigest()
    checksum_verified = expected_checksum is not None
    checksum_verified_count = 0
    if expected_checksum is not None:
        if actual_checksum.lower() != expected_checksum.lower():
            raise FileIntegrityError(
                f"Checksum mismatch for {resolved_filename!r}: "
                f"expected {expected_checksum.lower()} got "
                f"{actual_checksum.lower()}"
            )
        checksum_verified_count = 1

    is_archive = _is_archive_file(plan, resolved_file, payload)
    artifact_format = (
        "binary"
        if is_archive
        else _infer_handoff_format(
            resolved_file,
            fallback=plan.source_config.spark.input_format,
        )
    )
    persisted_original = raw_writer.write_bytes(
        plan,
        _raw_download_relative_path(version, resolved_filename),
        payload,
        metadata={
            "source_kind": resolved_file.source_kind,
            "source_location": _redacted_location(resolved_file),
            "resolved_version": version,
            "attempt_count": str(attempts_used),
        },
    )
    artifacts = [
        ExtractedArtifact(
            path=persisted_original.artifact.path,
            format=artifact_format,
            checksum=persisted_original.artifact.checksum,
        )
    ]
    if logger is not None:
        logger.info(
            "file_payload_persisted",
            file_index=file_index,
            filename=resolved_filename,
            source_kind=resolved_file.source_kind,
            resolved_version=version,
            artifact_path=persisted_original.artifact.path,
            artifact_format=artifact_format,
            checksum_verified=checksum_verified,
            is_archive=is_archive,
            payload_size_bytes=len(payload),
        )

    archive_member_count = 0
    if is_archive:
        extracted_artifacts = _extract_archive(
            plan,
            raw_writer,
            resolved_file,
            payload,
            version=version,
            file_hook=file_hook,
        )
        artifacts.extend(extracted_artifacts)
        archive_member_count = len(extracted_artifacts)
        if logger is not None:
            logger.info(
                "file_archive_extracted",
                file_index=file_index,
                archive_filename=resolved_file.filename,
                resolved_version=version,
                archive_member_count=len(extracted_artifacts),
                first_member_artifact_path=(
                    extracted_artifacts[0].path if extracted_artifacts else None
                ),
            )

    return FileCandidateResult(
        artifacts=tuple(artifacts),
        version=version,
        loaded_file_count=1,
        total_attempts=attempts_used,
        archive_member_count=archive_member_count,
        checksum_verified_count=checksum_verified_count,
    )

