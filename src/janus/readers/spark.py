from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.models import ExecutionPlan, ExtractionResult
from janus.utils.storage import StorageLayout

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

SUPPORTED_SPARK_READ_FORMATS = frozenset({"binary", "csv", "json", "jsonl", "parquet", "text"})
CORRUPT_RECORD_READ_FORMATS = frozenset({"csv", "json", "jsonl"})


class SparkDatasetReader:
    """Generic Spark reader for raw artifacts and explicit JANUS output zones."""

    supports_corrupt_record_read = True

    def read_paths(
        self,
        spark: SparkSession,
        paths: Sequence[str | Path],
        *,
        format_name: str,
        schema: Any | None = None,
        options: Mapping[str, Any] | None = None,
        corrupt_record_column: str | None = None,
    ) -> DataFrame:
        normalized_paths = _normalize_paths(paths)
        reader = spark.read.format(_spark_read_format(format_name))
        if schema is not None:
            reader = reader.schema(schema)
        for key, value in _resolved_read_options(
            format_name, options, corrupt_record_column=corrupt_record_column
        ).items():
            reader = reader.option(key, value)

        load_argument: str | list[str]
        if len(normalized_paths) == 1:
            load_argument = normalized_paths[0]
        else:
            load_argument = list(normalized_paths)
        return reader.load(load_argument)

    def read_extraction_result(
        self,
        spark: SparkSession,
        extraction_result: ExtractionResult,
        *,
        format_name: str | None = None,
        schema: Any | None = None,
        options: Mapping[str, Any] | None = None,
        corrupt_record_column: str | None = None,
    ) -> DataFrame:
        if extraction_result.is_empty:
            raise ValueError("extraction_result must contain at least one artifact")

        resolved_format = format_name or _single_artifact_format(extraction_result)
        return self.read_paths(
            spark,
            extraction_result.artifact_paths,
            format_name=resolved_format,
            schema=schema,
            options=options,
            corrupt_record_column=corrupt_record_column,
        )

    def read_plan_output(
        self,
        spark: SparkSession,
        plan: ExecutionPlan,
        storage_layout: StorageLayout,
        zone: str,
        *,
        format_name: str | None = None,
        schema: Any | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> DataFrame:
        resolved_target = storage_layout.resolve_output(plan, zone)
        return self.read_paths(
            spark,
            (resolved_target.resolved_path,),
            format_name=format_name or resolved_target.format,
            schema=schema,
            options=options,
        )


def _normalize_paths(paths: Sequence[str | Path]) -> tuple[str, ...]:
    normalized_paths = tuple(str(Path(path)) for path in paths)
    if not normalized_paths:
        raise ValueError("paths must not be empty")
    return _compact_complete_parent_paths(normalized_paths)


def _compact_complete_parent_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    file_groups: dict[Path, list[Path]] = defaultdict(list)
    for path_value in paths:
        path = Path(path_value)
        if path.is_file():
            file_groups[path.parent].append(path)

    compacted_parents: set[Path] = set()
    for parent, selected_files in file_groups.items():
        selected_file_set = set(selected_files)
        if len(selected_files) <= 1 or len(selected_file_set) != len(selected_files):
            continue
        discovered_files = {candidate for candidate in parent.rglob("*") if candidate.is_file()}
        if discovered_files == selected_file_set:
            compacted_parents.add(parent)

    if not compacted_parents:
        return paths

    compacted_paths: list[str] = []
    emitted_parents: set[Path] = set()
    for path_value in paths:
        path = Path(path_value)
        parent = path.parent
        if path.is_file() and parent in compacted_parents:
            if parent not in emitted_parents:
                compacted_paths.append(str(parent))
                emitted_parents.add(parent)
            continue
        compacted_paths.append(path_value)
    return tuple(compacted_paths)


def _resolved_read_options(
    format_name: str,
    options: Mapping[str, Any] | None,
    *,
    corrupt_record_column: str | None = None,
) -> dict[str, str]:
    resolved_options = _default_read_options(
        format_name, corrupt_record_column=corrupt_record_column
    )
    resolved_options.update(_normalize_options(options))
    return resolved_options


def _default_read_options(
    format_name: str, *, corrupt_record_column: str | None = None
) -> dict[str, str]:
    normalized = format_name.strip().lower()
    options: dict[str, str] = {}
    if normalized == "json":
        options["multiLine"] = "true"
    if corrupt_record_column is not None and normalized in CORRUPT_RECORD_READ_FORMATS:
        options.update(mode="PERMISSIVE", columnNameOfCorruptRecord=corrupt_record_column)
    return options


def _normalize_options(options: Mapping[str, Any] | None) -> dict[str, str]:
    if not options:
        return {}
    return {str(key): str(value) for key, value in options.items()}


def _single_artifact_format(extraction_result: ExtractionResult) -> str:
    formats = {artifact.format for artifact in extraction_result.artifacts}
    if len(formats) != 1:
        raise ValueError(
            "Artifacts contain multiple formats; pass format_name explicitly to read them"
        )
    return next(iter(formats))


def _spark_read_format(format_name: str) -> str:
    normalized = format_name.strip().lower()
    if normalized not in SUPPORTED_SPARK_READ_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_SPARK_READ_FORMATS))
        raise ValueError(f"format_name must be one of: {allowed}")
    if normalized == "binary":
        return "binaryFile"
    if normalized == "jsonl":
        return "json"
    return normalized
