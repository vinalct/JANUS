from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult, WriteResult
from janus.strategies.api import ApiHook, ApiRequest, ApiResponse
from janus.utils.environment import resolve_project_path

HOOK_ID = "ibge.sidra_flat"
LEGACY_HOOK_IDS = ("ibge.pib_brasil",)
NORMALIZED_ARTIFACT_NAME = "sidra_records.jsonl"
DIMENSION_KEY_PATTERN = re.compile(r"^D(?P<position>\d+)(?P<kind>[CN])$")
PERIOD_LABEL_HINTS = ("ano", "periodo", "mes", "trimestre", "semestre")
VARIABLE_LABEL_HINTS = ("variavel", "indicador")


class IbgeSidraFlatHook(ApiHook):
    """IBGE SIDRA flat-view adapter that keeps dimension parsing generic."""

    def on_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan.with_note(f"hook:{_configured_hook_id(plan)}")

    def checkpoint_params(
        self,
        plan: ExecutionPlan,
        checkpoint_value: str,
    ) -> Mapping[str, str]:
        del plan
        del checkpoint_value
        return {}

    def extract_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Sequence[Mapping[str, Any]]:
        del request
        del response
        return normalize_sidra_flat_payload(payload, aggregate_id=_aggregate_id(plan))

    def on_extraction_result(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
    ) -> ExtractionResult:
        del plan
        return (
            extraction_result.with_metadata("ibge_payload_shape", "sidra_flat_view")
            .with_metadata("ibge_customization", "header_driven_sidra_projection")
            .with_metadata("ibge_strategy_reuse", "api_http_retry_rate_limit_raw_json")
        )

    def on_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
    ) -> ExtractionResult:
        normalized_records, attribute_keys, dimension_ids = _collect_sidra_projection_details(
            plan,
            extraction_result.artifacts,
        )

        metadata = extraction_result.metadata_as_dict()
        metadata["ibge_normalized_record_count"] = str(len(normalized_records))
        metadata["ibge_sidra_attribute_keys"] = attribute_keys
        metadata["ibge_sidra_dimension_ids"] = dimension_ids

        if not normalized_records:
            metadata["ibge_normalized_artifact_count"] = "0"
            return ExtractionResult.from_plan(
                plan,
                (),
                records_extracted=0,
                checkpoint_value=extraction_result.checkpoint_value,
                metadata=metadata,
            )

        artifact = _write_normalized_jsonl(plan, normalized_records)
        metadata["ibge_normalized_artifact_count"] = "1"
        return ExtractionResult.from_plan(
            plan,
            (artifact,),
            records_extracted=len(normalized_records),
            checkpoint_value=extraction_result.checkpoint_value,
            metadata=metadata,
        )

    def metadata_fields(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
    ) -> Mapping[str, Any]:
        del write_results
        metadata = extraction_result.metadata_as_dict()
        attribute_keys = metadata.get("ibge_sidra_attribute_keys", "")
        dimension_ids = metadata.get("ibge_sidra_dimension_ids", "")
        if not attribute_keys or not dimension_ids:
            _, derived_attribute_keys, derived_dimension_ids = _collect_sidra_projection_details(
                plan,
                extraction_result.artifacts,
            )
            if not attribute_keys:
                attribute_keys = derived_attribute_keys
            if not dimension_ids:
                dimension_ids = derived_dimension_ids

        return {
            "ibge_hook": _configured_hook_id(plan),
            "ibge_aggregate_id": _aggregate_id(plan),
            "ibge_payload_shape": metadata.get("ibge_payload_shape", "sidra_flat_view"),
            "ibge_customization": metadata.get(
                "ibge_customization",
                "header_driven_sidra_projection",
            ),
            "ibge_reused_api_features": (
                "http_client,retries,rate_limit,raw_writer,checkpointing"
            ),
            "ibge_sidra_attribute_keys": attribute_keys,
            "ibge_sidra_dimension_ids": dimension_ids,
        }


def _collect_sidra_projection_details(
    plan: ExecutionPlan,
    artifacts: Sequence[ExtractedArtifact],
) -> tuple[list[Mapping[str, Any]], str, str]:
    normalized_records: list[Mapping[str, Any]] = []
    attribute_keys: set[str] = set()
    dimension_ids: set[str] = set()

    for artifact in artifacts:
        if artifact.format != "json":
            continue

        payload = _load_json_artifact(artifact)
        page_records = normalize_sidra_flat_payload(payload, aggregate_id=_aggregate_id(plan))
        normalized_records.extend(page_records)
        for record in page_records:
            for attribute in record.get("sidra_attributes", ()):
                attribute_key = str(attribute.get("key", "")).strip()
                if attribute_key:
                    attribute_keys.add(attribute_key)
            for dimension in record.get("sidra_dimensions", ()):
                dimension_id = str(dimension.get("id", "")).strip()
                if dimension_id:
                    dimension_ids.add(dimension_id)

    return (
        normalized_records,
        ",".join(sorted(attribute_keys)),
        ",".join(sorted(dimension_ids, key=_dimension_sort_key)),
    )


IbgePibBrasilHook = IbgeSidraFlatHook


def normalize_sidra_flat_payload(
    payload: Any,
    *,
    aggregate_id: str,
) -> tuple[Mapping[str, Any], ...]:
    if not _looks_like_sidra_flat_payload(payload):
        return ()

    header_row = _header_row(payload)
    if header_row is None:
        return ()

    records: list[Mapping[str, Any]] = []
    for row in payload:
        if not isinstance(row, Mapping) or _is_header_row(row):
            continue
        records.append(_normalize_row(row, header_row, aggregate_id=aggregate_id))
    return tuple(records)


def _looks_like_sidra_flat_payload(payload: Any) -> bool:
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes | bytearray):
        return False

    header_row = _header_row(payload)
    if header_row is None or "V" not in header_row:
        return False

    return bool(_dimension_ids(header_row)) or any(
        bool(_dimension_ids(row))
        for row in payload
        if isinstance(row, Mapping)
    )


def _normalize_row(
    row: Mapping[str, Any],
    header_row: Mapping[str, Any] | None,
    *,
    aggregate_id: str,
) -> Mapping[str, Any]:
    sidra_attributes = _sidra_attributes(row, header_row)
    sidra_dimensions = _sidra_dimensions(row, header_row)
    normalized = {
        "aggregate_id": aggregate_id,
        "value": _text(row.get("V")),
        "sidra_value_label": _text(header_row.get("V")) if header_row else "",
        "sidra_attributes": sidra_attributes,
        "sidra_attribute_count": len(sidra_attributes),
        "sidra_dimensions": sidra_dimensions,
        "sidra_dimension_count": len(sidra_dimensions),
    }
    normalized.update(_derived_dimension_fields(sidra_dimensions))
    return normalized


def _sidra_attributes(
    row: Mapping[str, Any],
    header_row: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    attributes: list[Mapping[str, Any]] = []
    for key in _attribute_keys(row, header_row):
        attributes.append(
            {
                "key": key,
                "label": _text(header_row.get(key)) if header_row else "",
                "value": _text(row.get(key)),
            }
        )
    return tuple(attributes)


def _sidra_dimensions(
    row: Mapping[str, Any],
    header_row: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    dimensions: list[Mapping[str, Any]] = []
    for dimension_id in _dimension_ids(header_row, row):
        dimensions.append(
            {
                "id": dimension_id,
                "position": _dimension_sort_key(dimension_id),
                "code": _text(row.get(f"{dimension_id}C")),
                "name": _text(row.get(f"{dimension_id}N")),
                "code_label": _text(header_row.get(f"{dimension_id}C")) if header_row else "",
                "label": _text(header_row.get(f"{dimension_id}N")) if header_row else "",
            }
        )
    return tuple(dimensions)


def _derived_dimension_fields(
    sidra_dimensions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    derived = {
        "sidra_period_code": "",
        "sidra_period_name": "",
        "sidra_variable_code": "",
        "sidra_variable_name": "",
    }

    for dimension in sidra_dimensions:
        normalized_hint = _normalized_hint(
            " ".join(
                part
                for part in (
                    _text(dimension.get("code_label")),
                    _text(dimension.get("label")),
                )
                if part
            )
        )
        if (
            not derived["sidra_period_code"]
            and any(hint in normalized_hint for hint in PERIOD_LABEL_HINTS)
        ):
            derived["sidra_period_code"] = _text(dimension.get("code"))
            derived["sidra_period_name"] = _text(dimension.get("name"))
        if (
            not derived["sidra_variable_code"]
            and any(hint in normalized_hint for hint in VARIABLE_LABEL_HINTS)
        ):
            derived["sidra_variable_code"] = _text(dimension.get("code"))
            derived["sidra_variable_name"] = _text(dimension.get("name"))

    return derived


def _header_row(payload: Sequence[Any]) -> Mapping[str, Any] | None:
    for row in payload:
        if isinstance(row, Mapping) and _is_header_row(row):
            return row
    return None


def _attribute_keys(
    row: Mapping[str, Any],
    header_row: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    attribute_keys = {
        str(key)
        for mapping in (header_row, row)
        if isinstance(mapping, Mapping)
        for key in mapping
        if str(key) != "V" and DIMENSION_KEY_PATTERN.fullmatch(str(key)) is None
    }
    return tuple(sorted(attribute_keys))


def _dimension_ids(*mappings: Mapping[str, Any] | None) -> tuple[str, ...]:
    dimension_ids = {
        f"D{match.group('position')}"
        for mapping in mappings
        if isinstance(mapping, Mapping)
        for key in mapping
        if (match := DIMENSION_KEY_PATTERN.fullmatch(str(key))) is not None
    }
    return tuple(sorted(dimension_ids, key=_dimension_sort_key))


def _dimension_sort_key(dimension_id: str) -> int:
    return int(dimension_id[1:])


def _is_header_row(row: Mapping[str, Any]) -> bool:
    return _normalized_hint(_text(row.get("V"))) == "valor"


def _configured_hook_id(plan: ExecutionPlan) -> str:
    configured_hook_id = _text(plan.source_config.source_hook)
    if configured_hook_id:
        return configured_hook_id
    return HOOK_ID


def _normalized_hint(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower().strip()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _aggregate_id(plan: ExecutionPlan) -> str:
    for value in (plan.source_config.access.path, plan.source_config.access.url):
        if not value:
            continue
        path_segments = [segment for segment in value.split("/") if segment]
        if "agregados" not in path_segments:
            continue
        index = path_segments.index("agregados") + 1
        if index < len(path_segments):
            return path_segments[index]
    return ""


def _load_json_artifact(artifact: ExtractedArtifact) -> Any:
    if artifact.format != "json":
        raise ValueError(f"IBGE hook expects json artifacts, received {artifact.format!r}")
    return json.loads(Path(artifact.path).read_text(encoding="utf-8"))


def _write_normalized_jsonl(
    plan: ExecutionPlan,
    records: Sequence[Mapping[str, Any]],
) -> ExtractedArtifact:
    raw_root = resolve_project_path(plan.run_context.project_root, plan.raw_output.path)
    path = raw_root / "normalized" / NORMALIZED_ARTIFACT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)

    body = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    payload = body.encode("utf-8")
    path.write_bytes(payload)
    checksum = sha256(payload).hexdigest()
    return ExtractedArtifact(path=str(path), format="jsonl", checksum=checksum)


__all__ = [
    "HOOK_ID",
    "LEGACY_HOOK_IDS",
    "IbgePibBrasilHook",
    "IbgeSidraFlatHook",
    "normalize_sidra_flat_payload",
]
