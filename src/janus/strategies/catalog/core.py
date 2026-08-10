"""The catalog strategy façade.

What remains here is the contract, not the machinery: the :class:`CatalogStrategy` lifecycle
methods the planner calls, the :class:`CatalogHook` extension points a source implements, and
compatibility re-exports for names that were defined in this module before the package was
split.

Where the machinery went:

* ``errors.py`` — the exception hierarchy.
* ``requests.py`` — per-request mechanics (build → send → persist → count).
* ``extraction.py`` — orchestration: request-input iteration, resume, dead-letters, totals.
* ``entities.py`` — record shaping, the governed normalized-record contract, entity merging.
* ``artifacts.py`` — raw/normalized path layout, page rediscovery, replay, persistence.
* ``metadata.py`` — dead-letter metadata and per-input parameter binding.
* ``document.py`` — the document walker.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    ExtractionProgressStore,
)
from janus.models import (
    ExecutionPlan,
    ExtractionResult,
    SourceConfig,
    WriteResult,
)
from janus.strategies.api import PaginationState
from janus.strategies.base import BaseStrategy, SourceHook
from janus.strategies.catalog.document import (  # noqa: F401  (compatibility re-exports)
    CATALOG_EDGES_FILE,
    CATALOG_NODES_FILE,
    COLLECTION_ALIASES,
    CREATED_AT_KEYS,
    DATASET_HINT_KEYS,
    DESCRIPTION_KEYS,
    ENTITY_FILE_NAMES,
    ENTITY_HINT_KEYS_BY_TYPE,
    ENTITY_TYPE_ORDER,
    FORMAT_KEYS,
    GENERIC_COLLECTION_KEYS,
    GROUP_HINT_KEYS,
    IDENTIFIER_KEYS,
    NAME_KEYS,
    ORGANIZATION_HINT_KEYS,
    RESOURCE_HINT_KEYS,
    ROOT_ENTITY_PRIORITY,
    STATE_KEYS,
    TITLE_KEYS,
    UNKNOWN_ENTITY_TYPE,
    UPDATED_AT_KEYS,
    URL_KEYS,
    WRAPPER_CONTAINER_KEYS,
    CatalogBatch,
    CatalogEntityReference,
    CatalogParseSummary,
    DocumentNode,
    NodeClassification,
    _batches_from_mapping,
    _build_entity_reference,
    _build_generic_catalog_edge,
    _build_generic_catalog_node,
    _classification_confidence,
    _coerce_records,
    _collect_document_nodes,
    _compute_parse_summary,
    _first_string,
    _infer_entity_type,
    _matched_signals,
    _nested_batches,
    _normalize_mapping,
    _payload_hash,
    _root_batches,
    _score_record,
    classify_catalog_node,
    walk_document,
)
from janus.strategies.common import _default_storage_layout
from janus.strategies.http import (
    ApiRequest,
    ApiResponse,
    ApiTransport,
    UrllibApiTransport,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter


from .artifacts import (
    _persist_generic_artifacts,
    _rediscover_catalog_input_artifacts,
    _replay_catalog_entities_from_dir,
)
from .entities import _normalize_catalog_record
from .errors import CatalogPayloadError, CatalogResponseError, CatalogStrategyError
from .extraction import (
    CatalogExtractionContext,
    build_extraction_result,
    load_scoped_request_inputs,
    run_catalog_extraction,
)
from .metadata import _apply_per_input_params
from .requests import _RETRY_POLICY, CatalogRequestExecutor

__all__ = [
    "SUPPORTED_CATALOG_INPUT_FORMATS",
    "SUPPORTED_CATALOG_PAYLOAD_FORMATS",
    "_RETRY_POLICY",
    "CatalogHook",
    "CatalogPayloadError",
    "CatalogRequestExecutor",
    "CatalogResponseError",
    "CatalogStrategy",
    "CatalogStrategyError",
    "_apply_per_input_params",
    "_normalize_catalog_record",
    "_persist_generic_artifacts",
    "_rediscover_catalog_input_artifacts",
    "_replay_catalog_entities_from_dir",
]

SUPPORTED_CATALOG_PAYLOAD_FORMATS = frozenset({"json"})
SUPPORTED_CATALOG_INPUT_FORMATS = frozenset({"jsonl"})


class CatalogHook(SourceHook):
    """Catalog-specific hook points for wrapper quirks and source-local iteration details."""

    def prepare_request(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        *,
        checkpoint_state: CheckpointState | None = None,
        pagination_state: PaginationState | None = None,
    ) -> ApiRequest:
        del plan
        del checkpoint_state
        del pagination_state
        return request

    def handle_response(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
    ) -> ApiResponse:
        del plan
        del request
        return response

    def transform_payload(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Any:
        del plan
        del request
        del response
        return payload

    def checkpoint_params(
        self,
        plan: ExecutionPlan,
        checkpoint_value: str,
    ) -> Mapping[str, str] | None:
        del plan
        del checkpoint_value
        return None

    def page_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Sequence[Any] | None:
        del plan
        del request
        del response
        del payload
        return None

@dataclass(slots=True)
class CatalogStrategy(BaseStrategy):
    """Reusable metadata-first strategy for public dataset catalogs."""

    transport_factory: Callable[[], ApiTransport] = UrllibApiTransport
    storage_layout_factory: Callable[[ExecutionPlan], StorageLayout] = field(
        default_factory=lambda: _default_storage_layout
    )
    raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter] = RawArtifactWriter
    checkpoint_store: CheckpointStore = field(default_factory=CheckpointStore)
    progress_store: ExtractionProgressStore = field(default_factory=ExtractionProgressStore)
    dead_letter_store: DeadLetterStore = field(default_factory=DeadLetterStore)
    env_reader: Callable[[str], str | None] = os.getenv
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    logger: StructuredLogger | None = None
    request_executor: CatalogRequestExecutor | None = None

    @property
    def strategy_family(self) -> str:
        return "catalog"

    def plan(
        self,
        source_config: SourceConfig,
        run_context,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        self._validate_source_config(source_config)
        plan = ExecutionPlan.from_source_config(source_config, run_context)
        plan = plan.with_note("strategy_family:catalog")
        plan = plan.with_note(f"strategy_variant:{source_config.strategy_variant}")
        if hook is not None:
            return hook.on_plan(plan)
        return plan

    def extract(
        self,
        plan: ExecutionPlan,
        hook: SourceHook | None = None,
        *,
        spark=None,
    ) -> ExtractionResult:
        executor = self._build_request_executor()
        context = CatalogExtractionContext.build(
            plan,
            catalog_hook=hook if isinstance(hook, CatalogHook) else None,
            executor=executor,
            storage_layout_factory=self.storage_layout_factory,
            raw_writer_factory=self.raw_writer_factory,
            checkpoint_store=self.checkpoint_store,
            logger=self._bind_logger(plan),
            clock=self.clock,
            sleeper=self.sleeper,
        )
        request_inputs = load_scoped_request_inputs(context, spark=spark)
        outcome = run_catalog_extraction(
            executor,
            context,
            request_inputs=request_inputs,
            progress_store=self.progress_store,
            dead_letter_store=self.dead_letter_store,
        )
        extraction_result = build_extraction_result(
            context,
            outcome,
            dead_letter_store=self.dead_letter_store,
        )
        if hook is not None:
            return hook.on_extraction_result(plan, extraction_result)
        return extraction_result

    def build_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        hook: SourceHook | None = None,
    ) -> ExtractionResult:
        entity_artifact_names = {
            f"{ENTITY_FILE_NAMES[entity_type]}.jsonl"
            for entity_type in ENTITY_TYPE_ORDER
        }
        handoff_artifacts = tuple(
            artifact
            for artifact in extraction_result.artifacts
            if artifact.format == plan.source_config.spark.input_format
            and Path(artifact.path).name in entity_artifact_names
        )
        if not handoff_artifacts:
            raise CatalogStrategyError(
                "No normalized catalog entity artifacts match the configured "
                f"spark.input_format {plan.source_config.spark.input_format!r}"
            )

        handoff = replace(extraction_result, artifacts=handoff_artifacts).with_metadata(
            "normalization_artifact_count",
            str(len(handoff_artifacts)),
        )
        if hook is not None:
            return hook.on_normalization_handoff(plan, handoff)
        return handoff

    def emit_metadata(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        hook: SourceHook | None = None,
    ) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "strategy_family": self.strategy_family,
            "strategy_variant": plan.source.strategy_variant,
            "pagination_type": plan.source_config.access.pagination.type,
            "request_timeout_seconds": plan.source_config.access.timeout_seconds,
            "input_format": plan.source_config.spark.input_format,
            "artifact_count": len(extraction_result.artifacts),
            "records_extracted": extraction_result.records_extracted or 0,
            "checkpoint_value": extraction_result.checkpoint_value or "",
            "write_result_count": len(write_results),
        }
        metadata.update(extraction_result.metadata_as_dict())
        if hook is not None:
            metadata.update(hook.metadata_fields(plan, extraction_result, write_results))
        return metadata

    def _build_request_executor(self) -> CatalogRequestExecutor:
        return self.request_executor or CatalogRequestExecutor(
            transport_factory=self.transport_factory,
            sleeper=self.sleeper,
            env_reader=self.env_reader,
        )

    def _validate_source_config(self, source_config: SourceConfig) -> None:
        if source_config.access.format not in SUPPORTED_CATALOG_PAYLOAD_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_CATALOG_PAYLOAD_FORMATS))
            raise ValueError(f"Catalog access.format must be one of: {allowed}")
        if source_config.outputs.raw.format not in SUPPORTED_CATALOG_PAYLOAD_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_CATALOG_PAYLOAD_FORMATS))
            raise ValueError(f"Catalog outputs.raw.format must be one of: {allowed}")
        if source_config.spark.input_format not in SUPPORTED_CATALOG_INPUT_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_CATALOG_INPUT_FORMATS))
            raise ValueError(f"Catalog spark.input_format must be one of: {allowed}")

    def _bind_logger(self, plan: ExecutionPlan) -> StructuredLogger | None:
        if self.logger is None:
            return None
        return self.logger.bind(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            strategy_family=self.strategy_family,
        )
