from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from janus.models.contracts import ExecutionPlan, ExtractionResult, RunContext, WriteResult
from janus.models.source_config import SourceConfig

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from janus.runtime.spark_lifecycle import SparkSessionProvider


class SourceHook:
    """Small, optional extension points for one source without changing a strategy family."""

    def on_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Adjust the plan for one source while keeping planner logic generic."""
        return plan

    def on_extraction_result(
        self, plan: ExecutionPlan, extraction_result: ExtractionResult
    ) -> ExtractionResult:
        """Adjust extracted artifacts or metadata before normalization handoff."""
        return extraction_result

    def on_normalization_handoff(
        self, plan: ExecutionPlan, extraction_result: ExtractionResult
    ) -> ExtractionResult:
        """Adjust the normalization handoff object without changing strategy-core behavior."""
        return extraction_result

    def metadata_fields(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
    ) -> Mapping[str, Any]:
        """Emit source-local metadata that should travel with the run metadata layer."""
        return {}


class BaseStrategy(ABC):
    """Common contract that API, file, and catalog strategies must all implement."""

    @property
    @abstractmethod
    def strategy_family(self) -> str:
        """Return the strategy family name exposed by the implementation."""

    @abstractmethod
    def plan(
        self,
        source_config: SourceConfig,
        run_context: RunContext,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        """Build an execution plan from a validated source config and run context."""

    @abstractmethod
    def extract(
        self,
        plan: ExecutionPlan,
        hook: SourceHook | None = None,
        *,
        spark: SparkSessionProvider | SparkSession | None = None,
    ) -> ExtractionResult:
        """Perform source-family extraction and return the raw-artifact handoff contract.

        ``spark`` carries a :class:`SparkSessionProvider` when the executor runs the
        strategy: extraction is driver-only work, so no live session is held across it.
        Families that load request inputs materialize a session from the provider only
        for that lookup (see ``scoped_request_input_session``). An externally-owned
        ``SparkSession`` is still accepted for embedding callers and tests.
        """

    @abstractmethod
    def build_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        hook: SourceHook | None = None,
    ) -> ExtractionResult:
        """Return the extraction result shape that the normalization layer should consume."""

    @abstractmethod
    def emit_metadata(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        hook: SourceHook | None = None,
    ) -> Mapping[str, Any]:
        """Emit strategy-level metadata without coupling the base layer to one store or format."""
