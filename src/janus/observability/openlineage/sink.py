"""One run, mapped to an OpenLineage event and handed to the configured transport."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from janus.lineage.models import LineageRecord, RunMetadata
from janus.models.dependencies import SourceDependencyGraph
from janus.observability.openlineage.facets import (
    OpenLineageDatasetContext,
    build_openlineage_run_event,
)
from janus.observability.openlineage.settings import OpenLineageTransportKind
from janus.observability.openlineage.transport import (
    NOT_CONFIGURED_REASON,
    DisabledOpenLineageTransport,
    OpenLineageEmissionOutcome,
    OpenLineageEmissionResult,
    OpenLineageTransport,
    WarningLogger,
    build_openlineage_transport,
)
from janus.observability.records import RunRecord
from janus.utils.catalog_properties import derive_pyiceberg_catalog_name
from janus.utils.environment import ICEBERG_WAREHOUSE_PATH_KEY, RuntimeLocation
from janus.utils.logging import StructuredLogger

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenLineageRunSink:
    """Map one lifecycle record and deliver it once, within the budget it is given."""

    transport: OpenLineageTransport
    dataset_context: OpenLineageDatasetContext | None = None
    context_error: str | None = None
    graph: SourceDependencyGraph | None = None

    def emit(
        self,
        run_metadata: RunMetadata,
        *,
        lineage_record: LineageRecord | None = None,
        run_record: RunRecord | None = None,
        budget_seconds: float,
        logger: StructuredLogger | WarningLogger | None = None,
    ) -> OpenLineageEmissionResult:
        """Emit one event, reporting rather than raising on every degradation path."""
        if self.transport.kind == OpenLineageTransportKind.DISABLED.value:
            return _report(self.transport.send({}, budget_seconds=budget_seconds), logger)

        if self.dataset_context is None:
            return _report(
                _failed(
                    self.transport,
                    step="dataset_context",
                    exception_type=self.context_error or "MissingDatasetContext",
                ),
                logger,
            )

        try:
            event = build_openlineage_run_event(
                run_metadata,
                self.dataset_context,
                lineage_record=lineage_record,
                run_record=run_record,
                graph=self.graph,
            )
        except Exception as exc:
            return _report(
                _failed(self.transport, step="mapping", exception_type=type(exc).__name__),
                logger,
            )

        try:
            result = self.transport.send(event, budget_seconds=budget_seconds)
        except Exception as exc:
            return _report(
                _failed(self.transport, step="transport", exception_type=type(exc).__name__),
                logger,
            )
        return _report(result, logger)


def build_openlineage_sink(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: StructuredLogger | WarningLogger | None = None,
    graph: SourceDependencyGraph | None = None,
) -> OpenLineageRunSink:
    """Select the transport and resolve the dataset identity once, for one run."""
    transport = build_openlineage_transport(config, resolved_paths, logger=logger)
    if isinstance(transport, DisabledOpenLineageTransport):
        return OpenLineageRunSink(transport=transport)

    try:
        context = _dataset_context(config, resolved_paths)
    except Exception as exc:
        return OpenLineageRunSink(transport=transport, context_error=type(exc).__name__)
    return OpenLineageRunSink(transport=transport, dataset_context=context, graph=graph)


def disabled_openlineage_sink() -> OpenLineageRunSink:
    """The default a directly-constructed emitter owns: a real object, never ``None``."""
    return OpenLineageRunSink(transport=DisabledOpenLineageTransport())


def _dataset_context(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
) -> OpenLineageDatasetContext:
    """Name datasets from the catalog identity the profile declares, doing no I/O."""
    warehouse = resolved_paths.get(ICEBERG_WAREHOUSE_PATH_KEY)
    if warehouse is None:
        spark = config.get("spark", {})
        iceberg = spark.get("iceberg", {}) if isinstance(spark, Mapping) else {}
        warehouse = iceberg.get("warehouse_dir") if isinstance(iceberg, Mapping) else None
    if warehouse is None:
        raise ValueError("the profile resolves no Iceberg warehouse to name datasets under")
    return OpenLineageDatasetContext(
        catalog_name=derive_pyiceberg_catalog_name(dict(config)),
        warehouse=str(warehouse),
    )


def _failed(
    transport: OpenLineageTransport,
    *,
    step: str,
    exception_type: str,
) -> OpenLineageEmissionResult:
    return OpenLineageEmissionResult(
        outcome=OpenLineageEmissionOutcome.FAILED,
        transport=transport.kind,
        target=getattr(transport, "target", None),
        reason=f"{step}_failed",
        step=step,
        exception_type=exception_type,
    )


def _report(
    result: OpenLineageEmissionResult,
    logger: StructuredLogger | WarningLogger | None,
) -> OpenLineageEmissionResult:
    """Degrade silently-but-loudly: one specific warning, already redacted, never a secret.

    A profile that never asked for events is not a degradation and says nothing; every other
    non-emitted outcome is one warning naming the transport, the step and the reason.
    """
    if result.emitted or result.reason == NOT_CONFIGURED_REASON:
        return result
    try:
        if logger is None:
            _LOG.warning(
                "openlineage_emission_degraded",
                extra={"event_fields": result.to_summary()},
            )
        else:
            logger.warning("openlineage_emission_degraded", **result.to_summary())
    except Exception:
        pass
    return result
