"""Global batch-planning refusals: problems that must stop a pipeline before it starts."""

from __future__ import annotations


class BatchPlanningError(ValueError):
    """Base class for a refusal that invalidates the whole batch request."""


class SelectionFilterError(BatchPlanningError):
    """Raised when the requested selectors cannot be interpreted as one filter."""


class EmptySelectionError(BatchPlanningError):
    """Raised when a selection matches no enabled source.

    Never a successful empty run: an operator who filtered on a tag that no longer
    exists asked for work, and silently doing none of it looks identical to success.
    """


class DisabledUpstreamError(BatchPlanningError):
    """Raised when a selected source requires a disabled producer.

    A batch never enables a source on an operator's behalf, and it never treats whatever
    that producer last wrote as this run's output.
    """


class GraphDriftError(BatchPlanningError):
    """Raised when planning changed the dependency graph the batch was ordered by.

    A hook may shape a plan; it may not move the bronze table a source produces or the
    tables it reads, because the order the batch is about to execute in was derived from
    those before any hook ran.
    """


class PipelineIdentityError(BatchPlanningError):
    """Raised when a pipeline or source-attempt identity is unusable."""
