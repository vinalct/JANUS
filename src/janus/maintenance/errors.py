"""Refusals raised by the maintenance package."""


class MaintenanceError(Exception):
    """Base for every refusal this package raises."""


class MaintenanceProfileError(MaintenanceError, ValueError):
    """The environment profile cannot name a usable retention policy.

    A ``ValueError`` for the same reason ``OpenLineageProfileError`` is one: the CLI
    already maps it to the configuration exit code without learning a new exception.
    """


class MaintenanceExecutionUnavailable(MaintenanceError):
    """A planned action has no executor yet; it must never be reported as applied."""


class MaintenanceInvariantError(MaintenanceError):
    """An executor was handed a protected path the planner must never select."""


class MaintenanceItemTimeout(MaintenanceError):
    """The caller stopped waiting; Spark cancellation is only a request."""

    def __init__(self, seconds: float, *, cancellation_requested: bool) -> None:
        self.cancellation_requested = cancellation_requested
        cancellation = (
            "cancellation requested" if cancellation_requested else "cancellation request failed"
        )
        super().__init__(
            f"timed out after {seconds:g}s; {cancellation}; operation may still be running"
        )
