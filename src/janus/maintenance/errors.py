"""Refusals raised by the maintenance package."""


class MaintenanceError(Exception):
    """Base for every refusal this package raises."""


class MaintenanceProfileError(MaintenanceError, ValueError):
    """The environment profile cannot name a usable retention policy.

    A ``ValueError`` for the same reason ``OpenLineageProfileError`` is one: the CLI
    already maps it to the configuration exit code without learning a new exception.
    """
