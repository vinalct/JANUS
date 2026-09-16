"""Dagster adapter errors with actionable installation and drift messages."""


class DagsterAdapterError(RuntimeError):
    """Base error raised by the optional Dagster adapter."""


class DagsterConfigurationDriftError(DagsterAdapterError):
    """The definition snapshot and worker configuration no longer agree."""


class DagsterRunCollectionError(DagsterAdapterError):
    """A Dagster run cannot be translated into a complete JANUS outcome."""
