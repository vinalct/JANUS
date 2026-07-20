from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.utils.environment import build_spark_session
from janus.utils.logging import StructuredLogger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class SparkSessionProvider:
    """Owns when a Spark session is alive, so callers never hold compute during I/O.

    The provider adds no session-construction logic of its own: it defers to
    :func:`build_spark_session`, which is config-driven and therefore works
    unchanged under ``local[*]`` or any cluster master.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        resolved_paths: Mapping[str, Path],
        logger: StructuredLogger | None = None,
        *,
        session_factory: Callable[[], SparkSession] | None = None,
    ) -> None:
        self._config = dict(config)
        self._resolved_paths = dict(resolved_paths)
        self._logger = logger
        self._session_factory = session_factory or self._build_session
        self._session: SparkSession | None = None
        self._owns_session = True
        self._session_info: dict[str, Any] | None = None
        self._was_started = False

    @classmethod
    def wrapping(
        cls,
        session: SparkSession,
        logger: StructuredLogger | None = None,
    ) -> SparkSessionProvider:
        """Adapt an externally-owned live session.

        ``get()`` hands back the wrapped session and ``stop()`` never stops it —
        the caller that built it keeps ownership. Because the provider never
        started this session, it reports neither ``was_started`` nor
        ``session_info``.
        """

        provider = cls({}, {}, logger, session_factory=_reject_rebuild)
        provider._session = session
        provider._owns_session = False
        return provider

    @property
    def session_info(self) -> dict[str, Any] | None:
        """Describe the most recent session this provider started, else ``None``."""

        return dict(self._session_info) if self._session_info is not None else None

    @property
    def was_started(self) -> bool:
        """Return whether this provider has ever started a session of its own."""

        return self._was_started

    def get(self) -> SparkSession:
        """Return the live session, building one on first call."""

        if self._session is not None:
            return self._session

        spark_config = self._config.get("spark", {})
        self._log(
            "spark_session_starting",
            app_name=spark_config.get("app_name"),
            master=spark_config.get("master"),
        )
        session = self._session_factory()
        self._session = session
        self._owns_session = True
        self._session_info = _describe_session(session)
        self._was_started = True
        self._log("spark_session_started", **self._session_info)
        return session

    def stop(self) -> None:
        """Release the session if this provider created it. Idempotent.

        A later :meth:`get` may build a fresh session — an ``iceberg_rows`` run
        legitimately acquires, stops, acquires and stops again.
        """

        session = self._session
        self._session = None
        if session is None or not self._owns_session:
            return

        try:
            session.stop()
        except Exception as exc:
            # Never let a teardown failure mask the run's real outcome.
            self._log_exception(
                "spark_session_stop_failed",
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )
            return
        self._log("spark_session_stopped")

    def _build_session(self) -> SparkSession:
        return build_spark_session(self._config, self._resolved_paths)

    def _log(self, event: str, **fields: Any) -> None:
        if self._logger is not None:
            self._logger.info(event, **fields)

    def _log_exception(self, event: str, **fields: Any) -> None:
        if self._logger is not None:
            self._logger.exception(event, **fields)


def _describe_session(session: SparkSession) -> dict[str, Any]:
    context = getattr(session, "sparkContext", None)
    return {
        "app_name": getattr(context, "appName", None),
        "master": getattr(context, "master", None),
    }


def _reject_rebuild() -> SparkSession:
    raise RuntimeError(
        "SparkSessionProvider.wrapping() cannot build a session: the wrapped "
        "session is owned externally and was already released"
    )
