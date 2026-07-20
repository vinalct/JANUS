from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO

from janus.runtime import SparkSessionProvider
from janus.utils.logging import build_structured_logger

ENVIRONMENT_CONFIG = {
    "spark": {
        "app_name": "janus-tests",
        "master": "local[*]",
    }
}


@dataclass(slots=True)
class StubSparkContext:
    appName: str  # matches the PySpark attribute name
    master: str


@dataclass(slots=True)
class StubSession:
    app_name: str = "janus-tests"
    master: str = "local[*]"
    stop_calls: int = 0

    @property
    def sparkContext(self) -> StubSparkContext:  # matches the PySpark attribute
        return StubSparkContext(appName=self.app_name, master=self.master)

    def stop(self) -> None:
        self.stop_calls += 1


@dataclass(slots=True)
class StubSessionFactory:
    sessions: list[StubSession] = field(default_factory=list)

    def __call__(self) -> StubSession:
        session = StubSession()
        self.sessions.append(session)
        return session


def test_provider_does_not_build_a_session_until_get_is_called():
    factory = StubSessionFactory()

    provider = SparkSessionProvider(ENVIRONMENT_CONFIG, {}, session_factory=factory)

    assert factory.sessions == []
    assert provider.was_started is False
    assert provider.session_info is None


def test_get_builds_once_and_returns_the_same_session():
    factory = StubSessionFactory()
    provider = SparkSessionProvider(ENVIRONMENT_CONFIG, {}, session_factory=factory)

    first = provider.get()
    second = provider.get()

    assert first is second
    assert len(factory.sessions) == 1
    assert provider.was_started is True
    assert provider.session_info == {"app_name": "janus-tests", "master": "local[*]"}


def test_stop_releases_an_owned_session_and_is_idempotent():
    factory = StubSessionFactory()
    provider = SparkSessionProvider(ENVIRONMENT_CONFIG, {}, session_factory=factory)
    session = provider.get()

    provider.stop()
    provider.stop()

    assert session.stop_calls == 1


def test_stop_without_a_session_is_a_no_op():
    factory = StubSessionFactory()
    provider = SparkSessionProvider(ENVIRONMENT_CONFIG, {}, session_factory=factory)

    provider.stop()

    assert factory.sessions == []


def test_get_after_stop_builds_a_fresh_session():
    factory = StubSessionFactory()
    provider = SparkSessionProvider(ENVIRONMENT_CONFIG, {}, session_factory=factory)

    first = provider.get()
    provider.stop()
    second = provider.get()

    assert first is not second
    assert len(factory.sessions) == 2
    assert provider.session_info == {"app_name": "janus-tests", "master": "local[*]"}


def test_session_info_reports_the_actual_session_not_the_configured_one():
    session = StubSession(app_name="resolved-app", master="yarn")
    provider = SparkSessionProvider(ENVIRONMENT_CONFIG, {}, session_factory=lambda: session)

    provider.get()

    assert provider.session_info == {"app_name": "resolved-app", "master": "yarn"}


def test_wrapping_returns_the_external_session_without_building_one():
    session = StubSession()

    provider = SparkSessionProvider.wrapping(session)

    assert provider.get() is session
    assert provider.get() is session


def test_wrapping_never_stops_the_session_it_does_not_own():
    session = StubSession()
    provider = SparkSessionProvider.wrapping(session)

    provider.stop()
    provider.stop()

    assert session.stop_calls == 0
    assert provider.was_started is False
    assert provider.session_info is None


def test_provider_logs_the_session_lifecycle_events():
    stream = StringIO()
    logger = build_structured_logger("janus.tests.spark_lifecycle", stream=stream)
    provider = SparkSessionProvider(
        ENVIRONMENT_CONFIG,
        {},
        logger,
        session_factory=StubSessionFactory(),
    )

    provider.get()
    provider.stop()

    events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    by_event = {event["event"]: event for event in events}
    assert [event["event"] for event in events] == [
        "spark_session_starting",
        "spark_session_started",
        "spark_session_stopped",
    ]
    assert by_event["spark_session_starting"]["fields"] == {
        "app_name": "janus-tests",
        "master": "local[*]",
    }
    assert by_event["spark_session_started"]["fields"] == {
        "app_name": "janus-tests",
        "master": "local[*]",
    }


def test_stop_failure_is_logged_and_swallowed():
    @dataclass(slots=True)
    class ExplodingSession(StubSession):
        def stop(self) -> None:
            raise RuntimeError("stop boom")

    stream = StringIO()
    logger = build_structured_logger("janus.tests.spark_lifecycle.failure", stream=stream)
    provider = SparkSessionProvider(
        ENVIRONMENT_CONFIG,
        {},
        logger,
        session_factory=ExplodingSession,
    )
    provider.get()

    provider.stop()

    events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    assert events[-1]["event"] == "spark_session_stop_failed"
    assert events[-1]["fields"]["failure_reason"] == "stop boom"
    assert events[-1]["fields"]["error_type"] == "RuntimeError"
