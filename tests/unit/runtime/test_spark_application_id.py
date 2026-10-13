"""Application diagnostics must never acquire Spark or alter lifecycle log fields."""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from janus.runtime import SparkSessionProvider
from janus.runtime.spark_lifecycle import _describe_session
from janus.utils.logging import build_structured_logger


def session(application_id):
    return SimpleNamespace(
        sparkContext=SimpleNamespace(
            appName="fixture", master="local[1]", applicationId=application_id
        ),
        stop=Mock(),
    )


def test_application_id_is_lazy_retained_after_stop_and_tracks_second_session():
    first, second = session("application-1"), session("application-2")
    factory = Mock(side_effect=[first, second])
    provider = SparkSessionProvider({}, {}, session_factory=factory)
    assert provider.application_id is None
    assert provider.was_started is False
    factory.assert_not_called()
    assert provider.get() is first
    assert provider.application_id == "application-1"
    assert provider.get() is first
    provider.stop()
    assert provider.application_id == "application-1"
    assert provider.get() is second
    assert provider.application_id == "application-2"
    provider.stop()
    assert provider.application_id == "application-2"
    first.stop.assert_called_once()
    second.stop.assert_called_once()
    assert factory.call_count == 2


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (0, None), (42, "42")])
def test_application_id_handles_missing_or_non_string_context_values(value, expected):
    provider = SparkSessionProvider({}, {}, session_factory=lambda: session(value))
    provider.get()
    assert provider.application_id == expected


@pytest.mark.parametrize(
    "spark",
    [SimpleNamespace(stop=Mock()), SimpleNamespace(sparkContext=SimpleNamespace(), stop=Mock())],
)
def test_application_id_tolerates_absent_context_or_attribute(spark):
    provider = SparkSessionProvider({}, {}, session_factory=lambda: spark)
    provider.get()
    assert provider.application_id is None


def test_wrapped_application_id_stays_none_and_owned_diagnostics_are_unchanged():
    spark = session("application-external")
    wrapped = SparkSessionProvider.wrapping(spark)
    assert wrapped.application_id is None
    assert wrapped.get() is spark
    wrapped.stop()
    assert wrapped.application_id is None and wrapped.was_started is False
    spark.stop.assert_not_called()
    stream = StringIO()
    logger = build_structured_logger("janus.tests.application_id", stream=stream)
    provider = SparkSessionProvider({}, {}, logger, session_factory=lambda: spark)
    provider.get()
    provider.stop()
    expected = {"app_name": "fixture", "master": "local[1]"}
    assert provider.session_info == _describe_session(spark) == expected
    started = next(
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if json.loads(line)["event"] == "spark_session_started"
    )
    assert json.dumps(started["fields"], sort_keys=True) == json.dumps(expected, sort_keys=True)
