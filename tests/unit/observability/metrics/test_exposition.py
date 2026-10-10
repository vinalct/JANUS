"""Independent wire examples and hostile detector cases for the strict parser."""

import math
from dataclasses import replace

import pytest

from janus.observability.metrics.catalogue import MetricDeclaration, MetricType
from janus.observability.metrics.exposition import ParsedSample, parse_exposition, render_exposition
from janus.observability.metrics.transport import MetricSample


def test_parser_accepts_independently_written_exposition_and_escapes():
    text = (
        '# HELP duration A "quoted" help\\nline and \\\\ slash.\n'
        "# TYPE duration gauge\n"
        'duration{status="succeeded",source_id="quoted \\"name\\"\\nC:\\\\data"} -1.25e2\n'
        "# HELP total Count.\n"
        "# TYPE total counter\n"
        "total 3\n"
    )
    assert parse_exposition(text) == (
        ParsedSample(
            "duration",
            MetricType.GAUGE,
            'A "quoted" help\nline and \\ slash.',
            (("source_id", 'quoted "name"\nC:\\data'), ("status", "succeeded")),
            -125.0,
        ),
        ParsedSample("total", MetricType.COUNTER, "Count.", (), 3.0),
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("metric 1\n", "preceding TYPE"),
        ("# TYPE metric gauge\n# TYPE metric gauge\nmetric 1\n", "duplicate TYPE"),
        ("# TYPE metric gauge\nmetric 1\nmetric 2\n", "duplicate metric"),
        ('# TYPE metric gauge\nmetric{source_id="bad"quote"} 1\n', "malformed label"),
        ('# TYPE metric gauge\nmetric{source_id="x",} 1\n', "trailing comma"),
        ('# TYPE metric gauge\nmetric{source_id="x"}\n', "missing value"),
        ("# TYPE metric gauge\nmetric{source_id=x} 1\n", "malformed label"),
        ('# TYPE metric gauge\nmetric{source_id="x" 1\n', "malformed sample"),
        ('# TYPE metric gauge\nmetric{source_id="x",source_id="y"} 1\n', "duplicate label"),
        (
            '# TYPE metric gauge\nmetric{source_id="x",status="ok"} 1\n'
            'metric{status="ok",source_id="x"} 2\n',
            "duplicate metric",
        ),
        ('# TYPE metric gauge\nmetric{source_id="bad\\q"} 1\n', "malformed label"),
        ("# HELP metric bad\\q\n# TYPE metric gauge\nmetric 1\n", "invalid exposition escape"),
        ("# HELP metric bad\\\n# TYPE metric gauge\nmetric 1\n", "invalid exposition escape"),
        ("# HELP metric First.\n# HELP metric Second.\n", "duplicate HELP"),
        ("# TYPE metric gauge\nmetric 1\n# HELP metric Late.\n", "precede"),
        ("# TYPE metric gauge\nmetric 1\n# TYPE metric counter\n", "precede"),
        ("# TYPE metric histogram\n", "gauge or counter"),
        ("# TYPE metric gauge extra\n", "gauge or counter"),
        ("# TYPE\n", "malformed HELP or TYPE"),
        ("# TYPE metric gauge\nmetric 1 12345\n", "timestamps"),
        ("# TYPE metric gauge\nmetric 1", "newline"),
        ("# TYPE metric gauge\nmetric 1,000\n", "invalid sample value"),
        ("# TYPE metric gauge\nmetric true\n", "invalid sample value"),
        ("# TYPE metric gauge\nmetric nan\n", "invalid sample value"),
        (
            "# TYPE first gauge\n# TYPE second gauge\nfirst 1\nsecond 2\n"
            'first{source_id="different"} 3\n',
            "grouped",
        ),
    ],
    ids=[
        "missing-type",
        "duplicate-type",
        "duplicate-series",
        "unescaped-quote",
        "trailing-comma",
        "missing-value",
        "unquoted-label",
        "unclosed-label-block",
        "duplicate-label",
        "reordered-duplicate-series",
        "bad-label-escape",
        "bad-help-escape",
        "incomplete-help-escape",
        "duplicate-help",
        "late-help",
        "late-type",
        "unsupported-type",
        "extra-type-token",
        "missing-type-name",
        "timestamp-suffix",
        "missing-newline",
        "thousands-separator",
        "boolean-value",
        "incorrect-nan-spelling",
        "ungrouped-samples",
    ],
)
def test_parser_rejects_deliberate_wire_violations(text, message):
    with pytest.raises(ValueError, match=message):
        parse_exposition(text)


@pytest.mark.parametrize("value", ["0", "-0.0", "+1.2", ".5", "1.", "1e-9", "+Inf", "-Inf"])
def test_parser_accepts_float_syntax(value):
    parsed = parse_exposition(f"# TYPE metric gauge\nmetric {value}\n")
    assert parsed[0].value == float(value)


def test_parser_accepts_nan_and_ignores_ordinary_comments_and_blank_lines():
    parsed = parse_exposition("# comment\n\n# TYPE metric gauge\n metric{} NaN \n")
    assert len(parsed) == 1
    assert math.isnan(parsed[0].value)
    assert parsed[0].labels == ()
    assert parse_exposition("") == ()


def _sample(name, value, *, labels=(), help_text="Measurement.", metric_type=MetricType.GAUGE):
    return MetricSample(
        MetricDeclaration(
            name, metric_type, help_text, "duration_seconds", "run", ("source_id", "status")
        ),
        labels,
        value,
    )


def test_renderer_round_trips_realistic_samples_including_special_values():
    samples = (
        _sample("duration", -5.25, labels=(("source_id", 'quotes " and slash \\ and\nline'),)),
        _sample("requests", 5, metric_type=MetricType.COUNTER),
        _sample("positive_infinity", math.inf),
        _sample("negative_infinity", -math.inf),
        _sample("missing_number", math.nan),
    )
    text = render_exposition(samples)
    parsed = parse_exposition(text)
    assert len(parsed) == len(samples)
    for expected, actual in zip(sorted(samples, key=lambda item: item.name), parsed, strict=True):
        assert (actual.name, actual.metric_type, actual.help_text, actual.labels) == (
            expected.name,
            expected.metric_type,
            expected.help_text,
            expected.labels,
        )
        assert actual.value == pytest.approx(expected.value, nan_ok=True)
    assert " NaN\n" in text and " +Inf\n" in text and " -Inf\n" in text
    assert render_exposition(parsed) == text
    assert parse_exposition(render_exposition(parsed)) == parsed


def test_renderer_groups_sorts_and_emits_metadata_once_before_samples():
    first = _sample("a_duration", 1.0, labels=(("status", "ok"), ("source_id", "b")))
    second = replace(first, labels=(("source_id", "a"), ("status", "ok")), value=2.0)
    third = _sample("z_total", 3.0, metric_type=MetricType.COUNTER)
    expected = (
        "# HELP a_duration Measurement.\n# TYPE a_duration gauge\n"
        'a_duration{source_id="a",status="ok"} 2.0\n'
        'a_duration{source_id="b",status="ok"} 1.0\n'
        "# HELP z_total Measurement.\n# TYPE z_total counter\nz_total 3.0\n"
    )
    assert render_exposition((third, first, second)) == expected
    assert render_exposition((second, third, first)) == expected
    assert len(parse_exposition(expected)) == 3


def test_renderer_escapes_help_but_leaves_quotes_unescaped():
    text = render_exposition((_sample("metric", 0.0, help_text='A "quote", \\slash\nand line.'),))
    assert text.startswith('# HELP metric A "quote", \\\\slash\\nand line.\n')
    assert parse_exposition(text)[0].help_text == 'A "quote", \\slash\nand line.'


def test_renderer_omits_absent_measurements_without_headers_or_placeholder_values():
    assert render_exposition((_sample("absent", None),)) == ""
    present = _sample("present", 0.0)
    text = render_exposition((_sample("absent", None), present))
    assert "absent" not in text
    assert text == render_exposition((present,))
    assert parse_exposition(text)[0].value == 0


@pytest.mark.parametrize("value", [True, "1", object()])
def test_renderer_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="numeric"):
        render_exposition((_sample("metric", value),))


@pytest.mark.parametrize(
    "labels", [(("run_id", "secret"),), (("source_id", "a"), ("source_id", "b"))]
)
def test_renderer_rejects_unbounded_or_duplicate_labels(labels):
    with pytest.raises(ValueError, match="allowed and unique"):
        render_exposition((_sample("metric", 1, labels=labels),))


def test_renderer_rejects_duplicate_series_and_conflicting_metadata():
    sample = _sample("metric", 1)
    with pytest.raises(ValueError, match="duplicate metric"):
        render_exposition((sample, sample))
    changed = replace(
        sample,
        declaration=replace(sample.declaration, help_text="Changed."),
        labels=(("source_id", "different"),),
    )
    with pytest.raises(ValueError, match="share type and help"):
        render_exposition((sample, changed))
