"""Characterization tests for file-strategy helpers in core and formats modules."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiRequest, ApiResponse
from janus.strategies.common import (
    _compare_checkpoint_values,
    _retry_delay_seconds,
)
from janus.strategies.files.formats import (
    _filename_from_content_disposition,
    _filename_from_url,
    _infer_format_name,
    _is_tarball_filename,
    _safe_path_segment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_plan(
    tmp_path: Path,
    *,
    retry_backoff_strategy: str = "fixed",
    retry_backoff_seconds: int = 2,
    retry_max_attempts: int = 3,
    rate_limit_backoff_seconds: int | None = None,
) -> ExecutionPlan:
    source_config = SourceConfig.from_mapping(
        {
            "source_id": "file_helper_test",
            "name": "file_helper_test",
            "owner": "janus",
            "enabled": True,
            "source_type": "file",
            "strategy": "file",
            "strategy_variant": "static_file",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "url": "https://example.invalid/data.csv",
                "method": "GET",
                "format": "csv",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "pagination": {"type": "none"},
                "rate_limit": {
                    "requests_per_minute": 10,
                    "concurrency": 1,
                    "backoff_seconds": rate_limit_backoff_seconds,
                },
            },
            "extraction": {
                "mode": "full_refresh",
                "retry": {
                    "max_attempts": retry_max_attempts,
                    "backoff_strategy": retry_backoff_strategy,
                    "backoff_seconds": retry_backoff_seconds,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "csv", "write_mode": "append"},
            "outputs": {
                "raw": {"path": "data/raw/example/file_helper_test", "format": "binary"},
                "bronze": {"path": "data/bronze/example/file_helper_test", "format": "iceberg"},
                "metadata": {"path": "data/metadata/example/file_helper_test", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / "file_helper_test.yaml",
    )
    run_context = RunContext.create(
        run_id="run-file-helper-test",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _make_response(retry_after: str | None = None) -> ApiResponse:
    headers: tuple[tuple[str, str], ...] = ()
    if retry_after is not None:
        headers = (("Retry-After", retry_after),)
    return ApiResponse(
        request=ApiRequest(method="GET", url="https://example.invalid/", timeout_seconds=30),
        status_code=429,
        body=b"",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# _retry_delay_seconds (file) — fixed
# ---------------------------------------------------------------------------


def test_file_retry_delay_fixed_returns_same_delay_for_all_attempts(tmp_path):
    plan = _build_plan(
        tmp_path, retry_backoff_strategy="fixed", retry_backoff_seconds=2, retry_max_attempts=5
    )

    delays = [_retry_delay_seconds(plan, attempt, None) for attempt in range(1, 5)]

    assert all(d == 2.0 for d in delays)


# ---------------------------------------------------------------------------
# _retry_delay_seconds (file) — exponential
# ---------------------------------------------------------------------------


def test_file_retry_delay_exponential_doubles_on_each_attempt(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="exponential",
        retry_backoff_seconds=2,
        retry_max_attempts=5,
    )

    assert _retry_delay_seconds(plan, 1, None) == 2.0
    assert _retry_delay_seconds(plan, 2, None) == 4.0
    assert _retry_delay_seconds(plan, 3, None) == 8.0


def test_file_retry_delay_exponential_capped_by_rate_limit_backoff(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="exponential",
        retry_backoff_seconds=2,
        retry_max_attempts=10,
        rate_limit_backoff_seconds=6,
    )

    assert _retry_delay_seconds(plan, 4, None) == 6.0


# ---------------------------------------------------------------------------
# _retry_delay_seconds (file) — Retry-After
# ---------------------------------------------------------------------------


def test_file_retry_delay_valid_retry_after_extends_delay(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="fixed",
        retry_backoff_seconds=1,
        rate_limit_backoff_seconds=60,
    )
    response = _make_response(retry_after="30")

    assert _retry_delay_seconds(plan, 1, response) == 30.0


def test_file_retry_delay_invalid_retry_after_is_ignored(tmp_path):
    plan = _build_plan(
        tmp_path,
        retry_backoff_strategy="fixed",
        retry_backoff_seconds=2,
        rate_limit_backoff_seconds=60,
    )
    response = _make_response(retry_after="not-a-number")

    assert _retry_delay_seconds(plan, 1, response) == 2.0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values (file) — datetime
# ---------------------------------------------------------------------------


def test_file_compare_checkpoint_earlier_datetime_is_less():
    assert _compare_checkpoint_values("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z") == -1


def test_file_compare_checkpoint_later_datetime_is_greater():
    assert _compare_checkpoint_values("2026-06-01T00:00:00Z", "2026-01-01T00:00:00Z") == 1


def test_file_compare_checkpoint_equal_datetimes_returns_zero():
    assert _compare_checkpoint_values("2026-04-10T12:00:00Z", "2026-04-10T12:00:00Z") == 0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values (file) — decimal
# ---------------------------------------------------------------------------


def test_file_compare_checkpoint_smaller_decimal_is_less():
    assert _compare_checkpoint_values("10", "20") == -1


def test_file_compare_checkpoint_larger_decimal_is_greater():
    assert _compare_checkpoint_values("100.5", "99.9") == 1


def test_file_compare_checkpoint_equal_decimals_return_zero():
    assert _compare_checkpoint_values("42", "42.0") == 0


# ---------------------------------------------------------------------------
# _compare_checkpoint_values (file) — text
# ---------------------------------------------------------------------------


def test_file_compare_checkpoint_text_uses_lexicographic_order():
    assert _compare_checkpoint_values("alpha", "beta") == -1
    assert _compare_checkpoint_values("beta", "alpha") == 1
    assert _compare_checkpoint_values("v1.0", "v1.0") == 0


# ---------------------------------------------------------------------------
# _infer_format_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, fallback, expected",
    [
        ("report.csv", "binary", "csv"),
        ("data.json", "binary", "json"),
        ("records.jsonl", "binary", "jsonl"),
        ("records.ndjson", "binary", "jsonl"),
        ("dump.parquet", "binary", "parquet"),
        ("readme.txt", "binary", "text"),
        ("tsv_file.tsv", "binary", "text"),
        ("archive.zip", "binary", "binary"),
        ("bundle.tar.gz", "binary", "binary"),
        ("bundle.tgz", "binary", "binary"),
        ("spreadsheet.xlsx", "binary", "binary"),
        ("spreadsheet.xls", "binary", "binary"),
    ],
)
def test_infer_format_name_known_extensions(filename: str, fallback: str, expected: str):
    assert _infer_format_name(filename, fallback=fallback) == expected


def test_infer_format_name_unknown_extension_uses_fallback_when_valid():
    assert _infer_format_name("data.xml", fallback="json") == "json"


def test_infer_format_name_unknown_extension_returns_binary_when_fallback_invalid():
    assert _infer_format_name("data.unknown", fallback="not_a_format") == "binary"


def test_infer_format_name_tar_gz_is_detected_as_binary_before_suffix_check():
    assert _infer_format_name("data.tar.gz", fallback="csv") == "binary"


def test_infer_format_name_tgz_is_detected_as_binary():
    assert _infer_format_name("data.tgz", fallback="csv") == "binary"


def test_infer_format_name_xls_is_binary():
    assert _infer_format_name("data.xls", fallback="binary") == "binary"


# ---------------------------------------------------------------------------
# _is_tarball_filename
# ---------------------------------------------------------------------------


def test_is_tarball_filename_tar_gz():
    assert _is_tarball_filename("archive.tar.gz") is True


def test_is_tarball_filename_tgz():
    assert _is_tarball_filename("bundle.tgz") is True


def test_is_tarball_filename_zip():
    assert _is_tarball_filename("file.zip") is False


def test_is_tarball_filename_case_insensitive():
    assert _is_tarball_filename("DATA.TAR.GZ") is True


# ---------------------------------------------------------------------------
# _filename_from_url
# ---------------------------------------------------------------------------


def test_filename_from_url_simple_path():
    assert _filename_from_url("https://example.gov.br/data/report.csv") == "report.csv"


def test_filename_from_url_encoded_chars():
    assert _filename_from_url("https://example.gov.br/files/my%20file.zip") == "my file.zip"


def test_filename_from_url_no_path_returns_fallback():
    assert _filename_from_url("https://example.gov.br/") == "download.bin"


def test_filename_from_url_strips_query_string():
    assert _filename_from_url("https://example.gov.br/data.csv?v=1") == "data.csv"


# ---------------------------------------------------------------------------
# _filename_from_content_disposition
# ---------------------------------------------------------------------------


def test_filename_from_content_disposition_plain():
    assert _filename_from_content_disposition('attachment; filename="report.csv"') == "report.csv"


def test_filename_from_content_disposition_encoded():
    assert (
        _filename_from_content_disposition("attachment; filename*=UTF-8''report%202026.csv")
        == "report 2026.csv"
    )


def test_filename_from_content_disposition_no_match():
    assert _filename_from_content_disposition("attachment") is None


def test_filename_from_content_disposition_empty_name():
    assert _filename_from_content_disposition('attachment; filename=""') is None


# ---------------------------------------------------------------------------
# _safe_path_segment
# ---------------------------------------------------------------------------


def test_safe_path_segment_clean_value():
    assert _safe_path_segment("2026-04-10") == "2026-04-10"


def test_safe_path_segment_replaces_unsafe_chars():
    assert _safe_path_segment("hello world!") == "hello-world"


def test_safe_path_segment_empty_becomes_current():
    assert _safe_path_segment("") == "current"


def test_safe_path_segment_only_unsafe_chars_becomes_current():
    assert _safe_path_segment("!!!") == "current"
