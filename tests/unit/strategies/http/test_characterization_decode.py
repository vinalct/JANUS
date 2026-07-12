"""Characterization: the api and catalog `_decode_payload` copies.
"""

from __future__ import annotations

import json

import pytest

from janus.strategies.api.http import ApiResponse

from .conftest import FAMILIES, build_plan, build_strategy, make_request

REQUEST_URL = "https://example.invalid/records"


def _decode(family_name, tmp_path, *, access_format, body, source_id):
    family = FAMILIES[family_name]
    plan = build_plan(family_name, tmp_path, source_id=source_id, access_format=access_format)
    strategy = build_strategy(family)
    response = ApiResponse(request=make_request(REQUEST_URL), status_code=200, body=body)
    return family.decode_payload(strategy, plan, response)


# ---------------------------------------------------------------------------
# api: binary | text | json | jsonl
# ---------------------------------------------------------------------------


def test_api_binary_format_returns_body_bytes_untouched(tmp_path):
    body = b"\x00\x01\xffPDF"
    decoded = _decode("api", tmp_path, access_format="binary", body=body, source_id="api_binary")
    assert decoded == body


def test_api_text_format_returns_decoded_text(tmp_path):
    decoded = _decode(
        "api",
        tmp_path,
        access_format="text",
        body="olá, café".encode(),
        source_id="api_text",
    )
    assert decoded == "olá, café"


def test_api_json_format_returns_parsed_object(tmp_path):
    decoded = _decode(
        "api",
        tmp_path,
        access_format="json",
        body=b'{"a": [1, 2], "b": null}',
        source_id="api_json",
    )
    assert decoded == {"a": [1, 2], "b": None}


def test_api_json_format_with_empty_body_returns_none(tmp_path):
    decoded = _decode(
        "api", tmp_path, access_format="json", body=b"", source_id="api_json_empty"
    )
    assert decoded is None


def test_api_jsonl_format_returns_list_of_parsed_lines(tmp_path):
    decoded = _decode(
        "api",
        tmp_path,
        access_format="jsonl",
        body=b'{"a": 1}\n{"b": 2}\n',
        source_id="api_jsonl",
    )
    assert decoded == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize("body", [b"", b"   \n\t  \n"], ids=["empty", "whitespace"])
def test_api_jsonl_format_with_empty_or_whitespace_body_returns_empty_list(tmp_path, body):
    decoded = _decode(
        "api", tmp_path, access_format="jsonl", body=body, source_id="api_jsonl_empty"
    )
    assert decoded == []


def test_api_jsonl_format_skips_blank_lines(tmp_path):
    decoded = _decode(
        "api",
        tmp_path,
        access_format="jsonl",
        body=b'{"a": 1}\n\n   \n {"b": 2} \n',
        source_id="api_jsonl_blanks",
    )
    assert decoded == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize(
    ("access_format", "body", "cause_type"),
    [
        ("json", b"{not valid", json.JSONDecodeError),
        ("jsonl", b'{"a": 1}\nnot json\n', json.JSONDecodeError),
        ("text", b"\xff\xfe\xfa", UnicodeDecodeError),
    ],
    ids=["malformed-json", "malformed-jsonl", "undecodable-text"],
)
def test_api_decode_failure_raises_payload_error_with_redacted_url(
    tmp_path, access_format, body, cause_type
):
    family = FAMILIES["api"]
    with pytest.raises(family.payload_error) as excinfo:
        _decode(
            "api",
            tmp_path,
            access_format=access_format,
            body=body,
            source_id=f"api_bad_{access_format}",
        )

    assert str(excinfo.value) == f"Failed to decode {access_format} payload from {REQUEST_URL}"
    assert isinstance(excinfo.value.__cause__, cause_type)


def test_api_unsupported_format_raises_payload_error(tmp_path):
    family = FAMILIES["api"]
    with pytest.raises(family.payload_error) as excinfo:
        _decode("api", tmp_path, access_format="csv", body=b"a,b\n1,2\n", source_id="api_csv")

    assert str(excinfo.value) == "Unsupported API payload format: csv"


# ---------------------------------------------------------------------------
# catalog: json only
# ---------------------------------------------------------------------------


def test_catalog_json_format_returns_parsed_object(tmp_path):
    decoded = _decode(
        "catalog",
        tmp_path,
        access_format="json",
        body=b'{"result": {"count": 3}}',
        source_id="catalog_json",
    )
    assert decoded == {"result": {"count": 3}}


def test_catalog_json_format_with_empty_body_returns_none(tmp_path):
    decoded = _decode(
        "catalog", tmp_path, access_format="json", body=b"", source_id="catalog_json_empty"
    )
    assert decoded is None


@pytest.mark.parametrize("access_format", ["text", "jsonl", "binary", "csv"])
def test_catalog_rejects_every_non_json_format_even_text(tmp_path, access_format):
    family = FAMILIES["catalog"]
    with pytest.raises(family.payload_error) as excinfo:
        _decode(
            "catalog",
            tmp_path,
            access_format=access_format,
            body=b'{"still": "json"}',
            source_id=f"catalog_{access_format}",
        )

    assert str(excinfo.value) == f"Unsupported catalog payload format: {access_format}"


def test_catalog_decode_failure_raises_payload_error_with_catalog_message(tmp_path):
    family = FAMILIES["catalog"]
    with pytest.raises(family.payload_error) as excinfo:
        _decode(
            "catalog",
            tmp_path,
            access_format="json",
            body=b"{not valid",
            source_id="catalog_bad_json",
        )

    assert str(excinfo.value) == f"Failed to decode catalog payload from {REQUEST_URL}"
    assert excinfo.value.__cause__ is not None
