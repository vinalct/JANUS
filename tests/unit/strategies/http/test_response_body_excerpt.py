"""A failed response carries its own explanation into the error that reports it.

Before this, every family's status error said only "status N for URL" — the status and the
URL are what happened, never why. The body holding the server's reason was discarded at the
raise site, which is also the text the dead-letter store persists, so the one artifact left
behind after a failed run explained nothing.
"""

from __future__ import annotations

import pytest

from janus.strategies.api.errors import ApiResponseError
from janus.strategies.catalog.errors import CatalogResponseError
from janus.strategies.files.download import _file_response_error
from janus.strategies.http import RESPONSE_BODY_EXCERPT_LIMIT, response_body_excerpt
from janus.strategies.http.transport import ApiResponse

from .conftest import make_request

#: The three families' status errors, as (factory, message prefix) pairs.
STATUS_ERROR_FACTORIES = [
    pytest.param(ApiResponseError, "API request failed with status", id="api"),
    pytest.param(CatalogResponseError, "Catalog request failed with status", id="catalog"),
    pytest.param(_file_response_error, "File request failed with status", id="file"),
]


def _response(body: bytes, *, status_code: int = 400) -> ApiResponse:
    return ApiResponse(
        request=make_request("https://example.invalid/records"),
        status_code=status_code,
        body=body,
    )


def test_an_error_document_is_returned_verbatim_when_it_fits() -> None:
    excerpt = response_body_excerpt(_response(b'{"message": "pagina invalida"}'))

    assert excerpt == '{"message": "pagina invalida"}'


def test_an_empty_body_yields_no_excerpt_rather_than_an_empty_one() -> None:
    assert response_body_excerpt(_response(b"")) is None
    assert response_body_excerpt(_response(b"   \n\t  ")) is None


def test_a_multiline_body_is_collapsed_onto_one_line() -> None:
    """Structured logs and dead-letter entries are line-oriented; a raw dump breaks both."""
    excerpt = response_body_excerpt(_response(b'{\n  "error": "bad",\n  "code": 12\n}'))

    assert excerpt == '{ "error": "bad", "code": 12 }'


def test_a_long_body_is_truncated_and_reports_its_true_size() -> None:
    body = b"x" * (RESPONSE_BODY_EXCERPT_LIMIT + 250)

    excerpt = response_body_excerpt(_response(body))

    assert excerpt is not None
    assert excerpt.startswith("x" * RESPONSE_BODY_EXCERPT_LIMIT)
    assert f"({len(body)} bytes total)" in excerpt
    assert len(excerpt) < len(body)


def test_an_undecodable_body_is_rendered_rather_than_raised() -> None:
    """A diagnostic helper that can throw would replace the failure it is describing."""
    excerpt = response_body_excerpt(_response(b"\xff\xfe not utf-8"))

    assert excerpt is not None
    assert "not utf-8" in excerpt


@pytest.mark.parametrize(("factory", "prefix"), STATUS_ERROR_FACTORIES)
def test_every_family_status_error_reports_the_body(factory, prefix: str) -> None:
    error = factory(_response(b'{"message": "Pagina invalida"}'))

    message = str(error)
    assert message.startswith(prefix)
    assert "https://example.invalid/records" in message
    assert '{"message": "Pagina invalida"}' in message


@pytest.mark.parametrize(("factory", "prefix"), STATUS_ERROR_FACTORIES)
def test_an_empty_body_leaves_the_message_shape_unchanged(factory, prefix: str) -> None:
    """No body means no trailing clause — not a dangling separator."""
    message = str(factory(_response(b"")))

    assert message.startswith(prefix)
    assert not message.endswith(":")
    assert not message.endswith(": ")


@pytest.mark.parametrize(
    ("error_cls", "prefix"),
    [
        pytest.param(ApiResponseError, "API request failed", id="api"),
        pytest.param(CatalogResponseError, "Catalog request failed", id="catalog"),
    ],
)
def test_the_excerpt_is_reachable_without_reparsing_the_message(error_cls, prefix: str) -> None:
    """Callers that want the body structured should not have to scrape ``str(exc)``."""
    error = error_cls(_response(b'{"message": "nope"}'))

    assert error.body_excerpt == '{"message": "nope"}'
    assert error_cls(_response(b"")).body_excerpt is None
