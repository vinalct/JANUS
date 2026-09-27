"""Two redaction gaps, both about values the *configuration* chose."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiResponse, ApiStrategy, ApiStrategyError
from janus.utils.logging import REDACTED_VALUE, build_structured_logger, redact_url
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

SECRET = "S3CRET-portal-key"
QUERY_PARAM = "chave"
HEADER_NAME = "chave-api-dados"
TOKEN_ENV_VAR = "JANUS_TEST_QUERY_TOKEN"


# ---------------------------------------------------------------------------
# Harness


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    payload: Any = None
    body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def rendered(self) -> bytes:
        if self.body is not None:
            return self.body
        return json.dumps(self.payload).encode("utf-8")


class FakeTransport:
    def __init__(self, responses: list[ResponseSpec | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request):
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No fake responses remain for this transport")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ApiResponse(
            request=request,
            status_code=response.status_code,
            body=response.rendered(),
            headers=tuple(sorted(response.headers.items())),
        )


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "runtime",
                "raw_dir": "runtime/raw",
                "bronze_dir": "runtime/bronze",
                "metadata_dir": "runtime/metadata",
            }
        },
        tmp_path,
    )


def _api_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    auth: dict[str, Any],
    dead_letter_max_items: int = 0,
    retry_max_attempts: int = 1,
) -> SourceConfig:
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "api",
            "strategy": "api",
            "strategy_variant": "page_number_api",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "base_url": "https://api.example.gov.br",
                "path": "/v1/records",
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": auth,
                "pagination": {
                    "type": "page_number",
                    "page_param": "pagina",
                    "size_param": "tamanhoPagina",
                    "page_size": 100,
                },
                "rate_limit": {
                    "requests_per_minute": None,
                    "concurrency": 1,
                    "backoff_seconds": 1,
                },
            },
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {
                    "max_attempts": retry_max_attempts,
                    "backoff_strategy": "fixed",
                    "backoff_seconds": 1,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "json", "write_mode": "append"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _plan(tmp_path: Path, source_config: SourceConfig) -> ExecutionPlan:
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=f"run-{source_config.source_id}",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        ),
    )


def _query_token_auth() -> dict[str, Any]:
    return {"type": "query_token", "env_var": TOKEN_ENV_VAR, "query_param": QUERY_PARAM}


def _strategy(tmp_path: Path, responses: list[ResponseSpec | Exception], *, logger=None):
    transport = FakeTransport(responses)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
        logger=logger,
    )
    return strategy, transport


@pytest.fixture
def token_env(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, SECRET)
    return SECRET


# ---------------------------------------------------------------------------
# FR-6 — redact_url learns the names the configuration chose


def test_redact_url_redacts_a_configured_query_parameter():
    """``chave`` is not in anybody's default list, which is the whole finding."""
    redacted = redact_url(
        f"https://api.example.gov.br/v1/a?{QUERY_PARAM}={SECRET}&pagina=1",
        extra_params=(QUERY_PARAM,),
    )

    assert SECRET not in redacted
    assert f"{QUERY_PARAM}={REDACTED_VALUE}" in redacted
    assert "pagina=1" in redacted


def test_extra_params_compare_case_insensitively():
    """A YAML that writes ``Chave`` and a server that echoes ``chave`` are the same secret."""
    redacted = redact_url(
        f"https://api.example.gov.br/v1/a?CHAVE={SECRET}", extra_params=("chave",)
    )

    assert SECRET not in redacted


def test_a_request_carries_the_names_needed_to_redact_its_own_url(token_env):
    """``redacted_url()`` is what makes the ~20 call sites impossible to get wrong."""
    from janus.models import AuthConfig
    from janus.strategies.http import ApiRequest, inject_auth

    request = inject_auth(
        ApiRequest(
            method="GET", url="https://api.example.gov.br/v1/records", timeout_seconds=30
        ),
        AuthConfig(type="query_token", env_var=TOKEN_ENV_VAR, query_param=QUERY_PARAM),
    )

    assert SECRET in request.full_url(), "the request must still send the credential"
    assert SECRET not in request.redacted_url()
    assert f"{QUERY_PARAM}={REDACTED_VALUE}" in request.redacted_url()


def test_redact_url_returns_a_url_with_nothing_sensitive_unchanged():
    """Green on arrival: a pin, not a red test.

    ``redact_url`` returns its input verbatim when nothing matched, which is why switching
    ~20 sites from ``full_url()`` to ``redacted_url()`` changes no byte of any existing
    fixture, golden or baseline (NFR-1). If this stops holding, AC-4 breaks silently.
    """
    url = "https://api.example.gov.br/v1/records?pagina=1&tamanhoPagina=100"

    assert redact_url(url) == url


# ---------------------------------------------------------------------------
# AC-5 — every site that renders a request URL


def test_the_structured_request_log_shows_no_configured_secret(tmp_path, token_env):
    """``api_request_started`` renders ``request_url``; the formatter's name-based net misses it."""
    stream = StringIO()
    logger = build_structured_logger("janus.tests.redaction.request", stream=stream)
    source_config = _api_source_config(
        tmp_path, source_id="redaction_log", auth=_query_token_auth()
    )
    strategy, _transport = _strategy(
        tmp_path, [ResponseSpec(200, {"records": []})], logger=logger
    )

    strategy.extract(_plan(tmp_path, source_config))

    rendered = stream.getvalue()
    assert SECRET not in rendered, "the configured credential reached a structured log line"
    assert REDACTED_VALUE in rendered


def test_the_raw_artifact_metadata_shows_no_configured_secret(tmp_path, token_env, monkeypatch):
    """``request_url`` in the raw write metadata reaches the run-metadata JSON verbatim."""
    captured: list[dict[str, str]] = []
    original = RawArtifactWriter._write_payload

    def _recording_write_payload(self, plan, relative_path, payload, **kwargs):
        captured.append(dict(kwargs.get("metadata") or {}))
        return original(self, plan, relative_path, payload, **kwargs)

    # Every ``write_*`` funnels through ``_write_payload``; this source's raw format is
    # ``json``, so patching ``write_bytes`` alone would capture nothing.
    monkeypatch.setattr(RawArtifactWriter, "_write_payload", _recording_write_payload)

    source_config = _api_source_config(
        tmp_path, source_id="redaction_raw_metadata", auth=_query_token_auth()
    )
    strategy, _transport = _strategy(tmp_path, [ResponseSpec(200, {"records": [{"id": 1}]})])

    strategy.extract(_plan(tmp_path, source_config))

    assert captured, "no raw artifact was written, so nothing was asserted"
    for metadata in captured:
        assert SECRET not in json.dumps(metadata), metadata
    assert any(REDACTED_VALUE in metadata.get("request_url", "") for metadata in captured)


def test_the_normalized_catalog_record_shows_no_configured_secret(token_env):
    """``catalog_request_url`` reaches **bronze**. A bronze column must never hold a credential.

    Redacting it is a data change only for a source that puts a secret in the URL — none
    today — and it is the right one: every other consumer of that column wants the endpoint,
    not the key.
    """
    from janus.models import AuthConfig, ExtractedArtifact
    from janus.strategies.api.pagination import PaginationState
    from janus.strategies.catalog.entities import _normalize_catalog_record
    from janus.strategies.http import ApiRequest, inject_auth
    from janus.strategies.http import ApiResponse as HttpApiResponse

    request = inject_auth(
        ApiRequest(
            method="GET", url="https://dados.gov.br/api/3/conjuntos", timeout_seconds=30
        ),
        AuthConfig(type="query_token", env_var=TOKEN_ENV_VAR, query_param=QUERY_PARAM),
    )
    response = HttpApiResponse(request=request, status_code=200, body=b"{}")

    record = _normalize_catalog_record(
        entity_type="dataset",
        record={"id": "abc"},
        collection_path="result",
        record_path="result[0]",
        request=request,
        response=response,
        pagination_state=PaginationState(request_index=1, page_number=1),
        raw_artifact=ExtractedArtifact(path="/data/raw/page-0001.json", format="json"),
        parent=None,
    )

    assert SECRET not in record["catalog_request_url"]
    assert REDACTED_VALUE in record["catalog_request_url"]


def test_the_dead_letter_record_shows_neither_the_url_nor_the_echoed_secret(tmp_path, token_env):
    """The dead letter is where a failed run's evidence is *persisted*, so it is the worst leak.

    The scripted body is what an API that validates credentials actually returns: the key,
    echoed back inside the error document. ``response_body_excerpt`` carries it into the
    error message verbatim today.
    """
    source_config = _api_source_config(
        tmp_path,
        source_id="redaction_dead_letter",
        auth=_query_token_auth(),
        dead_letter_max_items=1,
    )
    plan = _plan(tmp_path, source_config)
    strategy, _transport = _strategy(
        tmp_path,
        [ResponseSpec(500, body=json.dumps({"error": f"bad key {SECRET}"}).encode("utf-8"))],
    )

    with pytest.raises(ApiStrategyError):
        strategy.extract(plan)

    persisted = strategy.dead_letter_store.path(plan).read_text(encoding="utf-8")
    assert SECRET not in persisted, "the credential was persisted into the dead-letter store"
    entry = json.loads(persisted)["entries"][0]
    assert REDACTED_VALUE in entry["error_message"]
    assert REDACTED_VALUE in entry["metadata"]["request_url"], (
        "the dead-letter record's own request_url metadata is a second copy of the same leak"
    )


def test_a_basic_credential_is_scrubbed_in_both_its_forms(tmp_path, monkeypatch):
    """A body excerpt can echo ``user:pass`` or the base64 the header carried. Both are secrets."""
    import base64

    monkeypatch.setenv("JANUS_TEST_BASIC_USER", "portal-user")
    monkeypatch.setenv("JANUS_TEST_BASIC_PASS", "portal-pass")
    pair = "portal-user:portal-pass"
    encoded = base64.b64encode(pair.encode()).decode("ascii")

    source_config = _api_source_config(
        tmp_path,
        source_id="redaction_basic",
        auth={
            "type": "basic",
            "username_env_var": "JANUS_TEST_BASIC_USER",
            "password_env_var": "JANUS_TEST_BASIC_PASS",
        },
        dead_letter_max_items=1,
    )
    plan = _plan(tmp_path, source_config)
    strategy, _transport = _strategy(
        tmp_path,
        [ResponseSpec(500, body=f'{{"echo":"{pair}","header":"Basic {encoded}"}}'.encode())],
    )

    with pytest.raises(ApiStrategyError):
        strategy.extract(plan)

    persisted = strategy.dead_letter_store.path(plan).read_text(encoding="utf-8")
    assert pair not in persisted
    assert encoded not in persisted


# ---------------------------------------------------------------------------
# FR-6 — the scrubber itself


def _scrubber():
    from janus.strategies.http.scrubber import SecretScrubber

    return SecretScrubber()


def test_the_scrubber_replaces_the_longest_match_first():
    """``Bearer abc`` and ``abc`` overlap; replacing the short one first strands the prefix."""
    scrubber = _scrubber()
    scrubber.register("abcdefgh12345678", "Bearer abcdefgh12345678")

    scrubbed = scrubber.scrub_bytes(b'{"seen":"Bearer abcdefgh12345678"}')

    assert b"abcdefgh12345678" not in scrubbed
    assert b"***REDACTED***" in scrubbed


def test_the_scrubber_refuses_to_register_a_value_too_short_to_be_a_secret():
    """A two-character "secret" would mangle every diagnostic body it touched."""
    scrubber = _scrubber()
    scrubber.register("", "   ", "1234567")

    assert len(scrubber) == 0
    assert scrubber.scrub_bytes(b"1234567 is not a key") == b"1234567 is not a key"


def test_the_scrubber_never_renders_a_value():
    """It holds credentials; its ``repr`` is the one place they would escape by accident."""
    import pickle

    scrubber = _scrubber()
    scrubber.register("abcdefgh12345678")

    assert "abcdefgh12345678" not in repr(scrubber)
    assert "abcdefgh12345678" not in str(scrubber)
    assert len(scrubber) == 1
    with pytest.raises(Exception):  # noqa: B017 — any refusal; never a serialized secret
        pickle.dumps(scrubber)


def test_a_body_with_no_secret_comes_back_unchanged():
    """The scrub runs on every raise path; the no-op case must cost and change nothing."""
    scrubber = _scrubber()
    scrubber.register("abcdefgh12345678")
    body = b'{"error":"page not found"}'

    assert scrubber.scrub_bytes(body) == body


def test_a_request_without_auth_carries_no_scrubber(tmp_path):
    """``auth.type: none`` resolves no values, so there is nothing to hold and nothing to scrub.

    This is also what keeps the characterization retry suite byte-identical: it runs with no
    auth, so the scrub branch is never taken there.
    """
    from janus.models import AuthConfig
    from janus.strategies.http import ApiRequest, inject_auth

    del tmp_path
    request = ApiRequest(
        method="GET", url="https://api.example.gov.br/v1/a", timeout_seconds=30
    )

    assert inject_auth(request, AuthConfig(type="none")) is request
    assert request.scrubber is None
