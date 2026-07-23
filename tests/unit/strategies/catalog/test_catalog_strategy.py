from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from janus.checkpoints import CheckpointStore
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.planner import StrategyCatalog
from janus.strategies.api import ApiResponse
from janus.strategies.catalog import CatalogStrategy
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    payload: Any
    headers: dict[str, str] = field(default_factory=dict)


class FakeTransport:
    def __init__(self, responses: list[ResponseSpec | Exception]) -> None:
        self._responses = list(responses)
        self.requests = []
        self.opened = False
        self.closed = False
        # Optional shared trail, so session lifecycle and requests can be ordered together.
        self.events: list[str] | None = None

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request):
        self.requests.append(request)
        if self.events is not None:
            self.events.append("request")
        if not self._responses:
            raise AssertionError("No fake responses remain for this transport")

        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response

        return ApiResponse(
            request=request,
            status_code=response.status_code,
            body=json.dumps(response.payload).encode("utf-8"),
            headers=tuple(sorted(response.headers.items())),
        )


def test_default_strategy_catalog_uses_catalog_strategy_for_catalog_variants(tmp_path):
    source_config = _build_source_config(tmp_path, source_id="catalog_source")

    binding = StrategyCatalog.with_defaults().resolve(source_config)

    assert isinstance(binding.strategy, CatalogStrategy)
    assert binding.dispatch_path == "catalog.metadata_catalog"


def test_catalog_strategy_logs_page_progress(tmp_path):
    stream = StringIO()
    logger = build_structured_logger("janus.tests.catalog.progress", stream=stream)
    plan = _build_plan(tmp_path, source_id="logged_catalog_source", page_size=2)
    strategy, _transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {"id": "dataset-1", "title": "Dataset 1"},
                            {"id": "dataset-2", "title": "Dataset 2"},
                        ]
                    }
                },
            ),
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {"id": "dataset-3", "title": "Dataset 3"},
                        ]
                    }
                },
            ),
        ],
        logger=logger,
    )

    strategy.extract(plan)

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    events = [payload["event"] for payload in payloads]
    finished_pages = [
        payload for payload in payloads if payload["event"] == "catalog_request_finished"
    ]

    assert events[0] == "catalog_extraction_started"
    assert events[-1] == "catalog_extraction_finished"
    assert [page["fields"]["page_number"] for page in finished_pages] == [1, 2]
    assert finished_pages[0]["fields"]["records_extracted"] == 2
    assert finished_pages[0]["fields"]["entities_extracted"] == 2
    assert finished_pages[0]["fields"]["has_next_page"] is True
    assert finished_pages[0]["fields"]["next_page_number"] == 2
    assert finished_pages[1]["fields"]["records_extracted"] == 1
    assert finished_pages[1]["fields"]["entities_extracted"] == 1
    assert finished_pages[1]["fields"]["has_next_page"] is False
    assert payloads[-1]["fields"]["records_extracted"] == 3
    assert payloads[-1]["fields"]["request_count"] == 2


def test_catalog_strategy_page_number_extracts_raw_pages_and_normalized_entities(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="catalog_metadata",
        checkpoint_field="metadata_modified",
        checkpoint_strategy="max_value",
        page_size=2,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {
                                "id": "dataset-1",
                                "title": "Dataset 1",
                                "metadata_modified": "2026-04-10T01:00:00Z",
                                "organization": {
                                    "id": "org-1",
                                    "title": "Org 1",
                                },
                                "groups": [{"id": "grp-1", "title": "Group 1"}],
                                "resources": [
                                    {
                                        "id": "res-1",
                                        "format": "CSV",
                                        "url": "https://example.invalid/files/res-1.csv",
                                        "metadata_modified": "2026-04-10T01:30:00Z",
                                    }
                                ],
                            },
                            {
                                "id": "dataset-2",
                                "title": "Dataset 2",
                                "metadata_modified": "2026-04-10T03:00:00Z",
                                "organization": {
                                    "id": "org-1",
                                    "title": "Org 1",
                                },
                            },
                        ]
                    }
                },
            ),
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {
                                "id": "dataset-3",
                                "title": "Dataset 3",
                                "metadata_modified": "2026-04-10T05:00:00Z",
                                "organization": {
                                    "id": "org-1",
                                    "title": "Org 1",
                                },
                                "resources": [
                                    {
                                        "id": "res-2",
                                        "format": "JSON",
                                        "url": "https://example.invalid/files/res-2.json",
                                        "metadata_modified": "2026-04-10T05:30:00Z",
                                    }
                                ],
                            }
                        ]
                    }
                },
            ),
        ],
    )

    result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, result)
    metadata = result.metadata_as_dict()

    assert transport.opened is True
    assert transport.closed is True
    assert len(transport.requests) == 2

    first_query = parse_qs(urlsplit(transport.requests[0].full_url()).query)
    second_query = parse_qs(urlsplit(transport.requests[1].full_url()).query)
    assert first_query["page"] == ["1"]
    assert first_query["page_size"] == ["2"]
    assert second_query["page"] == ["2"]
    assert second_query["page_size"] == ["2"]

    assert result.records_extracted == 7
    assert result.checkpoint_value == "2026-04-10T05:30:00Z"
    assert metadata["raw_page_count"] == "2"
    assert metadata["organizations_extracted"] == "1"
    assert metadata["groups_extracted"] == "1"
    assert metadata["datasets_extracted"] == "3"
    assert metadata["resources_extracted"] == "2"
    assert metadata["normalized_artifact_count"] == "4"

    raw_page_one = Path(result.artifacts[0].path)
    raw_page_two = Path(result.artifacts[1].path)
    assert raw_page_one == _raw_run_dir(tmp_path, "catalog_metadata") / "pages" / "page-0001.json"
    assert raw_page_two == _raw_run_dir(tmp_path, "catalog_metadata") / "pages" / "page-0002.json"

    assert [Path(artifact.path).name for artifact in handoff.artifacts] == [
        "organizations.jsonl",
        "groups.jsonl",
        "datasets.jsonl",
        "resources.jsonl",
    ]

    organization_records = _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_metadata") / "normalized" / "organizations.jsonl"
    )
    resource_records = _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_metadata") / "normalized" / "resources.jsonl"
    )
    dataset_records = _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_metadata") / "normalized" / "datasets.jsonl"
    )

    assert len(organization_records) == 1
    assert organization_records[0]["entity_id"] == "org-1"
    assert organization_records[0]["entity_type"] == "organization"

    assert len(resource_records) == 2
    assert resource_records[0]["parent_entity_type"] == "dataset"
    assert resource_records[0]["parent_entity_id"] == "dataset-1"
    assert resource_records[0]["catalog_raw_artifact_path"] == str(raw_page_one)
    assert resource_records[1]["parent_entity_id"] == "dataset-3"

    assert [record["entity_id"] for record in dataset_records] == [
        "dataset-1",
        "dataset-2",
        "dataset-3",
    ]

    emitted_metadata = strategy.emit_metadata(plan, result)
    assert emitted_metadata["artifact_count"] == 6
    assert emitted_metadata["strategy_variant"] == "metadata_catalog"
    assert emitted_metadata["input_format"] == "jsonl"


def test_catalog_strategy_incremental_uses_checkpoint_param_and_filters_old_entities(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="catalog_incremental",
        extraction_mode="incremental",
        checkpoint_field="metadata_modified",
        checkpoint_strategy="max_value",
        lookback_days=1,
        page_size=50,
    )
    CheckpointStore().save(plan, "2026-04-10T06:00:00Z")
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {
                                "id": "dataset-old",
                                "title": "Old Dataset",
                                "metadata_modified": "2026-04-10T05:00:00Z",
                            },
                            {
                                "id": "dataset-new",
                                "title": "New Dataset",
                                "metadata_modified": "2026-04-10T08:00:00Z",
                                "resources": [
                                    {
                                        "id": "resource-new",
                                        "format": "CSV",
                                        "url": "https://example.invalid/files/resource-new.csv",
                                        "metadata_modified": "2026-04-10T09:00:00Z",
                                    }
                                ],
                            },
                        ]
                    }
                },
            )
        ],
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    query = parse_qs(urlsplit(transport.requests[0].full_url()).query)
    assert query["metadata_modified"] == ["2026-04-09T06:00:00Z"]
    assert result.records_extracted == 2
    assert result.checkpoint_value == "2026-04-10T09:00:00Z"
    assert metadata["checkpoint_loaded"] == "true"
    assert metadata["datasets_extracted"] == "1"
    assert metadata["resources_extracted"] == "1"

    dataset_records = _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_incremental") / "normalized" / "datasets.jsonl"
    )
    assert [record["entity_id"] for record in dataset_records] == ["dataset-new"]


def test_catalog_strategy_resource_catalog_uses_resource_root_results_for_handoff(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="catalog_resources",
        variant="resource_catalog",
        pagination_type="none",
    )
    strategy, _ = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "data": {
                        "results": [
                            {
                                "id": "resource-1",
                                "format": "CSV",
                                "url": "https://example.invalid/files/resource-1.csv",
                            },
                            {
                                "id": "resource-2",
                                "format": "JSON",
                                "url": "https://example.invalid/files/resource-2.json",
                            },
                        ]
                    }
                },
            )
        ],
    )

    result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, result)

    assert result.records_extracted == 2
    assert [Path(artifact.path).name for artifact in handoff.artifacts] == [
        "resources.jsonl",
    ]

    resource_records = _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_resources") / "normalized" / "resources.jsonl"
    )
    assert [record["entity_id"] for record in resource_records] == [
        "resource-1",
        "resource-2",
    ]
    assert all(record["entity_type"] == "resource" for record in resource_records)


def test_catalog_strategy_retries_on_malformed_payload_then_succeeds(tmp_path):
    plan = _build_plan(tmp_path, source_id="catalog_retry_payload", page_size=10)
    sleeps = []
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, None, headers={}),  # triggers CatalogPayloadError (invalid JSON)
            ResponseSpec(200, {"result": {"results": [{"id": "ds-1", "title": "DS 1"}]}}),
        ],
        sleeper=sleeps.append,
    )
    # Patch FakeTransport so the first response body is invalid JSON
    bad_body = b"not-json"
    original_send = transport.send

    call_count = [0]

    def patched_send(request):
        call_count[0] += 1
        if call_count[0] == 1:
            original_send.__func__.__get__(transport, type(transport)).__call__(request)
            from janus.strategies.api import ApiResponse

            return ApiResponse(
                request=request,
                status_code=200,
                body=bad_body,
                headers=(),
            )
        return original_send(request)

    transport.send = patched_send

    result = strategy.extract(plan)

    assert result.records_extracted == 1
    assert any(s == 1.0 for s in sleeps)  # retry backoff sleep fired


def test_catalog_strategy_retries_on_malformed_payload_exhausts_attempts(tmp_path):
    plan = _build_plan(tmp_path, source_id="catalog_retry_payload_fail", page_size=10)
    strategy, transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {}) for _ in range(3)],
    )
    bad_body = b"not-json"
    original_send = transport.send

    def patched_send(request):
        original_send(request)
        from janus.strategies.api import ApiResponse

        return ApiResponse(request=request, status_code=200, body=bad_body, headers=())

    transport.send = patched_send

    import pytest

    from janus.strategies.catalog import CatalogPayloadError

    with pytest.raises(CatalogPayloadError):
        strategy.extract(plan)


def test_catalog_strategy_checkpoint_as_path_param(tmp_path):
    from janus.checkpoints import CheckpointStore

    source_config = _build_source_config_with_url(
        tmp_path,
        source_id="catalog_path_param",
        url="https://example.invalid/catalog/{metadata_modified}/datasets",
        extraction_mode="incremental",
        checkpoint_field="metadata_modified",
        checkpoint_strategy="max_value",
    )
    from datetime import UTC, datetime

    from janus.models import RunContext

    run_context = RunContext.create(
        run_id="run-path-param",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    from janus.models import ExecutionPlan

    plan = ExecutionPlan.from_source_config(source_config, run_context)
    CheckpointStore().save(plan, "2026-04-10T06:00:00Z")

    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {"id": "ds-1", "metadata_modified": "2026-04-10T08:00:00Z"}
                        ]
                    }
                },
            )
        ],
    )

    result = strategy.extract(plan)

    sent_url = transport.requests[0].full_url()
    assert "2026-04-09T06:00:00Z" in sent_url  # lookback applied, substituted as path param
    assert "metadata_modified" not in sent_url.split("?", 1)[-1]  # NOT a query param
    assert result.records_extracted == 1


def test_catalog_strategy_resume_skips_completed_pages_and_rediscovers_artifacts(tmp_path):
    from datetime import UTC, datetime

    from janus.checkpoints import ExtractionProgressStore
    from janus.models import ExecutionPlan, RunContext

    source_id = "catalog_resume"
    plan_normal = _build_plan(tmp_path, source_id=source_id, page_size=10)

    # Write page-0001.json as if a prior run completed page 1
    pages_dir = tmp_path / "runtime" / "raw" / "example" / source_id / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page1_path = pages_dir / "page-0001.json"
    import json as _json

    page1_content = _json.dumps({"result": {"results": [{"id": "ds-1", "title": "DS 1"}]}})
    page1_path.write_text(page1_content, encoding="utf-8")

    # Save progress reflecting page 1 done
    ExtractionProgressStore().save(plan_normal, page_number=1, artifact_count=1)

    # Resume run context
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        attributes={"resume": "true"},
    )
    plan_resume = ExecutionPlan.from_source_config(plan_normal.source_config, run_context)

    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {"result": {"results": [{"id": "ds-2", "title": "DS 2"}]}},
            ),
        ],
    )

    result = strategy.extract(plan_resume)

    # Only page 2 was fetched over the network
    assert len(transport.requests) == 1
    sent_url = transport.requests[0].full_url()
    assert "page=2" in sent_url

    # Both artifacts are present (rediscovered page 1 + fetched page 2)
    raw_artifact_paths = [
        Path(a.path).name for a in result.artifacts if Path(a.path).suffix == ".json"
        and Path(a.path).parent.name == "pages"
    ]
    assert "page-0001.json" in raw_artifact_paths
    assert "page-0002.json" in raw_artifact_paths


def test_catalog_strategy_resume_clears_progress_on_success(tmp_path):
    source_id = "catalog_resume_clear"
    plan = _build_plan(tmp_path, source_id=source_id, page_size=10)

    strategy, _ = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"result": {"results": [{"id": "ds-1", "title": "DS 1"}]}})],
    )
    strategy.progress_store.save(plan, page_number=1, artifact_count=1)

    strategy.extract(plan)

    assert strategy.progress_store.load(plan) is None


def test_catalog_strategy_scopes_the_session_to_the_request_input_lookup(
    tmp_path,
    spy_spark_provider,
):
    events: list[str] = []
    plan = _build_plan(
        tmp_path,
        source_id="scoped_catalog_source",
        pagination_type="none",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "orgaos",
            "columns": {"org_id": "id"},
        },
        parameter_bindings={"id": {"from": "request_input.org_id"}},
    )
    provider = spy_spark_provider(
        {"bronze_test.orgaos": [{"id": "alpha"}, {"id": "beta"}]},
        events,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"result": {"results": [{"id": "ds-1", "title": "DS 1"}]}}),
            ResponseSpec(200, {"result": {"results": [{"id": "ds-2", "title": "DS 2"}]}}),
        ],
    )
    transport.events = events

    result = strategy.extract(plan, spark=provider)

    queries = [parse_qs(urlsplit(request.full_url()).query) for request in transport.requests]
    assert [query["id"] for query in queries] == [["alpha"], ["beta"]]
    assert result.records_extracted == 2
    assert events == ["spark_start", "spark_stop", "request", "request"]
    assert provider.start_count == 1
    assert provider.stop_count == 1


def _build_source_config_with_url(
    tmp_path,
    *,
    source_id: str,
    url: str,
    extraction_mode: str = "incremental",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    lookback_days: int | None = 1,
    dead_letter_max_items: int = 0,
) -> SourceConfig:
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "catalog",
            "strategy": "catalog",
            "strategy_variant": "metadata_catalog",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "url": url,
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "pagination": {"type": "none"},
                "rate_limit": {
                    "requests_per_minute": 60,
                    "concurrency": 1,
                    "backoff_seconds": 1,
                },
            },
            "extraction": {
                "mode": extraction_mode,
                "checkpoint_field": checkpoint_field,
                "checkpoint_strategy": checkpoint_strategy,
                "lookback_days": lookback_days,
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {
                    "max_attempts": 3,
                    "backoff_strategy": "fixed",
                    "backoff_seconds": 1,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {
                "input_format": "jsonl",
                "write_mode": "append",
            },
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _build_strategy(
    tmp_path: Path,
    responses: list[ResponseSpec | Exception],
    *,
    sleeper=None,
    logger=None,
):
    transport = FakeTransport(responses)
    strategy = CatalogStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=sleeper or (lambda seconds: None),
        clock=lambda: 0.0,
        logger=logger,
    )
    return strategy, transport


def _build_plan(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "metadata_catalog",
    pagination_type: str = "page_number",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    lookback_days: int | None = None,
    page_size: int = 100,
    dead_letter_max_items: int = 0,
    request_inputs: dict[str, Any] | None = None,
    parameter_bindings: dict[str, Any] | None = None,
) -> ExecutionPlan:
    source_config = _build_source_config(
        tmp_path,
        source_id=source_id,
        variant=variant,
        pagination_type=pagination_type,
        extraction_mode=extraction_mode,
        checkpoint_field=checkpoint_field,
        checkpoint_strategy=checkpoint_strategy,
        lookback_days=lookback_days,
        page_size=page_size,
        dead_letter_max_items=dead_letter_max_items,
        request_inputs=request_inputs,
        parameter_bindings=parameter_bindings,
    )
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _build_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "metadata_catalog",
    pagination_type: str = "page_number",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    lookback_days: int | None = None,
    page_size: int = 100,
    dead_letter_max_items: int = 0,
    request_inputs: dict[str, Any] | None = None,
    parameter_bindings: dict[str, Any] | None = None,
) -> SourceConfig:
    access: dict[str, Any] = {
        "base_url": "https://example.invalid",
        "path": "/catalog",
        "method": "GET",
        "format": "json",
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": _pagination_block(pagination_type, page_size),
        "rate_limit": {
            "requests_per_minute": 10,
            "concurrency": 1,
            "backoff_seconds": 5,
        },
    }
    if request_inputs is not None:
        access["request_inputs"] = request_inputs
    if parameter_bindings is not None:
        access["parameter_bindings"] = parameter_bindings
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "catalog",
            "strategy": "catalog",
            "strategy_variant": variant,
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": access,
            "extraction": {
                "mode": extraction_mode,
                "checkpoint_field": checkpoint_field,
                "checkpoint_strategy": checkpoint_strategy,
                "lookback_days": lookback_days,
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {
                    "max_attempts": 3,
                    "backoff_strategy": "fixed",
                    "backoff_seconds": 1,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {
                "input_format": "jsonl",
                "write_mode": "append",
            },
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _pagination_block(pagination_type: str, page_size: int) -> dict[str, Any]:
    if pagination_type == "page_number":
        return {
            "type": "page_number",
            "page_param": "page",
            "size_param": "page_size",
            "page_size": page_size,
        }
    if pagination_type == "offset":
        return {
            "type": "offset",
            "offset_param": "offset",
            "limit_param": "limit",
            "page_size": page_size,
        }
    if pagination_type == "cursor":
        return {
            "type": "cursor",
            "cursor_param": "cursor",
        }
    return {"type": "none"}


def _raw_run_dir(tmp_path: Path, source_id: str, *, run_id: str | None = None) -> Path:
    resolved_run_id = run_id or f"run-{source_id}"
    return (
        tmp_path
        / "runtime"
        / "raw"
        / "example"
        / source_id
        / "runs"
        / "ingestion_date=2026-04-10"
        / f"run_id={resolved_run_id}"
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# request_inputs + parameter_bindings tests
# ---------------------------------------------------------------------------


def _build_source_config_with_request_inputs(
    tmp_path: Path,
    *,
    source_id: str,
    path: str = "/catalog/{id}",
    request_inputs: dict,
    parameter_bindings: dict,
    pagination_type: str = "none",
    page_size: int = 100,
    dead_letter_max_items: int = 0,
) -> SourceConfig:
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "catalog",
            "strategy": "catalog",
            "strategy_variant": "metadata_catalog",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "base_url": "https://example.invalid",
                "path": path,
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "pagination": _pagination_block(pagination_type, page_size),
                "rate_limit": {
                    "requests_per_minute": 60,
                    "concurrency": 1,
                    "backoff_seconds": 1,
                },
                "request_inputs": request_inputs,
                "parameter_bindings": parameter_bindings,
            },
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {
                    "max_attempts": 1,
                    "backoff_strategy": "fixed",
                    "backoff_seconds": 1,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "jsonl", "write_mode": "overwrite"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def test_catalog_strategy_request_inputs_none_behaves_as_before(tmp_path):
    """Explicit type=none is equivalent to no request_inputs — single extraction loop."""
    source_config = _build_source_config(tmp_path, source_id="catalog_none_inputs")
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-none",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {"results": [{"id": "ds-1", "title": "Dataset 1"}]},
            ),
        ],
    )

    result = strategy.extract(plan)

    assert result.records_extracted == 1
    assert len(transport.requests) == 1


def test_catalog_strategy_iceberg_rows_request_inputs_iterates_over_inputs(tmp_path):
    """Each iceberg_rows context produces one request with the bound path param."""
    source_config = _build_source_config_with_request_inputs(
        tmp_path,
        source_id="catalog_iceberg_inputs",
        path="/catalog/{entity_id}",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze__test",
            "table_name": "entities",
            "columns": {"entity_id": "id"},
            "distinct": False,
        },
        parameter_bindings={
            "entity_id": {"from": "request_input.entity_id"},
        },
    )
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-iceberg",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )

    # Simulate two iceberg_rows contexts — same as load_request_inputs would return
    from unittest.mock import patch


    fake_inputs = ({"entity_id": "alpha"}, {"entity_id": "beta"})

    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"results": [{"id": "alpha", "title": "Alpha"}]}),
            ResponseSpec(200, {"results": [{"id": "beta", "title": "Beta"}]}),
        ],
    )

    with patch(
        "janus.strategies.catalog.core.load_request_inputs",
        return_value=fake_inputs,
    ):
        result = strategy.extract(plan)

    assert len(transport.requests) == 2
    assert "/catalog/alpha" in transport.requests[0].url
    assert "/catalog/beta" in transport.requests[1].url
    assert result.records_extracted == 2


def test_catalog_strategy_request_inputs_query_param_binding(tmp_path):
    """Parameter bindings injected as query params are sent on each request."""
    source_config = _build_source_config_with_request_inputs(
        tmp_path,
        source_id="catalog_query_binding",
        path="/catalog",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze__test",
            "table_name": "orgs",
            "columns": {"org_id": "id"},
        },
        parameter_bindings={
            "organization": {"from": "request_input.org_id"},
        },
    )
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-query",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )

    fake_inputs = ({"org_id": "org-1"}, {"org_id": "org-2"})

    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"results": [{"id": "ds-1", "title": "Dataset from org-1"}]}),
            ResponseSpec(200, {"results": [{"id": "ds-2", "title": "Dataset from org-2"}]}),
        ],
    )

    from unittest.mock import patch

    with patch(
        "janus.strategies.catalog.core.load_request_inputs",
        return_value=fake_inputs,
    ):
        strategy.extract(plan)

    assert len(transport.requests) == 2
    assert transport.requests[0].params_as_dict().get("organization") == "org-1"
    assert transport.requests[1].params_as_dict().get("organization") == "org-2"


def test_catalog_strategy_dead_letters_failed_request_input_without_partial_entities(
    tmp_path,
):
    from unittest.mock import patch

    source_config = _build_source_config_with_request_inputs(
        tmp_path,
        source_id="catalog_dead_letter",
        path="/catalog/{entity_id}",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "entities",
            "columns": {"entity_id": "id"},
        },
        parameter_bindings={
            "entity_id": {"from": "request_input.entity_id"},
        },
        pagination_type="page_number",
        page_size=2,
        dead_letter_max_items=1,
    )
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-catalog_dead_letter",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )
    fake_inputs = ({"entity_id": "alpha"}, {"entity_id": "beta"})
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "result": {
                        "results": [
                            {"id": "dataset-alpha-1", "title": "Alpha 1"},
                            {"id": "dataset-alpha-2", "title": "Alpha 2"},
                        ]
                    }
                },
            ),
            ResponseSpec(
                200,
                {"result": {"results": [{"id": "dataset-beta", "title": "Beta"}]}} ,
            ),
        ],
    )
    original_send = transport.send

    def patched_send(request):
        query = parse_qs(urlsplit(request.full_url()).query)
        if request.url.endswith("/catalog/alpha") and query.get("page") == ["2"]:
            transport.requests.append(request)
            return ApiResponse(request=request, status_code=200, body=b"not-json", headers=())
        return original_send(request)

    transport.send = patched_send

    with patch("janus.strategies.catalog.core.load_request_inputs", return_value=fake_inputs):
        result = strategy.extract(plan)

    metadata = result.metadata_as_dict()
    raw_artifact_paths = [
        Path(artifact.path)
        for artifact in result.artifacts
        if Path(artifact.path).suffix == ".json"
    ]
    dataset_records = _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_dead_letter") / "normalized" / "datasets.jsonl"
    )

    assert len(transport.requests) == 3
    assert [path.parent.name for path in raw_artifact_paths] == ["request-input-000002"]
    assert [record["entity_key"] for record in dataset_records] == ["dataset-beta"]
    assert result.records_extracted == 1
    assert metadata["dead_letter_count"] == "1"
    assert metadata["dead_letter_skipped_count"] == "1"


def test_catalog_strategy_continues_after_multiple_dead_letters_within_budget(tmp_path):
    from unittest.mock import patch

    source_config = _build_source_config_with_request_inputs(
        tmp_path,
        source_id="catalog_dead_letter_budget",
        path="/catalog/{entity_id}",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "entities",
            "columns": {"entity_id": "id"},
        },
        parameter_bindings={
            "entity_id": {"from": "request_input.entity_id"},
        },
        dead_letter_max_items=2,
    )
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-catalog_dead_letter_budget",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )
    fake_inputs = (
        {"entity_id": "alpha"},
        {"entity_id": "beta"},
        {"entity_id": "gamma"},
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(400, {"error": "bad alpha"}),
            ResponseSpec(400, {"error": "bad beta"}),
            ResponseSpec(200, {"result": {"results": [{"id": "dataset-gamma", "title": "Gamma"}]}}),
        ],
    )

    with patch("janus.strategies.catalog.core.load_request_inputs", return_value=fake_inputs):
        result = strategy.extract(plan)

    metadata = result.metadata_as_dict()

    assert len(transport.requests) == 3
    assert [record["entity_id"] for record in _read_jsonl(
        _raw_run_dir(tmp_path, "catalog_dead_letter_budget") / "normalized" / "datasets.jsonl"
    )] == ["dataset-gamma"]
    assert result.records_extracted == 1
    assert metadata["dead_letter_count"] == "2"
    assert metadata["dead_letter_skipped_count"] == "2"


def test_catalog_strategy_emits_no_entity_types_when_no_entities_returned(tmp_path):
    """Successful request returning no parseable entities still reports an
    empty entity handoff cleanly."""
    plan = _build_plan(tmp_path, source_id="catalog_empty_parse", pagination_type="none")
    strategy, _ = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {})],
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert metadata["normalized_artifact_count"] == "0"
    assert metadata["entity_types_emitted"] == "none"


def test_catalog_strategy_emits_entity_counts_for_ambiguous_payload(tmp_path):
    """Payloads with mixed confidence still emit only the entity artifacts
    discovered by the shared parser."""
    plan = _build_plan(tmp_path, source_id="catalog_ambiguous_parse", pagination_type="none")
    strategy, _ = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "results": [
                        {
                            "id": "ds-1",
                            "resources": ["r1"],
                            "owner_org": "org-1",
                            "metadata_created": "2026-01-01",
                        },
                        {"id": "ds-2", "title": "bare"},  # low confidence — no hint keys
                    ]
                },
            )
        ],
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert result.records_extracted > 0
    assert int(metadata["normalized_artifact_count"]) >= 1
    assert metadata["datasets_extracted"] == "2"
    assert "dataset" in metadata["entity_types_emitted"].split(",")


def test_catalog_source_config_rejects_request_inputs_for_file_source(tmp_path):
    """request_inputs is rejected at config load time for non-api/catalog sources."""
    from janus.models.source_config import SourceConfigValidationError

    with pytest.raises(SourceConfigValidationError, match="access.request_inputs"):
        SourceConfig.from_mapping(
            {
                "source_id": "file_src",
                "name": "file_src",
                "owner": "janus",
                "enabled": True,
                "source_type": "file",
                "strategy": "file",
                "strategy_variant": "static_file",
                "federation_level": "federal",
                "domain": "example",
                "public_access": True,
                "access": {
                    "url": "https://example.invalid/file.csv",
                    "method": "GET",
                    "format": "csv",
                    "timeout_seconds": 30,
                    "auth": {"type": "none"},
                    "pagination": {"type": "none"},
                    "rate_limit": {
                        "requests_per_minute": 10,
                        "concurrency": 1,
                        "backoff_seconds": 1,
                    },
                    "request_inputs": {
                        "type": "iceberg_rows",
                        "namespace": "ns",
                        "table_name": "t",
                        "columns": {"x": "y"},
                    },
                },
                "extraction": {
                    "mode": "full_refresh",
                    "checkpoint_strategy": "none",
                    "retry": {
                        "max_attempts": 1,
                        "backoff_strategy": "fixed",
                        "backoff_seconds": 0,
                    },
                },
                "schema": {"mode": "infer"},
                "spark": {"input_format": "csv", "write_mode": "overwrite"},
                "outputs": {
                    "raw": {"path": "data/raw/x", "format": "csv"},
                    "bronze": {"path": "data/bronze/x", "format": "iceberg"},
                    "metadata": {"path": "data/metadata/x", "format": "json"},
                },
                "quality": {"allow_schema_evolution": True},
            },
            tmp_path / "conf" / "sources" / "file_src.yaml",
        )
