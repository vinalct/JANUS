# Catalog Strategy Core

The short version is: JANUS now has a real catalog strategy instead of only knowing that a source belongs to the `catalog` family.

Before this step, the planner could classify a source as `metadata_catalog` or `resource_catalog`, but that was still only planning metadata. There was no reusable runtime layer for catalog traversal, metadata-first extraction, entity normalization, or deterministic persistence of catalog responses and discovery artifacts.

This step fills that gap with a reusable catalog strategy built on top of the JANUS contracts that were already in place: source config, execution plans, checkpoints, raw writers, and extraction metadata.

The result is not a `dados.gov.br` integration yet. It is the generic execution layer that a metadata-first source such as `dados.gov.br` is supposed to use.

## What was added

- `src/janus/strategies/catalog/core.py` now implements the `CatalogStrategy` itself, plus the catalog-specific hook points and orchestration used for request execution, checkpoint-aware metadata collection, raw persistence, and normalization handoff.
- `src/janus/strategies/catalog/document.py` contains the pure catalog document traversal, entity classification, entity batching, generic node and edge shaping, payload hashing, and parse-summary helpers.
- `src/janus/strategies/catalog/__init__.py` now exposes the catalog strategy package surface for downstream imports.
- `src/janus/planner/core.py` now resolves catalog variants to the real `CatalogStrategy` in the default strategy catalog.
- `tests/unit/strategies/catalog/test_catalog_strategy.py` now covers the main runtime behaviors introduced in this step.

## What the catalog layer is responsible for

This layer sits between the planner and the later Spark-facing read and write path.

Its job is not to ingest business facts like payments, census rows, or health records. Its job is to take a validated JANUS `ExecutionPlan`, traverse a metadata catalog safely, and hand the rest of the runtime a standard `ExtractionResult` that points at deterministic raw catalog pages and normalization-ready metadata artifacts.

That means the strategy now owns these responsibilities:

- building and sending catalog requests from the source contract;
- loading bounded request inputs and applying parameter bindings before each catalog request loop;
- applying the configured pagination strategy;
- respecting configured request pacing and retry rules;
- recording unrecoverable request inputs in source-scoped dead-letter state once request-level retries are exhausted;
- reading an existing checkpoint for incremental metadata refreshes;
- persisting exact raw catalog pages through the shared raw writer;
- traversing common catalog structures such as datasets, organizations, groups, and resources;
- normalizing those entities into line-oriented artifacts suitable for later Spark reads;
- emitting extraction metadata in the same contract shape the rest of JANUS already expects.

What it does **not** do is ingest the underlying dataset payloads themselves. If a catalog entry points to a CSV, ZIP, or API endpoint, that still belongs to the file or API strategy later.

## Why this needed its own strategy

A catalog source may also arrive over HTTP, but that does not make it the same as an API-record source.

The API strategy is designed around "call an endpoint and collect the records it serves." The catalog strategy is designed around "walk a registry and collect metadata about what datasets and resources exist."

That difference matters because catalog payloads are usually shaped around relationships:

- one organization can publish many datasets;
- one dataset can belong to groups;
- one dataset can expose several resources;
- one resource may be the real handoff point to a later file or API integration.

Trying to squeeze that into the API-record path would blur discovery logic with direct record ingestion. This strategy keeps those concerns separated.

## Traversal and entity extraction

This step adds a small traversal model built around `CatalogBatch` and a set of entity-specific normalization rules.

The strategy can now recognize and persist the most common metadata entities JANUS cares about:

- organizations;
- groups;
- datasets;
- resources.

At the root level, the strategy looks for common collection shapes such as:

- `result.results`;
- `data.results`;
- named collections like `organizations`, `groups`, `datasets`, or `resources`.

From there it keeps walking nested catalog structures. For example:

- dataset records can contain an embedded organization;
- dataset records can contain group membership;
- dataset records can contain one or more resources.

The strategy collects each of those entity types separately and keeps the parent-child links needed for lineage-friendly metadata. A resource record, for example, keeps the dataset identity that led to it.

## Document helper boundary

Catalog document traversal now lives in `janus.strategies.catalog.document`.

That module is deliberately side-effect free. It walks arbitrary JSON-like catalog payloads, recognizes common organization, group, dataset, and resource shapes, classifies generic catalog nodes, builds lineage-friendly document nodes and edges, hashes payload fragments, and reports parse quality metrics. It does not send requests, read checkpoints, write raw files, touch Spark, or record dead letters.

`catalog.core` remains the orchestration layer around those pure helpers. It owns HTTP execution, retry and throttling behavior, request inputs and parameter bindings, checkpoint and dead-letter state, raw page persistence, JSONL handoff writing, and metadata emission.

## Metadata normalization

The catalog strategy does a modest but useful normalization pass before the later Spark layer takes over.

For each extracted entity, the strategy now emits a consistent record shape that includes:

- `entity_type`;
- entity identifiers and titles when present;
- selected metadata fields such as URL, format, created time, and updated time;
- parent entity references;
- the collection path inside the payload;
- the originating request URL and page details;
- the raw artifact path and checksum;
- the original entity payload under `payload`, serialized as a JSON string.

That shape is intentionally practical.

It is enough for:

- discovery queries;
- provenance checks;
- later bronze modeling;
- debugging why JANUS believes one resource exists.

It is not trying to impose a grand canonical model on every catalog in phase 1.

## Raw persistence and handoff

The raw side of this step leans on the shared `RawArtifactWriter` introduced earlier.

Every successful catalog response is now written under a deterministic raw path:

- `pages/page-0001.json`
- `pages/page-0002.json`
- or the equivalent offset or cursor naming when other pagination modes are used.

When request inputs are configured, the same page naming is nested under stable input folders:

- `request-input-000001/page-0001.json`
- `request-input-000002/page-0001.json`

After traversal, the strategy also persists normalization-ready metadata artifacts under:

- `normalized/organizations.jsonl`
- `normalized/groups.jsonl`
- `normalized/datasets.jsonl`
- `normalized/resources.jsonl`

One detail matters here.

These normalized artifacts are the handoff produced by the strategy layer. They are not a hidden bronze writer. The strategy still stops at `ExtractionResult`, just like the API and file strategies do. A later runtime layer is still responsible for reading those artifacts with Spark and writing the bronze zone.

## Pagination, retries, and checkpoints

The catalog strategy reuses the shared request and pagination primitives already introduced for JANUS's HTTP-facing flows.

That means a catalog source now gets the same operational basics as the other runtime strategies:

- request inputs and parameter bindings for catalog detail endpoints;
- page-number, offset, cursor, or no-pagination request flow;
- bounded retry behavior for transient failures;
- single-threaded throttling through `requests_per_minute`;
- source-scoped dead-letter tracking for request inputs whose bounded retries are exhausted;
- incremental refresh support through the checkpoint layer.

Request inputs are handled as an outer loop. For each input, JANUS resolves configured bindings, fills matching `{name}` placeholders in the catalog URL, leaves non-placeholder bindings as query params, and then runs the normal paginator inside that bound request stream. This lets a catalog listing feed a later catalog-detail source without adding source-id branches to the catalog strategy.

If one request input still fails after the retry loop, the strategy records it in the dead-letter state and continues only while `extraction.dead_letter_max_items` allows it. When the operator runs with `--resume`, previously dead-lettered inputs are skipped instead of being retried again.

The checkpoint behavior here is metadata-oriented.

If a source config declares a checkpoint field such as `metadata_modified`, the strategy can:

- load the stored checkpoint;
- apply lookback days when configured;
- include the checkpoint value in the outgoing request;
- skip older catalog entities during an incremental run;
- return the highest observed metadata timestamp as the candidate checkpoint for the next run.

That keeps catalog refreshes rerun-safe without pretending the catalog strategy is ingesting the business data behind each resource.

## Hook points and what stays out of the core

This step also adds `CatalogHook`, which is the escape hatch for the catalog-family cases that cannot be described cleanly through config alone.

The hook can now intervene in a few bounded places:

- prepare a request before it is sent;
- adapt a response after it is received;
- transform a decoded payload before traversal starts;
- override the request parameters used for checkpoint-driven runs;
- provide a custom page-record view when a catalog uses an unusual wrapper.

That is enough room for real source quirks without hardcoding `dados.gov.br` behavior into the generic strategy.

## What the tests lock down

The unit tests added for this step focus on the behaviors that later catalog integrations will depend on.

They cover:

- planner resolution of catalog variants to the real runtime strategy;
- page-number traversal across more than one catalog page;
- request-input and parameter-binding execution for catalog detail loops;
- extraction of organizations, groups, datasets, and resources from one catalog flow;
- checkpoint-aware incremental filtering for metadata refreshes;
- `resource_catalog` behavior where resources are the primary root entity;
- normalization handoff limited to the JSONL artifacts that the Spark reader is supposed to consume next.

The focused verification for this step passed with:

- `python -m ruff check src/janus/strategies/catalog src/janus/planner/core.py tests/unit/strategies/catalog/test_catalog_strategy.py`
- `python -m pytest tests/unit/strategies/catalog/test_catalog_strategy.py tests/unit/planner/test_planner.py -q`
- `python -m pytest tests/unit/strategies/api/test_api_strategy.py tests/unit/strategies/files/test_file_strategy.py tests/unit/planner/test_planner.py tests/unit/strategies/catalog/test_catalog_strategy.py -q`

That last command matters because the catalog strategy now participates in the default planner catalog. The adjacent API, file, and planner checks still passed after that registration change.
