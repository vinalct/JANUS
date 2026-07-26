# API Strategy Core

The short version is: JANUS now has a real API strategy instead of a planning placeholder.

Before this step, the planner could tell us that a source belonged to the `api` family and which variant it wanted to use, but it still could not execute that plan. There was no shared HTTP request layer, no reusable pagination flow, no checkpoint-aware incremental behavior, and no standard way to persist raw API payloads while keeping the rest of the runtime contracts intact.

This step fills that gap with a reusable API strategy built around the JANUS contracts that were already in place: source config, execution plans, raw writers, checkpoints, and run metadata.

The result is not a source integration for Transparencia, IBGE, or any other one API. It is the common execution layer those integrations are supposed to stand on.

## What was added

- `src/janus/strategies/api/http.py` now defines the request and response transport objects, auth injection rules, and the default stdlib-backed transport.
- `src/janus/strategies/api/pagination.py` now defines the reusable paginator components for page-number, offset, cursor, and no-pagination flows.
- `src/janus/strategies/api/core.py` now implements the `ApiStrategy` itself, including request building, retry and throttling behavior, checkpoint-aware incremental execution, per-page extraction progress tracking, dead-letter tracking, raw persistence, and metadata emission.
- `src/janus/strategies/api/__init__.py` now exposes the API strategy package surface for downstream imports.
- `src/janus/planner/core.py` now resolves API variants to the real `ApiStrategy` in the default strategy catalog.
- `tests/unit/strategies/api/test_api_strategy.py` now covers the main runtime behaviors introduced in this step.

## What the API layer is responsible for

This layer sits above a plain HTTP client.

Its job is not only to send requests. Its job is to take a validated JANUS `ExecutionPlan`, execute an API extraction safely, and hand the rest of the project a standard `ExtractionResult`.

That means the layer now owns these responsibilities:

- building a request from the source contract;
- injecting auth from environment-managed secrets;
- applying the configured pagination strategy;
- respecting the configured request rate;
- retrying transient failures with bounded backoff;
- recording unrecoverable request inputs in source-scoped dead-letter state once request-level retries are exhausted;
- reading an existing checkpoint for incremental runs;
- persisting exact raw payloads through the shared raw writer;
- emitting extraction metadata in a shape the rest of JANUS can already consume.

What it does **not** do is normalize records into a Spark DataFrame. That handoff still belongs to later normalization and bronze-writing work.

## The transport model

The API module introduces two small runtime objects:

### `ApiRequest`

This is JANUS's normalized description of one outbound request.

It carries:

- HTTP method;
- URL;
- timeout;
- headers;
- query params;
- optional body.

The object is immutable. When auth or pagination modifies the request, the code produces a new request with `with_header(...)` or `with_params(...)` instead of mutating the original one in place.

That keeps request shaping easier to reason about when the strategy applies several layers of behavior to the same base request.

### `ApiResponse`

This is the normalized response object returned by the transport layer.

It keeps:

- the originating request;
- status code;
- raw body bytes;
- response headers;
- the response timestamp.

It also provides small helpers for decoding text and JSON payloads.

## Why the transport is separate from the strategy

This split is deliberate.

The transport deals with how to talk HTTP. The strategy deals with how JANUS should execute an API source.

That separation buys a few things:

- tests can replace the transport with a fake client and verify strategy behavior without real network calls;
- the strategy is not hard-wired to one HTTP library;
- a future `requests`-based or `httpx`-based transport could be added without rewriting pagination, checkpointing, or raw persistence logic.

At the moment the default implementation uses `urllib`, mainly because the project does not depend on `requests`.

## Auth and secret injection

This step adds a shared `inject_auth(...)` helper for the auth types already supported by the source contract:

- `none`
- `basic`
- `bearer_token`
- `header_token`
- `query_token`

The important design choice is that JANUS stores only the **name** of the environment variable in source config, not the secret value itself.

In practice the flow is:

1. the source config declares which auth mode applies;
2. the config also declares the env var name to read, such as `TRANSPARENCIA_API_TOKEN`;
3. the strategy builds the base request;
4. `inject_auth(...)` reads the secret from the environment at runtime;
5. the helper returns a new request with the correct header or query parameter applied.

For example:

- `bearer_token` produces `Authorization: Bearer <token>` by default;
- `header_token` uses the configured header name;
- `query_token` adds the token to the query string under the configured query parameter;
- `basic` reads a username env var and a password env var, then emits a standard Basic auth header.

The helper fails fast if a required env var name is missing from config or if the named secret is not present in the runtime environment.

## Pagination

This step adds reusable paginator components instead of burying pagination rules inside the strategy loop.

The current variants are:

- `NoPaginationPaginator`
- `PageNumberPaginator`
- `OffsetPaginator`
- `CursorPaginator`

Each paginator does two things:

1. apply its current state to a request;
2. decide whether another request is needed after a response has been processed.

### Page-number pagination

This variant injects:

- the configured page parameter;
- the configured page-size parameter;
- the configured page size.

It keeps requesting the next page while the number of extracted records is equal to the configured page size. A short page or an empty page ends the loop.

### Offset pagination

This variant injects:

- the configured offset parameter;
- the configured limit parameter;
- the configured page size.

It increments the offset by the configured page size after each full page and stops on a short or empty page.

### Cursor pagination

This variant supports a cursor parameter and can either:

- accept the next cursor from a source hook; or
- fall back to a small generic payload probe for fields such as `next_cursor`, `nextCursor`, or nested pagination metadata.

That is intentionally modest. Cursor APIs vary a lot, so the hook path is still the clean escape hatch when one API behaves differently.

## Retry, throttling, and safety defaults

The strategy now owns the common API execution safeguards that should not be rewritten per source.

### Request throttling

`ApiRequestThrottle` implements a simple single-threaded request pace using `requests_per_minute`.

This is not a concurrency framework. It is a safe default that keeps the first JANUS API integrations from hammering public endpoints just because the code runs quickly on the client side.

### Retry behavior

The retry loop currently treats these status codes as transient:

- `408`
- `429`
- `500`
- `502`
- `503`
- `504`

It also retries transport-level failures such as low-level connection errors.

Backoff is driven by the source config:

- `fixed`
- `exponential`

The computed delay is then bounded by the configured API-side backoff limit when one is present. If the response includes a `Retry-After` header, the strategy considers it but still keeps the delay inside the configured bound.

That gives JANUS safer default behavior without turning one public API integration into an unbounded sleep loop.

When one request input still fails after the bounded retry loop, the strategy records that input in the dead-letter state. The extraction continues only while `extraction.dead_letter_max_items` still allows more dead-lettered inputs; otherwise the failure is raised.

## Incremental extraction and checkpoints

This step is also where the API strategy starts using the checkpoint layer introduced earlier.

When the source runs in incremental mode:

- the strategy loads the current checkpoint state;
- it computes the outgoing checkpoint parameter value;
- it applies lookback days when configured;
- it injects that value into the request before the extraction loop starts.

The generic default is simple:

- if the source uses `checkpoint_field: updated_at`, the strategy sends that field name as the query parameter too.

That is good enough for config-first sources whose request semantics line up with their checkpoint field.

When they do not line up, the hook API allows a source integration to override only that part. In other words, the generic strategy handles the common case and the hook handles the naming mismatch without forcing source-specific code into the core loop.

The strategy also computes the highest observed checkpoint value while processing extracted records, so the downstream observer and checkpoint store can advance the source state after a successful run.

## Raw persistence

Every successful response is now persisted to the raw zone through the shared `RawArtifactWriter`.

The strategy writes deterministic file names under a raw subdirectory.

For sources with a single request input, files land under `pages/`:

- `pages/page-0001.json`
- `pages/offset-00000000.json`
- `pages/cursor-0001.json`
- `pages/response-0001.json`

For sources with multiple request inputs — `iceberg_rows`, `date_window`, or `combined` — each input gets its own subdirectory keyed by its position in the enumeration:

- `request-input-000001/page-0001.json`
- `request-input-000002/page-0001.json`

The file naming within each subdirectory follows the same page-number, offset, cursor, or response convention as the single-input case.

That gives JANUS an exact raw payload trail for API runs instead of treating HTTP responses as transient in-memory objects that disappear once parsing is done.

The raw write metadata currently includes basic operational context such as:

- request URL;
- response status;
- request index;
- page number, offset, or cursor when applicable.

## Extraction progress and resumable runs

Long-running API extractions fail mid-run. A source with 33,000 pages at 240 req/min still takes over two hours to complete. When the remote API returns an unexpected 400 or 500 at page 30,000, JANUS does not lose what it already wrote.

Every page the API strategy writes to the raw zone is durably persisted as a file with a deterministic name. The problem was not the raw files—they survive. The problem was that the strategy had no memory of where it stopped, so the next invocation restarted from page one.

The strategy now saves an `extraction_progress.json` entry after each page lands on disk. This file lives in the metadata zone alongside run records, checkpoint state, and dead-letter state. It is keyed by source id rather than run id, so it survives across separate CLI invocations.

When the operator runs the command with `--resume`, the strategy reads this progress file and the source-scoped dead-letter state. It resumes from where the previous run stopped without touching the API for any page or input combination it already has on disk, and it skips request inputs that were already dead-lettered in the interrupted run.

### Multi-input progress tracking

For sources that iterate over multiple request inputs — `iceberg_rows`, `date_window`, or `combined` — the progress file tracks each input by its **content**, not by its position in the list.

Each input gets a stable fingerprint derived from its field values:

```
orgao_codigo=170010|window_end=2025-01-31|window_start=2025-01-01
```

The file records completed inputs as a list of `{"key": ..., "index": ...}` objects plus the key and file-path index for the input that was in progress when the run stopped. On resume:

- inputs whose fingerprint is in `completed_inputs` are skipped and their raw files are re-discovered from disk;
- inputs whose fingerprint is in the dead-letter state are skipped immediately;
- the input whose fingerprint matches `current_input_key` has its raw files re-discovered up to the last page and pagination resumes from there;
- all remaining inputs are started fresh.

This makes resume correct even when upstream Iceberg data reorders between invocations or the total input count grows.

### Pagination mode support

Resume is supported for page-number and offset pagination. Cursor pagination does not support per-page resume because there is no way to derive the next cursor without the response of the previous page; cursor runs restart from the beginning when `--resume` is used, which is still better than corrupted state.

The progress file is removed at the end of every successful extraction. A normal run without `--resume` also removes any stale progress file and clears any stale dead-letter state before starting, so partial state from a previous run does not silently influence an intentional fresh run.

## Hooks and extension points

This step deliberately adds API-specific hook points without turning the strategy into a special-case switchboard.

`ApiHook` currently allows a source integration to override:

- request preparation;
- response handling;
- payload transformation;
- record extraction;
- next-cursor resolution;
- total-record resolution, which caps concurrent look-ahead exactly instead of speculating;
- checkpoint query parameter generation.

That is enough to support sources that mostly follow the shared strategy but still have one awkward detail, such as:

- a custom nested payload path;
- a non-standard incremental request parameter;
- a cursor hidden in a special metadata block;
- a record total published under a name the generic discovery does not recognise.

The strategy still controls the main execution loop. The hook only adjusts the edges that are genuinely source-specific.

## What the strategy returns

At the end of extraction, the strategy returns one JANUS `ExtractionResult`.

That result contains:

- the raw artifacts written during the run;
- the total extracted record count;
- the resolved checkpoint value when one was found;
- extraction metadata such as request count, retry count, pagination type, auth type, and whether a checkpoint was loaded.

That keeps the API layer aligned with the rest of the project instead of inventing a separate response shape that later runtime code would have to special-case.

## What the planner change does

Before this step, the default planner catalog mapped all strategy families to the planning-only placeholder implementation.

Now, the planner resolves API variants to the real `ApiStrategy`, while file and catalog variants still point at the placeholder until their own runtime steps land.

That means the planner no longer stops at “this source is an API source.” It can now hand API variants to an execution-ready strategy implementation.

## What the tests lock down

The new unit tests focus on behavior the next source-integration steps will rely on.

They cover:

- resolving API variants to the real `ApiStrategy` through the default planner catalog;
- page-number pagination and checkpoint-aware request shaping;
- offset pagination behavior;
- bounded retry behavior for transient failures;
- raw payload persistence under deterministic paths;
- hook-driven override of checkpoint request parameters and record extraction.

The focused verification for this step passed with:

- `python -m pytest tests/unit/strategies/api/test_api_strategy.py -q`
- `python -m pytest tests/unit/planner/test_planner.py -q`
- `python -m pytest tests/unit -q`

In the current environment, the Spark-backed unit tests are still skipped when `pyspark` is not available. The API strategy tests themselves do not depend on Spark.

