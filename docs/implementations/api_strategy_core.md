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

## Concurrent pagination and end of stream

The short version is: concurrency in the API strategy is **speculation**, and speculation needs an explicit answer to "what happens when I guess wrong."

When `access.rate_limit.concurrency` is above `1` and the paginator is page-number or offset, `_extract_concurrent_pages` runs instead of the sequential loop. It fills a window of in-flight requests by calling `_predicted_next_pagination_state`, which does exactly what its name says: it assumes the page currently in flight will come back full and computes the next page number or offset from that assumption. Nothing has confirmed that page exists.

That assumption is unavoidable — the whole point of fanning out is to issue page N+1 before page N answers — so the design question is not how to avoid guessing, but what a wrong guess is allowed to cost.

### Three signals that the stream ended

The loop now recognizes three:

1. **A short or empty page.** The paginator's existing rule, unchanged, and the only signal the sequential loop has ever had.
2. **A past-end status.** A response whose status is in `access.pagination.past_end_status_codes` (default `404`, `416`). Previously this raised `ApiResponseError` and dead-lettered the entire request input; now it ends the stream cleanly.
3. **A reported total.** When a payload or hook exposes a record count, the last useful request index is computed from it and speculation stops there — before the wasted requests are ever issued.

Signals 1 and 2 are terminal. Signal 3 is only a **cap on speculation**: past the reported total the loop degrades to a single in-flight request rather than stopping, so a stale or wrong total costs throughput and never completeness. That distinction is the safety property the whole feature rests on, and it is stated in `speculation.py`'s module docstring for the same reason.

### The first index is never speculative

The rule that keeps a genuine `404` loud is one line in `SpeculativePaginationPolicy.is_speculative`: a request index counts as speculative only when it is greater than the first index of that request input.

The first request of a request stream was issued on evidence — the config or an upstream Iceberg table said that input exists. A `404` there means the endpoint is broken, and it still raises exactly as before. Every later index exists only because JANUS guessed the previous page was full, so a `404` there is evidence about the *guess*, not about the endpoint. The first index is read from the initial pagination state rather than hard-coded to `1`, so a resumed run anchors on the page it actually restarted from.

There is a second guard for the case where the guess and the evidence disagree. Speculation often means futures for *higher* indexes have already resolved by the time a past-end response is committed. Before accepting end-of-stream, the loop inspects those already-done futures; if one of them returned records, the past-end read is contradicted and `ApiPastEndConflictError` is raised naming the conflicting index. Only futures that are already `done()` are inspected — never waited on, or cancellation would stop working. Since it is an `ApiStrategyError`, a contradicted stream dead-letters like any other failure. A truncated dataset stays impossible; a noisy failure is the acceptable cost.

### Where the classification lives

The knowledge that "some statuses are a normal terminal outcome" belongs to the shared retry loop, not to the API strategy. `send_with_retries` takes an opt-in `terminal_status_codes` set and, for a matching response, returns it undecoded instead of raising — checked after the `2xx` branch and before the retryable check, so a terminal status wins over a retry. The default is empty, so the catalog family, the file family, and the sequential API path are bit-for-bit unchanged.

This matters architecturally: the alternative — catching `ApiResponseError` in `api/core.py` and sniffing `exc.response.status_code` — would fork HTTP mechanics back into a family core, which is precisely what the shared `strategies/http/` layer exists to prevent.

### Bounding the over-fetch

Without a reported total, the bound is structural: at most `concurrency − 1` requests are outstanding past the end, all cancelled through `_cancel_pending` the moment either end signal fires, and the past-end page itself is requested exactly once.

With a total, the bound is exact. `total_records_from_payload` resolves the configured dotted path first, then a list of root hint keys, then the same hints nested inside `meta` / `metadata` / `pagination`. `count` is deliberately excluded from the hints — several APIs use it for records on the current page, which would yield a one-page ceiling and quietly serialize the run. `last_request_index_for_total` converts the count into the highest worthwhile request index using integer arithmetic, because page counts at CNPJ scale must not pass through binary floating point. The ceiling is monotone: it only ever rises, so a total that shrinks between pages cannot retro-truncate work already predicted.

`_may_submit` is where the ceiling is applied, and it gates **submission only** — above the ceiling an index is submitted when it is the next one to commit, which is the degradation-not-termination rule again, expressed in code.

### New hook point

`ApiHook.resolve_total_records(plan, request, response, payload)` returns the total for a request input when the API exposes one somewhere the generic discovery cannot reach. It defaults to `None`, which falls back to payload discovery. Hook values that are negative or not integers are rejected with an `api_total_records_invalid` warning and the payload value is used instead — a hook must not be able to truncate a run, though a legitimate `0` would be harmless anyway since the ceiling never terminates extraction.

### New structured events

- `api_pagination_past_end_detected` (WARNING) — a stream ended on an inferred signal rather than an observed empty page. Deliberately a warning: an operator should be able to grep for sources whose end-of-stream is inferred.
- `api_pagination_speculation_cancelled` (INFO) — outstanding speculative futures were dropped, with the count.
- `api_pagination_lookahead_bounded` (INFO) — a reported total produced a ceiling, with the total, the last request index, and where the total came from.
- `http_terminal_status_returned` (INFO, from the shared retry loop) — a caller-declared terminal status was returned instead of raised.
- `api_total_records_invalid` (WARNING) — a hook returned an unusable total.

Logged URLs go through `redact_url`, since auth tokens ride in query strings on some sources.

### New extraction metadata

`pagination_concurrency` is emitted on every run. These six appear only when the concurrent loop actually ran:

- `speculative_request_count`
- `speculative_discarded_count`
- `past_end_terminated_count`
- `past_end_status`
- `total_records_reported`
- `lookahead_ceiling_source` (`hook`, `payload`, or `none`)

They are exported as `CONCURRENCY_ONLY_METADATA_KEYS` and are the **only** legitimate differences between a concurrent and a sequential run of the same source. The equivalence suite imports that constant rather than restating the list, and asserts that everything else — artifact paths, checksums, bytes, record counts, checkpoints, and the remaining metadata — is identical.

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

