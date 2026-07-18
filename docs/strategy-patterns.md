# JANUS Strategy Patterns

Strategies are how JANUS stays realistic about source differences without falling back to script copy-paste.

The project is intentionally not "source-agnostic" in the naive sense. APIs, bulk files, and catalogs behave differently in practice. The clean answer is not to pretend they are the same. The clean answer is to give each family its own reusable execution layer and keep the family boundaries sharp.

## Pattern first, source second

Choose the family based on the unit of extraction, not on where the source is hosted.

An HTTP endpoint can still be a file source.
A JSON payload can still be a catalog source.
A source that publishes both metadata and downloadable resources may need more than one configured JANUS source.

## Family comparison

| Family | Use when the source is mainly... | Variants | Typical checkpoint unit | Normalization handoff |
| --- | --- | --- | --- | --- |
| `api` | record-serving HTTP endpoints | `page_number_api`, `offset_api`, `cursor_api`, `date_window_api` | record freshness field or request window | raw API response artifacts |
| `file` | files, packages, or discovered downloadable artifacts | `static_file`, `versioned_file`, `archive_package` | file version or publication token | persisted files filtered to reader-compatible artifacts |
| `catalog` | metadata about datasets, groups, organizations, or resources | `metadata_catalog`, `resource_catalog` | metadata freshness field | normalized JSONL entity artifacts |

## API family

Use the API family when the source is fundamentally about requesting pages or windows of records from an HTTP service.

The shared API strategy already owns:

- request construction from source config;
- bounded request-input loading and parameter binding before pagination when a source needs runtime request contexts;
- auth injection from environment variables;
- pagination flow;
- bounded retry and request pacing;
- checkpoint-aware incremental request shaping;
- deterministic raw response persistence;
- extraction metadata emission.

Typical fit:

- public APIs with page or offset parameters;
- APIs that return JSON records directly;
- endpoints where the main complexity is request/response behavior.

Keep the request-shaping layers separate:

- `access.params` for fixed literals;
- `access.request_inputs` for bounded outer request contexts;
- `access.parameter_bindings` for runtime values derived from the current request input or checkpoint;
- hooks for the cases that are still genuinely irregular.

Good hook use in this family:

- custom payload-to-record extraction;
- non-standard cursor lookup;
- source-specific checkpoint query parameters;
- small request or response adaptation.

What does not belong in the API family:

- archive discovery and extraction;
- file publication/version logic;
- metadata-catalog traversal of datasets and resources.

## File family

Use the file family when the source is fundamentally about locating, downloading, versioning, validating, or unpacking files.

The shared file strategy already owns:

- local file discovery and remote URL resolution;
- direct, HTML-link, and Nextcloud WebDAV link discovery;
- version selection;
- checksum validation when available;
- deterministic raw download persistence;
- ZIP and gzip tarball extraction with safe member handling;
- filtering extracted members for Spark handoff;
- version-oriented checkpoint behavior.

Typical fit:

- one direct CSV or Parquet file;
- periodic publications where the newest file matters;
- ZIP or `.tar.gz` packages that contain one or more tabular files.

Good hook use in this family:

- custom version-token extraction from filenames;
- source-specific discovered-file ordering;
- expected checksum lookup for a peculiar source;
- archive-member selection rules that are truly source-local.

What does not belong in the file family:

- API record pagination;
- catalog traversal semantics;
- source-specific row parsing hidden inside download logic.

## Catalog family

Use the catalog family when the source is fundamentally about ingesting metadata that points to datasets and resources.

The shared catalog strategy already owns:

- catalog request flow and pagination;
- bounded request-input loading and parameter binding for catalog detail loops;
- retry and request pacing;
- metadata-oriented checkpoint behavior;
- traversal of organizations, groups, datasets, and resources;
- deterministic raw page persistence;
- normalized JSONL handoff artifacts for later Spark reads.

Typical fit:

- metadata APIs such as open-data catalogs;
- registries where the important output is discovery metadata rather than business facts;
- sources where resources should be cataloged first and ingested later through another family.

Good hook use in this family:

- unusual root wrappers or nested collection paths;
- source-specific request shaping;
- one catalog's checkpoint-parameter mismatch;
- slight payload adaptation before traversal.

What does not belong in the catalog family:

- downloading the files behind a resource URL as part of catalog ingestion;
- treating resource payload rows as if they were the catalog itself;
- business-domain normalization for downstream facts.

## Mixed-source guidance

Some sources look ambiguous at first. Use the rule below.

### Catalog that points to files

Use `catalog` to ingest metadata about the resources.
Use a separate `file` source if JANUS also needs the actual resource payload.

### API endpoint that returns a downloadable file

If the important artifact is the file itself, use `file`.
If the important artifact is the response records and the file is incidental, use `api`.

### ZIP published behind an HTTP URL

Use `file`. The transport is HTTP, but the ingestion pattern is still file/package handling.

### Metadata endpoint that returns nested datasets and resources as JSON

Use `catalog`, even though the payload is JSON over HTTP.

## Shared HTTP mechanics

The `api`, `catalog`, and `file` families all speak HTTP, so the *behavioral* HTTP layer is single-sourced in `src/janus/strategies/http/` — the same way the bronze write path is single-sourced in `src/janus/runtime/materialize.py`. It is never copied into a family core:

- `transport.py` — `ApiClient` / `UrllibApiTransport` / `ApiRequest` / `ApiResponse`;
- `throttle.py` — one thread-safe `HttpRequestThrottle` for rate-limit pacing;
- `retry.py` — one `send_with_retries` loop (retryable status codes, `Retry-After`, backoff, family error mapping);
- `payload.py` — one `decode_payload` (`json` / `jsonl` / `text` / `binary`), shared by `api` and `catalog`;
- `binding.py` — one each of `resolve_url`, `split_path_and_query_params`, `default_checkpoint_params`, and `checkpoint_request_value`, shared by `api` and `catalog`.

New HTTP-shaped strategies **compose** these collaborators; they do not copy them. The throttle is thread-safe by construction, so any family that later adds concurrency keeps its rate-limit contract for free. Genuine family differences — which payload formats a family accepts, the error type raised on retry exhaustion, family-specific messages — are passed as explicit parameters, never forked into a private copy. The file family composes the throttle and retry loop only: it handles bytes and archives and does not JSON-decode.

## When a new pattern deserves promotion

Do not create a new family or a new reusable variant just because one source is awkward.

Promote behavior into the shared family when:

- the same need appears in more than one source;
- the behavior can be explained as part of the family contract;
- the improvement makes onboarding simpler instead of more magical.

Examples of good promotion:

- a new paginator shared by several APIs;
- a new archive format handled by more than one file source;
- a new catalog collection-discovery helper used by more than one metadata source.

Examples of bad promotion:

- a helper that exists only because one source has inconsistent field names;
- a generic function that really hardcodes one source's assumptions;
- a family-level option added only to avoid writing a small source hook.

## Strategy anti-patterns

Reject changes that do any of the following:

- branch on `source_id` inside shared strategy loops;
- make one family aware of another family's responsibilities;
- duplicate request, checkpoint, or raw-write logic in source code;
- copy a throttle, retry loop, payload decoder, or URL/param/checkpoint binding helper into a family core instead of composing `strategies/http/`;
- use hooks as a dumping ground for unmodeled generic behavior;
- flatten business payloads in shared runtime code when the logic is source-specific;
- copy an existing source implementation because it is "close enough."

The strategy layer is where JANUS either stays modular or starts to rot. Keep the family boundaries strict and the source-specific escape hatches small.
