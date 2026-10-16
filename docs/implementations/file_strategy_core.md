# File Strategy Core

The short version is: JANUS now has a real file strategy instead of only knowing that a source belongs to the `file` family.

Before this step, the planner could classify a source as `static_file`, `versioned_file`, or `archive_package`, but that was still only planning metadata. There was no reusable execution layer for local files, remote downloads, archive extraction, version selection, or deterministic raw persistence for bulk-file sources.

This step fills that gap with a reusable file strategy that sits on top of the JANUS contracts already in place: source config, execution plans, checkpoints, raw writers, and extraction metadata.

The result is the generic runtime layer used by bulk-file sources such as INEP and Receita Federal CNPJ. Source-specific package rules still stay in configuration or narrow hooks.

## What was added

- `src/janus/strategies/files/core.py` now implements the `FileStrategy` itself, plus the file-specific hook points and helper objects that support discovery, download, version handling, archive extraction, normalization handoff, and metadata emission.
- `src/janus/strategies/files/formats.py` is the file-family source of truth for filename handling, format inference, supported handoff formats, archive suffix constants, and safe path segment rendering.
- `src/janus/strategies/files/download_loop.py` separates each candidate into download and completion phases; `concurrent_extraction.py` orchestrates ordered consumption and the shared dead-letter policy.
- `src/janus/strategies/files/download_window.py` owns `DownloadOutcome`, the ordered download window and the family's only thread pool.
- `src/janus/strategies/files/resolvers.py` provides the reusable link resolver chain used when a file source starts from a landing page or public share instead of a direct download URL.
- `src/janus/strategies/files/__init__.py` now exposes the file strategy package surface for downstream imports.
- `src/janus/planner/core.py` now resolves file variants to the real `FileStrategy` in the default strategy catalog.
- `tests/unit/strategies/files/test_file_strategy.py` and `tests/unit/strategies/files/test_resolvers.py` now cover the main runtime behaviors introduced in this step.

## What the file layer is responsible for

This layer sits between the planner and the later Spark-facing read and write path.

Its job is not to parse business data into a final bronze dataset. Its job is to take a validated JANUS `ExecutionPlan`, execute a file-oriented extraction safely, and hand the rest of the runtime a standard `ExtractionResult` that points at deterministic raw artifacts.

That means the strategy now owns these responsibilities:

- discovering files from `access.path`, `access.discovery_pattern`, or a direct remote `access.url`;
- resolving remote links through direct URL, redirect, Nextcloud WebDAV, or HTML-link discovery;
- choosing which discovered file or files should actually run for the configured variant;
- downloading remote payloads when the source is HTTP-based;
- applying auth and retry behavior for remote file requests;
- recording irrecoverable file candidates in source-scoped dead-letter state;
- validating a checksum when one is available;
- persisting raw downloads through the shared raw writer;
- extracting ZIP and gzip tarball packages through reusable archive logic;
- filtering extracted members when the source config provides a file pattern;
- exposing only reader-compatible artifacts to the normalization handoff;
- emitting extraction metadata in the same contract shape the rest of JANUS already expects.

What it does **not** do is embed source-specific package rules for one dataset. That stays outside the core and belongs in a source hook when a source really does behave differently.

## Discovery and selection

This step adds a small discovery model built around `DiscoveredFile`.

A discovered file records the information the strategy needs before raw persistence starts:

- where the file came from;
- whether it is local, remote, or extracted from an archive;
- the resolved file name;
- the inferred format;
- the version token when one can be resolved.

The main moving parts are remote URL resolution and the three file variants.

### Remote URL resolution

When a file source uses `access.url`, the strategy resolves that URL before download. In `access.link_resolver: auto`, JANUS tries the reusable chain in order:

- direct file URLs with known file extensions;
- HTTP `HEAD` responses that resolve to a non-HTML payload;
- public Nextcloud shares through WebDAV `PROPFIND`;
- HTML pages with matching anchor links.

The resolver can be pinned with `direct`, `nextcloud_webdav`, or `html_links` when a source should avoid auto-detection. `access.remote_file_pattern` then filters the discovered remote candidates before the download loop starts.

### Static file sources

A static source is the simple case.

If a config points at one explicit file path or one direct file URL, the strategy persists that file deterministically and uses a stable checkpoint value of `current` unless a more specific version can be resolved.

That keeps reruns predictable without pretending a static publication has historical version semantics.

### Versioned file sources

A versioned source is where discovery matters more.

When JANUS sees multiple file candidates, the strategy now resolves an ordering key from the file name when it can. The default implementation looks for date-like version tokens such as `2026-03` or `2026-03-15`. If no version token is present for a local file, it falls back to the file modification timestamp.

From there the strategy behaves differently depending on extraction mode:

- `full_refresh` keeps only the newest candidate;
- `incremental` keeps every candidate strictly newer than the stored checkpoint.

That gives the file layer a sensible generic version-selection rule without hardcoding any one source’s naming convention.

### Archive package sources

For `archive_package`, the strategy still treats the original archive as the raw download to preserve, but it also extracts reusable member artifacts for downstream handoff.

The reusable archive support covers ZIP files and gzip tarballs (`.tar.gz` or `.tgz`).

## Format inference boundary

File-format inference now lives in `janus.strategies.files.formats`.

That module is intentionally narrow. It decides how JANUS derives a filename from a URL or content-disposition header, how a filename maps to a file-family format, which formats can be handed to Spark directly, which suffixes are archive candidates, and how version or archive path segments are made safe for raw persistence.

The file strategy calls those helpers instead of carrying its own extension checks. Resolvers use the same helpers when classifying remote candidates. That keeps CSV, JSONL, Parquet, binary archive, and text decisions consistent across local discovery, remote discovery, download persistence, archive extraction, and Spark handoff filtering.

## Remote download behavior

Not every file source is local. Some file sources are really just HTTP downloads that happen to return a file instead of a JSON API response.

Because of that, the file strategy reuses the existing HTTP request and response primitives already present in JANUS rather than inventing a second transport stack just for downloads.

In practice, that means remote file execution now supports:

- the configured HTTP method and timeout;
- header or token auth from the source contract;
- bounded retry behavior for transient response codes and transport failures;
- one shared, thread-safe request throttle across discovery, downloads and retries.

The important boundary here is that the file strategy is reusing generic HTTP plumbing, not API pagination or API payload logic. The file runtime still stays file-oriented.

When one selected file candidate still fails, the strategy records that candidate in the dead-letter state and continues only while `extraction.dead_letter_max_items` allows it. When the operator runs with `--resume`, previously dead-lettered candidates are skipped instead of being retried again.

### Download phase

`access.rate_limit.concurrency > 1` enables workers only for `_download_candidate`: each calls `FileDownloader.load_payload_with_dedicated_client` on a fresh transport, streams into the raw staging area and hashes while writing. `OrderedDownloadWindow` holds at most N submitted, unconsumed outcomes, including completed payloads, and exposes them in candidate order. `_complete_candidate` runs on the loop thread: remote-payload validation, every hook, version and checkpoint decisions, checksum verification, raw commit, sidecar creation and archive expansion stay sequential. `_consume_candidate`, shared by both paths, records dead letters and evaluates their budget there. On budget failure, in-flight downloads finish and are discarded without additional commits or dead letters before the original exception propagates. At concurrency 1, both phases run sequentially using the existing run-wide client, with no pool and no concurrency-only metadata keys.

Resume still re-downloads every selected non-dead-lettered candidate; an existing checkpoint can subsequently skip its completion. No file progress record or persisted-candidate rediscovery is introduced. See [Concurrent downloads](../source-onboarding.md#concurrent-downloads) for memory measurements, hook caveats, log fields and the live-smoke runbook.

## Raw persistence

The raw side of this step leans on the shared `RawArtifactWriter` introduced earlier.

Every selected file now lands under a deterministic raw path.

Downloaded or copied source files are written under:

- `downloads/<resolved-version>/<filename>`

Extracted archive members are written under:

- `extracted/<resolved-version>/<archive-stem>/<member-path>`

That structure is intentionally boring.

It gives later integrations a predictable place to look without tying the raw layout to one public dataset.

The strategy preserves the original archive as a raw artifact, then persists extracted members separately when archive handling is enabled. That makes reruns easier to reason about and keeps the raw zone honest about what was actually downloaded.

## Checkpoints and versions

This step is also where the file strategy starts using the checkpoint layer for versioned file execution.

When the source runs in incremental mode and a checkpoint is configured:

- the strategy loads the current checkpoint state;
- it compares discovered file versions against the stored value;
- it skips files that are not newer than the checkpoint;
- it returns the highest processed version as the candidate checkpoint for the rest of the runtime.

That keeps the file side aligned with the same rerun-safe contract already used by the API path.

One detail matters here: the checkpoint value for file sources is version-oriented, not record-oriented. In other words, this step is tracking file publication order, not row-level incremental logic inside the file contents.

## Checksum handling

The file strategy now performs integrity checks when it has something concrete to verify against.

At the moment there are two generic paths:

- a local sidecar file named `<filename>.sha256` beside the source file;
- a remote checksum value returned in headers such as `X-Checksum-Sha256`.

When a checksum is present, the strategy hashes the payload and fails the extraction if the values do not match.

If no checksum is available, the strategy still computes the persisted raw checksum through the shared writer for metadata purposes, but it does not pretend that this alone proves upstream integrity.

## Archive extraction and handoff

The archive extraction path is intentionally reusable and intentionally cautious.

The strategy rejects unsafe archive member paths such as absolute paths or members that try to escape with `..`. Safe members are extracted in memory and then persisted back through the shared raw writer just like any other artifact.

After extraction, the strategy filters the archive members using `access.file_pattern` when the config provides one. That is the main generic hook for packages that contain more than one file but only some of them belong in the Spark handoff.

The normalization handoff then narrows the full extraction result to only those artifacts whose format matches `spark.input_format`.

That distinction matters.

The full extraction result can include:

- the original ZIP or tarball file;
- extracted CSV files;
- extracted text side files;
- any other preserved raw members.

The normalization handoff keeps only the artifacts that the shared reader is actually supposed to consume next.

## Hook points and what stays out of the core

This step also adds `FileHook`, which is the escape hatch for the file-family cases that cannot be described cleanly through config alone.

The hook can now intervene in a few bounded places:

- override URL link resolution;
- reorder or filter discovered files;
- override version resolution;
- provide an expected checksum;
- adjust a downloaded payload before persistence;
- choose which archive members should survive extraction.

That is enough room for real source quirks without pushing INEP-specific naming rules or package layout hacks into the generic strategy.

## What the tests lock down

The unit tests added for this step focus on the behavior later file integrations will depend on.

They cover:

- planner resolution of file variants to the real runtime strategy;
- direct, HTML-link, and Nextcloud WebDAV resolver behavior;
- deterministic raw persistence for a simple local CSV source;
- latest-version selection for a versioned full-refresh source;
- checkpoint-aware filtering for an incremental versioned source;
- remote download retry behavior together with checksum validation;
- ZIP and tar.gz extraction with filtered CSV handoff;
- failure on checksum mismatch.

The focused verification for this step passed with:

- `python -m ruff check src/janus/strategies/files src/janus/planner/core.py tests/unit/strategies/files`
- `python -m pytest tests/unit/strategies/files tests/unit/planner/test_planner.py tests/unit/strategies/api/test_api_strategy.py -q`

That second command matters because the file strategy now participates in the default planner catalog. The adjacent planner and API checks still passed after that registration change.
