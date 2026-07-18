# JANUS Architecture Guide

JANUS is meant to stay a framework, not drift into a folder of one-off ingestion scripts. The architecture is organized around a simple rule:

`source YAML -> registry -> planner -> strategy -> raw artifacts -> Spark normalization/write -> metadata, checkpoints, and validation`

That rule matters because the project has to support different public-source patterns without hiding source-specific behavior in random places.

## Architectural stance

JANUS is:

- metadata-driven where behavior is declarative;
- strategy-based where source families behave differently;
- hook-friendly only for narrow source quirks;
- explicit about raw, bronze, and metadata outputs;
- reproducible through environment-driven runtime settings.

JANUS is not:

- a generic "one extractor fits everything" framework;
- a place to branch on source names in generic modules;
- a copy-paste script collection.

## Control flow

### 1. Source registry

The source registry starts with two checked-in configuration roots:

- `conf/app.yaml` points JANUS at the source-definition directory.
- `conf/sources/**/*.yaml` holds domain-organized source files; each file may contain one source contract or a top-level `sources:` list.

`janus.registry.loader` reads those files, validates them, and returns typed `SourceConfig` objects. Validation happens here on purpose so later layers do not have to parse raw YAML or guess at nested config structure.

Important rule: the registry owns contract validation, not source-specific behavior.

### 2. Planner

`janus.planner.core` turns one validated source into one deterministic runtime plan.

The planner is responsible for:

- loading the requested source from the registry;
- creating a run context with run id, environment, project root, and start time;
- resolving the strategy family and variant;
- resolving an optional source hook;
- returning a `PlannedRun` with the `ExecutionPlan`, resolved strategy, and pre-run metadata.

This is the execution boundary for JANUS. The planner decides how a source should run, but it does not perform extraction itself.

### 3. Strategy layer

`janus.strategies.base.interfaces` defines the shared lifecycle:

- `plan(...)`
- `extract(...)`
- `build_normalization_handoff(...)`
- `emit_metadata(...)`

The concrete families live under:

- `janus.strategies.api`
- `janus.strategies.files`
- `janus.strategies.catalog`

Each family owns the reusable behavior for that source pattern. Strategy code is where family-level logic belongs, such as pagination, file discovery, archive handling, or catalog traversal.

Shared support modules keep family code small without becoming source-specific:

- `janus.strategies.http` is the single-sourced *behavioral* HTTP layer shared by all three families: `transport` (`ApiClient` and the `urllib` transport), one thread-safe `HttpRequestThrottle`, one `send_with_retries` loop, one `decode_payload`, and one copy of each URL/param/checkpoint binding helper. Families **compose** it; they do not copy its mechanics. It is generic by construction and must not branch on source ids — the way `runtime/materialize.py` single-sources the bronze write path.
- `janus.strategies.common` is the family-neutral helper layer for retry-delay calculation, checkpoint value comparison, stable request-input keys, raw page naming, immutable mapping normalization, and default storage layout resolution. It must not import concrete strategy families or branch on source ids.
- `janus.strategies.files.formats` is the file-family source of truth for filename resolution, file-format inference, supported Spark handoff formats, archive suffix constants, and safe raw path segment rendering.
- `janus.strategies.catalog.document` contains pure catalog document traversal, node classification, entity batching, graph node and edge shaping, payload hashing, and parse-summary helpers. It has no transport, storage, Spark, checkpoint, or dead-letter side effects; `catalog.core` orchestrates those runtime concerns.

### 4. Optional source hooks

Hooks exist so JANUS can handle the small number of sources that do not fit a family cleanly through configuration alone.

Hooks are allowed to adjust edges of the flow, for example:

- request preparation;
- payload transformation;
- checkpoint parameter generation;
- discovered-file ordering;
- archive-member filtering;
- unusual catalog wrapper handling.

Hooks are not allowed to become a second strategy layer. If a hook is growing into generic behavior shared by more than one source, that behavior belongs in the strategy family instead.

### 5. Raw artifacts

The strategy runtime persists what it extracts before downstream normalization work starts.

Shared runtime components handle this:

- `janus.utils.storage.StorageLayout` resolves raw, bronze, and metadata roots from the active environment.
- `janus.writers.raw.RawArtifactWriter` persists raw payloads deterministically.
- `janus.models.ExtractionResult` records what was extracted and what downstream code should consume next.

The raw zone exists to preserve upstream payloads, not to hide parsing logic.

### 6. Spark-facing read and normalization path

Once a strategy has produced raw artifacts, JANUS uses the shared Spark-side runtime to move toward bronze data:

- `janus.readers.spark.SparkDatasetReader` reads raw artifacts or configured outputs.
- `janus.normalizers.base.BaseNormalizer` adds execution metadata columns such as run id, source id, strategy family, ingestion timestamp, and ingestion date.
- `janus.writers.spark.SparkDatasetWriter` writes structured datasets to the bronze or metadata zone.

This layer stays intentionally generic. It should not flatten one API's odd payload or embed one file source's business rules.

### 7. Validation, checkpoints, and lineage

Operational metadata is a first-class part of the architecture, not a later clean-up step.

- `janus.quality.validators.QualityGate` produces run-scoped validation reports.
- `janus.quality.store.ValidationReportStore` persists those reports in the metadata zone.
- `janus.checkpoints.store.CheckpointStore` manages rerun-safe checkpoint advancement.
- `janus.checkpoints.progress.ExtractionProgressStore` tracks per-page extraction progress so partial runs can resume where they stopped.
- `janus.checkpoints.dead_letters.DeadLetterStore` persists source-scoped dead-letter state so exhausted request inputs or file candidates can be skipped safely on resume and within a configured execution budget.
- `janus.lineage.store.RunObserver` persists run metadata and lineage artifacts.
- `janus.utils.logging` emits structured logs with secret redaction.

The metadata zone is where JANUS explains what happened during a run, not just whether a run returned exit code zero.

## Repository map

The main implementation areas are:

- `conf/`: app config, source definitions, schema contracts, certificates, and environment profiles.
- `src/janus/registry/`: config loading and typed source discovery.
- `src/janus/models/`: source contracts and runtime contracts.
- `src/janus/planner/`: deterministic plan construction and dispatch resolution.
- `src/janus/scripts/`: operational helpers that still run through the planner and shared runtime contracts.
- `src/janus/strategies/`: API, file, and catalog family behavior; `strategies/http/` single-sources the shared HTTP transport, throttle, retry, payload decode, and URL/param/checkpoint binding.
- `src/janus/readers/`, `src/janus/writers/`, `src/janus/normalizers/`: shared runtime I/O and normalization.
- `src/janus/quality/`: reusable validation checks and persisted reports.
- `src/janus/checkpoints/`: rerun-safe data checkpoints, dead-letter state, and per-page extraction progress.
- `src/janus/lineage/`: run metadata and lineage artifacts.
- `src/janus/utils/`: runtime config, storage resolution, Spark bootstrap, and logging.

## Extension boundaries

When adding or changing behavior, use these boundaries:

### Put it in source config when:

- the behavior is declarative;
- the value changes by source, not by code path;
- the existing contract already models it cleanly.

Examples: endpoints, auth mode, pagination type, checkpoint field, output paths, schema mode, quality rules.

### Put it in a strategy family when:

- the behavior belongs to a reusable source pattern;
- more than one source can benefit from it;
- it changes how a family executes, not just one source.

Examples: a new paginator, archive format support, a catalog traversal helper.

### Put it in a source hook when:

- one real source has a narrow mismatch with an existing family;
- the mismatch is too specific for generic strategy code;
- the hook can stay small, isolated, and testable.

### Put it nowhere until the design is clear when:

- the only way forward seems to be a copy of an older source implementation;
- the change would branch on `source_id` inside a generic module;
- the fix would make the registry or shared runtime lie about what it owns.

That pause is part of the architecture, not a delay.

## Current entry point

Today the public CLI in `src/janus/main.py` supports:

- environment loading and runtime-path preparation;
- optional Spark session bootstrap;
- deterministic planning for one configured source;
- full end-to-end execution through extraction, Spark normalization, bronze writing, quality validation, and metadata persistence;
- loading already-preserved raw artifacts into a requested bronze table with `--ingest-raw-to-bronze`;
- resuming a failed extraction run from saved progress or dead-letter state with `--resume`.

The architecture is real end-to-end for API, file, and catalog source families. The CLI is the primary execution interface for all three.

## Anti-patterns to reject

Reject changes that do any of the following:

- copy an old source file or test and edit names until it "works";
- add source-specific conditionals inside generic modules;
- move secrets from environment variables into checked-in YAML;
- bypass `StorageLayout` and hardcode machine-specific paths;
- write raw, bronze, or metadata outputs outside the configured zone layout;
- skip checkpoints, validation, or run metadata because a source is "simple";
- use hooks to smuggle in family-level behavior that should be shared properly.

If onboarding a new source starts feeling like script surgery, the architecture is telling you something is wrong.
