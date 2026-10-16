# Changelog

All notable changes to JANUS are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0,
a minor release may change the CLI, the source YAML schema or the `metadata.runs` schema.

## [0.1.0] - 2026-10-16

First tagged release of JANUS, a metadata-driven Apache Spark framework that ingests Brazilian
federal open data into `raw`, `bronze` and `metadata` zones in a reproducible way. This version is
the reference version for the post-graduation project evaluation.

### Added

#### Source declaration and validation

- Typed YAML source configurations under `conf/sources/`, parsed into frozen models that fail
  closed and report every validation issue at once, with field-path-prefixed messages.
- ODCS data contracts under `conf/contracts/` as the single schema declaration for each source.
  Contract `required` and `primaryKey` fields drive validation, incremental upserts and quality
  checks. Enabled sources must reference an `active` contract.
- A source registry that loads all configurations and contracts as one snapshot and checks
  semantic rules: primary-key columns are required, cross-source column references exist, hooks
  resolve, and partition columns are declared.
- Declared dependencies between sources, resolved at load time into a validated DAG.
- A phase-scope validation policy that limits this release to public, federal sources.

#### Extraction strategies

- Three strategy families with nine variants:
  - `api`: `page_number_api`, `offset_api`, `cursor_api` and `date_window_api`;
  - `catalog`: `metadata_catalog` and `resource_catalog`;
  - `file`: `static_file`, `versioned_file` and `archive_package`.
- API request shaping through static parameters, request inputs (`none`, `date_window`,
  `iceberg_rows` and `combined` Cartesian products) and parameter bindings resolved from the
  current request input or checkpoint.
- Concurrent page fetching for `page_number` and `offset` APIs, with past-end status detection
  and total-count look-ahead that never truncates a run.
- Concurrent file downloads with an ordered window. Results are consumed in candidate order, and
  every commit, hook call and dead-letter decision stays on a single thread.
- File link resolution for direct links, redirects, Nextcloud/WebDAV shares and HTML listings, and
  ZIP archive expansion with member filtering.
- Narrow source hooks for behaviour that configuration cannot express, with IBGE SIDRA and
  Receita Federal hooks included.
- A shared HTTP layer with one transport, one retry loop, one rate-limit throttle and one payload
  decoder. Each source declares which status codes are retryable.
- Resumable extraction: per-input progress records, raw-page rediscovery and a dead-letter store
  for inputs that keep failing.

#### Remote content safety

- An allow-list of `http` and `https` schemes, a redirect policy that strips credentials on any
  origin change and refuses `https` to `http` downgrades, and a cap on redirect hops.
- Payload size caps enforced before and during reads, and archive limits checked before any
  member is opened.
- Discovered links filtered by a same-origin and allowed-hosts policy, including links returned
  by hooks.
- Secret redaction in rendered URLs, logs and error excerpts.

#### Spark normalization and bronze writes

- One bronze materialization path shared by live execution and raw-zone replay, so the same raw
  input produces the same bronze tables.
- A Spark session that starts lazily, only for materialization and quality validation. Extraction
  runs without a session, and a run with nothing to materialize never starts one.
- Apache Iceberg bronze tables written with full refresh, append, merge on keys and
  skip-if-exists modes. Full refreshes use `INSERT OVERWRITE`, so table history and time travel
  are kept across runs.
- Contract enforcement before every bronze commit: required-field and malformed-row checks in
  `strict` or `lenient` mode, and a catalog preflight before extraction that needs no Spark
  session.
- Schema evolution decided from the contract's declared compatibility: added columns and safe
  type promotions are applied in place, and a breaking change replaces the table only on a
  declared major contract version.
- Malformed JSON, JSONL and CSV records captured and counted instead of silently nulled.
- Compaction of API pages into JSONL handoff files before the Spark read, and write partitioning
  sized from the handoff volume. Raw pages are never modified.
- SHA-256 sidecars for every raw artifact, and checksum verification on replay
  (`--verify-checksums`).

#### Command-line interface

- One `janus` command with the verbs `run`, `run-all`, `contract`, `validate`, `list`,
  `dead-letters`, `checkpoint`, `maintain` and `status`.
- `janus validate`, `list`, `dead-letters`, `checkpoint` and `status` never start Spark.
- `janus contract` drafts a contract from a sample of source data for review.
- Operator changes to dead letters and checkpoints write a history record, with operator and
  reason, before they change any state.
- Golden tests pin the output and exit code of every documented invocation.

#### Batch orchestration

- `janus run-all` runs a selection of sources (by tag or domain) together with their upstream
  dependencies, in topological order. A failed source skips its dependents and records why.
- Parallel execution of independent sources in spawned processes, set with
  `run-all --max-parallel` or `orchestration.max_parallel_default`.
- A kernel file lock (`flock`) for each source state directory, held by every command that
  changes source state.
- An atomically written pipeline summary for each batch run.
- An optional Dagster adapter (`pip install janus[dagster]`) that renders one op per source and
  calls the same execution service as the CLI.

#### Observability

- Run metadata, lineage, checkpoint and quality-validation JSON for every run in the metadata
  zone.
- A queryable `metadata.runs` Iceberg table, appended after each run without Spark through
  PyIceberg, with published example queries under `docs/queries/observability/`.
- OpenLineage events with `file`, `http` and `disabled` transports and a custom `janusRun` facet.
- Run and pipeline metrics with `file`, `textfile` (Prometheus) and `statsd` transports, disabled
  by default.
- Declared duration budgets and freshness expectations, reported by `janus status` and optionally
  enforced in batches with `run-all --fail-on-budget-breach`.
- Emission is best-effort and bounded, and it cannot change a run's outcome.

#### Retention and maintenance

- `janus maintain` applies retention declared per environment and per source: Iceberg snapshot
  expiry, metadata and lineage file retention, runs-table retention, and optional raw-zone
  cleanup.
- Dry run by default. Checkpoints, dead letters and progress records are never deleted, and every
  invocation writes a maintenance record.

#### Runtime and environments

- A container image and Docker/Podman Compose setup for local execution, with pinned Spark
  4.0.1, Iceberg 1.10.1 and vendored, checksum-verified JARs.
- Environment profiles for `local` (Iceberg JDBC catalog on SQLite) and `cluster` (Iceberg JDBC
  catalog on PostgreSQL, or a REST catalog), with object-store warehouse locations.
- Credentials read only from environment variables, never from tracked files.
- A hardened container: non-root user and no ports published on `0.0.0.0`.

#### Sources

- Portal da Transparência APIs: contratos, emendas parlamentares, gastos com cartões, licitações,
  órgãos, renúncias fiscais and servidores.
- dados.gov.br catalog: dataset listing and dataset details.
- IBGE SIDRA: PIB Brasil and pineapple production on Pronaf farms.
- INEP Censo Escolar microdata.
- Receita Federal CNPJ: empresas, estabelecimentos, sócios, Simples Nacional and lookup tables.
- All sources ship disabled. Live Portal da Transparência and dados.gov.br runs need an API token.

#### Quality and tooling

- CI on GitHub Actions running Ruff, mypy, unit tests and Spark integration tests. Required
  suites fail CI when skipped or deselected.
- Architecture guardrail tests: module size limits, import layering, and package-wide checks that
  keep source-specific logic, transport code and write paths in their own modules.
- Documentation: [architecture](docs/architecture.md),
  [source onboarding](docs/source-onboarding.md), [strategy patterns](docs/strategy-patterns.md),
  [data contracts](docs/data-contracts.md), [orchestration](docs/orchestration.md),
  [maintenance](docs/maintenance.md), [OpenLineage](docs/openlineage.md),
  [queryable observability](docs/queryable-observability.md) and
  [reproducibility](docs/reproducibility.md).

### Known limitations

- Several Portal da Transparência contracts are still drafts that need review against live data
  before those sources can be enabled.
- Checked-in file sources run downloads with `concurrency: 1` until each one has been tested live
  with more.
- Spark is the only compute engine.
- Execution is batch only, and the policy accepts only public, federal sources.

[0.1.0]: https://github.com/vinalct/janus/releases/tag/v0.1.0
