# Queryable observability operations

JANUS exposes run observability through three additive, best-effort surfaces:

- an attempted terminal-row append per run in the Iceberg runs table;
- OpenLineage lifecycle events through a file, HTTP, or disabled transport.
- metrics through a disabled, NDJSON file, Prometheus textfile, or StatsD transport.

The per-run JSON artifacts remain authoritative. The table, events and metrics are indexes over that
evidence, not replacements for it. **A missing runs-table row is not evidence that a run did not
happen.** Emission failures are logged and dropped so observability cannot change a run's status,
artifacts, or exit code.

## Runs table placement and identity

The table is in the same Iceberg catalog as bronze. Its catalog name is derived from
`spark.iceberg.catalog_name` by the shared `derive_pyiceberg_catalog_name` path. The table identifier
defaults to `metadata.runs`; an environment profile may override only that two-part identifier with
`observability.runs_table: <namespace>.<table>`. The runs namespace is independent of bronze's
`default_namespace`.

Spark uses the three-part `<catalog>.<namespace>.<table>` name. With the shipped defaults, the SQL
name is `janus.metadata.runs`. If `JANUS_ICEBERG_CATALOG_NAME` or `observability.runs_table` changes,
replace that name in the published queries with the resolved one.

The sink creates the namespace and table on first emission. It does not start Spark: it derives the
same catalog properties and appends with PyIceberg.

## Table semantics

The table is append-only. The writer does not merge, overwrite, or read before writing. Re-emitting
the same `run_id` therefore creates another physical row; every operator query must select the row
with the greatest `emitted_at` before filtering or aggregating:

```sql
WITH ranked_runs AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY emitted_at DESC
        ) AS row_rank
    FROM janus.metadata.runs
)
SELECT *
FROM ranked_runs
WHERE row_rank = 1;
```

The writer never deletes or expires anything in this table. It grows by one row per successful
terminal emission, plus any duplicate emissions, until `janus maintain` applies the profile's
declared `maintenance.runs_table` policy; see [Retention](#retention) for what that removes and
what it does to query windows.

The only partition field is Iceberg's hidden `day(emitted_at)` transform, named
`emitted_at_day`. `emitted_at` is when the terminal lineage record was emitted; `started_at` is when
the run began. An operator asking for "runs started in September" means a half-open
`started_at >= ... AND started_at < ...` window. Add `emitted_at >=` the same lower bound to prune
older day partitions safely: a terminal emission cannot precede its run start. Do not replace the
`started_at` predicate with an `emitted_at` predicate, because that changes the question to "rows
emitted in September" and moves long-running jobs across the boundary.

The shipped Spark profiles set `spark.sql.session.timeZone` to `UTC`. The example timestamp literals
therefore describe UTC boundaries. Preserve the half-open lower-inclusive, upper-exclusive shape
when changing a window.

## Column register

`NULL` means the event did not produce that fact. It is distinct from zero, `false`, and an empty
list. Timings are UTC `timestamptz` values; `duration_seconds` is fractional seconds; record and
check counts are counts of rows or checks, not bytes.

| Column | Type | Meaning and operational use |
|---|---|---|
| `run_id` | string | JANUS run identity; partition key for the latest-row window in every published query. |
| `source_id` | string | Stable source identity used for grouping and pipeline investigation. |
| `source_name` | string | Human-readable source label returned with operational results. |
| `environment` | string | Runtime environment dimension in the volume query. |
| `strategy_family` | string | Strategy-family dimension in the volume query. |
| `strategy_variant` | string | Strategy-variant dimension in the volume query. |
| `extraction_mode` | string | Full, incremental, or replay-related extraction mode dimension. |
| `source_hook` | string, nullable | Configured hook module; `NULL` when no hook participated. |
| `pipeline_run_id` | string, nullable | Order-14 batch correlation key used by the pipeline-failure query. |
| `pipeline_attempt` | int, nullable | Attempt number within a pipeline run. |
| `trigger` | string, nullable | Pipeline trigger recorded by order-14. |
| `status` | string | Terminal `succeeded` or `failed` outcome. |
| `started_at` | timestamptz | Run-start time and the timestamp operator windows apply to. |
| `ended_at` | timestamptz, nullable | Terminal time; `NULL` only when the source record has no terminal time. |
| `emitted_at` | timestamptz | Terminal emission time, latest-row ordering key, and day-partition source. |
| `duration_seconds` | double, nullable | End-to-end run duration in seconds. |
| `config_version` | string | Existing SHA-256 config version used by the drift query; never recomputed by observability. |
| `source_config_path` | string | Config provenance returned at each version-change point. |
| `records_extracted` | long, nullable | Extracted row count; `NULL` means no count was reported, not zero. |
| `artifact_count` | int | Number of persisted raw artifacts. |
| `records_written` | long, nullable | Sum of reported bronze rows; `NULL` means no bronze output reported a count. |
| `bronze_table_identifier` | string, nullable | First materialized bronze table, used to split volume by destination. |
| `bronze_write_mode` | string, nullable | Bronze write mode used by the run. |
| `checkpoint_field` | string, nullable | Configured checkpoint field. |
| `checkpoint_strategy` | string, nullable | Configured checkpoint strategy. |
| `checkpoint_value` | string, nullable | Candidate value associated with the latest checkpoint decision. |
| `checkpoint_decision` | string, nullable | `advanced`, `retained`, `reused`, `skipped`, or `NULL` when no write was attempted. |
| `checkpoint_advanced` | boolean, nullable | Whether the decision advanced progress; `NULL` means no decision. |
| `quality_outcome` | string | Closed vocabulary: `passed`, `failed`, or `not_run`. |
| `quality_checks_passed` | int, nullable | Passed-check count; `NULL` when validation did not run. |
| `quality_checks_failed` | int, nullable | Failed-check count; `NULL` when validation did not run. |
| `quality_checks_skipped` | int, nullable | Skipped-check count; `NULL` when validation did not run. |
| `quality_failed_checks` | list<string>, nullable | Actionable `<phase>.<name>` check identities; `NULL` for `not_run`, empty for a passing report. |
| `failure_reason` | string, nullable | Failure message, bounded to 2,000 characters; `NULL` for success. |
| `failure_reason_truncated` | boolean, nullable | Whether `failure_reason` was truncated. |
| `failure_reason_length` | int, nullable | Original character count before truncation. |
| `error_type` | string, nullable | Exception class or failure category. |
| `run_metadata_path` | string, nullable | Direct link to the authoritative per-run metadata JSON. |
| `lineage_path` | string, nullable | Link to the authoritative lineage JSON. |
| `checkpoint_history_path` | string, nullable | Link to the checkpoint-history JSON when a checkpoint write was attempted. |
| `validation_report_path` | string, nullable | Link to the validation report when validation ran. |
| `record_schema_version` | int | Row-contract version for schema evolution and mixed-version investigations; v2 adds the three nullable contract-identity columns below, v3 the three contract enforcement columns after them. |
| `schema_version` | string, nullable | SHA-256 of the data contract used for the run; `NULL` when no contract was declared. The drift query detects contract changes. |
| `contract_id` | string, nullable | Contract identity, `<domain>.<table>`; join with `schema_version` and `contract_version` to identify the declaration. `legacy:<path>` appears only in rows written before retired schema files. |
| `contract_version` | string, nullable | Declared semver; `NULL` only in rows from legacy or inferred runs before. |
| `contract_preflight_outcome` | string, nullable | What the session-free preflight found before extraction: `ok`, `will_evolve`, `refused`, `table_missing`, `catalog_unavailable`; `NULL` for runs before. |
| `schema_evolution` | string, nullable | What the writer changed on the bronze table for this run: `none`, `added:<cols>`, `promoted:<col>(<from>-><to>)`, `added:…;promoted:…`, `breaking_replace`; `NULL` when no bronze output was written. |
| `malformed_rows` | long, nullable | Rows Spark could not parse into the contract's types across the run's batches (JSON/CSV handoffs); `0` when counted and clean; `NULL` when not counted (Parquet, or no validation). |
| `duration_budget_seconds` | int, nullable | Duration window declared in `operations.max_duration_seconds` at the time of this run; `NULL` when undeclared. |
| `duration_budget_outcome` | string, nullable | `within`, `exceeded`, or `undeclared`; a breach records evidence and never changes the run's status. `NULL` in older rows. |
| `freshness_max_age_seconds` | int, nullable | Window declared in `operations.freshness.max_age` at the time of this run, carried so SQL can evaluate staleness without the source registry; `NULL` when undeclared. |
| `request_count` | long, nullable | Successful HTTP requests reported during extraction, aggregated in the source-volume query; `NULL` for the file family, so `SUM` and `AVG` skip those rows rather than count them as zero. |
| `retry_count` | long, nullable | HTTP retries during extraction, used to investigate transport instability; `NULL` for the file family, so `SUM` and `AVG` skip those rows rather than count them as zero. |
| `attempt_count` | long, nullable | Extraction attempts reported by the HTTP strategy, used to compare retry pressure over time; `NULL` for the file family, so `SUM` and `AVG` skip those rows rather than count them as zero. |
| `dead_letter_count` | long, nullable | Dead letters reported during extraction, including by the file family; `NULL` only when the strategy did not report a count. Used to investigate skipped work. |

Adding a nullable column is the only routine schema evolution and follows the declared
`record_schema_version` and stable field IDs. Version 2 appends `schema_version`, `contract_id`, and
`contract_version` with stable field IDs 44–46. Existing v1 rows read `NULL` for those fields.
Version 3 appends the contract enforcement columns `contract_preflight_outcome`,
`schema_evolution`, and `malformed_rows` with stable field IDs 47–49; v1 and v2 rows read `NULL`
for them, and a v1 table gains all six in one transaction. Order-23 adds the three operations
fields with stable IDs 50–52 and the four HTTP counters with IDs 53–56, retaining row-contract
version 3; older rows read `NULL` for all seven. The
sink applies all missing declared nullable columns in one Iceberg schema transaction, reloads the
table, and verifies the declared schema before appending. A type change, rename, dropped or extra
field, required addition, or field-ID mismatch is still refused. Removing a column or changing its
type requires an explicit migration.

## Published Spark SQL

These files are the executable interface. They target Spark SQL 4 against the shared catalog and
all de-duplicate on `run_id` before answering the question:

- [failed runs in a window](queries/observability/failed-runs-in-window.sql) — the first AC-2 query;
- [quality breaches by source](queries/observability/quality-breaches-by-source.sql) — the second AC-2 query;
- [runs by source over time with volume and HTTP counters](queries/observability/runs-by-source-over-time.sql);
- [checkpoint decisions by source](queries/observability/checkpoint-decisions-by-source.sql);
- [config and contract version drift](queries/observability/config-version-drift.sql);
- [failed sources in one pipeline run](queries/observability/pipeline-failures.sql);
- [schema drift by source](queries/observability/schema-drift-by-source.sql) — what each run's
  contract decided: preflight outcome, schema evolution, malformed rows, and enforcement errors;
- [source freshness and duration-budget breaches](queries/observability/stale-sources.sql) — the
  latest successful completion, declared windows, exceeded-run count, age, and freshness outcome.

The source-volume query sums `request_count`, `retry_count`, `attempt_count`, and
`dead_letter_count` after de-duplication. File-family runs carry `NULL` for the three HTTP
counters and can report a real dead-letter count. `SUM` and `AVG` skip `NULL` values;
a group containing only unreported values remains `NULL`.

The freshness query reports one row per `source_id` and `environment`. It takes the greatest
`ended_at` among successful runs, so a later failed run cannot make a source fresh. Its outcome
precedence matches `janus status`: no declared window is `undeclared`, then no success is
`never_ran`, then an age **greater than** the window is `stale`, otherwise `fresh`. An age
exactly equal to the window is fresh; fractional seconds and negative ages are preserved.
Filter `outcome IN ('stale', 'never_ran')` when consuming the result as a breach list.

Bind all four named parameters when executing the checked-in SQL through Spark:

```python
from datetime import UTC, date, datetime
from pathlib import Path

sql = Path("docs/queries/observability/stale-sources.sql").read_text(encoding="utf-8")
spark.sql(sql, args={
    "window_start_day": date(2026, 9, 1),
    "window_start": datetime(2026, 9, 1, tzinfo=UTC),
    "window_end": datetime(2026, 10, 1, tzinfo=UTC),
    "evaluated_at": datetime(2026, 9, 18, 12, tzinfo=UTC),
}).show()
```

This query's half-open window applies to **`emitted_at`**, since it asks about available
terminal evidence rather than runs started in a month. Set `window_start_day` to the UTC date
of `window_start`. The timestamp predicates prune the hidden `emitted_at_day` partitions;
that partition field is not a selectable runs-table column. Choose a window covering the
history being compared: if a source's last success predates the window but a failed run is
inside it, SQL reports `never_ran`, while `janus status` scanning the whole metadata directory
finds the older success and evaluates its age. `status --from-runs-table` also reads without
this query window.

The query uses the greatest declared freshness and duration windows recorded in the selected
history. Both status readers use the source config's current freshness declaration. To compare
the three answers, use matching declarations, the same environment, and the same available
history. SQL can report only sources with rows in the window; `janus status` also lists registry
sources with empty history. Missing best-effort table rows cannot prove that a run never happened.

The drift query emits the first observed run for each source and every later change in either
`config_version` or `schema_version`. Compare `contract_id` and `contract_version` to tell a
new declaration from an in-place byte edit; a version bump alone does not enforce compatibility.
Old v1 rows have `NULL` contract fields. See [Data contracts](data-contracts.md) for the identity
and versioning rules.

The schema drift query answers "what did the contract decide, and where did it bite?". Per source
and in start order, it returns every latest run row where the preflight reported `will_evolve`,
`refused` or `catalog_unavailable`; the writer changed the table (`schema_evolution` other than
`none`); malformed rows were counted; the contract bytes changed since the previous run
(`previous_contract_version` beside `contract_version`); or the run stopped on
`ContractViolationError`, `MalformedRowsError`, `ContractPreflightError` or
`SchemaEvolutionRefusedError`. A retried run appears only when its latest row still matches. The
query applies no window; add one with the half-open `started_at` shape and `emitted_at` pruning
described above. Follow `validation_report_path` for the samples behind `malformed_rows`, and see
[Reading a failed preflight or `malformed_rows` check](data-contracts.md#reading-a-failed-preflight-or-malformed_rows-check)
for what each value means. The failure stage is not a column; `error_type` names the enforcement
error, and `failure_stage` is in the run-metadata JSON and the `janusRun` facet.

## OpenLineage operator surface

JANUS pins OpenLineage core specification `2-0-2`; the mapping contract and vendored schema are
documented in [OpenLineage mapping contract](openlineage.md). Lifecycle mapping is fixed:

| Observer hook | OpenLineage `eventType` | Runs-table action |
|---|---|---|
| `start_run` | `START` | None; the table is terminal-only. |
| `record_success` | `COMPLETE` | Append the terminal row. |
| `record_failure` | `FAIL` | Append the terminal row. |

The file transport is the shipped default. It appends one JSON event per line to
`<metadata_dir>/<JANUS_OPENLINEAGE_EVENTS_DIR>/events-YYYY-MM-DD.ndjson`; by default that is
`data/metadata/lineage/openlineage/events-YYYY-MM-DD.ndjson`. The date is the event's UTC
`eventTime`. These files are append-only, and the transport never rotates or removes them beyond the
daily file boundary. `janus maintain --zone lineage` removes day files older than
`maintenance.lineage_events.older_than_days`, never today's file or the most recent one; see
[retention and maintenance](maintenance.md#zone-by-zone).

To opt a local or cluster profile into HTTP without editing tracked YAML, export:

```sh
export JANUS_OPENLINEAGE_TRANSPORT=http
export JANUS_OPENLINEAGE_URL=https://lineage.example.org
export JANUS_OPENLINEAGE_API_KEY=replace-with-runtime-secret
```

`JANUS_OPENLINEAGE_ENDPOINT` optionally overrides the default `api/v1/lineage`, and
`JANUS_OPENLINEAGE_TIMEOUT_SECONDS` optionally overrides the two-second request timeout. An empty
`JANUS_OPENLINEAGE_API_KEY` emits no `Authorization` header. Set
`JANUS_OPENLINEAGE_TRANSPORT=disabled` to emit no events. Supported values are exactly `disabled`,
`file`, and `http`; an unknown value is a profile error.

OpenLineage requires a UUID run id. JANUS deterministically derives UUIDv5 from the readable JANUS
`run_id` under a project-specific namespace. The same JANUS id therefore maps to the same UUID, and
the readable id remains in `run.facets.janusRun.run_id`.

The versioned `janusRun` custom run facet carries the complete lineage projection plus
`started_at`, checkpoint decision and advancement, quality outcome/counts/failed names,
the contract preflight outcome, schema evolution, malformed-row count and failure stage,
metadata-zone evidence paths, and declared input provenance. Standard facets carry documentation,
source-code/config version, job type, error message, and output row counts. A consumer should use
the custom facet for JANUS-specific operational detail and the standard datasets/facets for
cross-platform lineage.

## Metrics

Metrics use the same run projection as the runs table. Terminal delivery follows OpenLineage,
then the runs-table append, then metrics, within one five-second budget. START emits no metrics.
A metrics failure is recorded under `run_event_emission.metrics` in the returned summary and
does not change the aggregate emission outcome, the run's status, its exit code, or its JSON.
Batch metrics come from the finished `PipelineOutcome`; see [Pipeline metrics](orchestration.md#pipeline-metrics).

Every shipped profile, including overlays, declares metrics **disabled** by default. An absent
block also means disabled. An unknown `observability.metrics.transport` is a profile error
(exit `2`), naming `disabled`, `file`, `textfile`, and `statsd`; inside an already-running
observer it logs one warning and degrades to disabled. There is no new runtime dependency,
HTTP endpoint, daemon, alert delivery, or dashboard.

Select a transport through environment variables, without editing tracked YAML:

```sh
export JANUS_METRICS_TRANSPORT=textfile
export JANUS_METRICS_DIR=metrics/textfile
```

`JANUS_METRICS_DIR` is relative to the profile's metadata root (`metadata_dir`), rather than
each source's `outputs.metadata.path`. Absolute paths and paths escaping that root are refused
by the metadata zone's containment rule. The shipped profiles default this variable to
`metrics`; set it explicitly as above for the collector directory used below.

| Transport | Output and intended use |
|---|---|
| `disabled` | A real no-op transport; no metrics files or sockets. The shipped default. |
| `file` | One NDJSON batch per terminal run or finished pipeline, appended to a UTC day file. Mainly a hermetic test and inspection target. |
| `textfile` | Atomic Prometheus exposition files, with the latest run per source and latest pipeline per environment/trigger. For the node exporter textfile collector. |
| `statsd` | Tagged StatsD UDP, without retries. For a receiver that supports DogStatsD-style tags. |

### Catalogue and labels

The [catalogue](../src/janus/observability/metrics/catalogue.py) declares each metric's type,
help, projection and labels once. Missing numeric values are omitted, never converted to
measured zero. Counters are **per-run totals**, reset on each invocation; textfiles retain the
latest value rather than a cumulative process registry. Do not assume that `rate()` of these
textfile counters measures all intervening runs. Use NDJSON history or the runs table for
complete counts.

| Metric | Type | Projection |
|---|---|---|
| `janus_run_duration_seconds` | gauge | `RunRecord.duration_seconds` |
| `janus_run_total` | counter | One terminal record, labelled by its status |
| `janus_records_extracted` | gauge | `RunRecord.records_extracted` |
| `janus_records_written` | gauge | `RunRecord.records_written` |
| `janus_http_requests_total` | counter | `RunRecord.request_count` (successful requests) |
| `janus_http_retries_total` | counter | `RunRecord.retry_count` |
| `janus_dead_letters_total` | counter | `RunRecord.dead_letter_count` |
| `janus_quality_checks_failed` | gauge | `RunRecord.quality_checks_failed` |
| `janus_duration_budget_exceeded_total` | counter | `duration_budget_outcome`: exceeded `1`, within `0`, undeclared omitted |
| `janus_last_run_timestamp_seconds` | gauge | `RunRecord.ended_at`, Unix seconds |
| `janus_last_success_timestamp_seconds` | gauge | `ended_at` for a succeeded run; textfiles preserve it after failure |
| `janus_emission_failures_total` | counter | Failed OpenLineage and runs-table destinations; excludes its own metrics send |
| `janus_pipeline_duration_seconds` | gauge | `PipelineOutcome.totals().duration_seconds` |
| `janus_pipeline_sources_total` | gauge | Finished batch counts with `status="succeeded"`, `"failed"`, or `"skipped"`, including zero |

The six allowed labels are `source_id`, `strategy_family`, `strategy_variant`, `environment`,
`status`, and `trigger`. Run samples use those labels when present. Pipeline duration uses
`environment` and `trigger`; source counts also use `status` (rather than a seventh `outcome`
label). Registry identities and controlled outcomes keep the series count bounded. `run_id`,
pipeline IDs, URLs, failure messages and other unbounded values must never become labels.
The two-way drift test accounts for every numeric record/summary field, including deliberate
omissions such as attempt counts and declared thresholds.

### Prometheus textfile collector

With the exports above and the shipped local metadata root, start the node exporter with:

```sh
node_exporter --collector.textfile.directory=/path/to/janus/data/metadata/metrics/textfile
```

Use the same host directory as JANUS's container mount. JANUS needs write/search permissions
on the directory; the exporter needs read/search permissions on it and every parent and read
permission on the `.prom` files. Staged files are mode `0644` and replaced with `os.replace`, so
the collector sees a complete old or new exposition. Run files are `<sanitized_source_id>.prom`;
pipeline files are `pipeline-<environment>-<trigger>.prom` after identifier sanitization.

Each failed run reads the previous exposition, preserves its last-success timestamp, and
rewrites the other gauges. It retains the original success labels, including `status="succeeded"`,
so the freshness series keeps its identity across failures. A first-ever failure has no success
gauge. Invalid previous exposition is discarded with `metrics_textfile_existing_discarded`;
an I/O error fails that delivery and leaves the previous file in place. Simultaneous writers
are not serialized: atomic replacement guarantees complete files, not newest-run ordering.

The intended alert relationship is:

```promql
time() - janus_last_success_timestamp_seconds > janus_freshness_max_age_seconds
```

**`janus_freshness_max_age_seconds` is not emitted by JANUS.** The freshness window is a
runs-table column and a source declaration, deliberately unmapped in the catalogue. Supply
the threshold in an operator-managed rule, for example a literal for a source declared `24h`:

```promql
time() - janus_last_success_timestamp_seconds{source_id="transparencia__orgaos__siafi__full_refresh"} > 86400
```

If an operator supplies a threshold series instead, its labels must match the success series
or the rule must specify vector matching. The exporter and Prometheus own collection and
alerting. Use `janus status` for `never_ran`: an absent success series does not satisfy the
comparison above.

### StatsD receiver

```sh
export JANUS_METRICS_TRANSPORT=statsd
export JANUS_METRICS_STATSD_HOST=127.0.0.1
export JANUS_METRICS_STATSD_PORT=8125
export JANUS_METRICS_PREFIX=janus
```

The host is required; the default port is `8125` and prefix is `janus`. Hostnames are resolved
once when building the sink. Lines have the form
`<prefix>.<name_without_janus_>:<value>|g|#key:value,key:value` for gauges, and `|c|` for counters:

```text
janus.run_duration_seconds:12.5|g|#environment:local,source_id:transparencia__orgaos__siafi__full_refresh,status:succeeded
```

Tags are sorted; commas, pipes, colons, newlines and carriage returns in values become `_`.
One batch normally fits one datagram; larger batches split at whole-line boundaries within
the 1,432-byte ceiling. A single oversized line fails the delivery. Each send has a timeout
of at most `0.5s`, within the remaining shared budget, and there is no retry or backoff.
**UDP is fire-and-forget:** a successful send to a black hole reports success. It proves
only that the local socket accepted the bytes, not that a receiver ingested them. Detected
failures produce one redacted warning without host/secret-bearing exception messages.

### NDJSON file target

```sh
export JANUS_METRICS_TRANSPORT=file
export JANUS_METRICS_DIR=metrics
```

Files are `<metadata_dir>/metrics/metrics-YYYY-MM-DD.ndjson`, using the record's UTC
`emitted_at` (pipeline completion time for a batch). Each line is one `O_APPEND` single write
and validates against the [pinned NDJSON schema](../tests/fixtures/observability/metrics_ndjson.schema.json).
A shortened example batch has this shape:

```json
{"emitted_at":"2026-10-10T12:00:00+00:00","scope":"run","labels":{"source_id":"transparencia__orgaos__siafi__full_refresh","environment":"local","status":"succeeded"},"samples":[{"name":"janus_run_duration_seconds","type":"gauge","value":12.5},{"name":"janus_run_total","type":"counter","value":1}]}
```

Run labels are shared at batch level. Pipeline batches have `scope="pipeline"`, shared
environment/trigger labels, and a per-sample `labels.status` for the three source counts.
Partial writes report failure; no transport rotates, removes, or retries a metrics file.

## Source freshness with `janus status`

```sh
janus status --environment local
janus status --environment local --source-id transparencia__orgaos__siafi__full_refresh --format json
janus status --environment local --from-runs-table
```

This command loads the registry, including disabled sources, and evaluates each source's
**current** `operations.freshness.max_age`. It starts no Spark session, plans no run, and
does not evaluate duration budgets. By default it reads `runs/*.json` in each source's own
`outputs.metadata.path`, resolved from the registry; there is no shared `runs/` root. It
selects the greatest successful `ended_at` across the whole directory, not file mtime.
Malformed files are counted as `unreadable_run_files` and scanning continues. A `never_ran`
answer with unreadable files can mean that successful history could not be read.
If sources share a metadata path, the command notes that fact and scans all files there for
each source; it does not filter embedded source IDs. Give sources separate metadata paths
when their histories must remain independent.

| Flag | Meaning |
|---|---|
| `--environment`, `--project-root` | Shared profile and registry location selectors |
| `--source-id` | Select one source; repeat for several, including disabled sources |
| `--from-runs-table` | Read latest successful evidence from the profile-selected catalog through PyIceberg, without Spark; useful for deployed profiles and long histories |
| `--format text\|json` | Table by default, or structured ages, thresholds, outcomes and reader diagnostics |
| `--now` | Timezone-aware ISO-8601 evaluation instant, normalized to UTC; defaults to current UTC time |

| Outcome | Meaning | Breach |
|---|---|---|
| `undeclared` | No freshness window in the current source config | No |
| `never_ran` | A window is declared but no successful completion is available | Yes |
| `stale` | Age is strictly greater than the declared window | Yes |
| `fresh` | Age is at most the window, including equality | No |

Exit `0` means no selected source breaches; `1` names at least one stale/never-ran source;
`2` means invalid arguments/configuration or a reader failure. The text and JSON views carry
the same result. Ages retain fractional seconds and may be negative under clock skew.
The metadata reader is authoritative local evidence; the table reader is best-effort evidence
and can lack rows. Both status readers use current config, unlike the query's recorded
windows; see [Published Spark SQL](#published-spark-sql) for history-window differences.
The table reader filters successful rows by source ID, without an environment predicate
or the query's latest-row de-duplication. `--environment` selects the profile/catalog, not
a row filter. Use isolated profile histories, or account for shared environments and
conflicting re-emissions when comparing its result with SQL.

An hourly cron invocation can hand any nonzero outcome to the operator's alert script:

```cron
0 * * * * cd /path/to/janus && janus status --environment local >>/var/log/janus-status.log 2>&1 || /path/to/notify-janus-status
```

The notification script is operator-owned. Distinguish exit `1` (freshness breach) from
exit `2` (the check could not be evaluated) when routing alerts.

## Retention

The table is retained only by an explicit `janus maintain` (the `runs-table` zone, selected by
default), under the profile's `maintenance.runs_table.older_than_days`: 365 days in the shipped
`local` and `cluster` profiles. The full policy, the dry run and the evidence record are in
[retention and maintenance](maintenance.md).

- **Whole partitions only.** One Spark `DELETE … WHERE emitted_at < TIMESTAMP '<cutoff>'`, where
  the cutoff is UTC midnight of today minus `older_than_days`. The predicate is the partition
  source column, so it drops whole `emitted_at_day` partitions and rewrites no surviving file. A
  dry run lists every day it would delete, with its row count. The delete goes through Spark;
  the PyIceberg sink stays append-only.
- **No de-duplication.** Retention removes days, not rows. Retried `run_id` rows stay until their
  day ages out, and selecting the latest row stays the queries' job.
- **Snapshot expiry reuses the bronze floor.** After the delete, the table's own snapshots expire
  with the same cutoff, keeping at least `maintenance.bronze.retain_last`. The runs table has no
  `retain_last` of its own. Snapshots committed after the cutoff are kept too, so more than
  `retain_last` can remain.
- **The row can outlive its JSON.** `run_metadata_path`, `lineage_path`, `checkpoint_history_path`
  and `validation_report_path` point into the metadata zone, which `maintenance.metadata` retains
  separately: 90 days in the shipped profiles, except each source's latest 20 runs. For an older run
  the row can then be the only record left, and those paths no longer resolve. That is by design; see
  [reproducibility](reproducibility.md#retention-and-what-stays-authoritative).

**Retention silently shortens a query's window.** A published query asks for a window; it cannot
know the table has forgotten part of it. After retention, a query over the last 400 days returns
the rows of the last 365 at most, with no error and no warning row. On the order-21 runs-table
fixture, with `older_than_days: 3`, the published September window of
[failed runs in a window](queries/observability/failed-runs-in-window.sql) returned 4 runs instead
of 7 after the apply. Before reading a short or empty answer as "nothing happened", compare the
window's lower bound with the oldest day still present:

```sql
SELECT MIN(partition.emitted_at_day) AS oldest_retained_day
FROM janus.metadata.runs.partitions;
```

## Batch and replay behavior

A `run-all` batch attempts one terminal table row and lifecycle events per attempted source. A
skipped descendant never reaches `RunObserver`, so it emits no row, OpenLineage event, or run metric.
Use the order-14 pipeline summary for skipped descendants; a smaller runs-table row count than the
planned batch is not by itself data loss.
Skips are counted only in the finished pipeline's `janus_pipeline_sources_total{status="skipped"}`.

A raw-to-bronze replay is its own run and emits its own terminal row and events, distinct from the
original extraction. A retry or explicit re-emission that reuses a `run_id` creates duplicate
physical rows; the latest-row pattern makes the published answers logical-run safe.

## Troubleshooting

Start with the structured `run_event_emission_finished` event. Its `outcome`, `reason`, `stage`,
`exception_type`, `table_identifier`, and nested `openlineage` and `metrics` results describe what happened. A
terminal `outcome=failed` or `skipped` does not change the ingestion outcome.

| Symptom | Actual signal and action |
|---|---|
| No runs-table row under `local-hadoop` | `runs_table_append_degraded` reports `reason=hadoop_catalog_unrepresentable`, `step=catalog_properties`, `exception_type=HadoopCatalogUnrepresentableError`. This throwaway Spark-only profile cannot be represented by PyIceberg; use `local` for queryable observability. |
| PyIceberg or PyArrow is missing | `runs_table_append_degraded` reports `reason=pyiceberg_or_pyarrow_unavailable`, `step=dependency_import`, `exception_type=ImportError`. Install the project runtime dependencies. |
| Catalog is unreachable or credentials are refused | The same event reports `reason=catalog_load_failed`, `step=catalog_load`, and the concrete exception type. Check the profile URI, credentials, service, and network. Namespace/table bootstrap failures similarly report `namespace_create_failed`, `table_create_failed`, or `table_load_failed`. |
| Live schema differs from the declaration | `reason=live_schema_does_not_match_declaration` or `live_schema_does_not_match_declaration_after_evolution`, `step=schema_validation`, `exception_type=RunsTableSchemaMismatch`. Compare the live schema with `RUNS_TABLE_SCHEMA`; type changes, renames, dropped fields, extra fields, required additions, and ID mismatches need an explicit migration. |
| Runs table evolution conflicts with a live schema | Check `step=schema_validation` for a non-additive mismatch or `step=schema_evolution` for a failed nullable-column update. Stop retries of a persistent mismatch and plan an explicit migration. |
| Two runs try to add the contract columns at once | Iceberg commits one schema update; the other reports `reason=schema_evolution_failed`, `step=schema_evolution`, and its commit-conflict exception type. The next run reloads the table, sees the columns, and appends without another evolution. |
| Runs table evolution to v3 | The first run against a v2 table adds `contract_preflight_outcome`, `schema_evolution` and `malformed_rows` (field IDs 47–49) in one schema update, then appends a row with `record_schema_version = 3`; a v1 table gains IDs 44–49 in the same single update. Older rows read `NULL` in the new columns. No action is needed; if the update fails, the rows above for `step=schema_evolution` apply. |
| Emission exceeded its budget | `run_event_emission_finished` reports `reason=budget_failed`, `stage=budget`, normally with `exception_type=EmissionTimeoutError`. The total START or terminal fan-out budget is five seconds; investigate a slow catalog or endpoint. |
| Table exists but a query returns no rows | Confirm the catalog name and `observability.runs_table` override, then confirm the half-open `started_at` window. `emitted_at` is the partition-pruning timestamp, not a substitute for the operator's run-start window. |
| One `run_id` has several rows | This is the append-only retry/re-emission consequence. Use the published `ROW_NUMBER() ... PARTITION BY run_id ORDER BY emitted_at DESC` pattern; do not de-duplicate in the writer. |
| A window returns fewer runs than it used to | `janus maintain` has deleted the days older than `maintenance.runs_table.older_than_days`, without any error or warning row. Check the oldest retained day with the query under [Retention](#retention), and the `metadata/maintenance/` records for the apply that removed them. |
| No OpenLineage events by design | The nested result shows `transport=disabled`, `reason=transport_not_configured`, `step=transport_selection`. Select `file` or `http`. This intentionally produces no degradation warning. |
| The transport was misspelled | Profile loading reports `Environment config has an unsupported observability.openlineage.transport` and lists `disabled, file, http`. If encountered inside a run, `openlineage_transport_unavailable` reports `reason=transport_profile_error` and emission degrades to disabled. |
| HTTP receiver gets no auth header | An unset or empty `JANUS_OPENLINEAGE_API_KEY` deliberately produces no header. Export a non-empty token in the runtime environment; never put it in tracked YAML. |
| HTTP returns 401, 500, or another non-2xx | `openlineage_emission_degraded` reports `reason=unexpected_status`, `step=response`, the `status_code`, and a redacted `target`. The event is logged and dropped with no retry. Refused or timed-out requests report `reason=request_failed`, `step=request`, and their exception type. |

When the table and JSON disagree, inspect `run_metadata_path` first. JSON is written before
emission and remains the authoritative record for that run.
