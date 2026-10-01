# Queryable observability operations

JANUS exposes run observability through two additive, best-effort surfaces:

- an attempted terminal-row append per run in the Iceberg runs table;
- OpenLineage lifecycle events through a file, HTTP, or disabled transport.

The per-run JSON artifacts remain authoritative. The table and events are indexes over that
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

There is no retention, snapshot expiry, or compaction policy for this table. It grows by one row per
successful terminal emission, plus any duplicate emissions. Storage and metadata growth must be
monitored externally until JANUS defines a maintenance schedule.

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
| `contract_id` | string, nullable | Contract identity such as `<domain>.<table>` or `legacy:<path>`; join with `schema_version` and `contract_version` to identify the declaration. |
| `contract_version` | string, nullable | Declared semver; `NULL` for legacy or inferred runs. |
| `contract_preflight_outcome` | string, nullable | What the session-free preflight found before extraction: `ok`, `will_evolve`, `refused`, `table_missing`, `catalog_unavailable`. |
| `schema_evolution` | string, nullable | What the writer changed on the bronze table for this run: `none`, `added:<cols>`, `promoted:<col>(<from>-><to>)`, `added:…;promoted:…`, `breaking_replace`; `NULL` when no bronze output was written. |
| `malformed_rows` | long, nullable | Rows Spark could not parse into the contract's types across the run's batches (JSON/CSV handoffs); `0` when counted and clean; `NULL` when not counted (Parquet, or no validation). |

Adding a nullable column is the only routine schema evolution and requires a
`record_schema_version` bump. Version 2 appends `schema_version`, `contract_id`, and
`contract_version` with stable field IDs 44–46. Existing v1 rows read `NULL` for those fields.
Version 3 appends the contract enforcement columns `contract_preflight_outcome`,
`schema_evolution`, and `malformed_rows` with stable field IDs 47–49; v1 and v2 rows read `NULL`
for them, and a v1 table gains all six in one transaction. The
sink applies all missing declared nullable columns in one Iceberg schema transaction, reloads the
table, and verifies the declared schema before appending. A type change, rename, dropped or extra
field, required addition, or field-ID mismatch is still refused. Removing a column or changing its
type requires an explicit migration.

## Published Spark SQL

These files are the executable interface. They target Spark SQL 4 against the shared catalog and
all de-duplicate on `run_id` before answering the question:

- [failed runs in a window](queries/observability/failed-runs-in-window.sql) — the first AC-2 query;
- [quality breaches by source](queries/observability/quality-breaches-by-source.sql) — the second AC-2 query;
- [runs by source over time with volume](queries/observability/runs-by-source-over-time.sql);
- [checkpoint decisions by source](queries/observability/checkpoint-decisions-by-source.sql);
- [config and contract version drift](queries/observability/config-version-drift.sql);
- [failed sources in one pipeline run](queries/observability/pipeline-failures.sql);
- [schema drift by source](queries/observability/schema-drift-by-source.sql) — what each run's
  contract decided: preflight outcome, schema evolution, malformed rows, and enforcement errors.

The drift query emits the first observed run for each source and every later change in either
`config_version` or `schema_version`. Compare `contract_id` and `contract_version` to tell a
new declaration from an in-place byte edit; a version bump alone does not enforce compatibility.
Old v1 rows have `NULL` contract fields. See [Data contracts](data-contracts.md) for the identity
and versioning rules.

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
`eventTime`. These files are append-only and unbounded: JANUS defines no rotation beyond the daily
file boundary and no expiry.

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
metadata-zone evidence paths, and declared input provenance. Standard facets carry documentation,
source-code/config version, job type, error message, and output row counts. A consumer should use
the custom facet for JANUS-specific operational detail and the standard datasets/facets for
cross-platform lineage.

## Batch and replay behavior

A `run-all` batch attempts one terminal table row and lifecycle events per attempted source. A
skipped descendant never reaches `RunObserver`, so it emits neither a row nor an OpenLineage event.
Use the order-14 pipeline summary for skipped descendants; a smaller runs-table row count than the
planned batch is not by itself data loss.

A raw-to-bronze replay is its own run and emits its own terminal row and events, distinct from the
original extraction. A retry or explicit re-emission that reuses a `run_id` creates duplicate
physical rows; the latest-row pattern makes the published answers logical-run safe.

## Troubleshooting

Start with the structured `run_event_emission_finished` event. Its `outcome`, `reason`, `stage`,
`exception_type`, `table_identifier`, and nested `openlineage` result describe what happened. A
terminal `outcome=failed` or `skipped` does not change the ingestion outcome.

| Symptom | Actual signal and action |
|---|---|
| No runs-table row under `local-hadoop` | `runs_table_append_degraded` reports `reason=hadoop_catalog_unrepresentable`, `step=catalog_properties`, `exception_type=HadoopCatalogUnrepresentableError`. This throwaway Spark-only profile cannot be represented by PyIceberg; use `local` for queryable observability. |
| PyIceberg or PyArrow is missing | `runs_table_append_degraded` reports `reason=pyiceberg_or_pyarrow_unavailable`, `step=dependency_import`, `exception_type=ImportError`. Install the project runtime dependencies. |
| Catalog is unreachable or credentials are refused | The same event reports `reason=catalog_load_failed`, `step=catalog_load`, and the concrete exception type. Check the profile URI, credentials, service, and network. Namespace/table bootstrap failures similarly report `namespace_create_failed`, `table_create_failed`, or `table_load_failed`. |
| Live schema differs from the declaration | `reason=live_schema_does_not_match_declaration` or `live_schema_does_not_match_declaration_after_evolution`, `step=schema_validation`, `exception_type=RunsTableSchemaMismatch`. Compare the live schema with `RUNS_TABLE_SCHEMA`; type changes, renames, dropped fields, extra fields, required additions, and ID mismatches need an explicit migration. |
| Runs table evolution conflicts with a live schema | Check `step=schema_validation` for a non-additive mismatch or `step=schema_evolution` for a failed nullable-column update. Stop retries of a persistent mismatch and plan an explicit migration. |
| Two runs try to add the contract columns at once | Iceberg commits one schema update; the other reports `reason=schema_evolution_failed`, `step=schema_evolution`, and its commit-conflict exception type. The next run reloads the table, sees the columns, and appends without another evolution. |
| Emission exceeded its budget | `run_event_emission_finished` reports `reason=budget_failed`, `stage=budget`, normally with `exception_type=EmissionTimeoutError`. The total START or terminal fan-out budget is five seconds; investigate a slow catalog or endpoint. |
| Table exists but a query returns no rows | Confirm the catalog name and `observability.runs_table` override, then confirm the half-open `started_at` window. `emitted_at` is the partition-pruning timestamp, not a substitute for the operator's run-start window. |
| One `run_id` has several rows | This is the append-only retry/re-emission consequence. Use the published `ROW_NUMBER() ... PARTITION BY run_id ORDER BY emitted_at DESC` pattern; do not de-duplicate in the writer. |
| No OpenLineage events by design | The nested result shows `transport=disabled`, `reason=transport_not_configured`, `step=transport_selection`. Select `file` or `http`. This intentionally produces no degradation warning. |
| The transport was misspelled | Profile loading reports `Environment config has an unsupported observability.openlineage.transport` and lists `disabled, file, http`. If encountered inside a run, `openlineage_transport_unavailable` reports `reason=transport_profile_error` and emission degrades to disabled. |
| HTTP receiver gets no auth header | An unset or empty `JANUS_OPENLINEAGE_API_KEY` deliberately produces no header. Export a non-empty token in the runtime environment; never put it in tracked YAML. |
| HTTP returns 401, 500, or another non-2xx | `openlineage_emission_degraded` reports `reason=unexpected_status`, `step=response`, the `status_code`, and a redacted `target`. The event is logged and dropped with no retry. Refused or timed-out requests report `reason=request_failed`, `step=request`, and their exception type. |

When the table and JSON disagree, inspect `run_metadata_path` first. JSON is written before
emission and remains the authoritative record for that run.
