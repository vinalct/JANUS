# Batch Orchestration Guide

How to run many sources in one dependency-ordered batch, how the pipeline summary is
shaped, and how to diagnose a batch that refuses to start.

The single-source CLI remains the unit of execution. `janus run-all` schedules it; it does
not replace it, and it never reimplements planning, extraction, materialization, quality
validation, or metadata persistence.

- Runnable end-to-end example, including Dagster: [examples/orchestration/](../examples/orchestration/README.md)
- Architecture of the batch layer: [architecture guide](architecture.md#8-batch-orchestration)
- Declaring a dependency on another source: [source onboarding](source-onboarding.md#step-5b-declare-the-producer-of-every-iceberg-input)

## The dependency graph

An `iceberg_rows` request input makes one source read a bronze table another source wrote.
That relationship used to be implicit — an operator had to know to run A before B. It is
now **declared** on every Iceberg leaf and **validated for the whole registry at load**:

```yaml
access:
  request_inputs:
    type: iceberg_rows
    upstream_source_id: transparencia__orgaos__siafi__full_refresh
    namespace: bronze__transparencia
    table_name: orgaos__siafi
    columns:
      orgao_codigo: codigo
```

The declaration is checked against the table the named producer actually writes, derived
with `bronze_table_identifier` — the writer's own identity, never a warehouse lookup. A
declaration naming a source that does not produce the referenced table is rejected, so the
graph cannot drift away from the data.

Edges come only from Iceberg leaves, including every leaf inside a `combined` input. A
`date_window` sub-input creates no edge, because no other source produces a calendar.

## Running a batch

```bash
janus run-all --environment local
```

Everything after `run-all` belongs to the batch command. It runs once and exits; it is not a
daemon and has no scheduler, calendar, or automatic retry loop.

| Option | Meaning |
|---|---|
| `--environment NAME` | Profile under `conf/environments/` (default `local`). |
| `--project-root PATH` | Root used to resolve `conf/` and `data/`. Defaults to `JANUS_PROJECT_ROOT`, then the current project. |
| `--tag TAG` | Select enabled **roots** carrying this tag. Repeatable; repeats are OR-ed. |
| `--domain DOMAIN` | Select enabled **roots** in this domain. Repeatable; repeats are OR-ed. |
| `--pipeline-run-id ID` | Identity correlating every source in this batch. Must be new per standalone run. |
| `--started-at TS` | Timezone-aware ISO-8601 *logical planning* instant, for reproducible ids. |
| `--resume` | Let each source consume its own existing extraction progress. |

`--tag` and `--domain` cannot be combined — that is an argument error, not a silent
intersection. With no filter, every enabled source is a root.

### Selectors pick roots; the graph picks the rest

A filter decides **roots only**. Each selected root drags in its transitive upstreams,
whatever tag or domain those carry, because a consumer scheduled without its producer reads
whatever the table happened to hold last time.

```bash
janus run-all --project-root examples/orchestration --environment example --tag consumer
```

In the shipped example the tag `consumer` belongs only to B. Selecting it expands to A then
B; unrelated C stays out unless it is selected too. The summary records both halves
separately, so an operator can see what was asked for and what was added:

```json
"selection": {
  "requested": { "domains": [], "tags": ["consumer"] },
  "root_ids": ["B"],
  "included_upstream_ids": ["A"],
  "source_ids": ["A", "B"]
}
```

A selection matching nothing is a configuration error with zero executions — never an empty
success. A batch never enables a disabled source: `--include-disabled` stays a single-source
facility and is rejected here by name.

### Deterministic order

Order is the induced subgraph's own topological order, with ties broken lexicographically by
the **original** `source_id` — never by discovery order, directory order, or position in a
grouped YAML file. Permuting the configuration files yields the same order, the same ids and
the same normalized summary.

### Identity and time

Two identities, deliberately separate:

- **Pipeline run id** correlates the batch. Supplied with `--pipeline-run-id`, or derived as
  `pipeline-<environment>-<YYYYMMDDTHHMMSSZ>`. It is validated before it is ever used as a
  path component.
- **Source attempt run id** is derived per source and attempt as
  `<pipeline>-<normalized source>-a<attempt>-<digest>`, where the 10-character digest is
  taken over the *original* source id. Without it `a.b`, `a-b` and `a b` would normalize
  alike and three sources would share one run directory.

`--started-at` sets the **logical planning instant** that pins run ids and run contexts. It
is not the clock durations are measured against, and it does not change any source's
extraction window — see [backfills](#backfills) below.

## The pipeline summary

The complete aggregate is printed on stdout as JSON and persisted under the configured
metadata root:

```
<metadata_dir>/pipelines/<pipeline_run_id>/summary.json
```

The write is atomic (temp file plus `replace`) and refuses to overwrite a finalized summary:
reusing a pipeline id is an error, so a re-execution or backfill takes a new identity. A
fresh invocation also preflights that path *before* any source writes, so a batch does not
spend an hour extracting only to find it cannot record the result.

`schema_version` is `1`. Top-level keys:

| Key | Contents |
|---|---|
| `pipeline` | `pipeline_run_id`, `attempt`, `environment`, `trigger`, `planned_at`, `started_at`, `ended_at` |
| `selection` | `requested` filters, `root_ids`, `included_upstream_ids`, ordered `source_ids` |
| `graph` | the edges actually used: `consumer_id`, `producer_id`, `table`, `input_paths` |
| `config_versions` | `compute_config_version` per source — the same lineage hash, hashed once per file |
| `sources` | one record per planned source (below) |
| `totals` | `selected`, `expanded`, `attempted`, `succeeded`, `failed`, `skipped`, `duration_seconds`, `status` |
| `summary_persistence` | `status` and `path` of this document's own write |

Each source record carries `source_id`, `status`, `attempted`, `selected_directly`,
`upstream_ids`, `config_version`, `run_id`, `timing`, its `attempts` history, and either a
`failure` or a `skip`. An attempted source also carries `evidence`: raw artifacts with
checksums, extraction metadata, materialized outputs, and the per-run metadata paths
(`run_metadata_path`, `lineage_path`, `validation_report_path`, checkpoint paths). Nothing is
fabricated for a node that never ran — a skipped source has an empty `attempts` list and null
timings.

Terminal statuses are `succeeded`, `failed`, and `skipped`. Only a successful upstream
releases a dependent.

Queryable run observability follows attempts, not planned nodes. Every attempted source emits
its own terminal row with `pipeline_run_id`, `pipeline_attempt`, and `trigger` copied from the
run attributes. A skipped source never reaches `RunObserver`, so it emits neither a
`metadata.runs` row nor an OpenLineage event. This is intentional: the pipeline summary is
the authoritative record for skips. A batch with five planned sources and three attempts
therefore produces three run rows, not five.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Every planned source succeeded and the summary was persisted. |
| `1` | The batch ran and something failed: a source failed, or the summary could not be persisted. The aggregate is still printed. |
| `2` | The batch never started: argument error, missing/unreadable config, invalid graph, empty selection, unusable pipeline id, or plan/snapshot mismatch. |
| `130` | Interrupted. Partial evidence is reported on stderr; no summary is finalized. |

Exit `2` is the important one: it means **zero sources executed**. Graph validation runs
before any planning, and every plan is built before the first source runs.

## Failure isolation

A source that fails blocks only its descendants. Independent sources continue and finish.
Every skipped node records why, naming both its direct blockers and the root failures behind
them:

```json
{
  "source_id": "B",
  "status": "skipped",
  "attempted": false,
  "skip": {
    "reason_code": "upstream_failed",
    "direct_blocking_upstream_ids": ["A"],
    "root_failed_source_ids": ["A"]
  }
}
```

All of these are failures, and none lets stale upstream data pass as fresh success: a source
planning error, an `ExecutedRun` returned with `status: "failed"`, a raised exception, and a
compute-session cleanup error after an otherwise successful run.

The runner cannot roll back a successful upstream write when a downstream source later
fails. A partial pipeline is a real outcome: A's new bronze is committed, B never ran, and
the summary says so.

## Resume, retries, and backfills

Three different mechanisms, deliberately kept apart.

**Request retries** stay inside source extraction, governed by
`extraction.retry.retryable_status_codes` as before. Orchestration does not wrap them.

**Whole-source retries belong to the orchestrator.** `run-all` executes once and exits.
Dagster (or any scheduler) decides whether to retry an op; each retry receives a distinct
attempt identity — `…-a2-…` — while keeping the pipeline correlation.

**`--resume`** lets each source consume its own existing extraction progress. It does not
mark an earlier pipeline successful, and it does not suppress a required upstream: a resumed
consumer still waits for its producer to succeed in *this* batch.

### Backfills

A backfill is a normal batch with a **fresh pipeline id** and an explicitly configured
extraction window:

```bash
janus run-all \
  --environment local \
  --tag monthly \
  --pipeline-run-id backfill-2026-08-attempt-1
```

Changing `--started-at` alone does **not** move any extraction window — it moves the logical
planning instant only. The window is `access.request_inputs` (`date_window`, or a
`date_window` member of a `combined` input). There is no partition or date-range CLI flag,
and none is implied.

### Concurrency and overlap

The core runner is sequential. An atomic Iceberg catalog makes table commits safe, but it
does not make every other runtime artifact safe for concurrent writers: checkpoint state and
extraction progress are source-scoped files. Serialize conflicting runs — Dagster pools or
concurrency limits, or an exclusive lock around a cron invocation. Do not rely on the catalog
lock for this.

## The Dagster adapter

Dagster is optional and installed separately:

```bash
pip install 'janus[dagster]'
```

Core JANUS never imports it. `import janus`, `janus.main`, and `janus.orchestration` pull no
orchestrator, and a package-scoped guardrail sweep fails the build if any core module
imports one.

The adapter renders **one op per selected source**, wired by the same validated graph the CLI
uses, and each op delegates to the same `SourceExecutionService` / `SourceExecutor` seam. It
does not touch HTTP transport, the materializer, the normalizer, the writer, quality, or
checkpoints directly.

```python
from pathlib import Path
from janus.adapters.dagster import build_definitions

defs = build_definitions(Path("/srv/janus"), environment="local")
```

A run is planned once, at definition time, and the selected graph travels with the run as an
immutable manifest, so a code reload mid-run cannot silently change the batch. Terminal
aggregation happens outside the job's dependency gating, which is why a node Dagster never
started still appears in the final summary with its skip cause.

Scheduling, whole-run retries, and backfill windows are the orchestrator's job. The example
ships one `ScheduleDefinition` with an explicit timezone and
`DefaultScheduleStatus.STOPPED`; an operator activates it deliberately.

## Troubleshooting

Every diagnostic below is actual output, with the project root replaced by `<PROJECT>`. All
of them exit `2` and execute nothing.

### Dependency cycle

```
Invalid source dependency graph: <PROJECT>/conf/sources
- A → B → D → A: is a source dependency cycle and can never be scheduled (B.access.request_inputs reads bronze.a; D.access.request_inputs reads bronze.b; A.access.request_inputs reads bronze.d)
```

The chain is printed with the leaf that creates each edge. A self-cycle reads the same way
(`A → A`). Break the cycle in configuration; there is no ordering that satisfies it.

### Missing producer

```
- B (<PROJECT>/conf/sources/sources.yaml:sources[0]).access.request_inputs: declares upstream_source_id 'missing', which is missing from the registry: no configured source has that id
```

Upstreams must be **managed**: a table that happens to exist in the warehouse is not a
producer. Add the producing source, or correct the id.

### Disabled producer

```
- B (<PROJECT>/conf/sources/sources.yaml:sources[0]).access.request_inputs: is enabled, but its upstream source 'A' (<PROJECT>/conf/sources/sources.yaml:sources[1]) is disabled; a run never enables an upstream on its behalf
```

Enable the producer deliberately, or disable the consumer. A batch will not enable a source
on your behalf. A fully disabled acyclic subgraph is fine and simply stays out of the run.

### Wrong declared producer

```
- B (<PROJECT>/conf/sources/sources.yaml:sources[2]).access.request_inputs: reads 'bronze.a', but its declared upstream source 'C' (<PROJECT>/conf/sources/sources.yaml:sources[1]) produces 'bronze.c'; the referenced table is produced by A (<PROJECT>/conf/sources/sources.yaml:sources[3])
```

The message names the source that actually writes the referenced table, so the fix is a copy
of the id it printed.

### Ambiguous producer (normalized table collision)

```
- A (<PROJECT>/conf/sources/sources.yaml:sources[0]).outputs.bronze.shared_with: writes 'bronze.shared_target', which B (<PROJECT>/conf/sources/sources.yaml:sources[1]) also writes: an undeclared collision is an ambiguous producer target. Two pipelines may deliberately share one bronze table — a full-refresh rebuild and an incremental delta job for one dataset — but each must then name the other in shared_with
```

Note `Shared-Target` and `shared target` both sanitize to `shared_target`. If the sharing is
deliberate, declare it on both sides with `outputs.bronze.shared_with`; otherwise give one
source its own table.

### Empty selection

```
No enabled source matches tag in (absent). Enabled sources: ['A', 'B', 'C', 'D']. A selection that matches nothing is a configuration problem, not an empty success.
```

### Legacy single-source flags

```
janus run-all: error: --include-disabled cannot be used with run-all; use the existing single-source command instead
```

`--source-id`, `--execute`, `--run-id`, `--bronze-table`, `--ingest-raw-to-bronze`, and
`--with-spark` are rejected the same way, by name.

### A failed quality gate

A source whose quality validation fails returns a failed `ExecutedRun` rather than raising.
The batch records it as `status: "failed"` with the validation report path in its evidence,
and its descendants are skipped with `reason_code: "upstream_failed"`. Read
`validation_report_path` from that source's `evidence.metadata_outputs`.

### Summary write failure

If the aggregate cannot be persisted, the batch still prints the complete in-memory summary
on stdout and exits `1`, with the reason on stderr:

```
Could not persist pipeline summary at <path>: <reason>
```

The summary's own `summary_persistence` block carries `status: "failed"` and the failure
details. Check permissions on the metadata root, and whether the pipeline id was already
used.

## Compatibility

Single-source runs and replay are unchanged. `janus --source-id … --execute`,
`--ingest-raw-to-bronze`, `--include-disabled`, exit codes, run metadata, lineage,
checkpoints, quality reports, and observer behavior keep their contracts. A single-source run
still does **not** run its dependencies — that is what `run-all` is for.

Two intentional changes came with this work:

1. Every `iceberg_rows` leaf now **requires** `upstream_source_id`. Existing consumers must
   declare it; see [migrating an existing consumer](source-onboarding.md#migrating-an-existing-consumer).
2. Registry load now validates the **whole** graph, so a problem in one source's declaration
   fails the load for every entry point, including single-source runs.

Config hashes change where YAML changed, which is expected and visible in `config_versions`.

