# Batch Orchestration Guide

How to run many sources in one dependency-ordered batch, how the pipeline summary is
shaped, and how to diagnose a batch that refuses to start.

The single-source CLI remains the unit of execution. `janus run-all` schedules it; it does
not replace it, and it never reimplements planning, extraction, materialization, quality
validation, or metadata persistence.

- Runnable end-to-end example, including Dagster: [examples/orchestration/](../examples/orchestration/README.md)
- Architecture of the batch layer: [architecture guide](architecture.md#9-batch-orchestration)
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

To see the graph the registry resolved, without running anything:

```bash
janus list --graph
```

It prints the topological order `run-all` uses, then every edge with the table and the input
path that create it. `janus list` prints one row per source with its upstream ids, and
`janus validate` loads the registry and plans every source.

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
| `--max-parallel N` | Run at most N source attempts at once. Positive integer; overrides `orchestration.max_parallel_default`, else defaults to 1. |
| `--fail-on-budget-breach` | Treat a successful run with an exceeded duration budget as a failed batch attempt, skipping its dependents. Off by default. |

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
| `pipeline` | `pipeline_run_id`, `attempt`, `environment`, `trigger`, `planned_at`, `started_at`, `ended_at`; `max_parallel` only when N > 1 |
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

Queryable run observability follows executed sources, not planned nodes. Each source that
reaches `RunObserver` projects its terminal row with `pipeline_run_id`, `pipeline_attempt`, and
`trigger` copied from the run attributes. Lock and snapshot-drift refusals have empty attempt
evidence and create no run metadata or terminal row. A skipped source never reaches `RunObserver`, so it emits neither a
`metadata.runs` row nor an OpenLineage event. This is intentional: the pipeline summary is
the authoritative record for skips. A batch with five planned sources and three attempts
therefore projects three run rows when all three attempts reach the observer. Delivery remains
best-effort; the summary also records attempts refused before execution.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Every planned source succeeded and the summary was persisted. |
| `1` | The batch ran and something failed: a source failed, or the summary could not be persisted. The aggregate is still printed. |
| `2` | The batch never started: argument error, missing/unreadable config, invalid `orchestration` profile block, invalid graph, empty selection, unusable pipeline id, or plan/snapshot mismatch. |
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
compute-session cleanup error after an otherwise successful run. A data-contract refusal is a
returned failure like any other: a refused preflight, a batch refused by its pre-write check or
malformed-row count, and a refused schema evolution each carry a `failure_stage`
(`contract_preflight`, `contract_check`, `malformed_rows`, `schema_evolution`) and skip their
dependents.

### Duration budget policy

By default, a run that exceeds `operations.max_duration_seconds` remains a successful
batch attempt and its dependents run. The budget is recorded as a scheduling signal.
Opt in when a breach should stop dependent work:

```bash
janus run-all --environment local --fail-on-budget-breach
```

The batch converts only a successful attempt whose recorded `duration_budget_outcome`
is `exceeded`. Its failure has `phase: "budget"`, `error_type: "RunDurationBudgetExceeded"`,
and a reason naming the declared budget in seconds. Dependents are skipped with the
usual direct and root causes; independent sources continue. The aggregate is still
printed and persisted, and the existing failed-batch exit code is `1`.

The run's own metadata and lineage JSON still say `status: "succeeded"`: bronze was
written, quality passed, and any checkpoint advance remains valid. Only the batch's
view of that run changes. Evidence is preserved, the observer is not called again,
and no second runs-table row or lineage event is emitted. Runs with `within` or
`undeclared` outcomes are unaffected, and an already failed run retains its original
failure phase. Single-source `janus run` rejects `--fail-on-budget-breach` by name.

The policy travels on `BatchPlanRequest.fail_on_budget_breach` (default `False`).
Dagster uses that same request through the shared attempt seam; an injected
`DagsterAdapterServices.batch_planner` can supply requests with the policy enabled.

### Failure phases

A failed source's `failure.phase` identifies where the attempt stopped. Phases are
documented strings, not a validated enum:

| Phase | Meaning | First thing to check |
|---|---|---|
| `planning` | The source could not be planned. | Source configuration and hook binding. |
| `lock` | Another process holds the resolved metadata directory's lock, or locking is unsupported; nothing ran. | Holder pid, hostname, run id and command in the reason; local POSIX filesystem support. |
| `snapshot_drift` | A worker's fresh registry, source or contract no longer matches the parent's plan; nothing ran. | YAML/contract edits or graph/selection changes since planning; deploy one snapshot and start a fresh batch. |
| `execution` | Source execution failed. | The source's failure reason and evidence paths. |
| `cleanup` | Session cleanup failed after execution. | Spark shutdown diagnostics. |
| `result_translation` | The executed result could not be converted into an attempt. | Result shape at the execution-service boundary. |
| `worker_crash` | A worker exited without a valid result; dependents are skipped and attempt evidence is empty. | Exit code, signal or envelope error in the reason, worker logs and host memory. Partial source files may remain. |
| `budget` | The data is fine and the schedule's duration assumption is not. | The declared duration budget and measured run duration. |

`lock`, `snapshot_drift` and `worker_crash` come from
[concurrency and overlap](#concurrency-and-overlap). An interrupted parallel batch records
none of them: its unfinished nodes stay pending (see [parallel batches](#parallel-batches)).

The runner cannot roll back a successful upstream write when a downstream source later
fails. A partial pipeline is a real outcome: A's new bronze is committed, B never ran, and
the summary says so.

### Pipeline metrics

After persisting the finished pipeline summary, the batch projects `PipelineOutcome` through
the profile-selected metrics sink. The adapter uses the same seam, and collecting an already
finished result emits nothing again. Delivery has the same five-second, non-raising guard as
run metrics; failures remain logs and cannot change the summary or batch exit code.

`janus_pipeline_duration_seconds` is the finished batch duration. The gauge
`janus_pipeline_sources_total{status="succeeded|failed|skipped"}` emits three separate samples,
including measured zeros. Both metrics carry `environment` and `trigger`; the counts also
carry `status`, staying inside the six-label allow-list. With `--fail-on-budget-breach`, the
pipeline counts the converted attempt as failed while its run metrics still describe a
succeeded run. The two views answer different questions.

A skipped node has no run record and emits no per-run metric, table row, or OpenLineage event.
Skips are counted at pipeline level only. Metrics ship disabled; collector setup and the full
catalogue are in the [observability guide](queryable-observability.md#metrics).

## Resume, retries, and backfills

Three different mechanisms, deliberately kept apart.

**Request retries** stay inside source extraction, governed by
`extraction.retry.retryable_status_codes` as before. Orchestration does not wrap them.

**Whole-source retries belong to the orchestrator.** `run-all` executes once and exits.
Dagster (or any scheduler) decides whether to retry an op; each retry receives a distinct
attempt identity — `…-a2-…` — while keeping the pipeline correlation.

**`--resume`** lets each source consume its own existing extraction progress. It does not
mark an earlier pipeline successful, and it does not suppress a required upstream: a resumed
consumer still waits for its producer to succeed in *this* batch. Under `--resume`, an item a
previous run dead-lettered stays skipped until it is released with `janus dead-letters release`.
A run without `--resume` starts over: it clears the dead-letter state and retries every item.

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

## Concurrency and overlap

There are two kinds of overlap, and each has one protection. Overlap nobody planned (a cron
tick, a second operator, another Dagster run) meets the kernel-held source-state lock, which
refuses after its configured wait. Overlap a parallel batch plans for itself is ordered by
the batch's ready queue, so the batch does not contend with itself. An atomic Iceberg catalog still protects table
commits only. It does not serialise the source-scoped state files.

### Source-state locking

`runtime/locking.py::SourceStateLock` holds an exclusive, advisory `flock` on
`<resolved source metadata directory>/.lock`. The key is the resolved directory: two source
ids sharing that directory share the lock; distinct nested directories have distinct locks.
It protects checkpoint state/history, extraction progress, dead letters, and run evidence
through the whole mutation. Live execution and replay acquire it before observer creation
and retain it through terminal persistence and emission.

The seven registered entry points in `LOCKED_ENTRY_POINTS` are live `run --execute`, raw
replay `run --ingest-raw-to-bronze`, `dead-letters release`, `dead-letters replay --execute`,
`checkpoint set`, `checkpoint clear`, and `maintain` for source-scoped metadata/raw retention.
Replay holds one reentrant lock across release and resumed execution. Read-only `validate`,
`list`, `status`, `checkpoint show`, `dead-letters list`, replay previews, and single-source
planning/bootstrap do not acquire it: atomic state writes let readers see a complete version.

| Entry point | Refused lock |
|---|---|
| Live execution or raw replay | Exit **2**, holder on stderr, no stdout summary or run metadata. |
| Batch/adapter node | Failed attempt with `failure.phase = "lock"`, empty evidence; dependents skipped with their recorded causes. A finished failed batch exits **1**. |
| Mutating operator verb | Exit **2**, holder on stderr and **"Nothing was changed."**; no state/history mutation. |
| Metadata/raw maintenance | Skip the source as `source_locked`, leave it untouched, and continue its peers; contention alone exits **0**. |

The holder record includes pid, hostname, source id, run id, pipeline run id, acquisition
time and command. Logs use `source_lock_acquired`, `source_lock_refused`, and
`source_lock_released`. The kernel releases the lock on process death, including `SIGKILL`.
The file stays in place; normal release truncates its diagnostics, while a crash may leave
old diagnostics. Its existence or contents never prove that the lock is held. Never delete,
rename or replace it to clear contention.

Profiles can set a bounded polling wait:

```yaml
orchestration:
  max_parallel_default: 1
  lock_wait_seconds: 0
```

Both are YAML integers (`max_parallel_default >= 1`, `lock_wait_seconds >= 0`); quoted
strings, booleans, floats and unknown keys are refused. An absent block uses the values
above. `lock_wait_seconds` reaches every mutating entry point, including adapter attempts;
zero means fail fast, and a positive value waits at most that many seconds for external
contention. This is a single-host, local-filesystem guarantee. NFS, cross-host locking and
object-store metadata zones are unsupported; unavailable POSIX locking refuses execution.

### Cron overlap

Two single-source schedules that overlap no longer need an external lock for source state:
the second invocation refuses and exits **2**, which cron can distinguish from **1** (ran
and failed). A wrapper can treat **2** as "skip this tick", retaining stderr for diagnosis:

```sh
#!/bin/sh
# Save as /srv/janus/run-source-tick.sh; pass an enabled source id as the first argument.
cd /srv/janus || exit 1
janus run --environment local --source-id "$1" --execute
status=$?
case "$status" in
  2) printf '%s\n' 'JANUS did not start; skipping this tick (see stderr).' >&2; exit 0 ;;
  *) exit "$status" ;;
esac
```

```cron
0 * * * * /srv/janus/run-source-tick.sh my_enabled_source >>/var/log/janus-source.log 2>&1
```

Exit 2 also covers configuration/argument refusals, so retain and inspect stderr. For
`run-all`, contention is a node failure and the finished aggregate exits 1; it is not a
whole-batch skip. Each invocation needs a fresh pipeline id. Maintenance of shared lineage
files or orphan data has its separate [scheduling constraints](maintenance.md#scheduling-with-cron-or-dagster).

### Parallel batches

```bash
janus run-all --environment local --max-parallel 3
```

The flag overrides `orchestration.max_parallel_default`; both supported profiles ship 1.
At N=1 the command uses the existing `BatchExecutor`, with byte-identical stdout, summary
and nonvolatile metadata goldens. At N>1, `runtime/parallel.py::ParallelBatchExecutor`
uses one spawned process per attempt through `execute_source_attempt`, never threads or
forked JVMs. The pure `orchestration/scheduling.py` ready queue dispatches eligible nodes
in plan order. A node waits until **all** upstreams are terminal; any failed/skipped upstream
then skips it with the same full blocker/root causes as N=1. Nodes sharing a resolved
metadata directory never overlap inside this batch; independent lock domains can overlap.

The parent starts no Spark session. Worker startup, registry loading, planning, locking and
provider construction also start none. Ordinary extraction finishes before that source
acquires Spark; empty handoffs stay JVM-free. The existing scoped `iceberg_rows` lookup is
the only pre-download acquisition and stops before the first request. Workers reload the
registry, plan through `BatchPlanner`, and verify their `config_version`, contract digest,
run id and upstreams against the parent. Refusal is `snapshot_drift`, with no execution.

Results return as JSON summaries and aggregate in plan order, independent of completion
order. `schema_version` stays 1; `pipeline.max_parallel` appears only above 1. A worker
that dies without a valid result becomes `worker_crash`, with its exit diagnostics and
empty attempt evidence. On parent `SIGINT` or `SIGTERM`, dispatch stops, live workers get
`SIGTERM`, and completed results are collected for a bounded grace (default 30 seconds).
Survivors get `SIGKILL`; a second interrupt skips the grace. Exit is **130**, with completed
and pending counts and no finalized summary. Interrupted attempts without results remain
pending, along with undispatched nodes; they are not recorded as worker crashes. The runner
restores its previous signal handlers after dispatch and shutdown. A caller embedding it
outside the main thread keeps its own handlers and receives one
`parallel_signal_handlers_not_installed` diagnostic.

### Operating guidance

Budget peak memory at approximately **N × `spark.driver.memory` plus the parent**, plus each
driver's JVM native and Python overhead. More exactly, it follows the number of sources
*running at once*: at most N, and never more than the graph's width at that moment. Each
running source may start its own driver. The `local` profile's `spark.driver.memory` is
8g (`JANUS_SPARK_DRIVER_MEMORY`), so choose N for the host's available memory. Parallel
execution does not necessarily make a small batch faster.

Recorded local measurements (2026-10-11/12, pinned Python 3.13.12 / Spark 4.0.1):

| Measurement | Result |
|---|---|
| TASK-01 spawn + registry + ten-source batch planning, 20 samples | Median **0.298295 s**, p95 **0.312488 s** per worker. |
| TASK-15 A → B plus C, 1g drivers, N=3 | A **16:37:54.675559–16:38:04.671807**, C **16:37:54.672859–16:38:04.567663**, B **16:38:05.080838–16:38:13.227238**, America/Sao_Paulo: A/C overlap, B follows A. |
| Same TASK-15 success capture, N=1 / N=3 | **13.794403 s / 18.952177 s** total; normalized summaries and bronze business rows match. No `SQLITE_BUSY` observed. |
| TASK-15's parity case re-run three times, peak PSS of the CLI process tree, sampled every 0.1 s | N=1: **959.6–1170.4 MiB**, one JVM, inside the CLI process (parent ≈ 241 MiB RSS). N=3: **1567.9–1851.1 MiB**, two concurrent JVMs (A and C; the graph is two wide), parent ≈ 44 MiB RSS. One driver JVM peaked at **707.0–979.5 MiB** RSS with a 1g heap. |
| TASK-19 full local CI, sampled Python + Java RSS every 0.5 s | Peak aggregate **5712.9 MiB**, peak Java **4955.6 MiB**. This is CI-wide RSS, not a per-batch or hosted-runner measurement. |

The parity-case memory row is a small fixture's per-driver floor, not a workload ceiling: a source that
caches or shuffles large frames grows toward its heap. The full artifacts and measurement
scope are in the order-24 evidence record: TASK-01's spawn-cost experiment, TASK-15's parity
captures, TASK-20's peak-memory sampling and TASK-19's gate run.

Pre-seed the shared Ivy cache before starting parallel workers:

```bash
make seed-ivy
```

TASK-01 ran **30 concurrent cache-hit trials** (20 with two JVMs, 10 with four), with
**80 successful workers, zero failures and unchanged resolution reports**. The decision
is to keep `spark.jars.packages`; no worker-only `spark.jars` override was needed. This
proof depends on the pinned, pre-seeded jars: a cache miss would make concurrent drivers
resolve over the network into one Ivy directory, which the experiment did not test.

The `local` JDBC SQLite URI uses WAL and **`busy_timeout=30000`**. Keep **N ≤ 4** on
that profile; this is operating guidance, not an enforced CLI cap or a throughput guarantee.
For N > 4 use the `cluster` profile with Postgres and size that host's drivers deliberately.
TASK-15 observed no `SQLITE_BUSY` at N=3; it does not establish a higher SQLite ceiling.
Provision a new local catalog before its first parallel batch: run once at N=1, so the JDBC
catalog tables and every bronze namespace exist. TASK-13's early captures let workers
bootstrap a fresh SQLite catalog concurrently and hit `SQLITE_BUSY_SNAPSHOT` and
namespace-creation races. Every parallel proof since then ran on a provisioned catalog, and
JANUS does not serialise first-time catalog bootstrap.

The CLI warns once when N exceeds **`os.process_cpu_count()`**, which respects CPU affinity
(falling back to `os.cpu_count()`, then 1). It emits one structured warning and one stderr
line, and leaves N unchanged.

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

The default executor runs in one process and keeps the `janus/concurrency: single-process`
tag. Pass `executor="multiprocess", max_concurrent=3` to `build_definitions`, `build_job`,
or `build_dagster_adapter` to run independent steps in spawned processes, tagged
`janus/concurrency: multiprocess:3`. Multiprocess execution requires an explicit positive
integer limit: each running source may start its own Spark driver. The adapter's
`execute_in_process` convenience always runs in the calling process; use a reconstructable
job and a persistent Dagster instance with `dagster.execute_job` for multiprocess execution.

Each source holds its source-state lock through the shared executor. Overlap with another
Dagster run or cron invocation records a failed attempt with `phase: "lock"` and skips its
dependents. An opt-in Dagster `RetryPolicy` can retry that failure after its configured
delay, allowing an operator to wait for the competing run to finish.

A run is planned once, at definition time, and the selected graph travels with the run as an
immutable manifest, so a code reload mid-run cannot silently change the batch. Terminal
aggregation happens outside the job's dependency gating, which is why a node Dagster never
started still appears in the final summary with its skip cause.

Scheduling, whole-run retries, and backfill windows are the orchestrator's job. The example
ships one `ScheduleDefinition` with an explicit timezone and
`DefaultScheduleStatus.STOPPED`; an operator activates it deliberately.

## Troubleshooting

Every diagnostic below is actual output, with the project root replaced by `<PROJECT>`. The
graph, selection and flag refusals exit `2` and execute nothing. The entries from
[a failed quality gate](#a-failed-quality-gate) on start from a batch that ran.

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
`--with-spark` are rejected the same way, by name. `--verify-checksums` belongs to single-source
replay too, and `run-all` refuses it as an unknown option:

```
janus run-all: error: unrecognized arguments: --verify-checksums
```

### A failed quality gate

A source whose quality validation fails returns a failed `ExecutedRun` rather than raising.
The batch records it as `status: "failed"` with the validation report path in its evidence,
and its descendants are skipped with `reason_code: "upstream_failed"`. Read
`validation_report_path` from that source's `evidence.metadata_outputs`.

### A refused contract preflight

Before extracting, every source compares its live bronze table with its data contract. When a
`strict` source's table differs outside the contract's compatibility mode, or its catalog cannot
be read, the source fails before it starts:

- The summary shows the source `failed` with `failure.error_type: "ContractPreflightError"`; each
  dependent is `skipped` with `reason_code: "upstream_failed"` and the source in
  `direct_blocking_upstream_ids`.
- Nothing was extracted: the attempt sent no request, wrote no raw artifact, and started no Spark
  session. Its bronze table is untouched.
- The attempt's `evidence.metadata_outputs.run_metadata_path` links the run-metadata JSON. Its
  `run_attributes.contract_preflight_outcome` is `refused` (a schema difference) or
  `catalog_unavailable` (the catalog raised or exceeded the five-second budget), `failure_stage`
  is `contract_preflight`, and `failure_reason` lists each refused column and why, for example
  `amount: retyped`.

There are two fixes, both in the contract. For a difference the table can absorb — a new nullable
column, or a promotion under `janus.compatibility: backward` — declare it in the contract and
rerun; the preflight reports `will_evolve`. For a breaking change, bump the contract's MAJOR version
and run the source as a full refresh, which replaces the table; see
[the breaking-change workflow](data-contracts.md#the-breaking-change-workflow). A `lenient` source
logs `contract_preflight_warning` instead and proceeds; a refused difference then fails at the
write with `failure_stage: "schema_evolution"`, after extraction has run.

### A malformed-row refusal

For `failure_stage: "malformed_rows"`, open the source's validation report and inspect
`data.malformed_rows.details`: `count`, `threshold`, `batch_index`, and up to five bounded,
scrubbed `samples`. A refused batch commits no bronze snapshot. JSON pages read with `multiLine`
can flag every row when one record drifts; JSONL and CSV identify individual lines. A CSV header
mismatch can flag every data row. In a batch, the source fails and its descendants are skipped
with `reason_code: "upstream_failed"`.

### A source failed and its dependents were skipped

The summary shows the source `failed` and each dependent `skipped` with
`reason_code: "upstream_failed"`. When one item exhausted its retries — a request input, or a file
candidate — the run recorded it as a dead letter before it gave up. Start there:

```
$ janus dead-letters list --source-id semantics_clean_producer
semantics_clean_producer - 1 dead letter(s), recorded by run local-20260918-060000-semantics_clean_producer
  state    data/metadata/semantics/clean_producer/dead_letters/current.json
  updated  2026-09-18T06:04:11+00:00

  item_key                                       type           recorded_at
  window_end=2026-09-02|window_start=2026-09-02  request_input  2026-09-18T06:04:11+00:00
      error                RuntimeError: API request failed with status 503 for https://example.invalid/reference?page=1&window_start=2026-09-02: {"message": "upstream maintenance"}
      request_input_count  2
      request_input_index  2
      request_url          https://example.invalid/reference?window_start=2026-09-02
```

If it reports `no dead letters recorded`, the failure was not an item's: read the source's
`failure` in the summary and the entries above.

Once the cause is gone upstream — a token renewed, an endpoint back, a window republished —
release the entries with a reason, then resume:

```
$ janus dead-letters release --source-id semantics_clean_producer --item-key 'window_end=2026-09-02|window_start=2026-09-02' --reason 'upstream maintenance over'
semantics_clean_producer - released 1 dead letter(s); 0 remain
  released       window_end=2026-09-02|window_start=2026-09-02  (request_input, RuntimeError)
  remaining      -
  operator       ops-tester
  reason         upstream maintenance over
  history        data/metadata/semantics/clean_producer/dead_letters/history/20261003T102235Z-local-20260918-060000-semantics-clean-producer.json
  state          data/metadata/semantics/clean_producer/dead_letters/current.json (deleted: nothing remains)

The next resuming run retries the released item(s) and keeps skipping the rest:
  janus --environment local --source-id semantics_clean_producer --execute --resume
```

Then re-run. `janus run-all --environment local --resume` resumes the failed source and runs the
dependents it skipped, in order. `janus dead-letters replay … --reason R` without `--execute`
prints what that resume would retry and writes nothing; with `--execute` it releases and runs the
one source in a single step.

What these do **not** do:

- `release` re-runs nothing. It removes the named entries from the state and writes the history
  record; the next `--resume` run is what retries them.
- A release does not fix the cause. An item that fails again is dead-lettered again.
- Only a `--resume` run honours the state. A run without `--resume` clears it and retries every
  item, with no history record.

### A run started from the wrong position

An incremental source starts where its checkpoint says. If a run skipped rows it should have
read, or the checkpoint was pushed too far by a bad value, inspect it and its history:

```
$ janus checkpoint show --source-id semantics_clean_producer --history 10
semantics_clean_producer - checkpoint
  field     updated_at            strategy  max_value
  value     2026-09-30T00:00:00Z  updated   2026-09-17T06:00:00+00:00
  run       local-20260917-060000-semantics_clean_producer
  state     data/metadata/semantics/clean_producer/checkpoints/current.json

history (last 3 of 3, newest first)
  recorded_at                decision  advanced  previous              stored                run
  2026-09-17T06:00:00+00:00  advanced  no        2026-09-15T23:40:00Z  2026-09-30T00:00:00Z  local-20260917-060000-semantics_clean_producer
  2026-09-16T06:00:00+00:00  advanced  no        2026-09-14T23:10:00Z  2026-09-15T23:40:00Z  local-20260916-060000-semantics_clean_producer
  2026-09-15T06:00:00+00:00  advanced  yes       -                     2026-09-14T23:10:00Z  local-20260915-060000-semantics_clean_producer
```

Read the `decision` column. The `advanced` flag is currently `no` for every advancing run after a
source's first, a known defect in what the store records. Then move the checkpoint, with a reason:

```
$ janus checkpoint set --source-id semantics_clean_producer --to 2026-09-15T23:40:00Z --reason 'run of 2026-09-17 read a future-dated row'
semantics_clean_producer: updated_at 2026-09-30T00:00:00Z -> 2026-09-15T23:40:00Z (max_value; this moves the checkpoint backwards)
  history   data/metadata/semantics/clean_producer/checkpoints/history/manual-20261003T102223Z-ops-tester.json
  state     data/metadata/semantics/clean_producer/checkpoints/current.json
  operator  ops-tester
  reason    run of 2026-09-17 read a future-dated row
```

The next run advances from the value set. `janus checkpoint clear --source-id S --reason R`
forgets the checkpoint instead, and the next run starts with none.

What `set` does **not** do:

- It does not check that the value corresponds to real upstream data. It checks only that the
  next run can compare it with what it extracts: the value must be the same kind as the stored one
  (a timestamp, a number or text). Moving a `max_value` checkpoint *forward* past rows that were
  never extracted makes every later run skip them. The history entry records who did it and why;
  it does not make the move right.
- It runs nothing and does not touch bronze. The next run does the extraction.

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

Order-24 adds one refusal to single-source execution. `--execute` and
`--ingest-raw-to-bronze` exit `2`, with the holder on stderr, when another process holds the
source's state lock. Nothing ran, so no run metadata exists. `run-all` at N=1, the shipped
default, is the unchanged sequential runner and keeps its output bytes; see
[concurrency and overlap](#concurrency-and-overlap).

Config hashes change where YAML changed, which is expected and visible in `config_versions`.
