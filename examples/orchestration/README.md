# JANUS orchestration example and operations runbook

This project is an isolated, credential-free demonstration of the same source graph and
execution services used by `janus run-all` and the Dagster adapter. It never loads the
repository's production-like source registry.

```text
A (reference producer) ──iceberg_rows(reference_id)──> B (consumer)

C (independent date-window source)
```

Run them from the JANUS repository root. Each new invocation needs a new pipeline ID; the examples use explicit IDs
so the persisted evidence is easy to find.

For the option reference, the pipeline summary schema, exit codes, and the full troubleshooting
catalogue behind the diagnoses below, see the
[batch orchestration guide](../../docs/orchestration.md).

## What is isolated

The example owns `conf/app.yaml`, its three source definitions, a local fixture service, and
an `example` environment profile. The profile uses JANUS's existing JDBC catalog/session
construction with the repository's pinned Iceberg 1.10.1 and SQLite 3.53.2.1 jars; it does not
copy connection-building logic.

Every generated file is below `examples/orchestration/runtime/`:

- `data/raw/`, `data/bronze/`, and `data/metadata/` contain source artifacts and summaries;
- `catalog/` and `data/bronze/iceberg/` contain the SQLite catalog and Iceberg warehouse;
- `spark/` and `ivy/` contain local Spark state and copies of the vendored jars;
- `dagster/` contains the local Dagster instance; and
- `fixture/` contains the fixture readiness marker and logs.

For a genuinely fresh demonstration, use a fresh checkout or remove the ignored runtime with
your normal recoverable cleanup procedure. Do not reuse finalized pipeline IDs.

## Host setup and healthy fixture

Prerequisites are Python 3.13, Java 17, and a POSIX shell. The web UI package is pinned to the
same 1.13.22 release as the adapter. It remains example-only and is not a default JANUS
dependency.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dagster]' 'dagster-webserver==1.13.22'
python examples/orchestration/bootstrap.py
```

The bootstrap performs no network access. It prints the runtime root, Dagster home, and the
two copied jar paths. The expected jar basenames are:

```text
org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar
org.xerial_sqlite-jdbc-3.53.2.1.jar
```

Start the fixture in terminal 1:

```sh
python -u examples/orchestration/fixture_service.py \
  > examples/orchestration/runtime/fixture/service.log 2>&1
```

Its readiness file is `examples/orchestration/runtime/fixture/ready.json`; it records
`{"fail_a": false, "host": "127.0.0.1", "port": 8765}`. The service reads the checked-in
reference/detail payloads and synthesizes C deterministically from its requested window. It
requires no token or external endpoint.

## One CLI batch

In terminal 2, run all three enabled sources once:

```sh
janus run-all \
  --project-root examples/orchestration \
  --environment example \
  --pipeline-run-id cli-success-20260914-live \
  --started-at 2026-09-14T12:00:00Z
```

The command exits `0`. Its single stdout document and the persisted summary both contain this
stable portion (durations, source run IDs, and absolute paths vary):

```json
{
  "selection": {
    "included_upstream_ids": [],
    "root_ids": ["A", "B", "C"],
    "source_ids": ["A", "B", "C"]
  },
  "totals": {
    "failed": 0,
    "skipped": 0,
    "status": "succeeded",
    "succeeded": 3
  }
}
```

The complete persisted document is:

```text
examples/orchestration/runtime/data/metadata/pipelines/cli-success-20260914-live/summary.json
```

### Selector expansion: select B as the root

`consumer` belongs only to B. The dependency closure adds A, while unrelated C is excluded:

```sh
janus run-all \
  --project-root examples/orchestration \
  --environment example \
  --tag consumer \
  --pipeline-run-id cli-b-only-20260914 \
  --started-at 2026-09-14T12:05:00Z
```

The printed and persisted selection is:

```json
{
  "requested": {"domains": [], "tags": ["consumer"]},
  "root_ids": ["B"],
  "included_upstream_ids": ["A"],
  "source_ids": ["A", "B"]
}
```

## Dagster execution and graph

The local execution wrapper runs the native Dagster job in-process and then invokes JANUS's
terminal collector outside dependency-success gating. It prints and persists the same summary
schema as `run-all`:

```sh
python examples/orchestration/run_dagster.py \
  > examples/orchestration/runtime/dagster-success.json
```

The command exits `0`; `totals.status` is `succeeded`, and the pipeline ID is Dagster's run ID.
The summary is also stored under
`runtime/data/metadata/pipelines/<dagster-run-id>/summary.json`.

To inspect and launch the graph in Dagster's local UI:

```sh
export DAGSTER_HOME="$PWD/examples/orchestration/runtime/dagster"
dagster dev \
  -f examples/orchestration/definitions.py \
  -a defs \
  -h 127.0.0.1 \
  -p 3000
```

Open `http://127.0.0.1:3000`, select the `janus_sources` job, and open its graph. There is one
node for each configured source. Dagster-safe node names carry metadata with the original
`janus/source_id`; the only edge is A→B and C is disconnected. Automated tests assert that
node/edge contract, so the UI is not the sole evidence.

`janus_orchestration_example_daily` is registered for 06:00 in the explicit
`America/Sao_Paulo` timezone and has `DefaultScheduleStatus.STOPPED`. Activate it deliberately
from Automation → Schedules in the local UI. `dagster dev` runs the daemon needed for schedule
ticks and the adapter's terminal-summary sensors.

JANUS itself contains no polling loop. A host scheduler may instead call the one-shot
`janus run-all ...` command; it executes one batch, emits one summary, and exits. When using
cron, wrap it in the platform's exclusive-lock facility and generate a fresh pipeline ID per
invocation.

## Deliberate A failure

Stop the healthy fixture with Ctrl-C and restart terminal 1 in failure mode:

```sh
python -u examples/orchestration/fixture_service.py --fail-a \
  > examples/orchestration/runtime/fixture/service-fail-a.log 2>&1
```

Run a fresh CLI batch:

```sh
janus run-all \
  --project-root examples/orchestration \
  --environment example \
  --pipeline-run-id cli-fail-a-20260914 \
  --started-at 2026-09-14T12:10:00Z
```

The expected exit status is `1`. A records the fixture's HTTP 503 as a source failure; B never
calls its endpoint and is skipped with direct/root cause A; C still succeeds:

```json
{
  "sources": [
    {"source_id": "A", "status": "failed"},
    {
      "source_id": "B",
      "status": "skipped",
      "skip": {
        "direct_blocking_upstream_ids": ["A"],
        "root_failed_source_ids": ["A"]
      }
    },
    {"source_id": "C", "status": "succeeded"}
  ]
}
```

Run the same failure through Dagster:

```sh
python examples/orchestration/run_dagster.py \
  > examples/orchestration/runtime/dagster-fail-a.json
```

Dagster marks the run failed, blocks B natively, completes C, and the terminal collector emits
the same three JANUS statuses. The wrapper exits `1`.

## Following source evidence

Start with the pipeline summary. Each attempted source embeds its evidence paths. For source
`<source-id>` and source attempt `<source-run-id>`, the underlying files are:

```text
runtime/data/metadata/<source-id>/runs/<source-run-id>.json
runtime/data/metadata/<source-id>/lineage/<source-run-id>.json
runtime/data/metadata/<source-id>/validations/<source-run-id>.json
runtime/data/metadata/<source-id>/checkpoints/current.json
runtime/data/metadata/<source-id>/checkpoints/history/<source-run-id>.json
runtime/data/metadata/<source-id>/extraction_progress.json
```

Full-refresh sources in this example use `checkpoint_strategy: none`, so checkpoint files are
normally absent; extraction progress is still source-scoped. A skipped B has no attempt-level
metadata, lineage, quality, checkpoint, or progress write. Its reason lives in the pipeline
summary. Raw pages are under
`runtime/data/raw/<source-id>/runs/ingestion_date=<date>/run_id=<source-run-id>/`; committed tables
are `orchestration_example.reference_a`, `.details_b`, and `.independent_c` in the local
catalog.

## Overlap and concurrency policy

Two invocations must not race on source-scoped `extraction_progress.json`, checkpoints, raw
paths, or other metadata. An atomic Iceberg catalog protects table commits; it does not make
those other files safe for concurrent writers.

The example's `conf/dagster/dagster.yaml`, copied into `DAGSTER_HOME` by the bootstrap, sets
the adapter's shared `janus_source_execution` pool limit to one with `granularity: run`. Thus
only one run containing those source ops holds the pool at a time, serializing conflicting
scheduled/manual Dagster runs across the instance. The adapter also uses its conservative
single-process executor. Run monitoring frees abandoned slots after a bounded delay. Keep an
equivalent deployment-wide lock for cron or direct `run-all`; do not rely on the catalog lock.

## Retry and resume ownership

There are three different mechanisms:

- `extraction.retry` in source YAML owns bounded retries of an individual HTTP request inside
  one source attempt. The fixture sources set `max_attempts: 1` so failure evidence is quick.
- Dagster owns whole-source retries. Defaults are off. To demonstrate one bounded retry while
  the `--fail-a` fixture is active, run:

  ```sh
  python examples/orchestration/run_dagster.py --max-retries 1 \
    > examples/orchestration/runtime/dagster-retry-a.json
  ```

  A has attempts 1 and 2 under one Dagster pipeline ID. Each attempt has a distinct JANUS
  source run ID and `pipeline_attempt`; after the final failure B is skipped and C succeeds.
- `janus run-all --resume` consumes existing per-source `extraction_progress.json` and reuses
  already fetched raw pages after an interrupted extraction. It does not retry a completed
  pipeline, suppress required upstream execution, or change a source's configured window.
  Always provide a fresh `--pipeline-run-id` when invoking it.

Whole-source retry idempotency is not granted by Dagster. It depends on each source's existing
write mode and keys. These fixtures use full-refresh overwrite; production operators must
review append/merge keys, external side effects, and checkpoint state before enabling retries.

## Backfill runbook: configure the extraction window

C is the backfill example because it has an explicit, supported `date_window` request input.
For a seven-day backfill, edit C in `conf/sources/sources.yaml` before loading/planning the run:

```yaml
request_inputs:
  type: date_window
  start: 2026-08-01
  end: 2026-08-07
  step: day
```

The existing `window_start` and `window_end` parameter bindings send those bounds to the
fixture once per daily context. Then run only C with a fresh identity:

```sh
janus run-all \
  --project-root examples/orchestration \
  --environment example \
  --tag independent \
  --pipeline-run-id backfill-c-20260801-20260807-attempt-1 \
  --started-at 2026-09-14T12:20:00Z
```

Restore the default bounds after the run, or keep each backfill's source config in a separately
versioned deployment. Reload Dagster definitions after a config edit: the adapter correctly
rejects definition/worker snapshot drift.

`--started-at` is a logical planning/correlation timestamp only. Changing it does not alter
the extraction window. JANUS has no date-range/partition CLI and does not rewrite window YAML.
Do not combine `--resume` state produced for one window definition with a different window.

## Deployment assumptions

Every node/worker must load the same source registry snapshot and logical catalog/profile.
Workers need the source configuration and any referenced hook code. Multi-worker deployments
need genuinely shared raw and metadata paths; the local filesystem profile is intentionally a
single-host example, not a distributed-storage promise. Supply credentials through environment
variables, never YAML. Keep Spark/source concurrency conservative and configure the shared
Dagster pool at the deployment level. Runtime behavior branches on the catalog/profile contract,
not on vendor names.

## Failure diagnosis and recovery

- **Missing upstream declaration or producer:** registry loading fails before execution. Add a
  configured source with the declared ID and matching physical table, or correct
  `upstream_source_id`/table identity.
- **Disabled upstream:** an enabled consumer is invalid. Enable the producer explicitly or
  disable/narrow the consumer; selection never enables it implicitly and never accepts stale
  output as this pipeline's upstream result.
- **Dependency cycle:** the registry reports the concrete cycle and request-input provenance.
  Remove or redesign at least one `iceberg_rows` edge; no node runs.
- **Ambiguous producer:** undeclared multiple writers of one physical table are rejected. Give
  each source a unique table or make the deliberate co-writer contract symmetric with
  `shared_with`.
- **Runtime lookup failure:** if B cannot find A's committed table/column despite a valid static
  declaration, B fails at request-input loading. Inspect A's attempt evidence and actual catalog,
  correct profile/schema/access, reload definitions, and use a fresh pipeline ID.
- **Partial pipeline:** use the persisted summary to separate succeeded, failed, and skipped
  nodes. Repair the root failure, assess the successful write's mode/keys, then launch a new
  batch. The runner cannot roll back A after A commits successfully and B later fails; recovery
  is a forward repair/re-execution, not a pipeline transaction.

## Container execution

Build the normal image, then the example-only layer with the pinned Dagster web runtime:

```sh
docker build -t janus:local -f docker/Dockerfile .
docker build -t janus:orchestration-example \
  -f examples/orchestration/Dockerfile .
mkdir -p examples/orchestration/runtime
```

Run a healthy CLI batch entirely inside one container (fixture and JANUS share loopback):

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/examples/orchestration:/workspace/examples/orchestration" \
  -v "$PWD/deps:/workspace/deps:ro" \
  janus:orchestration-example sh -lc '
    set -eu
    python examples/orchestration/bootstrap.py
    python -u examples/orchestration/fixture_service.py \
      > examples/orchestration/runtime/fixture/container-service.log 2>&1 &
    fixture_pid=$!
    trap '\''kill "$fixture_pid" 2>/dev/null || true'\'' EXIT INT TERM
    while [ ! -f examples/orchestration/runtime/fixture/ready.json ]; do sleep 0.1; done
    python -m janus.main run-all \
      --project-root examples/orchestration \
      --environment example \
      --pipeline-run-id container-cli-success-20260914-live \
      --started-at 2026-09-14T12:30:00Z
  '
```

Replace the final `python -m janus.main ...` command with
`python examples/orchestration/run_dagster.py` to execute the Dagster path in the same image.
The expected terminal status is the same successful A/B/C summary shown above.

## Snapshot maintenance follow-up

Full-refresh overwrite preserves Iceberg history under the order-09 contract. This example does
not expire snapshots, compact files, or delete orphans: its `example` profile declares no
`maintenance:` block, so `janus maintain` refuses on it. A deployment that wants retention
declares the window in its profile and schedules `janus maintain`, dry run first, serialized with
ingestion through the same `janus_source_execution` pool. That work stays separate from
ingestion and orchestration; see [retention and maintenance](../../docs/maintenance.md).
