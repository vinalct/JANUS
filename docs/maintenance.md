# Retention and maintenance

JANUS keeps history on purpose. A full refresh keeps the previous bronze snapshot so time travel
and rollback work. Every run also leaves run metadata, lineage, checkpoint history and
a validation report. The runs table gains a row per terminal emission, and the OpenLineage file
transport writes one file per UTC day. None of that shrinks by itself. `janus maintain` is how it
shrinks: it applies a retention policy you declare, shows you the plan first, and records every
deletion as evidence.

## What `janus maintain` is and is not

- **A mechanism, not a schedule.** JANUS core contains no scheduler. cron or a Dagster job invokes
  `janus maintain` (see [Scheduling](#scheduling-with-cron-or-dagster)).
- **A dry run unless you say `--apply`.** The default invocation prints the plan, persists it as a
  maintenance record, and deletes nothing. `--apply` executes the same planner's output.
- **Policy is declared, never defaulted.** The command reads the environment profile's
  `maintenance:` block. A profile without one is refused with exit `2`:
  `Environment config has no 'maintenance' block; 'janus maintain' applies only a declared policy`.
- **Only this command reads the block.** `run` and `run-all` never consult it, so a typo in the
  policy cannot fail an ingestion run.
- **The write path is untouched.** The bronze writer, the materializer and the run observer never
  expire, compact or delete. No Iceberg table property makes the engine expire snapshots on its
  own. A package-scoped sweep
  ([`test_no_deletion_outside_maintenance.py`](../tests/unit/toolchain/test_no_deletion_outside_maintenance.py))
  fails the build if a retention procedure or deletion appears outside `janus/maintenance/`.

```text
janus maintain [--environment E] [--project-root P] [--dry-run | --apply]
               [--zone bronze|metadata|lineage|runs-table|raw]... [--source-id S]...
               [--format text|json]
```

| Zone | What it retains | Engine |
|---|---|---|
| `bronze` | Iceberg snapshots of every registry bronze table, disabled sources included | Spark |
| `metadata` | `runs/`, `lineage/`, `checkpoints/history/` and `validations/` per source, plus `pipelines/*/summary.json` | none |
| `lineage` | OpenLineage `events-<day>.ndjson` files of the `file` transport | none |
| `runs-table` | whole `emitted_at_day` partitions of `metadata.runs` | Spark |
| `raw` | run-scoped raw prefixes, only when `maintenance.raw.enabled: true` | none |

Without `--zone`, every zone is selected, with `raw` only when it is enabled. `--source-id`
narrows `bronze`, `metadata` and `raw`; the lineage files and runs-table partitions are shared, so
they cannot be filtered by source, and `--zone runs-table --source-id …` is refused. Selecting
only `metadata`, `lineage` or `raw` never starts Spark. Selecting `bronze` or `runs-table` starts
one session lazily and stops it in `finally`.

Exit codes: `0` when everything planned succeeded, including an empty plan; `1` when any item
failed or a record-level step failed (a Spark teardown failure included); `2` when nothing was
attempted: an absent or invalid `maintenance:` block, an unknown source, `--zone raw` while raw is
disabled, `--source-id` with `--zone runs-table`, or a flag that belongs to `run`.

In the container, use the Makefile targets. Both refuse unless `ENVIRONMENT` is typed on the
command line or exported, so `maintain-apply` can never fall back to the Makefile's `local`
default:

```sh
make maintain-dry-run ENVIRONMENT=local
make maintain-dry-run ENVIRONMENT=local MAINTAIN_ARGS="--zone metadata --source-id ibge_pib_brasil"
make maintain-apply ENVIRONMENT=local
```

## The policy

`conf/environments/local.yaml` and `conf/environments/cluster.yaml` ship this block, every key
written out. The `cluster-rest` overlay reads `cluster.yaml`, so it ships the same policy.
`local-hadoop.yaml` has no block on purpose, and `maintain` refuses on it.

```yaml
maintenance:
  item_timeout_seconds: 1800
  bronze:
    retain_last: 3
    older_than_days: 30
    remove_orphan_files: false
    orphan_older_than_days: 3
    compact:
      enabled: false
      target_file_size_mb: 512
  metadata:
    keep_last_runs: 20
    older_than_days: 90
  lineage_events:
    older_than_days: 90
  runs_table:
    older_than_days: 365
  raw:
    enabled: false
    keep_last_runs: 3
    older_than_days: 365
```

| Key | Rule | Shipped | Meaning |
|---|---|---|---|
| `item_timeout_seconds` | optional; positive, finite | `1800` | Bound for each Spark procedure or statement. A timeout fails the item, asks Spark to cancel it, and says the operation may still be running. |
| `bronze.retain_last` | required; integer ≥ 1 | `3` | The newest snapshots of each table that are always kept. |
| `bronze.older_than_days` | required; integer ≥ 0 | `30` | Only snapshots older than this are expired. |
| `bronze.remove_orphan_files` | optional boolean | `false` | Also run `remove_orphan_files` on each table. |
| `bronze.orphan_older_than_days` | optional; integer ≥ 1 | `3` | Only unreferenced files older than this are orphan candidates. |
| `bronze.compact.enabled` | optional boolean | `false` | Run `rewrite_data_files` after expiration. |
| `bronze.compact.target_file_size_mb` | integer ≥ 1; required when compaction is enabled | `512` | Target data-file size for compaction. |
| `metadata.keep_last_runs` | required; integer ≥ 1 | `20` | The newest runs of each source whose files are always kept. |
| `metadata.older_than_days` | required; integer ≥ 0 | `90` | Only history records older than this are candidates. |
| `lineage_events.older_than_days` | required; integer ≥ 0 | `90` | Only day files older than this many UTC days are candidates. |
| `runs_table.older_than_days` | required; integer ≥ 0 | `365` | Only partitions older than this many UTC days are deleted. |
| `raw.enabled` | optional boolean | `false` | Opt in to raw retention. |
| `raw.keep_last_runs` | required when enabled; at least `bronze.retain_last` | `3` | The newest successful runs of each source whose raw prefixes are kept. |
| `raw.older_than_days` | required when enabled; integer ≥ 0 | `365` | Only raw prefixes older than this are candidates. |

The five sub-blocks are required even when you accept every default inside them. Values are
strict: an integer must be a YAML integer and a flag a YAML boolean, so `"3"` and `"false"` are
refused. So is a `${VAR:-3}` expansion, because expansion produces a string. That is why the
shipped profiles hold literal values: changing retention is an edit to a tracked file, reviewed
like code. An unknown key is refused naming the key. The first problem found is the one reported.

Ages are strict. Something exactly at the cutoff stays. `older_than_days: 0` means "older than
the instant the command started"; the cutoff passed to Spark is always a full timestamp, never a
procedure's own default. A bronze snapshot is expired only when it is outside the newest
`retain_last` **and** older than the cutoff. Lineage day files and runs-table partitions are aged
by UTC day.

The header of every plan prints the policy digest, a SHA-256 over the resolved values. Both
shipped profiles resolve to `sha256:7312626fb3bd3cd825aff0e9c40f29ecb9c6cc21c8479c49a9719a73a05cbc4e`.

### Per-source bronze override

A source can replace the two snapshot-window keys for its own table:

```yaml
outputs:
  bronze:
    path: data/bronze/receita_federal/cnpj_estabelecimentos
    format: iceberg
    retention:
      retain_last: 10
      older_than_days: 90
```

Both keys are required. `retain_last` must be a positive integer and `older_than_days` a
non-negative integer. The block is accepted only on `outputs.bronze` with `format: iceberg`;
anywhere else it is a load error (`outputs.raw.retention: is only supported for outputs.bronze`,
`outputs.bronze.retention: requires format='iceberg'`). The override replaces the profile's
`retain_last` and `older_than_days` for that table only; orphan removal and compaction still
follow the profile. Sources that share a table through `shared_with` must declare the same
override, or none. If they disagree, that table is skipped as `retention_conflict`, naming both
declarations, and the other tables proceed. No shipped source declares an override.

## The protected set

Nothing below can become a deletion candidate. The planner computes the set, and the executor
checks it again before each deletion, so even a hand-built plan cannot remove a state file. The
reason shown in the plan and the record is in parentheses.

- **State** (`state_file`): each planned source's `checkpoints/current.json`,
  `dead_letters/current.json` and `extraction_progress.json`, whether or not the file exists yet.
- **Bronze**: the current snapshot of every table, read from its `main` ref, after a rollback too
  (`current_snapshot`); the newest `retain_last` snapshots (`retain_last`); snapshots inside the
  age window (`within_window`).
- **Lineage events**: today's file (`todays_file`), and the file with the most recent day present
  (`most_recent_file`), even when that day is older than the policy.
- **The latest runs** (`keep_last_runs`): the newest `keep_last_runs` runs of each source, by the
  `started_at` inside `runs/<run_id>.json`, ties broken by run id, and every file of those runs in
  `runs/`, `lineage/`, `checkpoints/history/` and `validations/`. Each record counts for the
  `source_id` it names, so two sources that declare one metadata directory keep their runs
  separately.
- **Live progress** (`live_progress`): any run whose id matches the `run_id=` segment of a live
  extraction progress record, compared after the same sanitization the raw writer applies.
- **Anything without a trustworthy age or owner**: a record whose timestamp is missing, null,
  unparseable or not timezone-aware (`unaged`); a record that cannot be read, whose `run_id` does
  not match its file name, or that sits in a shared directory and names none of the sources
  declaring it (`unreadable`). Age never comes from a file's mtime.
- **Shared pipeline summaries under `--source-id`** (`source_filter`): a summary cannot be
  attributed to the selected sources, so a filtered run leaves every summary alone.
- **Runs-table partitions** inside the window (`within_window`).
- **Raw, when enabled**: the newest `raw.keep_last_runs` successful runs of each source
  (`keep_last_runs`); a run whose status is unknown because its run record is missing, running,
  malformed or contradictory (`unknown_run_status`); live progress (`live_progress`); every prefix
  of a source whose live progress record has no raw prefix (`legacy_progress_prefix`); a flat,
  pre-prefix raw layout (`flat_layout_present`).

Some files are never listed at all: atomic-write `.tmp` leftovers, files outside the history
families above (dead-letter history, for example), and directories.

Some items are recorded as skipped rather than failed: `absent_table` (a registry table that was
never written), `retention_conflict`, `snapshot_read_failed: <type>`, `invalid_event_filename`,
`invalid_raw_prefix`, `already_absent` (removed between planning and deletion) and
`source_locked`.

## Zone by zone

**Bronze.** The table set is every Iceberg bronze target in the registry, keyed by the writer's
own identifier derivation, never a warehouse listing. For each table, in this order:

```sql
CALL `janus`.system.expire_snapshots(table => '`ns`.`table`', older_than => TIMESTAMP '…', retain_last => N)
CALL `janus`.system.remove_orphan_files(table => '`ns`.`table`', older_than => TIMESTAMP '…')   -- only when enabled
CALL `janus`.system.rewrite_data_files(table => '`ns`.`table`', options => map('target-file-size-bytes', '…'))   -- only when enabled
```

The executor reads the snapshots table before and after expiration and records the snapshot ids
that actually expired. Compaction writes a new snapshot of its own, so a compacted table keeps
`retain_last + 1` snapshots. One table's failure does not stop the others.

**Metadata.** One walk per declared metadata directory, plus the shared `<metadata>/pipelines/`.
Files are removed one at a time and each removal is measured before it happens. A failed removal
is recorded and the next file proceeds; empty directories stay.

**Lineage events.** For a `file` OpenLineage transport, maintenance resolves the same configured
directory as emission, through the same metadata-zone containment rule. An `http`, `disabled` or
absent transport has no local files. Files are selected by the UTC day in
`events-YYYY-MM-DD.ndjson`; maintenance never opens a file or reads its mtime. A file is eligible
only when its day is strictly before today minus `older_than_days`. Unparseable names, such as
`events-undated.ndjson`, are kept and recorded as `invalid_event_filename`.

**Runs table.** Old days go in one statement, then the table's snapshots expire with the same
cutoff and `bronze.retain_last`:

```sql
DELETE FROM `janus`.`metadata`.`runs` WHERE emitted_at < TIMESTAMP '<UTC midnight cutoff>';
CALL `janus`.system.expire_snapshots(table => '`metadata`.`runs`', older_than => TIMESTAMP '<same cutoff>', retain_last => 3);
```

The delete is partition-aligned, so it drops whole `emitted_at_day` partitions and rewrites no
surviving file. If the delete fails, expiration is skipped. The pass never de-duplicates retried
rows; that stays the published queries' job. What retention does to query windows is in
[queryable observability](queryable-observability.md#retention).

**Raw.** See [Raw retention is opt-in](#raw-retention-is-opt-in).

## What a dry run prints

A dry run with a candidate, from the fixture warehouse the CI smoke seeds
(`make test-maintenance-smoke`), whose profile overrides the bronze policy to `retain_last: 1`,
`older_than_days: 0`. Recorded on 2026-10-06:

```text
$ make maintain-dry-run ENVIRONMENT=local \
    MAINTAIN_ARGS="--zone bronze --source-id full_refresh_history_unpartitioned --project-root data/metadata/maintenance-smoke"
janus maintain — DRY RUN (nothing will be deleted)
environment: local    policy: sha256:0c8c0f8ca3aa9dcac1c2c2b430fbfdcae21e5c9f5643f259b515e51993952ad8    lock: none
zones: bronze

bronze
  bronze_full_refresh_history.unpartitioned_fixture
    expire_snapshots  older_than=2026-10-06T12:20:23.375122+00:00  retain_last=1
      would expire 5 snapshots: 4083521127184152782, 6265174292015691267, 7432748858934937766, 7877257162043268241, 9216039607224097948

protected (not candidates)
  bronze  bronze_full_refresh_history.unpartitioned_fixture#1813141971757592012  current_snapshot
```

The shipped `local` policy on one CNPJ table of a working copy, the same day. Nine snapshots, none
of them old enough:

```text
$ make maintain-dry-run ENVIRONMENT=local \
    MAINTAIN_ARGS="--zone bronze --source-id receita_federal__cnpj__cnaes_full_refresh"
janus maintain — DRY RUN (nothing will be deleted)
environment: local    policy: sha256:7312626fb3bd3cd825aff0e9c40f29ecb9c6cc21c8479c49a9719a73a05cbc4e    lock: none
zones: bronze

nothing would be deleted in: bronze (no retention candidates)

protected (not candidates)
  bronze  bronze__receita_federal.cnpj_cnaes#1188051418261715645  within_window
  bronze  bronze__receita_federal.cnpj_cnaes#1758147972221899554  within_window
  bronze  bronze__receita_federal.cnpj_cnaes#1952317059964641498  within_window
  bronze  bronze__receita_federal.cnpj_cnaes#2378567533407233999  retain_last
  bronze  bronze__receita_federal.cnpj_cnaes#360366831132879611  within_window
  bronze  bronze__receita_federal.cnpj_cnaes#3775944155659606843  retain_last
  bronze  bronze__receita_federal.cnpj_cnaes#6160722927759736017  within_window
  bronze  bronze__receita_federal.cnpj_cnaes#7716782744043520066  within_window
  bronze  bronze__receita_federal.cnpj_cnaes#810811524196593688  current_snapshot
```

Reading the output:

- The header names the mode, the environment, the policy digest and the lock in use.
- When `metadata` or `raw` is selected and no source lock is held, the header carries the
  no-lock warning shown under [Operational rules](#operational-rules).
- When a plan contains orphan-file removal, the header adds
  `⚠ do not remove orphan files while an extraction is in flight`, and the item is marked `⚠`.
- Each selected zone lists its items with their arguments, or says
  `nothing would be deleted in: <zone> (no retention candidates)`. Skipped items show
  `skipped: <reason>`.
- `protected (not candidates)` lists every protected target once, with all its reasons.

`--format json` prints exactly the record that is persisted. With JSON, the no-lock warning goes
to stderr so stdout stays identical to the file.

## The maintenance record

Every invocation, dry run included, writes one record atomically to
`<metadata>/maintenance/<maintenance_run_id>.json` under the shared metadata root
(`data/metadata/maintenance/` with the shipped profiles). The id is
`maintenance-<UTC second>-<first eight hex digits of the plan digest>`.

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `maintenance_run_id`, `environment` | identity of the invocation |
| `dry_run` | `true` for a plan, `false` for an apply |
| `zones`, `source_ids` | the selection; `source_ids: []` means unrestricted |
| `policy_digest` | SHA-256 of the resolved policy |
| `plan_digest` | SHA-256 of the sorted actions and arguments; a dry run and an apply of the same inventory share it |
| `lock` | the source lock in use; `"none"` until implement one |
| `started_at`, `ended_at`, `duration_seconds` | timing |
| `zone_summaries` | per zone: items planned, applied, skipped and failed, files removed, bytes removed |
| `items` | per item: zone, target, action, `status` (`planned`, `applied`, `skipped`, `failed`), `detail` with the arguments and any `skipped_reason`, `removed_count`, `removed_bytes`, `expired_snapshot_ids`, `failure_type`, `failure_message`, `duration_seconds` |
| `protected` | every protected target with its reasons |
| `failures` | record-level failures with a `stage`: `maintenance`, `spark_cleanup` or `interrupted` |

A count the executor cannot know stays `null` rather than being guessed. Snapshot expiration, for
example, reports files removed but not bytes. Failure messages are bounded and credential-redacted.
One item's failure is recorded on that item and the rest still run. An interrupted apply persists
a partial record whose unfinished items stay `planned`. A second `--apply` straight after the first
has `items: []`. Each item also emits one structured log event: `maintenance_item_planned`,
`maintenance_item_applied`, `maintenance_item_skipped` or `maintenance_item_failed`.

A maintenance run is not a source run. It is not projected into `metadata.runs`.

## Raw retention is opt-in

The raw zone is the reproducibility record, so raw retention is off in every shipped profile, and
`--zone raw` is refused (exit `2`, naming `maintenance.raw.enabled: true`) until you enable it.
When enabled, the candidates are run-scoped prefixes, `runs/ingestion_date=<date>/run_id=<run>/`,
older than `raw.older_than_days` by the date in the directory name. The raw guards in the
protected set apply, and `raw.keep_last_runs` must be at least `bronze.retain_last`. The profile
is refused otherwise, so every retained bronze snapshot keeps the raw it was built from.

Deleting a run's raw prefix removes the only local way to rebuild that run's bronze with
`--ingest-raw-to-bronze`. Keep raw for at least as long as you want to be able to rebuild any
bronze snapshot you retain; see
[reproducibility](reproducibility.md#retention-and-what-stays-authoritative).

## Operational rules

### Source locks and scheduling

Do not run `janus maintain` while a run of the same source is in flight. This is
an operational requirement until provides source-state locking. Cron
and other schedulers must serialise maintenance with extraction and replay for
the same sources; separate schedules alone do not prevent a long run from
overlapping maintenance.

The current `NullMaintenanceLock` acquires nothing. Every maintenance record
reports `lock: "none"`, and both dry-run and apply output show this warning when
`metadata` or `raw` is selected (on stderr for JSON output):

```text
warning: no source lock is held — do not run maintain while a run of the same source is in flight
```

The command accepts an injected `MaintenanceLock` by argument. It acquires each
source before collecting metadata/raw inventory, holds acquired locks through
execution, and releases them in `finally`. A refused acquisition leaves that
source unread and untouched, with one skipped item per selected metadata/raw
zone carrying `source_id` and `skipped_reason: "source_locked"`. Other sources
proceed, and contention alone exits 0. Shared pipeline summaries are preserved
when any source is locked. The record reports the injected lock's `name`.

Bronze, shared lineage event files and the runs table do not acquire source
locks. The scheduling requirement for concurrent back-dated lineage emission
below still applies when a real source lock becomes available.

### Orphan files

Iceberg cannot tell a data file a writer has not committed yet from an orphan. Do not enable
`remove_orphan_files` for a run that may overlap a long extraction, and keep
`orphan_older_than_days` at three days or more, which is Iceberg's own guidance. It ships off.

### Roll back before `maintain`, not after

Once a snapshot is expired, `VERSION AS OF` and rollback to it fail. If you might need yesterday's
bronze, roll back first. The dry run lists every snapshot id it would expire.

### Concurrent and back-dated events

Appends to today's file and to the most recent file survive lineage maintenance.
The latter protection also covers the midnight transition when today's file has
not been created yet.

The filename comes from the event's own `eventTime`. A run emitting an event with a
back-dated `eventTime` into a day file older than the retention window, concurrently
with maintenance, can lose that event when the file is neither today's nor the
most recent. Maintenance does not inspect payloads or detect every active append.
Do not run `maintain` during an extraction. The metadata/raw lock seam does not
lock shared lineage event files.

## Scheduling with cron or Dagster

Both examples write the plan first and apply it only when the plan step succeeded. The apply
re-reads the inventory at its own instant, so its record, not the dry run's, is the evidence of
what was removed.

**cron.** Share one lock file between the ingestion entries and the maintenance entry, so the two
can never overlap:

```cron
# Ingestion, as before, under the shared lock.
0 2 * * *   cd /srv/janus && flock /var/lock/janus-cluster.lock janus run-all --environment cluster
# Weekly retention under the same lock: plan, then apply.
30 4 * * 0  cd /srv/janus && flock /var/lock/janus-cluster.lock sh -c 'janus maintain --environment cluster --format json > /var/log/janus/maintain-plan.json && janus maintain --environment cluster --apply --format json > /var/log/janus/maintain-apply.json'
```

**Dagster.** The adapter does not generate a maintenance op; JANUS core stays free of
scheduling. Write a small job next to the adapter's definitions. Its ops join the adapter's
`janus_source_execution` pool. With the pool limited to one at run granularity, as the
[orchestration example](../examples/orchestration/README.md#overlap-and-concurrency-policy)
configures it, maintenance and ingestion runs cannot overlap.

```python
import subprocess

from dagster import DefaultScheduleStatus, In, Nothing, ScheduleDefinition, job, op

from janus.adapters.dagster import SOURCE_EXECUTION_POOL

MAINTAIN = ["janus", "maintain", "--environment", "cluster", "--project-root", "/srv/janus"]


@op(pool=SOURCE_EXECUTION_POOL)
def maintain_plan() -> None:
    # Exit 1 (an item failed) or 2 (nothing attempted) fails the op.
    subprocess.run([*MAINTAIN, "--dry-run", "--format", "json"], check=True)


@op(pool=SOURCE_EXECUTION_POOL, ins={"after_plan": In(Nothing)})
def maintain_apply() -> None:
    subprocess.run([*MAINTAIN, "--apply", "--format", "json"], check=True)


@job
def janus_maintenance():
    maintain_apply(after_plan=maintain_plan())


weekly_maintenance = ScheduleDefinition(
    job=janus_maintenance,
    cron_schedule="30 4 * * 0",
    execution_timezone="America/Sao_Paulo",
    default_status=DefaultScheduleStatus.STOPPED,
)
```

Add `janus_maintenance` and `weekly_maintenance` to your `Definitions`. The schedule ships
stopped, like the example's ingestion schedule, so an operator turns it on deliberately.

## Known limitations

- **A back-dated event can be lost.** A run emitting an event with a back-dated `eventTime` into
  an already-expired day file, concurrently with `maintain`, can lose that event when the file is
  neither today's nor the most recent. Maintenance does not open event files, and the source lock
  does not cover them. Do not run `maintain` during an extraction.
- **Raw retention depends on metadata retention.** A raw prefix's status comes from its run
  record. Once metadata retention has removed that record, the prefix has no status to read, so it
  is protected as `unknown_run_status` and stays forever. Aggressive raw retention therefore needs
  longer metadata retention: keep `metadata.older_than_days` at or above `raw.older_than_days`.
  With the shipped values (90 and 365 days), turning on `raw.enabled` alone would remove almost
  nothing, because most run records are gone long before their raw becomes eligible.
