# Maintenance

## Source locks and scheduling

Do not run `janus maintain` while a run of the same source is in flight. This is
an operational requirement until order-24 provides source-state locking. Cron
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

## OpenLineage event files

Declare `maintenance.lineage_events.older_than_days` in the environment's maintenance
policy. The policy is required; an absent or invalid maintenance block is refused.
Inspect the plan before applying it:

```sh
janus maintain --environment local --zone lineage
janus maintain --environment local --zone lineage --apply --format json
```

The default invocation is a dry run. Both invocations persist evidence under
`<metadata>/maintenance/<maintenance_run_id>.json`. The `lineage` zone does not start
Spark, and `--source-id` does not filter its shared files.

For a `file` OpenLineage transport, maintenance uses the same configured directory
and metadata-zone containment helper as emission. The CLI resolves it from the
shared runtime metadata root. An `http`, `disabled`, or absent OpenLineage transport
produces no local candidates and does not walk a directory.

Files are selected by the UTC day in `events-YYYY-MM-DD.ndjson`. Maintenance never
opens an event file to inspect its contents and never uses mtime to determine age.
A file is eligible only when its day is strictly before the current UTC day minus
`older_than_days`. The exact cutoff day stays within the retention window.

Today's file is protected with reason `todays_file`. The file with the most recent
parsed day is protected with reason `most_recent_file`, even when that day is older
than the policy. When both rules protect the same path, text and JSON evidence list
the path once with both reasons. JSON records carry a `protected` list with `zone`,
`target`, and `reasons` fields.

Unparseable filenames, including `events-undated.ndjson`, are preserved and recorded
as skipped items with reason `invalid_event_filename`. Deletions use the shared
metadata file executor: each result records status, count, and actual removed
bytes. A missing file is skipped; a removal failure is recorded and other items
continue. A repeated apply has no remaining deletion candidates, while malformed
files still appear as skipped evidence.

## Concurrent and back-dated events

Appends to today's file and to the most recent file survive lineage maintenance.
The latter protection also covers the midnight transition when today's file has
not been created yet.

The filename comes from the event's own `eventTime`. A run emitting an event with a
back-dated `eventTime` into a day file older than the retention window, concurrently
with maintenance, can lose that event when the file is neither today's nor the
most recent. Maintenance does not inspect payloads or detect every active append.
Do not run `maintain` during an extraction. The metadata/raw lock seam does not
lock shared lineage event files.
