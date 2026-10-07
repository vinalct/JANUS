# JANUS Reproducibility Guide

Reproducibility in JANUS is a product requirement, not a nice-to-have. A contributor should be able to understand which runtime is expected, which settings are environment-driven, and what commands produce the same baseline behavior on another machine.

This guide covers the execution paths that exist in the repository today.

## Pinned baseline

The checked-in baseline is:

- Python `3.13.12`
- PySpark `4.0.1`
- OpenJDK `17`
- Apache Iceberg Spark runtime `1.10.1`

These come from:

- `pyproject.toml`
- `requirements.txt`
- `docker/Dockerfile`
- `conf/environments/local.yaml`
- `conf/environments/cluster.yaml`

## Environment-driven configuration

JANUS keeps runtime settings in checked-in environment profiles:

| Profile | Catalog | Warehouse | What it is for |
|---|---|---|---|
| `conf/environments/local.yaml` | `jdbc`, a SQLite file under `data/metadata/iceberg-catalog/` | `data/bronze/iceberg` | the default. Every `make` target, CI job and test session uses it |
| `conf/environments/cluster.yaml` | `jdbc` over Postgres, or `rest` via an env overlay | `s3://…` through `S3FileIO` | the cloud-agnostic profile, run against the compose stack |
| `conf/environments/local-hadoop.yaml` | `hadoop` | `data/bronze/iceberg-hadoop` | a throwaway, for host-local experiments only |

`local-hadoop.yaml` exists so the old filesystem catalog stays reachable, not so it stays
supported. It is referenced by no Makefile target, no CI job, no compose service and no test,
and it writes to a **different** warehouse root on purpose — a Hadoop run and a JDBC run must
never arrive at the same table directories. The Hadoop catalog commits by filesystem rename:
single-writer, no concurrency control, and unsafe on object storage.

Each file supports `${ENV_VAR:-default}` interpolation through `janus.utils.environment.expand_env_vars`, which means the repository can define sane defaults while still allowing environment-specific overrides outside source code.

The companion example env files are:

- `conf/environments/local.env.example`
- `conf/environments/cluster.env.example`

### The Iceberg catalog keys

Everything about the catalog lives under `spark.iceberg` in the profile. `catalog_type` is
**required** whenever that block is present, and an unrecognised value is refused by name rather
than defaulted — a profile that forgets the key fails loudly instead of quietly regressing to the
unsafe catalog.

| Key | Applies to | Meaning |
|---|---|---|
| `catalog_type` | all | `jdbc`, `rest`, or `hadoop`. Required; no default |
| `catalog_name` | all | the Spark catalog name (`janus`), which every `spark.sql.catalog.<name>.*` key is built from |
| `warehouse_dir` | all | a filesystem path or an `s3://…` URI under `jdbc`/`hadoop`; under `rest` it is an **identifier** the catalog resolves, not a location |
| `default_namespace` | all | the namespace bronze tables are created in (`bronze`) |
| `runtime_package` | all | the pinned Iceberg Spark runtime coordinate |
| `uri` | `jdbc`, `rest` | the JDBC URL, or the catalog service's base URL |
| `driver_package` | `jdbc` | the JDBC driver coordinate — `org.xerial:sqlite-jdbc` locally, `org.postgresql:postgresql` on the cluster |
| `credentials.user` / `credentials.password` | `jdbc` | the database login, from the environment only |
| `auth.token` / `auth.credential` / `auth.oauth2_server_uri` / `auth.scope` | `rest` | the REST spec's authentication surface, from the environment only |
| `object_store.*` | any warehouse on object storage | `io_impl` (`S3FileIO`), `io_package`, `endpoint`, `region`, `path_style_access` |

Two rules the profiles follow, and yours should too:

- **Credentials never appear in a tracked file.** Every one of them is an `${ENV_VAR:-}`
  expansion. `conf/environments/*.env` is gitignored; the `.env.example` files carry
  placeholders, and `make up-cluster` generates the real values into `conf/environments/cluster.env`
  on first use.
- **A warehouse that is a URI stays a URI.** `prepare_runtime` creates directories for the paths
  it resolves, but it never touches a value carrying a `scheme://` prefix — otherwise
  `s3://janus-bronze/warehouse` would become a local directory literally named `s3:`.

A `jdbc` profile pins one more Maven coordinate than a `hadoop` one did: the JDBC driver. Both
drivers are committed under `deps/`, seeded into the container's Ivy cache by `make seed-ivy`, and
sha256-verified in CI — so the test gate resolves nothing from Maven. That is the "one extra jar"
the move off Hadoop costs.

The source registry path is configured separately in `conf/app.yaml`, and the example source contract lives in `conf/sources/example/example_source.yaml`. The registry scans that tree recursively, so provider folders under `conf/sources/` are part of the supported layout.

## Data layout and runtime paths

At runtime, JANUS materializes explicit roots for:

- `raw`
- `bronze`
- `metadata`
- Spark warehouse
- Iceberg warehouse
- the local Ivy cache when configured

`janus.utils.environment.prepare_runtime(...)` creates those paths before the CLI continues, and `janus.utils.storage.StorageLayout` resolves configured source outputs against the active environment.

That means:

- source configs can keep repository-relative logical paths;
- local and cluster-shaped environments can relocate the physical storage roots;
- the execution code does not need hardcoded machine paths.

If the configured Spark warehouse or local Iceberg warehouse is not writable, JANUS relocates
that warehouse under a user-private fallback root. An explicit `JANUS_RUNTIME_SCRATCH_DIR` takes
precedence; otherwise JANUS uses `$XDG_RUNTIME_DIR/janus`, or a unique `janus-runtime-*` temporary
directory created once per process when `XDG_RUNTIME_DIR` is unset. JANUS refuses a fallback root
that is owned by another user or writable by group or other.

The Ivy jar cache is never relocated automatically because Spark loads executable code from it.
Set `JANUS_SPARK_IVY_DIR` to an explicitly chosen writable location if the configured cache cannot
be created. This is the only supported way to move the jar cache.

## Recommended workflow: containerized local mode

The repository is set up to make containerized local mode the default reproducible path.

### Prerequisites

Install one compose-capable engine:

- Docker Compose
- `docker-compose`
- `podman compose`
- `podman-compose`

### Build the runtime image

```bash
make bootstrap
```

This builds the `janus` image defined by `docker/docker-compose.yml`.

### Start the development container

```bash
make up
```

The image runs as uid/gid `1000:1000` by default. The Make targets set
`JANUS_CONTAINER_USER` to your host uid/gid so writable bind mounts retain host ownership.
Podman maps that identity through `keep-id`; Docker uses `nss_wrapper` when the selected uid has
no account in the image, giving the process and its children a user lookup without modifying
`/etc/passwd`. Set `JANUS_CONTAINER_USER=<uid>:<gid>` explicitly when invoking Compose directly
with Docker and a non-default identity.

The compose service mounts:

- `src/`
- `tests/`
- `conf/`
- `data/`
- `pyproject.toml`

Source code and config are mounted read-only; the data directory stays writable for runtime outputs.

### Validate the local profile without Spark startup

```bash
make run-local-config
```

This runs:

```bash
python -m janus.main --environment local
```

The command loads the local environment profile, prepares runtime paths, and prints a JSON summary of resolved settings. This validates the *environment profile* only; `make validate` validates the source registry.

### Validate the source registry

```bash
make validate
```

This runs:

```bash
python -m janus.main validate --environment local
```

It loads every source, disabled ones included, resolves the dependency graph, applies the
registry's semantic rules and plans every source. Then it checks the profile: its catalog type,
its OpenLineage transport, the Spark options it renders and the runtime paths it resolves. It
starts no Spark session and creates no directory; `--prepare` creates the runtime directories.
It exits `0` when everything resolves and `2` otherwise, with the loader's or the profile's
refusal printed verbatim on stderr. The rules are listed in
[source onboarding](source-onboarding.md#what-the-registry-checks-across-sources).

`make ci` runs the same command last. CI's fast job runs `python -m janus.main validate
--project-root .` on the host, with no engine installed.

### Validate the local profile and start Spark

```bash
make run-local
```

This runs:

```bash
python -m janus.main --environment local --with-spark
```

Use this when you want to verify that the pinned Spark runtime and Iceberg configuration are actually bootstrapping correctly.

### Run checks inside the container

```bash
make lint
make test
```

### Open a shell

```bash
make shell
```

That shell is the easiest place to run ad hoc CLI commands or inspect Spark interactively with `pyspark`.

### Stop the environment

```bash
make down
```

## Deterministic source planning

The CLI can also plan one configured source deterministically.

From inside the container shell, or from a host environment with JANUS installed, run:

```bash
python -m janus.main \
  --environment local \
  --source-id federal_open_data_example \
  --run-id run-20260409-demo \
  --started-at 2026-04-09T12:00:00+00:00
```

That command:

- loads the selected environment profile;
- prepares runtime paths;
- loads the requested source from the registry;
- resolves its strategy family and variant;
- prints a stable JSON planning summary, including the contract identity block loaded with the registry snapshot. `schema_version` is the SHA-256 of the contract file bytes and can be derived without a run.

Why pass `--run-id` and `--started-at` explicitly?

- it makes the planning output deterministic;
- it is easier to compare runs between machines;
- it avoids confusion about relative timing when documenting a run.

## Host-local workflow

Containerized local mode is the recommended baseline, but host-local execution is still possible if your machine already provides the pinned toolchain.

Minimum host requirements:

- Python `3.13`
- Java `17`
- project dependencies installed from the checked-in requirements

Typical host-local commands:

```bash
python -m janus.main --environment local
python -m janus.main --environment local --with-spark
```

If you use host-local mode, keep the versions aligned with the checked-in baseline. Otherwise "it works on my laptop" stops being meaningful.

## Cluster-compatible mode

`conf/environments/cluster.yaml` describes the cluster-shaped runtime contract. It is not tied to one cloud vendor.

The important assumptions are:

- the Spark master is provided by environment config;
- storage roots are absolute paths appropriate for the cluster runtime;
- the same JANUS package and pinned dependencies are available in that environment;
- the Iceberg catalog and warehouse settings are driven by the cluster profile.

A typical validation command in a cluster-like environment is:

```bash
python -m janus.main --environment cluster
```

And when Spark should be created as part of the validation:

```bash
python -m janus.main --environment cluster --with-spark
```

For reproducible cluster work:

- start from `conf/environments/cluster.env.example`;
- provide the `JANUS_*` values through your scheduler or runtime environment;
- keep storage roots stable across reruns;
- do not move cluster-specific secrets or absolute paths into checked-in source configs.

### Running the cluster stack locally

The repository ships the cluster profile's dependencies as an **opt-in** compose profile, so
`make up` stays a one-container experience and nothing below runs unless you ask for it.

All published cluster ports bind to loopback by default; the MinIO console is available at
`http://127.0.0.1:9001`. To expose the stack on another interface, set the host-side Compose
variable explicitly, for example `JANUS_CLUSTER_BIND_ADDRESS=0.0.0.0 make up-cluster` (or
`make up-cluster-rest`).

```bash
make up-cluster                                        # MinIO + Postgres + the janus service
make run-cluster RUN_ARGS="--source-id <id> --execute" # a real run against object storage
make test-cluster                                      # the catalog suites, on that stack
make down-cluster                                      # stop it; the named volumes survive
```

`make up-cluster` does three things before starting anything, all of them once:

- **`cluster-secrets`** generates `conf/environments/cluster.env` with random Postgres and MinIO
  credentials. The file is gitignored and mode-600. Delete it and re-run to rotate.
- **`seed-ivy`** copies the vendored jars from `deps/` into the container's Ivy cache.
- **`seed-cluster-jars`** fetches `iceberg-aws-bundle` (59.8 MiB) and verifies it against a pinned
  sha256. This is the one jar the repository does **not** commit: it is larger than everything
  under `deps/` put together, and no CI job needs it, because no CI job starts this stack. The
  download happens once per machine and is skipped on every later `up-cluster`.

The named volumes outliving `down-cluster` is deliberate: the catalog rows and the bronze objects
must survive the containers that wrote them, which is the whole claim an atomic catalog makes.
Wipe them explicitly with `make down-cluster` followed by
`<engine> volume rm janus-postgres-data janus-minio-data`.

The raw and metadata zones stay on the local volume in this profile. Only bronze moves to object
storage — that is what AC-3 asked for, and moving the other two is separate work.

## Choosing the catalog: JDBC or REST

Both profiles configure an **atomic** Iceberg catalog — that is the point of the storage layer,
and it is not the thing you choose between. What you choose is who opens the connection to the
metadata store.

| | `cluster` (default) | `cluster-rest` |
|---|---|---|
| `catalog_type` | `jdbc` | `rest` |
| Who talks to the metadata store | JANUS, over JDBC | a catalog service, over HTTP |
| Services the stack runs | MinIO + Postgres | MinIO + Postgres + Nessie |
| Client classpath | Iceberg runtime + Postgres driver + AWS bundle | Iceberg runtime + AWS bundle |
| `warehouse` means | the location `s3://janus-bronze/warehouse` | the identifier `janus`, resolved by the catalog |
| Credentials | a database login (`spark.iceberg.credentials`) | the REST spec's `token` / `credential` / `oauth2_server_uri` / `scope`, from the environment |
| Start it with | `make up-cluster` | `make up-cluster-rest` |
| Run against it with | `make run-cluster` | `make run-cluster-rest` |
| Re-run the catalog suites | `make test-cluster` | `make test-cluster-rest` |

Both read the **same** `conf/environments/cluster.yaml`. There is no second profile: every key
in that file is an environment expansion, and `conf/environments/cluster-rest.env.example`
overrides four of them. Bronze identifiers, schema, partitioning and the S3FileIO write path
are identical either way — the catalog coordinates commits, it does not decide what the data
looks like.

**Choose JDBC** — the default — when JANUS is the thing writing the tables. It needs no extra
service, it is atomic on any SQL database, and both Spark and `pyiceberg` speak it natively.
Locally the same code runs against a SQLite file, so a laptop exercises production commit
semantics with nothing running.

**Choose REST** when the catalog is somebody else's: a managed service (Glue, Snowflake Open
Catalog, R2), or a self-hosted one that several engines and teams already point at. The
argument is not atomicity — JDBC is atomic too — it is that credentials, storage layout and
access control stop being every client's business. Pointing JANUS at a managed catalog is then
`JANUS_ICEBERG_CATALOG_TYPE`, `JANUS_ICEBERG_CATALOG_URI`, and whichever of the auth variables
that service wants, all exported rather than written into a tracked file.

The self-hosted REST service in the compose stack is Nessie, chosen because it reaches a
working Iceberg REST catalog from a pinned image and environment variables alone, and stores
its state in the Postgres the `cluster` stack already runs. It runs unauthenticated: it is a
proof that the protocol swap is configuration, not a deployment blueprint.

## Re-materializing bronze after a catalog change

Local bronze data is disposable. There is **no migration** from the old Hadoop catalog to the
JDBC one, by decision: the raw zone is the reproducible artifact, and bronze is a function of it
for every run whose raw is retained (see
[Retention and what stays authoritative](#retention-and-what-stays-authoritative)).
If your `data/bronze/iceberg` predates the catalog switch, the tables are still on disk but the
new catalog has no rows describing them, so a run will not find them.

Rebuild, from inside the container (`make shell`):

```bash
rm -rf data/bronze/iceberg data/metadata/iceberg-catalog     # 1. drop bronze and the catalog db
python -m janus.main --environment local --source-id <id> \
  --include-disabled --ingest-raw-to-bronze \
  --bronze-table <namespace.table>                            # 2. replay the raw zone
```

Step 2 rehydrates the existing raw artifacts and re-materializes bronze through exactly the same
`BronzeMaterializer` a live run uses — no re-extraction, no network. `--bronze-table` is required;
to rebuild the table a live run writes, pass the `bronze_table` that
`python -m janus.main list --format json` prints for the source. If the raw zone is empty too,
replace step 2 with a full run (`--execute` instead of `--ingest-raw-to-bronze --bronze-table …`).

Deleting `data/metadata/iceberg-catalog` is safe because the catalog holds only pointers: table
identifiers and the current metadata location. The data and the Iceberg metadata itself live under
the warehouse. Deleting it without also deleting the warehouse leaves orphaned files that the next
run ignores, which is why step 1 removes both.

### Verifying the raw zone before a replay

Extraction writes a `<path>.sha256` sidecar beside every raw artifact, holding the digest of the
bytes it just persisted. A replay trusts those sidecars by default: it reads the digest instead of
re-hashing the file, which keeps replaying a multi-GB zone cheap. Add `--verify-checksums` to
re-hash every artifact and compare it with its sidecar before anything is materialized:

```bash
python -m janus.main --environment local --source-id <id> \
  --include-disabled --ingest-raw-to-bronze \
  --bronze-table <namespace.table> --verify-checksums
```

- **A mismatch fails the run** with `RawArtifactIntegrityError`. The message names the first
  mismatching artifact in sorted order, the sidecar digest and the computed digest, and ends
  `The raw zone has changed since extraction; bronze was not written.` Verification happens
  while the raw zone is rediscovered, before materialization, so no bronze snapshot is committed
  and the run is recorded as failed like any other rehydration error. One case is weaker: a
  catalog source whose request inputs are `iceberg_rows` opens its scoped request-input Spark
  session before rediscovery, so for that source "before Spark" means before materialization.
- **A verified replay says so.** On success, `checksums_verified: "true"` appears in the printed
  `raw_to_bronze_run` summary, in the run-metadata JSON's `run_attributes`, and in the lineage
  JSON's `extraction_metadata`. Without the flag the key is absent everywhere, and replay behaves
  exactly as it did before the flag existed.
- **All three families honour it.** API, file and catalog rediscovery read a digest through the
  same resolver.
- **A zone written before sidecars existed still replays.** An artifact with no sidecar has no
  recorded digest to compare against, so it is hashed and loaded as it always was. Verification
  can only prove what extraction recorded.
- **It belongs to replay.** `--verify-checksums` without `--ingest-raw-to-bronze` is an argument
  error (exit `2`), and `run-all` rejects it as an unrecognized argument.

The cost is one full read of the raw zone. Measured on 2026-10-02 over a Portal da Transparência
zone of 35,812 artifacts and 1,158.7 MiB, on a 16-core machine with NVMe storage: 16.95 s with a
mostly cold page cache (about 15 s per GiB) and 1.28 s warm, against 0.4–0.5 s to read the
sidecars alone. That is why the flag is opt-in. Use it when the raw zone may have changed since
extraction: copied between machines, restored from a backup, or edited by hand.

## What the current CLI does and does not do

Be explicit about the current project state.

`janus` is one command with eight verbs. `src/janus/main.py` only delegates to the verb table in
`src/janus/cli/dispatch.py`, and a command line that starts with an option is the `run` verb, so
every form in this guide works as it did before the other verbs existed.

`janus run` is a reproducible runtime entry point for:

- loading environment profiles;
- materializing runtime paths;
- validating Spark bootstrap;
- planning one configured source;
- executing one configured source end to end with `--execute`;
- resuming interrupted extraction state with `--resume`;
- loading already-preserved raw artifacts into a requested bronze table with `--ingest-raw-to-bronze --bronze-table ...`;
- re-hashing every raw artifact against its extraction-time checksum before that load, with
  `--verify-checksums` ([above](#verifying-the-raw-zone-before-a-replay)).

For example, a live framework run looks like:

```bash
python -m janus.main \
  --environment local \
  --source-id ibge_pib_brasil \
  --include-disabled \
  --execute
```

And a raw-to-bronze reload looks like:

```bash
python -m janus.main \
  --environment local \
  --source-id inep_censo_escolar_microdados \
  --include-disabled \
  --ingest-raw-to-bronze \
  --bronze-table bronze_inep.censo_escolar_microdados
```

The other seven verbs:

| Verb | What it is for | Runs a source? |
|---|---|---|
| `janus run-all` | executing the enabled sources once, in dependency order ([batch orchestration](orchestration.md)) | yes |
| `janus contract draft` | drafting a data contract from a raw run or a fixture ([data contracts](data-contracts.md)) | no |
| `janus validate` | checking that the registry loads, means something executable, and plans; with `--environment`, the profile too | no |
| `janus list` | listing sources, their dispatch, their state and the dependency graph | no |
| `janus dead-letters` | inspecting, releasing or replaying the items a run gave up on | only `replay --execute` |
| `janus checkpoint` | showing, setting or clearing where a source's next run starts | no |
| `janus maintain` | planning, and with `--apply` applying, the profile's declared retention ([retention and maintenance](maintenance.md)) | no |

Cross-source scheduling is `janus run-all`: it runs one batch in dependency order and
exits. Inspecting and correcting what a run leaves behind is the operator verbs below.
Calendars, triggers and whole-run retries stay outside JANUS: use cron, or the optional Dagster
adapter.

### Operator commands

`validate`, `list`, `dead-letters` and `checkpoint` start no Spark session, open no catalog and
send no request; `dead-letters replay --execute` is the exception, because it runs the source.
Each plans or lists disabled sources too, since state exists whether or not a source is enabled.
A refusal or an argument error exits `2` before anything is written. A `--format json` option,
where offered, prints one JSON document with sorted keys.

**`janus validate [--source-id S] [--format text|json] [--environment E [--prepare]]`** loads the
registry, resolves the graph, applies the
[semantic rules](source-onboarding.md#what-the-registry-checks-across-sources) and plans every
source with one planner against one registry snapshot. A registry the loader refuses prints the
loader's error on stderr and nothing on stdout. A source the planner refuses is reported on its
own line, and its peers are still planned. `--source-id` narrows the planning step, never the
semantic pass. The report holds no run id and no timestamp, so two runs print the same bytes.
Exit `0` with no issue.

**`janus list [--tag T … | --domain D …] [--family F] [--enabled-only] [--graph] [--format table|json]`**
prints one row per source, sorted by `source_id`: family, variant, extraction mode, enabled state,
hook, upstream ids and tags. The JSON form adds the name, domain, downstream ids and the bronze table
the source writes. Unlike `run-all`, a filter never adds upstreams; the `UPSTREAMS` column names
them. `--graph` prints the graph's own topological order, then every edge with the table and input
path that create it. An empty listing exits `0`. Only the registry is read.

**`janus dead-letters list|release|replay --source-id S`** works on the state a run leaves at
`<metadata>/dead_letters/current.json` when an item exhausts its retries.

- `list [--item-key K …]` prints every entry with its error and metadata. No state exits `0`.
- `release (--item-key K … | --all) --reason R` removes exactly those entries. It first writes a
  history record at `dead_letters/history/<UTC timestamp>-<recording run>.json`, holding the
  released entries whole, the remaining keys, the operator and the reason. Then it rewrites
  `current.json` atomically, or deletes it when nothing remains. An unknown key refuses the whole
  release. **Release runs nothing.** The next `janus --environment E --source-id S --execute
  --resume` (plus `--include-disabled` for a disabled source) retries the released items and keeps
  skipping the rest.
- `replay (--item-key K … | --all) --reason R` without `--execute` is a dry run: what a resume
  would retry, what stays skipped and where it would pick up, with nothing written. With
  `--execute` it releases, records `replay: "true"` in the history record, and runs exactly what
  `run --execute --resume` runs, returning that run's exit code. A disabled source needs
  `--include-disabled`.

**`janus checkpoint show|set|clear --source-id S`** works on `<metadata>/checkpoints/current.json`.

- `show [--history N]` prints the stored value, its field and strategy, the run that wrote it, and
  the last `N` history entries, newest first (default 5). A source with no checkpoint exits `0`.
- `set --to V --reason R` moves the checkpoint to `V`, backwards or forwards. It prints the
  previous value and the direction before it writes. `V` must be the same kind as the stored value
  (timestamp, number or text), and comparable by the store's own comparator, or it is refused: the
  next run compares the two to decide whether it advanced. The history entry is written first, with
  decision `reset`, then `current.json`. The next run advances from `V`.
- `clear --reason R` records the forgotten value in a `reset` history entry
  (`metadata.cleared: "true"`), then deletes `current.json`, so the next run starts with no
  checkpoint.

Operator history entries are named `checkpoints/history/manual-<UTC timestamp>-<operator>.json`.
`reset` is never a run's own decision, so the runs table's `checkpoint_decision` shows only what
runs decided; the history entry is the record of the operator's change.

Two known limits:

- A stored checkpoint that no longer matches the plan, for example after the contract moved
  `checkpoint_field`, is refused by `show`, `set` and `clear` alike, with the store's message.
  Clearing it still means deleting `checkpoints/current.json` by hand, with no history entry.
- On a **catalog** source, the resume after a release also re-extracts the inputs the first run
  completed, because the catalog family clears its progress record at the end of every run. The
  result is correct; the completed inputs cost a second extraction.

## Safe reruns and stable outputs

Reproducibility is not only about installing the same packages. It is also about getting predictable runtime behavior.

JANUS already enforces several pieces of that:

- output roots come from checked-in environment profiles plus explicit env overrides;
- raw, bronze, and metadata zones are resolved through `StorageLayout`;
- checkpoints are persisted through a monotonic checkpoint store, which only an operator's
  recorded `janus checkpoint set|clear` can move backwards;
- run metadata and lineage artifacts are written under the metadata zone, and stay there until
  `janus maintain` applies a declared retention policy;
- logs are structured and redact secret-bearing fields by default.

If you are comparing runs across environments, compare:

- the environment profile used;
- the explicit env-var overrides;
- the run id and start time;
- the resolved storage roots;
- the source config path and version.

## Retention and what stays authoritative

Two claims this guide rests on hold for exactly as long as declared retention keeps their
evidence. That is a precise bound, not a hole: only `janus maintain` removes anything, only
under a policy written in the environment profile, never as a side effect of a run, and every
removal is itself recorded under `<metadata>/maintenance/`. See
[retention and maintenance](maintenance.md).

**The per-run JSON is authoritative for as long as `maintenance.metadata` keeps it, and the
runs-table row outlives it by design.** A run's metadata, lineage, checkpoint-history and
validation JSON are the record of what happened. `maintenance.metadata.keep_last_runs` keeps
every file of each source's newest runs at any age (20 in the shipped profiles), and
`maintenance.metadata.older_than_days` keeps everything younger (90 days). Outside those runs,
each file goes once its own timestamp is older than the window. The run's row in `metadata.runs`,
which
`maintenance.runs_table.older_than_days` keeps for 365 days, is then the record that remains:
what ran, under which config and contract version, how it ended and what it wrote. The row's
links to the removed files stop resolving. Checkpoint, dead-letter and extraction-progress state
is never a retention candidate, so resuming and the next run never depend on retention.

**Bronze can be rebuilt from raw for every run whose raw is retained.** The re-materialization
above, and every `--ingest-raw-to-bronze` replay, reads the run's raw prefix. Raw retention is
off in every shipped profile (`maintenance.raw.enabled: false`), so today every extracted run
stays rebuildable. An operator who enables it decides how far back that holds with
`maintenance.raw.keep_last_runs` and `maintenance.raw.older_than_days`. The first is validated to
be at least `maintenance.bronze.retain_last`, so the raw behind each retained bronze snapshot
stays. A run whose raw was removed can no longer be rebuilt locally; re-extracting it is the only
way back, and the upstream may have changed since.

**Time travel reaches the snapshots `maintenance.bronze` keeps.** `VERSION AS OF` and rollback
work for the newest `retain_last` snapshots and anything younger than `older_than_days` (three
snapshots and 30 days in the shipped profiles), unless a source declares its own
`outputs.bronze.retention`. Roll back before running maintenance, not after.

## Known limits at this stage

A few facts are important for anyone reproducing the project today:

- `conf/sources/example/example_source.yaml` is a contract example, not a live integration. Its URL points to `example.invalid` on purpose.
- Some Spark-backed tests are skipped automatically when `pyspark` is not available outside the container runtime.
- The containerized workflow is the most stable path because it already pins Python, Java, and dependency installation in one place.

If you keep those limits explicit, contributors can reproduce the current state without false expectations.
