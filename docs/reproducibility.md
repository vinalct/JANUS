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

The command loads the local environment profile, prepares runtime paths, and prints a JSON summary of resolved settings.

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
- prints a stable JSON planning summary.

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
JDBC one, by decision: the raw zone is the reproducible artifact, and bronze is a function of it.
If your `data/bronze/iceberg` predates the catalog switch, the tables are still on disk but the
new catalog has no rows describing them, so a run will not find them.

Rebuild, from inside the container (`make shell`):

```bash
rm -rf data/bronze/iceberg data/metadata/iceberg-catalog     # 1. drop bronze and the catalog db
python -m janus.main --environment local --source-id <id> \
  --include-disabled --ingest-raw-to-bronze --with-spark      # 2. replay the raw zone
```

Step 2 rehydrates the existing raw artifacts and re-materializes bronze through exactly the same
`BronzeMaterializer` a live run uses — no re-extraction, no network. If the raw zone is empty too,
replace it with a full run (`--execute` instead of `--ingest-raw-to-bronze`).

Deleting `data/metadata/iceberg-catalog` is safe because the catalog holds only pointers: table
identifiers and the current metadata location. The data and the Iceberg metadata itself live under
the warehouse. Deleting it without also deleting the warehouse leaves orphaned files that the next
run ignores, which is why step 1 removes both.

## What the current CLI does and does not do

Be explicit about the current project state.

Today `src/janus/main.py` is a reproducible runtime entry point for:

- loading environment profiles;
- materializing runtime paths;
- validating Spark bootstrap;
- planning one configured source;
- executing one configured source end to end with `--execute`;
- resuming interrupted extraction state with `--resume`;
- loading already-preserved raw artifacts into a requested bronze table with `--ingest-raw-to-bronze --bronze-table ...`.

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

The CLI still runs one selected source at a time. Cross-source scheduling, dependency orchestration, and production job control belong outside this entry point for now.

## Safe reruns and stable outputs

Reproducibility is not only about installing the same packages. It is also about getting predictable runtime behavior.

JANUS already enforces several pieces of that:

- output roots come from checked-in environment profiles plus explicit env overrides;
- raw, bronze, and metadata zones are resolved through `StorageLayout`;
- checkpoints are persisted through a monotonic checkpoint store;
- run metadata and lineage artifacts are written under the metadata zone;
- logs are structured and redact secret-bearing fields by default.

If you are comparing runs across environments, compare:

- the environment profile used;
- the explicit env-var overrides;
- the run id and start time;
- the resolved storage roots;
- the source config path and version.

## Known limits at this stage

A few facts are important for anyone reproducing the project today:

- `conf/sources/example/example_source.yaml` is a contract example, not a live integration. Its URL points to `example.invalid` on purpose.
- Some Spark-backed tests are skipped automatically when `pyspark` is not available outside the container runtime.
- The containerized workflow is the most stable path because it already pins Python, Java, and dependency installation in one place.

If you keep those limits explicit, contributors can reproduce the current state without false expectations.
