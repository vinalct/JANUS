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

- `conf/environments/local.yaml`
- `conf/environments/cluster.yaml`

Each file supports `${ENV_VAR:-default}` interpolation through `janus.utils.environment.expand_env_vars`, which means the repository can define sane defaults while still allowing environment-specific overrides outside source code.

The companion example env files are:

- `conf/environments/local.env.example`
- `conf/environments/cluster.env.example`

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
