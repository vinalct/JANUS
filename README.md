
![alt text](janus.png)

# JANUS

[![CI](https://github.com/vinalct/janus/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vinalct/janus/actions/workflows/ci.yml)

JANUS is a metadata-driven Apache Spark framework for ingesting Brazilian federal open data in a reproducible, safe, and extensible way.

In simple terms, it helps turn public datasets published as APIs, catalogs, and bulk files into repeatable, traceable data products. Technically, it combines typed YAML source contracts, a planner, strategy families (`api`, `catalog`, `file`), optional narrow hooks, Spark normalization, and metadata, checkpoint, lineage, and validation persistence.

## Objective

The goal is not to collect every public dataset in Brazil. The goal is to prove that heterogeneous federal sources can be ingested without turning the repository into a collection of one-off scripts.

JANUS is also the implementation core of a post-graduation data engineering project, so the repository is designed to be both academically defensible and practically runnable.

The current phase is intentionally focused on:

- federal public data;
- batch ingestion;
- reproducible local, containerized, and cluster-shaped execution;
- explicit `raw`, `bronze`, and `metadata` zones;
- configuration-first onboarding, with source-specific code only when needed.

Deeper context: [PRD](docs/PRD.md), [foundation](docs/foundation.md), and [architecture](docs/architecture.md).

## How It Works

1. A source is declared in a YAML file under `conf/sources/`, optionally grouped by domain folders and a top-level `sources:` list.
2. The registry validates the contract and loads a typed configuration.
3. The planner resolves the strategy family, variant, and optional hook.
4. The strategy extracts the source and preserves raw artifacts.
5. Spark reads the handoff, applies shared normalization, and writes structured bronze outputs.
6. JANUS persists validation results, checkpoints, lineage, and run metadata in the metadata zone.
7. After those authoritative JSON artifacts exist and Spark has stopped, JANUS best-effort
   projects the terminal run into `metadata.runs` and emits an OpenLineage lifecycle event.

Control flow:

`source YAML -> registry -> planner -> strategy -> raw -> Spark normalization/write -> metadata`

More detail: [strategy patterns](docs/strategy-patterns.md), [source onboarding](docs/source-onboarding.md), and [implementation notes](docs/implementations/).

## Current Scope

The checked-in source contracts already cover the three core strategy families.

| Source group | Family | Notes |
| --- | --- | --- |
| Portal da Transparencia APIs | `api` | Contratos, emendas, gastos com cartoes, licitacoes, orgaos, renuncias fiscais, and servidores; disabled by default; require `TRANSPARENCIA_API_TOKEN` for live execution |
| dados.gov.br catalog | `catalog` | Catalog listing and detail sources; disabled by default; require `DADOS_GOV_BR_API_TOKEN` for live execution |
| IBGE SIDRA | `api` | `ibge_pib_brasil` and `ibge_agro_abacaxi_pronaf`; disabled by default; public execution path; use the `ibge.sidra_flat` hook |
| INEP microdata | `file` | `inep_censo_escolar_microdados`; disabled by default; exercises ZIP extraction and configured CSV read options |
| Receita Federal CNPJ | `file` | Declarative CNPJ entity ingestions for empresas, estabelecimentos, socios, simples, and lookup tables; disabled by default; use WebDAV discovery, archive filtering, and explicit schemas |
| `federal_open_data_example` | `api` | Contract example only; points to `example.invalid` and is not a live source |

Output zones live under `data/`:

- `data/raw`
- `data/bronze`
- `data/metadata`

## How To Reproduce

The recommended reproducible path is containerized local mode.

Pinned baseline:

- Python `3.13.12`
- PySpark `4.0.1`
- OpenJDK `17`
- Iceberg runtime `1.10.1`

Prerequisite: Docker Compose, `docker-compose`, `podman compose`, or `podman-compose`.

### 1. Build and start the local runtime

```bash
make bootstrap
make up
```

### 2. Validate the runtime without starting Spark

```bash
make run-local-config
```

This validates the `local` environment profile and prepares the runtime paths.

### 3. Validate Spark startup

```bash
make run-local
```

This boots the pinned local Spark runtime and verifies the Iceberg configuration.

### 4. Run the test suite

```bash
make test
```

### 5. Plan one configured source deterministically

```bash
make shell
janus \
  --environment local \
  --source-id ibge_pib_brasil \
  --include-disabled
```

This does not hit the live source. It validates the source contract, resolves the strategy, and prints a stable JSON planning summary.

### 6. Execute one live source end to end

```bash
make shell
janus \
  --environment local \
  --source-id ibge_pib_brasil \
  --include-disabled \
  --execute
```

This runs planning, extraction, Spark normalization, bronze writing, validation, and metadata persistence. Outputs are materialized under the configured `raw`, `bronze`, and `metadata` paths.

Notes:

- Token-backed sources need their env vars available at runtime.
- `--include-disabled` is required for the checked-in live sources because they are disabled by default for safer development.
- The `local` and `cluster` profiles live in `conf/environments/`; `cluster` serves both
  the JDBC and the REST catalog variants, selected by an env overlay
  ([choosing a catalog](docs/reproducibility.md#choosing-the-catalog-jdbc-or-rest)).
- Host-local execution is also possible if you match the pinned toolchain; see [reproducibility](docs/reproducibility.md).

### 7. Run many sources in dependency order

```bash
make shell
janus run-all --environment local
```

`run-all` executes enabled sources once, in validated dependency order. An `iceberg_rows`
request input declares the source that produces the table it reads, and the registry
validates that graph at load, so a consumer is never scheduled before its producer. Filters
select roots only — `--tag` and `--domain` are each repeatable — and the required upstreams
come along automatically. A failed source skips only its dependents; independent sources
continue. The aggregate is printed as JSON and persisted under
`<metadata>/pipelines/<pipeline_run_id>/summary.json`.

The command runs one batch and exits. Scheduling stays outside JANUS core: use cron, or the
optional Dagster adapter, which renders one task per source over the same graph and delegates
to the same planner and executor.

```bash
pip install 'janus[dagster]'
```

Core JANUS never imports Dagster; only requesting the adapter crosses that boundary.

Full option reference, summary schema, exit codes, retry/backfill ownership, and
troubleshooting: [batch orchestration guide](docs/orchestration.md). A self-contained,
runnable A → B plus independent C project — CLI and Dagster, including a deliberate-failure
variant — lives in [examples/orchestration/](examples/orchestration/README.md).

### Query run observability

Completed runs are projected into the append-only Iceberg table `metadata.runs`, in the same
catalog as bronze, and each observer lifecycle emits OpenLineage `START`, `COMPLETE`, or `FAIL`.
Both destinations are additive to the per-run JSON and best-effort: JSON remains authoritative,
and an emission problem is logged without changing run status or exit code. Consequently, a
missing table row is not proof that a run did not happen.

Table columns, the tested failed-run and quality-breach queries, OpenLineage transports,
retention limits, retry semantics, and troubleshooting are in the
[queryable observability guide](docs/queryable-observability.md).

## Documentation Map

- [Architecture guide](docs/architecture.md): control flow and extension boundaries.
- [Strategy patterns](docs/strategy-patterns.md): when to use `api`, `file`, or `catalog`.
- [Source onboarding](docs/source-onboarding.md): how to add a new source without script sprawl.
- [Batch orchestration guide](docs/orchestration.md): `run-all`, the source DAG, pipeline summaries, and the Dagster adapter.
- [Orchestration example](examples/orchestration/README.md): a runnable dependency graph and an operations runbook.
- [Queryable observability guide](docs/queryable-observability.md): `metadata.runs`, tested SQL, OpenLineage, and operational limits.
- [Reproducibility guide](docs/reproducibility.md): environment profiles, container workflow, and cluster-shaped runs.
- [Implementation notes](docs/implementations/): component-level notes for planner, strategies, Spark I/O, quality, lineage, and source integrations.
