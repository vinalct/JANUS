
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
- The `local` and `cluster` profiles live in `conf/environments/`.
- Host-local execution is also possible if you match the pinned toolchain; see [reproducibility](docs/reproducibility.md).

## Documentation Map

- [Architecture guide](docs/architecture.md): control flow and extension boundaries.
- [Strategy patterns](docs/strategy-patterns.md): when to use `api`, `file`, or `catalog`.
- [Source onboarding](docs/source-onboarding.md): how to add a new source without script sprawl.
- [Reproducibility guide](docs/reproducibility.md): environment profiles, container workflow, and cluster-shaped runs.
- [Implementation notes](docs/implementations/): component-level notes for planner, strategies, Spark I/O, quality, lineage, and source integrations.
